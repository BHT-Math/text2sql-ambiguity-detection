import argparse
import os
import json
import time
import threading
import traceback
import logging  # Add logging import

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from bird_interact.llm.config import model_config
from bird_interact.llm.api_util import read_full_response
import os 

# Configure logging
def setup_logging(log_level=logging.INFO):
    """Configure logging with the specified level."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Suppress verbose logging from other libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('anthropic').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def new_directory(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def write_response(results, data_list, output_path):
    # This function seems designed for writing all results at once,
    # but the current multi-threaded approach writes incrementally.
    # Keep it for potential future use, but it's not called by collect_response_from_api.
    formatted_data = []
    # Assuming results is a list matching data_list
    for i, data in enumerate(data_list):
        if i < len(results):
            data["response"] = results[
                i
            ]  # Assuming results contains only the content part
            formatted_data.append(data)

    if output_path:
        directory_path = os.path.dirname(output_path)
        new_directory(directory_path)
        with open(output_path, "w") as f:
            for instance in formatted_data:
                f.write(json.dumps(instance, ensure_ascii=False) + "\n")


def api_request(messages, engine, client, backend, **kwargs):
    """
    Calls the underlying LLM endpoint depending on the 'backend'.
    Includes more robust error handling and retry logic.
    """
    retries = 6  # Max retries for transient errors
    retry_delay = 10 # Initial delay in seconds

    for attempt in range(retries):
        try:
            # Route by backend. Only OpenAI-compatible and Anthropic endpoints
            # are exercised to reproduce the paper; call_api_model() maps every
            # model to one of these two.
            if backend == "openai":
                reasoning = kwargs.get("reasoning", False)
                max_tokens = kwargs.get("max_tokens", 512) if not reasoning else 10000
                # Gemini's OpenAI-compat endpoint rejects frequency_penalty /
                # presence_penalty (returns 400 INVALID_ARGUMENT). Strip them
                # for Gemini engines; they're zero-default for everyone else,
                # so omitting is equivalent.
                _is_gemini = isinstance(engine, str) and engine.startswith("gemini-")
                _create_kwargs = dict(
                    model=engine,
                    messages=messages,
                    temperature=kwargs.get("temperature", 0),
                    max_tokens=max_tokens,
                    top_p=kwargs.get("top_p", 1),
                )
                # Gemini's OpenAI-compat endpoint rejects `"stop": null` with
                # 400 'Value is not a string: null' (OpenAI/vLLM tolerate it).
                # Omitting the field is semantically identical to stop=None for
                # all backends, so only include it when actually set.
                _stop = kwargs.get("stop", None)
                if _stop is not None:
                    _create_kwargs["stop"] = _stop
                if not _is_gemini:
                    _create_kwargs["frequency_penalty"] = kwargs.get("frequency_penalty", 0)
                    _create_kwargs["presence_penalty"] = kwargs.get("presence_penalty", 0)
                completion = client.chat.completions.create(**_create_kwargs)

                reasoning_content = getattr(
                    completion.choices[0].message, "reasoning_content", None
                )
                if reasoning_content is None:
                    reasoning_content = getattr(
                        completion.choices[0].message, "reasoning", None
                    )
                content = completion.choices[0].message.content
                token_usage = {
                    "completion_tokens": completion.usage.completion_tokens,
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }
                logging.debug(f"Token usage (OpenAI): {token_usage}")
                return reasoning_content, content, token_usage

            elif backend == "anthropic":
                # Claude 4+ models reject both temperature and top_p in the same request.
                # Only send temperature (top_p defaults are fine).
                anth_kwargs = {
                    "model": engine,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", 0),
                    "max_tokens": kwargs.get("max_tokens", 512),
                }
                if kwargs.get("stop") is not None:
                    anth_kwargs["stop_sequences"] = kwargs.get("stop")
                message = client.messages.create(**anth_kwargs)
                usage_data = message.usage
                token_usage = {
                    "prompt_tokens": usage_data.input_tokens,
                    "completion_tokens": usage_data.output_tokens,
                    "total_tokens": usage_data.input_tokens + usage_data.output_tokens,
                    "cache_creation_input_tokens": usage_data.cache_creation_input_tokens,
                    "cache_read_input_tokens": usage_data.cache_read_input_tokens,
                }
                content = message.content[0].text
                logging.debug(f"Token usage (Anthropic): {token_usage}")
                return None, content, token_usage
            else:
                raise ValueError(f"Unsupported backend: {backend!r}")

        except Exception as e:
            is_retryable = True
            if is_retryable and attempt < retries - 1:
                logging.error(f"ERROR: {e}")
                logging.info(
                    f"Retryable error detected. Waiting {retry_delay} seconds before retry..."
                )
                time.sleep(min(retry_delay, 40))
                retry_delay *= 1.5  # Exponential backoff (optional)
                continue  # Go to next attempt
            else:
                logging.error("Non-retryable error or max retries reached. Failing request.")
                # Re-raise the exception to be caught by worker_function
                raise e

    # Should not be reached if retries are exhausted and exception is raised
    return None, "Error: Max retries exceeded", None


def call_api_model(
    messages,
    model_name,
    temperature=0,
    max_tokens=2048,  # Default max_tokens used if not overridden
    top_p=1,
    frequency_penalty=0,  # Note: May not be supported by all backends
    presence_penalty=0,  # Note: May not be supported by all backends
    # timeout=10, # Timeout not directly used in current api_request logic
    stop=None,
    return_format="",  # Note: May not be supported by all backends
):
    """
    Sets up the correct backend client + model engine, then calls 'api_request'.
    """
    client = None
    backend = None
    reasoning = False
    # engine usually becomes the specific model ID/name for the API call
    engine = model_name  # Default assumption, override as needed below

    # --- Backend/Client Setup Logic ---
    #
    # Three routes, special cases first; everything else is OpenAI-compatible.
    #   1. Gemini 3.1 Flash Lite -- user-simulator encoder + Gemini detection
    #      row (Vertex Express by default, OpenAI-compat dev endpoint fallback).
    #   2. Claude Haiku -- independent detection re-judge (direct Anthropic API).
    #   3. else -- any OpenAI-compatible endpoint. The three paper backbones
    #      (GLM-4.5-Air, MiniMax-M2.5, Qwen3.5-122B) resolve here through
    #      config.py, and a reviewer can swap in any served model the same way:
    #      pass its key as --agent_model / --user_model / --user_encoder_model
    #      with the endpoint in config.py or OPENAI_API_BASE / OPENAI_API_KEY.

    # 1) Gemini 3.1 Flash Lite. Vertex Express (paid tier) avoids the free-tier
    # RPM/RPD throttling that would silently corrupt a multi-thousand-call
    # encoder run; keyed by NEW_GEMINI / new_gemini. The wrapper exposes an
    # OpenAI-shaped chat.completions.create(...), so the backend stays "openai"
    # and the Gemini-rejects-`stop:null` bug cannot recur. Fallback (no Vertex
    # key): developer OpenAI-compat endpoint with GEMINI_API_KEY.
    if model_name == "gemini-3-1-flash-lite":
        config = model_config.get("gemini-3-1-flash-lite", {})
        engine = config.get("model_id", "gemini-3.1-flash-lite-preview")
        vertex_key = os.environ.get("NEW_GEMINI") or os.environ.get("new_gemini")
        client = None
        if vertex_key:
            try:
                from bird_interact.llm.gemini_client import build_gemini_client
                client = build_gemini_client(api_key=vertex_key, backend="vertex_express")
                backend = "openai"
                logging.info("gemini-3-1-flash-lite: using Vertex Express backend (new_gemini)")
            except Exception as _e:
                logging.warning(
                    "Vertex Express init failed (%s: %s); falling back to "
                    "developer OpenAI-compat endpoint", type(_e).__name__, str(_e)[:160]
                )
                client = None
        if client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Neither NEW_GEMINI (Vertex Express) nor GEMINI_API_KEY set; "
                    "one is required for model_name='gemini-3-1-flash-lite'"
                )
            client = OpenAI(
                base_url=config.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai/"),
                api_key=api_key,
                timeout=600.0,
            )
            backend = "openai"
            logging.info("gemini-3-1-flash-lite: using developer OpenAI-compat endpoint (GEMINI_API_KEY)")

    # 2) Claude Haiku -- direct Anthropic API (reads ANTHROPIC_API_KEY). Used
    # only for the independent detection-recall re-evaluation reported in the
    # results; not required for the main 3x3 reproduction.
    elif model_name in ("claude-haiku-4-5", "claude-haiku-4-5-20251001"):
        engine = model_name
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                f"ANTHROPIC_API_KEY env var not set; required for model {model_name}"
            )
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        backend = "anthropic"

    # 3) Any OpenAI-compatible endpoint: the three paper backbones plus any
    # reviewer-supplied model. A registered key uses its config.py entry; an
    # unregistered name is treated as a custom model served at OPENAI_API_BASE
    # (engine = the name itself), so no code edit is needed to swap models.
    else:
        config = model_config.get(model_name)
        if config is None:
            config = {
                "model_id": model_name,
                "base_url": os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
                "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
            }
            logging.info(
                "model '%s' not in config.py; using generic OpenAI-compatible "
                "endpoint %s", model_name, config["base_url"]
            )
        engine = config.get("model_id", model_name)
        client = OpenAI(
            base_url=config.get("base_url", os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")),
            api_key=config.get("api_key", "EMPTY"),
            timeout=600.0,
        )
        backend = "openai"

    # Ensure client and backend were set
    if client is None or backend is None:
        raise RuntimeError(
            f"Could not configure client or backend for model: {model_name}"
        )

    kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stop": stop,
        "return_format": return_format,  # Pass along, api_request handles if supported
        "reasoning": reasoning,
    }
    # Call the API request function
    return api_request(messages, engine, client, backend, **kwargs)


def worker_function(task, data_list, output_path, lock, stop=None, max_tokens=2048):
    """
    Processes a single prompt with robust error handling and logging.
    Writes result (success or error) to the output file incrementally.
    """
    prompt, idx, model_name, return_format = task
    messages = [{"role": "user", "content": prompt}]
    reasoning_content = None
    token_usage = None
    # Default to error message in case of failure
    content = f"Error: Processing failed unexpectedly for index {idx}"
    processed_successfully = False  # Flag to track success
    current_thread_name = threading.current_thread().name

    try:
        logging.debug(f"Worker {current_thread_name} START processing index {idx} with model {model_name}")

        # Core API call logic
        response_tuple = call_api_model(
            messages,
            model_name,
            return_format=return_format,
            # You might need to pass temperature, max_tokens etc. here if
            # call_api_model doesn't get them from args or has fixed defaults
            # that need overriding per task.
            max_tokens=max_tokens, # Example if needed
            stop=stop,
        )

        # Validate response structure
        if isinstance(response_tuple, tuple) and len(response_tuple) == 3:
            reasoning_content, content_result, token_usage = response_tuple
            # Check if the content itself indicates an error (e.g., blocked response)
            if (
                content_result is None
                or "Model response blocked" in str(content_result)
                or "Error:" in str(content_result)
            ):
                # logging.warning(f"Worker {current_thread_name} received API-level error/block for index {idx}: {content_result}")
                logging.warning(f"Worker {current_thread_name} received API-level error/block for index {idx}: ...")
                content = str(content_result)  # Store the error message as content
                processed_successfully = (
                    False  # Treat API error/block as failure for success flag
                )
            else:
                content = content_result  # Store successful content
                processed_successfully = True  # Mark as successful API call
            logging.debug(f"Worker {current_thread_name} received response for index {idx}. Success: {processed_successfully}. Content snippet: {str(content)[:100]}...")
        else:
            # Handle cases where call_api_model returns unexpected format
            error_msg = f"Error: Unexpected response format from call_api_model for index {idx}. Type: {type(response_tuple)}, Value: {response_tuple}"
            logging.error(f"Worker {current_thread_name} {error_msg}")
            content = error_msg
            processed_successfully = False

    except Exception as e:
        # Catch ANY exception during the process (init, client creation, API call)
        logging.error(f"FATAL ERROR in worker {current_thread_name} for index {idx}: {e}")
        traceback.print_exc()  # Print full traceback for debugging
        content = f"Error: Exception during processing for index {idx}: {type(e).__name__} - {e}"
        processed_successfully = False

    # --- Write result (success or error) to file safely ---
    try:
        # Use lock to ensure thread-safe file writing
        with lock:
            # Open in append mode ('a')
            with open(output_path, "a", encoding="utf-8") as f:
                # Get corresponding original data item
                # Use a copy to avoid modifying the shared data_list object
                row = data_list[idx].copy()
                # Update with results or error messages
                row["response"] = content
                row["reasoning_content"] = (
                    reasoning_content if reasoning_content else ""
                )
                row["token_usage"] = (
                    token_usage if token_usage else {}
                )  # Use empty dict for consistency
                # Add index for final sorting
                row["_index"] = idx
                # Write as JSON line
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as write_e:
        # Log critical error if writing fails
        logging.critical(f"Worker {current_thread_name} FAILED to write result for index {idx} to file {output_path}: {write_e}")

    # Return index and success status (optional, useful for tracking)
    return idx, processed_successfully


def final_sort_jsonl_by_index(file_path):
    """
    Reads an existing JSONL file, sorts it by the '_index' field,
    removes the '_index' field, and overwrites the file.
    Handles potential errors during file reading/writing.
    """
    all_data = []
    try:
        logging.info(f"Attempting to read and sort file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as fin:
            for line_num, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if "_index" not in row:
                        logging.warning(
                            f"Warning: Missing '_index' in line {line_num+1}. Skipping line: {line}"
                        )
                        continue
                    all_data.append(row)
                except json.JSONDecodeError as json_err:
                    logging.warning(
                        f"Warning: Failed to decode JSON in line {line_num+1}. Error: {json_err}. Skipping line: {line}"
                    )
                    continue

        if not all_data:
            logging.warning("Warning: No valid data with '_index' found to sort.")
            return

        # Sort by '_index'
        all_data.sort(key=lambda x: x["_index"])
        logging.info(f"Successfully read {len(all_data)} records. Sorting complete.")

        # Overwrite the file, removing the '_index' field
        with open(file_path, "w", encoding="utf-8") as fout:
            for row in all_data:
                row.pop("_index", None)  # Remove index field
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        logging.info(f"Successfully overwrote {file_path} with sorted data.")

    except FileNotFoundError:
        logging.error(f"Error: File not found for sorting: {file_path}")
    except Exception as sort_err:
        logging.error(f"Error during final sorting of {file_path}: {sort_err}")


def collect_response_from_api(
    prompt_list,
    model_name,
    data_list,  # Pass the original data list to worker
    output_path,
    num_threads=8,
    start_index=0,
    return_format="",
    stop=None,
    max_tokens=2048,
):
    """
    Uses ThreadPoolExecutor to process prompts concurrently.
    Writes results incrementally and sorts the final file.
    """
    # Validate start_index
    if start_index < 0 or start_index >= len(prompt_list):
        logging.warning(
            f"Warning: start_index {start_index} is out of bounds (0-{len(prompt_list)-1}). Setting to 0."
        )
        start_index = 0

    # Prepare tasks only for the required range
    tasks = [
        (prompt_list[i], i, model_name, return_format)
        for i in range(start_index, len(prompt_list))
    ]

    if not tasks:
        logging.warning("No tasks to process based on start_index and prompt_list length.")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Check if output_path includes a directory
        new_directory(output_dir)

    # --- File Handling ---
    # If starting fresh (or start_index is 0), clear the file with 'w' mode first.
    # Otherwise, we rely on append ('a') mode in the worker.
    if start_index == 0 and os.path.exists(output_path):
        logging.info(f"Clearing existing output file: {output_path}")
        try:
            open(output_path, "w").close()
        except IOError as e:
            logging.warning(
                f"Warning: Error clearing output file {output_path}: {e}. Appending may lead to duplicates if run was interrupted."
            )

    # Lock for protecting the write operation in worker_function
    lock = threading.Lock()

    # --- Thread Pool Execution ---
    logging.info(f"Starting processing {len(tasks)} tasks with {num_threads} threads...")
    successful_tasks = 0
    failed_tasks = 0
    MULTI_THREAD = True
    if MULTI_THREAD:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit tasks
            futures = {
                executor.submit(worker_function, t, data_list, output_path, lock, stop=stop, max_tokens=max_tokens): t
                for t in tasks
            }

            # Process results as they complete
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing prompts"
            ):
                task_info = futures[future]  # Get original task info
                task_idx = task_info[1]
                try:
                    # Get result from worker (idx, success_flag)
                    result_idx, success_flag = future.result()
                    if success_flag:
                        successful_tasks += 1
                    else:
                        failed_tasks += 1
                        logging.warning(f"Task for index {result_idx} completed with failure.")
                except Exception as exc:
                    # Catch unexpected errors from the future itself (less likely with worker handling)
                    logging.error(
                        f"FATAL: Task for index {task_idx} generated an exception in future: {exc}"
                    )
                    failed_tasks += 1
    else:
        for task in tasks:
            worker_function(task, data_list, output_path, lock, stop=stop, max_tokens=max_tokens)


    logging.info(
        f"Processing complete. Successful tasks: {successful_tasks}, Failed tasks: {failed_tasks}"
    )

    # --- Final Sort ---
    # Perform a final sort of the output file based on '_index'
    if successful_tasks + failed_tasks > 0:  # Only sort if something was processed
        logging.info("Sorting the output file...")
        final_sort_jsonl_by_index(output_path)
    else:
        logging.info("No tasks were processed, skipping final sort.")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--prompt_path", type=str, default="input.jsonl")  # Make required
    args_parser.add_argument("--output_path", type=str, default="output.jsonl")  # Make required
    args_parser.add_argument(
        "--model_name", type=str, default="gemini-2.0-flash-001"
    )  # Sensible default
    args_parser.add_argument(
        "--num_threads", type=int, default=8
    )  # Add threads argument
    args_parser.add_argument("--return_format", type=str, default="")
    args_parser.add_argument("--start_index", type=int, default=0)
    args_parser.add_argument("--limit", type=int, default=None)
    args_parser.add_argument("--stop", type=str, default=None)
    # Add logging level argument
    args_parser.add_argument("--log_level", type=str, default="INFO",
                           choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                           help="Set the logging level")
    args = args_parser.parse_args()

    # Setup logging with the specified level
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level)

    # --- Load Data ---
    try:
        data_list = load_jsonl(args.prompt_path)
        if not data_list:
            logging.error(
                f"Error: Prompt file {args.prompt_path} is empty or could not be loaded."
            )
            exit(1)
        prompts = [data["prompt"] for data in data_list]
        logging.info(f"Loaded {len(prompts)} prompts. First prompt: {prompts[0][:100]}...")
    except FileNotFoundError:
        logging.error(f"Error: Prompt file not found: {args.prompt_path}")
        exit(1)
    except Exception as load_err:
        logging.error(f"Error loading prompts from {args.prompt_path}: {load_err}")
        exit(1)

    # --- Apply Limit (if any) ---
    # Limit affects the list *before* passing to collect_response_from_api
    effective_data_list = data_list
    effective_prompts = prompts
    if args.limit is not None and args.limit > 0:
        logging.info(f"Applying limit: processing first {args.limit} prompts.")
        effective_data_list = data_list[: args.limit]
        effective_prompts = prompts[: args.limit]
        if args.start_index >= args.limit:
            logging.error(
                f"Error: start_index ({args.start_index}) >= limit ({args.limit}). No prompts to process."
            )
            exit(1)

    # --- Start API Collection ---
    collect_response_from_api(
        effective_prompts,  # Use potentially limited list
        args.model_name,
        effective_data_list,  # Pass corresponding data list
        args.output_path,
        num_threads=args.num_threads,  # Use argument
        start_index=args.start_index,
        return_format=args.return_format,
        stop=args.stop,
    )

    logging.info("Script finished.")
