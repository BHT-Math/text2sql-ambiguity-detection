"""Gemini API client.

Two backends:
- **Vertex Express Mode (default, 2026-05-14)**: `google.genai.Client(vertexai=True,
  api_key=...)` with the `new_gemini` API key from `.env`. Wrapped with an
  OpenAI-compatible shim so existing call sites (`client.chat.completions.create(...)`)
  keep working without refactor. Express Mode uses Vertex's quota tier (much higher
  daily caps than the developer API free tier).
- **Developer API (legacy fallback)**: `openai.OpenAI(base_url=generativelanguage...)`
  with the older `gemini_api` / `gemini_api2` keys. Subject to the 500-req/day
  free-tier cap. Kept reachable via `build_openai_compat_client()` for tools that
  haven't been ported.

The default `build_gemini_client()` returns the Vertex Express wrapper.

Usage (unchanged for callers):
    from gemini_client import build_gemini_client, DEFAULT_MODEL_ID
    client = build_gemini_client()
    response = client.chat.completions.create(
        model=DEFAULT_MODEL_ID,
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
        max_tokens=512,
    )
    text = response.choices[0].message.content
    in_t = response.usage.prompt_tokens
    out_t = response.usage.completion_tokens

Caveats vs. raw OpenAI:
- Pass `seed` via `extra_body={"seed": N}` exactly as before — the wrapper
  extracts it and routes it to genai's `GenerateContentConfig.seed`.
- The wrapper retries transient errors (429/5xx/timeout) up to `max_retries=15`
  with exponential backoff (via tenacity), matching the prior OpenAI SDK behavior.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeoutError

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_MODEL_ID = "gemini-3.1-flash-lite-preview"

KNOWN_MODELS = {
    "gemini-3.1-flash-lite-preview": {"input_per_1m": 0.25, "output_per_1m": 1.50},
    "gemini-3.1-flash": {"input_per_1m": 0.30, "output_per_1m": 2.50},
    "gemini-2.5-flash": {"input_per_1m": 0.30, "output_per_1m": 2.50},
    "gemini-2.5-flash-lite": {"input_per_1m": 0.10, "output_per_1m": 0.40},
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI-response-shape stubs — just enough surface for our call sites.
# ---------------------------------------------------------------------------

class _Message:
    __slots__ = ("content", "role")
    def __init__(self, content, role="assistant"):
        self.content = content
        self.role = role


class _Choice:
    __slots__ = ("message", "finish_reason", "index")
    def __init__(self, message, finish_reason=None, index=0):
        self.message = message
        self.finish_reason = finish_reason
        self.index = index


class _Usage:
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens")
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _ChatCompletion:
    __slots__ = ("choices", "usage", "model")
    def __init__(self, content, prompt_tokens, completion_tokens, finish_reason=None, model=None):
        self.choices = [_Choice(_Message(content), finish_reason=finish_reason)]
        self.usage = _Usage(prompt_tokens, completion_tokens)
        self.model = model


# ---------------------------------------------------------------------------
# Vertex Express wrapper.
# ---------------------------------------------------------------------------

class _Completions:
    """Mimics `openai.resources.chat.completions.Completions`."""

    def __init__(self, genai_client, max_retries=15, hard_timeout_s=90.0):
        self._client = genai_client
        self._max_retries = max_retries
        self._hard_timeout_s = hard_timeout_s
        # Long-lived thread pool so we don't pay setup cost per call.
        self._pool = ThreadPoolExecutor(max_workers=1)

    def create(self, *, model, messages, temperature=0.0, max_tokens=2048,
               extra_body=None, **_ignored):
        """OpenAI-compatible chat.completions.create() over Vertex Express genai."""
        # Lazy import — keeps gemini_client importable in environments where
        # google-genai isn't installed (e.g., for callers that only use the
        # OpenAI-compat fallback).
        from google.genai import types as gtypes
        from google.genai import errors as gerrors

        # ----- translate OpenAI messages → genai contents + system_instruction
        system_parts = []
        contents = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                contents.append(gtypes.Content(
                    role="user",
                    parts=[gtypes.Part(text=content)],
                ))
            elif role == "assistant":
                contents.append(gtypes.Content(
                    role="model",
                    parts=[gtypes.Part(text=content)],
                ))
            else:
                raise ValueError(f"Unsupported role: {role}")

        # ----- build config
        cfg_kwargs = dict(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_parts:
            cfg_kwargs["system_instruction"] = "\n\n".join(system_parts)
        if extra_body and "seed" in extra_body:
            cfg_kwargs["seed"] = int(extra_body["seed"])
        config = gtypes.GenerateContentConfig(**cfg_kwargs)

        # ----- retry loop with hard wall-clock timeout per attempt
        # The genai SDK's http_options.timeout is not always honored when the
        # underlying httpx socket hangs (observed 2026-05-14: process stayed
        # blocked in wait_woken for 8+ minutes despite a 60s timeout setting).
        # We enforce a hard wall-clock timeout via a thread pool — if the call
        # doesn't return in self._hard_timeout_s seconds, we abandon it and
        # let the retry loop try again.
        last_exc = None
        for attempt in range(self._max_retries + 1):
            try:
                future = self._pool.submit(
                    self._client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=config,
                )
                try:
                    resp = future.result(timeout=self._hard_timeout_s)
                except _FutureTimeoutError:
                    logger.info("Vertex Express hard-timeout (%.0fs) on attempt %d/%d — abandoning request",
                                self._hard_timeout_s, attempt + 1, self._max_retries)
                    # Recreate the pool: the abandoned thread is still holding
                    # the httpx connection. A fresh pool gets a fresh worker.
                    try:
                        self._pool.shutdown(wait=False)
                    except Exception:
                        pass
                    self._pool = ThreadPoolExecutor(max_workers=1)
                    if attempt >= self._max_retries:
                        raise TimeoutError(f"Vertex Express call timed out after {self._max_retries} retries")
                    delay = min(30.0, 2.0 + attempt)
                    time.sleep(delay)
                    continue
                break
            except gerrors.APIError as e:
                last_exc = e
                # Retry on transient errors
                code = getattr(e, "code", None)
                retriable = code in (429, 500, 502, 503, 504) or "503" in str(e) or "429" in str(e)
                if not retriable or attempt >= self._max_retries:
                    raise
                delay = min(60.0, 2.0 ** attempt + 0.1 * attempt)
                logger.info("Vertex Express retry attempt %d/%d after %.1fs (%s)",
                            attempt + 1, self._max_retries, delay, str(e)[:80])
                time.sleep(delay)
            except Exception as e:
                last_exc = e
                # Non-API exceptions (network, connection) — retry with caution
                if attempt >= self._max_retries:
                    raise
                delay = min(60.0, 2.0 ** attempt + 0.1 * attempt)
                logger.info("Vertex Express non-API retry %d/%d after %.1fs (%s: %s)",
                            attempt + 1, self._max_retries, delay,
                            type(e).__name__, str(e)[:80])
                time.sleep(delay)
        else:
            raise last_exc  # safeguard

        # ----- extract text and usage
        text = resp.text or ""
        um = getattr(resp, "usage_metadata", None)
        in_t = getattr(um, "prompt_token_count", 0) if um else 0
        out_t = getattr(um, "candidates_token_count", 0) if um else 0
        finish_reason = None
        if resp.candidates:
            fr = getattr(resp.candidates[0], "finish_reason", None)
            finish_reason = str(fr) if fr is not None else None
        return _ChatCompletion(text, in_t, out_t, finish_reason=finish_reason, model=model)


class _Chat:
    def __init__(self, genai_client, max_retries=15):
        self.completions = _Completions(genai_client, max_retries=max_retries)


class VertexGeminiOpenAIWrapper:
    """Adapter that exposes an OpenAI-style `client.chat.completions.create(...)`
    backed by `google.genai.Client(vertexai=True, api_key=...)`.

    Construct via `build_gemini_client()` — do not instantiate directly.
    """

    def __init__(self, api_key: str, max_retries: int = 15, timeout_s: float = 60.0):
        from google import genai
        from google.genai import types as gtypes
        # Explicit HTTP timeout in milliseconds. Without this, a stuck connection
        # can hang the process indefinitely (observed 2026-05-14 on Arm1/Arm3a).
        http_options = gtypes.HttpOptions(timeout=int(timeout_s * 1000))
        self._genai = genai.Client(vertexai=True, api_key=api_key, http_options=http_options)
        self.chat = _Chat(self._genai, max_retries=max_retries)

    @property
    def backend(self):
        return "vertex_express"


# ---------------------------------------------------------------------------
# Factory.
# ---------------------------------------------------------------------------

def build_gemini_client(api_key: str | None = None, timeout: float = 600.0,
                        backend: str | None = None):
    """Construct a Gemini client.

    Default backend (2026-05-14+): **Vertex Express** via `new_gemini` env var.
    Set `backend="openai_compat"` to fall back to the developer-API OpenAI-compat
    endpoint with `gemini_api` / `gemini_api2` (legacy).

    Args:
        api_key: explicit key. Falls back to env vars based on backend.
        timeout: request timeout in seconds (openai_compat backend only).
        backend: "vertex_express" (default) or "openai_compat".

    Returns:
        `VertexGeminiOpenAIWrapper` or raw `openai.OpenAI` — both expose
        `client.chat.completions.create(...)` with the same surface and return
        shape.

    Raises:
        RuntimeError: if no API key could be resolved.
    """
    backend = backend or os.environ.get("GEMINI_BACKEND", "vertex_express")

    if backend == "vertex_express":
        key = api_key or os.environ.get("NEW_GEMINI") or os.environ.get("new_gemini")
        if not key:
            raise RuntimeError(
                "No Vertex Express API key resolved. Export `new_gemini=...` "
                "from .env or pass api_key= explicitly. "
                "Get one at https://aistudio.google.com/apikey (then use it in "
                "Vertex Express mode)."
            )
        return VertexGeminiOpenAIWrapper(api_key=key, max_retries=15)

    if backend == "openai_compat":
        from openai import OpenAI
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Export it or pass api_key= explicitly. "
                "Get one at https://aistudio.google.com/apikey"
            )
        return OpenAI(api_key=key, base_url=GEMINI_OPENAI_BASE_URL,
                      timeout=timeout, max_retries=15)

    raise ValueError(f"Unknown backend: {backend!r}. "
                     "Use 'vertex_express' or 'openai_compat'.")


def build_openai_compat_client(api_key: str | None = None, timeout: float = 600.0):
    """Explicit alias for the developer-API OpenAI-compat path. Same behavior
    as `build_gemini_client(backend='openai_compat')`. Kept for clarity."""
    return build_gemini_client(api_key=api_key, timeout=timeout, backend="openai_compat")
