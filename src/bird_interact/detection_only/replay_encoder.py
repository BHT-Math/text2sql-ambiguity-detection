"""Replay an existing detection JSON with a different encoder.

Given an output from ``bird_interact.detection_only.pipeline`` (which
already contains the per-sample ``extracted_detections`` from a Direct /
MGA / SE+AST / Direct-Multi run), this script re-runs ONLY the encoder
calls against a (possibly different) endpoint + encoder kind and writes
a new JSON.

The detection LLM is not invoked. For a typical lite-300 SE+AST run that
took 11 LLM calls per sample, this is roughly a 10x cost reduction when
you want to compare two encoder choices on the same detection output.

Example uses
------------

Re-run an existing legacy-encoder Direct output with the **multi-label**
encoder (one question can credit up to 3 GT terms):

    python -m bird_interact.detection_only.replay_encoder \\
        --input    results/det_qwen_direct.json \\
        --output   results/det_qwen_direct_multi_label.json \\
        --base_url $QWEN35_122B_BASE_URL --model_id qwen3.5-122b \\
        --encoder_kind multi_label

Re-judge with a **different** encoder model (e.g. swap MiniMax for
Qwen as encoder on a Qwen-detection run — paper-canonical cross-routing):

    python -m bird_interact.detection_only.replay_encoder \\
        --input  results/det_qwen_direct.json \\
        --output results/det_qwen_direct_mm_encoder.json \\
        --base_url $MINIMAX_M25_BASE_URL --model_id minimax-m2.5

The output JSON keeps the original ``model_id`` / ``method`` /
``extracted_detections`` per record but replaces ``detection.gt_terms``,
``detection.recall``, etc. with the new encoder's verdict, and records
``encoder_model_id`` + ``encoder_kind`` in the top-level header.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from .data import filter_kb_for_sample, get_gt_terms, load_kb, load_schema, load_samples
from .encoder import default_encoder_extra_body
from .pipeline import _evaluate_detections

logger = logging.getLogger("detection_only.replay_encoder")


def _load_sample_by_id(samples: list[dict]) -> dict[str, dict]:
    return {s.get("instance_id"): s for s in samples if s.get("instance_id")}


def _replay_one(
    rec: dict, sample: dict, schema: str,
    *, enc_client: OpenAI, enc_model_id: str,
    encoder_extra_body: dict | None, user_sim_prompt_version: str,
    encoder_kind: str,
) -> dict:
    iid = rec.get("instance_id", "?")
    detections = rec.get("extracted_detections") or []
    new = dict(rec)
    try:
        if detections:
            new["detection"] = _evaluate_detections(
                enc_client, enc_model_id, detections, sample, schema,
                encoder_extra_body=encoder_extra_body,
                user_sim_prompt_version=user_sim_prompt_version,
                encoder_kind=encoder_kind,
            )
        else:
            gt = get_gt_terms(sample)
            new["detection"] = {
                "total_gt": len(gt), "detected_gt": 0, "recall": 0.0,
                "valid": 0, "fp": 0, "total_detections": 0,
                "labeled_matches": [],
                "gt_terms": [{**g, "detected": False} for g in gt],
                "raw_judgments": [],
            }
    except Exception as e:
        logger.exception(f"[{iid}] replay failed: {e}")
        new["detection"] = {"error": str(e)}
    return new


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="existing detection JSON to replay against")
    p.add_argument("--output", required=True, help="new JSON to write")
    p.add_argument("--data_path", default="data/bird-interact-lite/bird_interact_data.jsonl",
                   help="bird_interact_data.jsonl (needed to fetch ground-truth terms by instance_id)")
    p.add_argument("--data_dir",  default="data/bird-interact-lite",
                   help="dir with per-DB schema/KB/columns")
    p.add_argument("--num_threads", type=int, default=8)

    # New encoder
    p.add_argument("--base_url", required=True, help="encoder endpoint")
    p.add_argument("--api_key", default="EMPTY")
    p.add_argument("--model_id", required=True, help="encoder model id (was --encoder_model_id in pipeline)")
    p.add_argument("--encoder_kind", choices=("legacy", "multi_label"), default="legacy",
                   help=("legacy (default) = single-label BIRD-Interact encoder. "
                         "multi_label = opt-in: one question can credit up to 3 GT terms."))
    p.add_argument("--user_sim_prompt_version", default="v2", choices=("v1", "v2"))
    p.add_argument("--encoder_no_thinking", action="store_true", default=True)
    p.add_argument("--log_level", default="INFO")

    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s")

    blob = json.loads(Path(args.input).read_text())
    records = blob.get("results") or []
    logger.info(f"Loaded {len(records)} records from {args.input} "
                f"(method={blob.get('method')}, model_id={blob.get('model_id')})")

    samples = load_samples(args.data_path)
    sample_by_id = _load_sample_by_id(samples)

    enc_client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    enc_extra = default_encoder_extra_body(force_no_thinking=args.encoder_no_thinking)

    # Per-DB schema/KB cache. Same shape as the pipeline cache.
    schema_cache: dict[str, str] = {}
    kb_cache: dict[str, list[dict]] = {}

    def _load_ctx(db: str):
        if db not in schema_cache:
            schema_cache[db] = load_schema(db, args.data_dir)
            kb_cache[db] = load_kb(db, args.data_dir)
        return schema_cache[db], kb_cache[db]

    start = time.time()
    out_records: list[dict] = []
    todo: list[tuple[dict, dict, str]] = []
    skipped = 0
    for rec in records:
        iid = rec.get("instance_id")
        sample = sample_by_id.get(iid)
        if not sample:
            logger.warning(f"[{iid}] not found in data_path; copying record unchanged")
            out_records.append(rec)
            skipped += 1
            continue
        schema, _ = _load_ctx(sample.get("selected_database", ""))
        todo.append((rec, sample, schema))

    with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
        futs = {
            ex.submit(_replay_one, rec, sample, schema,
                      enc_client=enc_client, enc_model_id=args.model_id,
                      encoder_extra_body=enc_extra,
                      user_sim_prompt_version=args.user_sim_prompt_version,
                      encoder_kind=args.encoder_kind): rec
            for (rec, sample, schema) in todo
        }
        done = 0
        for fut in as_completed(futs):
            try:
                entry = fut.result()
            except Exception as e:
                entry = {**futs[fut], "error": f"worker crashed: {e}"}
            out_records.append(entry)
            done += 1
            if done % max(1, len(todo) // 20) == 0 or done == len(todo):
                logger.info(f"  replayed {done}/{len(todo)}")

    out_blob = {
        "method": blob.get("method"),
        "model_id": blob.get("model_id"),
        "encoder_model_id": args.model_id,
        "encoder_kind": args.encoder_kind,
        "seed": blob.get("seed"),
        "n_total": blob.get("n_total"),
        "n_done": len(out_records),
        "elapsed_sec": round(time.time() - start, 1),
        "source": str(Path(args.input).resolve()),
        "source_encoder_model_id": blob.get("encoder_model_id"),
        "source_encoder_kind": blob.get("encoder_kind", "unknown"),
        "results": out_records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out_blob, indent=2))
    logger.info(f"DONE — wrote {len(out_records)} records (skipped {skipped} not-in-dataset) "
                f"to {args.output} in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
