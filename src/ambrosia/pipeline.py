"""Unified AMBROSIA pipeline entry.

One CLI for the four canonical arms in the paper appendix (paper name → flag):

  Baseline                  --method baseline    AMBROSIA-native 1-call prompt (T=0)
  Si-1                      --method direct      Two-stage Self-Introspection
  Union (Si-1 ∪ Mcs-10)     --method union_mcs   Si-1 ∪ 10 independent SQL samples @T=0.7
  Union (Si-1 ∪ Mcs-1)      --method union_spmi  Si-1 ∪ single-pass Multi-Candidate (paper headline)

Each invocation runs the channels end-to-end against an OpenAI-compatible
endpoint, evaluates per-sample coverage with the AMBROSIA execution
protocol, and writes intermediate artifacts + a final ``eval.json``
summary into ``--output-dir``. ``eval.json`` schema:
``{config, summary, per_type, results}``.

The companion shell wrapper at ``scripts/run_ambrosia.sh`` maps friendly
``--model {glm,mm,qwen}`` flags to the right base-URL / temperature /
``--no_thinking`` combination per the paper-canonical recipe.

Examples::

    python -m ambrosia.pipeline \\
        --method union_spmi --model_id qwen3.5-122b --no_thinking \\
        --base_url http://localhost:8000/v1 \\
        --ambrosia_dir data/ambrosia \\
        --output_dir results/ambrosia/qwen_union_spmi

    python -m ambrosia.pipeline \\
        --method baseline --model_id glm-4.5-air --no_thinking \\
        --base_url http://localhost:8000/v1 \\
        --ambrosia_dir data/ambrosia \\
        --output_dir results/ambrosia/glm_baseline \\
        --max_samples 5    # smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from openai import OpenAI

from . import channels
from .data import load_ambrosia, resolve_db_path, stratified_subsample
from .eval_lib import evaluate_coverage_execution
from .union import pool_records

logger = logging.getLogger("ambrosia.pipeline")


METHODS = ("baseline", "direct", "union_mcs", "union_spmi")


def _atomic_dump(path: str, blob: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(blob, f, indent=2, default=str)
    os.replace(tmp, path)


def _wrap_summary(records: list[dict], config: dict) -> dict:
    """Wrap a list of pipeline records into the {config, summary, results} schema."""
    is_ambig = lambda r: r["is_ambiguous"]
    ambig = [r for r in records if is_ambig(r)]
    unambig = [r for r in records if not is_ambig(r)]
    summary = {
        "total_samples": len(records),
        "ambiguous": len(ambig),
        "unambiguous": len(unambig),
    }
    return {"config": config, "summary": summary, "results": records}


def _evaluate_records(
    records: list[dict], ambrosia_dir: str, equivalence_threshold: float,
) -> tuple[list[dict], dict, dict]:
    """Score each ambiguous record's predicted_sqls against ambig_sqls.

    Returns (out_records_with_coverage, aggregate_summary, per_type_summary).
    """
    out_records: list[dict] = []
    for i, rec in enumerate(records):
        coverage = None
        preds = rec.get("predicted_sqls", []) or []
        ambig_sqls = rec.get("ambig_sqls", []) or []
        if rec["is_ambiguous"] and ambig_sqls and preds:
            db_path = resolve_db_path(ambrosia_dir, rec.get("db_file", ""))
            if os.path.exists(db_path):
                coverage = evaluate_coverage_execution(
                    db_path=db_path,
                    sampled_sqls=preds,
                    gt_sqls=ambig_sqls,
                    ast_threshold=equivalence_threshold,
                )
        out_records.append({**rec, "coverage": coverage})
        if (i + 1) % 100 == 0 or (i + 1) == len(records):
            logger.info(f"  eval {i+1}/{len(records)}")

    cov = [r for r in out_records if r.get("coverage")]
    recalls = [r["coverage"]["recall"] for r in cov]
    precisions = [r["coverage"]["precision"] for r in cov]
    f1s = [r["coverage"]["f1"] for r in cov]
    afs = [r["coverage"]["all_found"] for r in cov]
    aggregate = {
        "total_samples": len(out_records),
        "ambiguous": sum(1 for r in out_records if r["is_ambiguous"]),
        "unambiguous": sum(1 for r in out_records if not r["is_ambiguous"]),
        "coverage_samples": len(cov),
        "mean_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "mean_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "mean_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "all_found_rate": sum(afs) / len(afs) if afs else 0.0,
        "total_exec_errors": sum(r["coverage"]["execution_errors"] for r in cov),
    }
    per_type: dict[str, list[float]] = {}
    for r in cov:
        t = r["ambig_type"] or "unknown"
        per_type.setdefault(t, []).append(r["coverage"]["recall"])
    per_type_summary = {
        t: {"n": len(v), "recall": sum(v) / len(v)} for t, v in per_type.items()
    }
    return out_records, aggregate, per_type_summary


def _print_summary(label: str, summary: dict, per_type: dict) -> None:
    print(f"\n{'='*60}\nEVAL: {label}\n{'='*60}")
    print(f"Samples: {summary['total_samples']} "
          f"({summary['ambiguous']} ambig, {summary['unambiguous']} unambig)")
    print(f"Coverage samples: {summary['coverage_samples']}")
    print(f"Mean Recall:    {summary['mean_recall']:.1%}")
    print(f"Mean Precision: {summary['mean_precision']:.1%}")
    print(f"Mean F1:        {summary['mean_f1']:.1%}")
    print(f"AllFound rate:  {summary['all_found_rate']:.1%}")
    print(f"Exec errors:    {summary['total_exec_errors']}")
    print("Per-type recall:")
    for t, v in sorted(per_type.items()):
        print(f"  {t:14s} (n={v['n']:3d}): {v['recall']:.1%}")


def _run_method(args, samples: list[dict], client: OpenAI) -> list[dict]:
    """Dispatch on --method, producing predicted_sqls per record."""
    out_dir = Path(args.output_dir)
    if args.method == "baseline":
        recs = channels.run_baseline(
            client, args.model_id, samples,
            no_thinking=args.no_thinking,
            max_tokens=args.baseline_max_tokens,
            num_threads=args.num_threads,
        )
        _atomic_dump(str(out_dir / "baseline.json"),
                     _wrap_summary(recs, {"stage": "baseline", "model_id": args.model_id}))
        return recs

    if args.method == "direct":
        stage1 = channels.run_direct_stage1(
            client, args.model_id, samples,
            no_thinking=args.no_thinking,
            max_tokens=args.direct_stage1_max_tokens,
            num_threads=args.num_threads,
        )
        _atomic_dump(str(out_dir / "stage1.json"),
                     _wrap_summary(stage1, {"stage": "direct.stage1", "model_id": args.model_id}))
        stage2 = channels.run_direct_stage2(
            client, args.model_id, stage1,
            no_thinking=args.no_thinking,
            max_tokens=args.direct_stage2_max_tokens,
            num_threads=args.num_threads,
        )
        _atomic_dump(str(out_dir / "stage2.json"),
                     _wrap_summary(stage2, {"stage": "direct.stage2", "model_id": args.model_id}))
        return stage2

    # union_mcs or union_spmi — run Direct AND the chosen sampling channel.
    stage1 = channels.run_direct_stage1(
        client, args.model_id, samples,
        no_thinking=args.no_thinking,
        max_tokens=args.direct_stage1_max_tokens,
        num_threads=args.num_threads,
    )
    _atomic_dump(str(out_dir / "stage1.json"),
                 _wrap_summary(stage1, {"stage": "direct.stage1", "model_id": args.model_id}))
    direct = channels.run_direct_stage2(
        client, args.model_id, stage1,
        no_thinking=args.no_thinking,
        max_tokens=args.direct_stage2_max_tokens,
        num_threads=args.num_threads,
    )
    _atomic_dump(str(out_dir / "stage2.json"),
                 _wrap_summary(direct, {"stage": "direct.stage2", "model_id": args.model_id}))

    if args.method == "union_mcs":
        sampling = channels.run_mcs(
            client, args.model_id, samples,
            no_thinking=args.no_thinking,
            num_samples=args.mcs_num_samples,
            temperature=args.mcs_temperature,
            max_tokens=args.mcs_max_tokens,
            num_threads=args.num_threads,
            seed=args.seed,
        )
        sampling_path = "mcs.json"
        sampling_stage = "mcs"
    else:  # union_spmi
        sampling = channels.run_spmi(
            client, args.model_id, samples,
            no_thinking=args.no_thinking,
            temperature=args.spmi_temperature,
            max_tokens=args.spmi_max_tokens,
            num_threads=args.num_threads,
            seed=args.seed,
        )
        sampling_path = "spmi.json"
        sampling_stage = "spmi"
    _atomic_dump(
        str(out_dir / sampling_path),
        _wrap_summary(sampling, {"stage": sampling_stage, "model_id": args.model_id,
                                 "temperature": (args.mcs_temperature if args.method == "union_mcs"
                                                 else args.spmi_temperature)}),
    )

    pooled = pool_records(direct, sampling, args.ambrosia_dir, mode="union")
    _atomic_dump(
        str(out_dir / "union.json"),
        _wrap_summary(pooled, {"stage": "union", "method": args.method,
                               "model_id": args.model_id}),
    )
    return pooled


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="BIRD-Interact AMBROSIA cross-benchmark pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--base_url", required=True)
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model_id", required=True)
    p.add_argument("--ambrosia_dir", default=os.environ.get("AMBROSIA_DIR", "data/ambrosia"),
                   help="Dir containing ambrosia.csv + per-domain SQLite files.")
    p.add_argument("--output_dir", required=True,
                   help="All artifacts (baseline.json/stage1/stage2/mcs/spmi/union + eval.json) go here.")
    p.add_argument("--split", default="test", help="'test' or 'few_shot_examples'.")
    p.add_argument("--max_samples", type=int, default=0,
                   help="0 = full split; otherwise stratified subsample.")
    p.add_argument("--seed", type=int, default=42)

    # threading / budget
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--no_thinking", action="store_true",
                   help="Pass chat_template_kwargs.enable_thinking=False on every call. "
                        "Required for Qwen3.5 + reasoning models without a server-side parser.")

    # evaluator
    p.add_argument("--equivalence_threshold", type=float, default=1.01,
                   help="AST similarity threshold. Default ≥1 means AST fallback is "
                        "unreachable (paper-canonical exec-only protocol). Use 0.70 "
                        "to enable the fallback if you need a recall upper bound.")

    # method-specific knobs (sane defaults match the paper)
    p.add_argument("--baseline_max_tokens", type=int, default=2048)
    p.add_argument("--direct_stage1_max_tokens", type=int, default=2048)
    p.add_argument("--direct_stage2_max_tokens", type=int, default=1024)
    p.add_argument("--mcs_num_samples", type=int, default=10)
    p.add_argument("--mcs_temperature", type=float, default=0.7)
    p.add_argument("--mcs_max_tokens", type=int, default=1024)
    p.add_argument("--spmi_temperature", type=float, default=1.3,
                   help="Per-model: GLM=1.0, MM/Qwen=1.3 (matches MGA_TEMP). "
                        "Lower than 1.0 collapses the 10 interpretations to one.")
    p.add_argument("--spmi_max_tokens", type=int, default=4096)
    p.add_argument("--log_level", default="INFO")

    args = p.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("openai", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    samples = load_ambrosia(args.ambrosia_dir, split=args.split)
    if args.max_samples and args.max_samples < len(samples):
        samples = stratified_subsample(samples, args.max_samples, seed=args.seed)
        logger.info(f"Subsampled {len(samples)} (stratified, seed={args.seed}).")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=120.0)

    logger.info(
        f"AMBROSIA pipeline: method={args.method} model={args.model_id} "
        f"split={args.split} n={len(samples)} → {args.output_dir}")

    records = _run_method(args, samples, client)

    logger.info("Scoring coverage...")
    out_records, summary, per_type = _evaluate_records(
        records, args.ambrosia_dir, args.equivalence_threshold,
    )
    _print_summary(f"{args.method} ({args.model_id})", summary, per_type)

    eval_blob = {
        "config": {
            "stage": "eval",
            "method": args.method,
            "model_id": args.model_id,
            "split": args.split,
            "max_samples": args.max_samples,
            "equivalence_threshold": args.equivalence_threshold,
            "ambrosia_dir": args.ambrosia_dir,
        },
        "summary": summary,
        "per_type": per_type,
        "results": out_records,
    }
    _atomic_dump(str(Path(args.output_dir) / "eval.json"), eval_blob)
    logger.info(f"Wrote eval.json -> {Path(args.output_dir) / 'eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
