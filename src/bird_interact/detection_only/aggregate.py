"""Aggregator for detection-only output → Tables 5–7 of the paper.

Reads the per-record JSON written by `pipeline.py` and rolls it up into:

  - **Overall recall**: detected GT terms / total GT terms across all samples
    (the headline number in Table 5 of `tab:overall`).
  - **Per-subtype recall** (Tables 5/6 — `tab:intent`, `tab:impl`):
    `knowledge_linking_ambiguity`, `intent_ambiguity`, `schema_linking_ambiguity`,
    `semantic_ambiguity`, `lexical_ambiguity`, `null_ambiguity`, `sort_ambiguity`,
    `decimal_ambiguity`, `join_ambiguity`, `distinct_ambiguity`,
    `divide_zero_ambiguity`, plus `knowledge_ambiguity`.
  - **Per-tier recall** (Table 7 — `tab:knowledge`): Intent / Implementation /
    Masked-knowledge rollups via micro-averaged recall.

You can also pass multiple `--input` files (one per method) and get a
side-by-side Markdown table.

Single-file usage::

    python -m bird_interact.detection_only.aggregate \
        --input results/det_se_ast_qwen.json

Multi-file (one column per method)::

    python -m bird_interact.detection_only.aggregate \
        --input results/det_direct_qwen.json:direct \
        --input results/det_mga_qwen.json:mga \
        --input results/det_se_ast_qwen.json:se_ast \
        --markdown results/detection_summary.md

Per-term union of two runs (paper's ``\\Direct ∪ \\MultiGen`` row in
Tables 5–7). For each shared (instance_id, GT term) the union counts the
term as detected iff *either* input detected it. The denominator is the
intersection of both runs' GT term universes — typically identical when
both runs evaluated the same lite-300 split with the same encoder
mapping::

    python -m bird_interact.detection_only.aggregate \
        --input results/det_direct_qwen.json:direct \
        --input results/det_mga_qwen.json:mga \
        --union direct+mga:union \
        --markdown results/det_union.md

The ``:union`` label is what the column will be called in the output;
omit it to default to ``A+B``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Category membership matches the paper's taxonomy table (setup section):
# syntactic is INTENT-level (multiple grammatical parses), NOT implementation.
INTENT_TYPES = [
    "knowledge_linking_ambiguity",
    "intent_ambiguity",
    "schema_linking_ambiguity",
    "semantic_ambiguity",
    "lexical_ambiguity",
    "syntactic_ambiguity",
]
IMPL_TYPES = [
    "sort_ambiguity",
    "decimal_ambiguity",
    "null_ambiguity",
    "join_ambiguity",
    "distinct_ambiguity",
    "divide_zero_ambiguity",
    "date_format_ambiguity",
    "rank_ambiguity",
]
KNOWLEDGE_TYPES = ["knowledge_ambiguity"]

CATEGORIES = [
    ("Intent", INTENT_TYPES),
    ("Implementation", IMPL_TYPES),
    ("Masked-knowledge", KNOWLEDGE_TYPES),
]


def _per_sample_term_detections(blob: dict) -> dict:
    """{instance_id: {term_key: (term_type, detected_bool)}}.

    Used by ``union_file_metrics`` to OR detection flags between two runs
    per (sample, GT term). The term key is ``term.lower().strip()`` to match
    the encoder's labelled-matches normalisation.
    """
    out: dict[str, dict[str, tuple[str, bool]]] = {}
    for r in (blob.get("results") or []):
        iid = r.get("instance_id")
        if not iid:
            continue
        det = r.get("detection") or {}
        gts = det.get("gt_terms") or []
        out[iid] = {}
        for gt in gts:
            term = (gt.get("term") or "").lower().strip()
            if not term:
                continue
            out[iid][term] = (gt.get("type", "unknown"), bool(gt.get("detected", False)))
    return out


def compute_file_metrics(blob: dict) -> dict:
    """Return {overall, by_type, by_category} recall + sample count."""
    records = blob.get("results") or []
    n_records = sum(1 for r in records if (r.get("detection") or {}).get("gt_terms") is not None)

    overall = {"detected": 0, "total": 0}
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"detected": 0, "total": 0})
    total_dets = 0
    total_valid = 0

    for r in records:
        det = r.get("detection") or {}
        gts = det.get("gt_terms") or []
        total_dets += int(det.get("total_detections", 0) or 0)
        total_valid += int(det.get("valid", 0) or 0)
        for gt in gts:
            t = gt.get("type", "unknown")
            detected = bool(gt.get("detected", False))
            overall["total"] += 1
            by_type[t]["total"] += 1
            if detected:
                overall["detected"] += 1
                by_type[t]["detected"] += 1

    def rate(d: dict) -> float:
        return d["detected"] / d["total"] if d["total"] else 0.0

    cat: dict[str, dict] = {}
    for cat_name, type_list in CATEGORIES:
        d = {"detected": 0, "total": 0}
        for t in type_list:
            d["detected"] += by_type[t]["detected"]
            d["total"] += by_type[t]["total"]
        cat[cat_name] = {"recall": rate(d), "total_gt": d["total"]}

    return {
        "method": blob.get("method"),
        "model_id": blob.get("model_id"),
        "encoder_model_id": blob.get("encoder_model_id"),
        "seed": blob.get("seed"),
        "n_records": n_records,
        "overall_recall": rate(overall),
        "overall_total_gt": overall["total"],
        "overall_detected_gt": overall["detected"],
        "by_type": {t: {"recall": rate(v), "total": v["total"], "detected": v["detected"]}
                    for t, v in by_type.items()},
        "by_category": cat,
        "precision_at_sample": (total_valid / total_dets) if total_dets else 0.0,
        "total_detections": total_dets,
        "total_valid_detections": total_valid,
        "n_failed_records": sum(1 for r in records if r.get("error")),
        # Raw per-sample term map kept so union_file_metrics can OR the flags.
        "_per_sample_terms": _per_sample_term_detections(blob),
    }


def union_file_metrics(label_a: str, label_b: str, metrics: dict[str, dict]) -> dict:
    """Per-term union of two already-computed file metrics.

    Walks the intersection of (instance_id, term) keys; counts a term as
    detected iff *either* input detected it. Precision is not defined for a
    union (the underlying questions sets are not jointly available), so it
    is left as ``None``.
    """
    a = metrics[label_a]["_per_sample_terms"]
    b = metrics[label_b]["_per_sample_terms"]
    shared_ids = sorted(set(a) & set(b))
    only_a = set(a) - set(b)
    only_b = set(b) - set(a)
    if only_a or only_b:
        # Same lite-300 split should give identical key sets; warn if not.
        print(f"  union({label_a}+{label_b}): {len(only_a)} ids only in {label_a}, "
              f"{len(only_b)} only in {label_b}; keeping intersection of "
              f"{len(shared_ids)} ids.", file=sys.stderr)

    overall = {"detected": 0, "total": 0}
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"detected": 0, "total": 0})
    for iid in shared_ids:
        terms_a, terms_b = a[iid], b[iid]
        for term in set(terms_a) | set(terms_b):
            # Prefer A's type label when both exist (they agree in practice).
            t = (terms_a.get(term) or terms_b[term])[0]
            d_a = terms_a.get(term, ("", False))[1]
            d_b = terms_b.get(term, ("", False))[1]
            detected = d_a or d_b
            overall["total"] += 1
            by_type[t]["total"] += 1
            if detected:
                overall["detected"] += 1
                by_type[t]["detected"] += 1

    def rate(d: dict) -> float:
        return d["detected"] / d["total"] if d["total"] else 0.0

    cat: dict[str, dict] = {}
    for cat_name, type_list in CATEGORIES:
        d = {"detected": 0, "total": 0}
        for t in type_list:
            d["detected"] += by_type[t]["detected"]
            d["total"] += by_type[t]["total"]
        cat[cat_name] = {"recall": rate(d), "total_gt": d["total"]}

    return {
        "method": f"union({label_a}+{label_b})",
        "model_id": metrics[label_a].get("model_id"),
        "encoder_model_id": metrics[label_a].get("encoder_model_id"),
        "seed": None,
        "n_records": len(shared_ids),
        "overall_recall": rate(overall),
        "overall_total_gt": overall["total"],
        "overall_detected_gt": overall["detected"],
        "by_type": {t: {"recall": rate(v), "total": v["total"], "detected": v["detected"]}
                    for t, v in by_type.items()},
        "by_category": cat,
        "precision_at_sample": None,    # undefined for a per-term union
        "total_detections": None,
        "total_valid_detections": None,
        "_per_sample_terms": None,
    }


def parse_input_spec(spec: str) -> tuple[str, str]:
    """`path` or `path:label` → (path, label).

    The label is whatever follows the last colon, unless that part contains a
    path separator — so a Windows drive prefix such as ``C:\\results\\x.json``
    is kept as a plain path.
    """
    path, sep, label = spec.rpartition(":")
    if sep and path and label and not any(c in label for c in "/\\"):
        return path, label
    return spec, Path(spec).stem


def render_markdown(metrics_by_label: dict[str, dict]) -> str:
    """Render side-by-side table with one column per method/file."""
    labels = list(metrics_by_label.keys())
    rows: list[list[str]] = []

    header = ["Section / Term"] + labels
    rows.append(header)
    rows.append(["---"] * len(header))

    # Overall
    rows.append(["**Overall**"] + [f"{metrics_by_label[l]['overall_recall'] * 100:.1f}" for l in labels])
    rows.append(["  n_records"] + [str(metrics_by_label[l]["n_records"]) for l in labels])
    rows.append(["  total_gt"] + [str(metrics_by_label[l]["overall_total_gt"]) for l in labels])
    rows.append(["  precision@sample"] + [
        "—" if metrics_by_label[l]["precision_at_sample"] is None
        else f"{metrics_by_label[l]['precision_at_sample'] * 100:.1f}"
        for l in labels
    ])

    # Categories (Table 7)
    rows.append(["**Tier (Table 7)**"] + [""] * len(labels))
    for cat_name, _ in CATEGORIES:
        rows.append([f"  {cat_name}"] +
                    [f"{metrics_by_label[l]['by_category'][cat_name]['recall'] * 100:.1f}"
                     for l in labels])

    # Per-subtype (Tables 5 + 6)
    rows.append(["**Intent subtypes (Table 5)**"] + [""] * len(labels))
    for t in INTENT_TYPES:
        if any(metrics_by_label[l]["by_type"].get(t, {}).get("total", 0) for l in labels):
            cells = []
            for l in labels:
                bt = metrics_by_label[l]["by_type"].get(t, {})
                if bt.get("total", 0):
                    cells.append(f"{bt['recall'] * 100:.1f}")
                else:
                    cells.append("—")
            rows.append([f"  {t}"] + cells)

    rows.append(["**Implementation subtypes (Table 6)**"] + [""] * len(labels))
    for t in IMPL_TYPES:
        if any(metrics_by_label[l]["by_type"].get(t, {}).get("total", 0) for l in labels):
            cells = []
            for l in labels:
                bt = metrics_by_label[l]["by_type"].get(t, {})
                if bt.get("total", 0):
                    cells.append(f"{bt['recall'] * 100:.1f}")
                else:
                    cells.append("—")
            rows.append([f"  {t}"] + cells)

    rows.append(["**Knowledge (Table 7)**"] + [""] * len(labels))
    for t in KNOWLEDGE_TYPES:
        if any(metrics_by_label[l]["by_type"].get(t, {}).get("total", 0) for l in labels):
            cells = []
            for l in labels:
                bt = metrics_by_label[l]["by_type"].get(t, {})
                if bt.get("total", 0):
                    cells.append(f"{bt['recall'] * 100:.1f}")
                else:
                    cells.append("—")
            rows.append([f"  {t}"] + cells)

    lines = []
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Aggregate detection-only outputs into Tables 5–7.",
    )
    p.add_argument("--input", action="append", required=True,
                   help="path/to/result.json or path:label (repeatable)")
    p.add_argument("--union", action="append", default=[],
                   help=("per-term union of two existing input labels, e.g. "
                         "'direct+mga' or 'direct+mga:my_union' (repeatable)"))
    p.add_argument("--out_json", default=None, help="write aggregator state as JSON")
    p.add_argument("--markdown", default=None, help="write Markdown summary table")
    args = p.parse_args(argv)

    metrics_by_label: dict[str, dict] = {}
    for spec in args.input:
        path, label = parse_input_spec(spec)
        with open(path) as f:
            blob = json.load(f)
        m = compute_file_metrics(blob)
        metrics_by_label[label] = m
        ovr = m["overall_recall"] * 100
        print(
            f"{label:>20}  n={m['n_records']:<3}  R={ovr:5.1f}%  "
            f"GT={m['overall_total_gt']}  "
            f"P@sample={m['precision_at_sample']*100:5.1f}%  "
            f"(Intent {m['by_category']['Intent']['recall']*100:4.1f}% / "
            f"Impl {m['by_category']['Implementation']['recall']*100:4.1f}% / "
            f"Mask-kn {m['by_category']['Masked-knowledge']['recall']*100:4.1f}%)"
        )
        if m["n_failed_records"]:
            print(f"  WARNING: {label}: {m['n_failed_records']} record(s) failed during generation "
                  f"or encoding, so their recall is missing or understated. Rerun the same "
                  f"run_detection.sh command to retry them.", file=sys.stderr)

    # Per-term unions over already-loaded labels.
    for spec in args.union:
        pair, _, union_label = spec.partition(":")
        if "+" not in pair:
            print(f"  --union {spec!r}: must be A+B or A+B:label", file=sys.stderr)
            sys.exit(2)
        a, b = pair.split("+", 1)
        if a not in metrics_by_label or b not in metrics_by_label:
            missing = [x for x in (a, b) if x not in metrics_by_label]
            print(f"  --union {spec!r}: unknown label(s) {missing}; "
                  f"available: {list(metrics_by_label)}", file=sys.stderr)
            sys.exit(2)
        union_label = union_label or f"{a}+{b}"
        u = union_file_metrics(a, b, metrics_by_label)
        metrics_by_label[union_label] = u
        ovr = u["overall_recall"] * 100
        print(
            f"{union_label:>20}  n={u['n_records']:<3}  R={ovr:5.1f}%  "
            f"GT={u['overall_total_gt']}  "
            f"P@sample=    —  "
            f"(Intent {u['by_category']['Intent']['recall']*100:4.1f}% / "
            f"Impl {u['by_category']['Implementation']['recall']*100:4.1f}% / "
            f"Mask-kn {u['by_category']['Masked-knowledge']['recall']*100:4.1f}%)"
        )

    # Strip the per-sample term map before serialising / rendering so output
    # JSON / Markdown stays the same size as before.
    for m in metrics_by_label.values():
        m.pop("_per_sample_terms", None)

    md = render_markdown(metrics_by_label)
    print()
    print(md)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(metrics_by_label, indent=2))
        print(f"Saved JSON → {args.out_json}")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(md)
        print(f"Saved Markdown → {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
