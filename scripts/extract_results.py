#!/usr/bin/env python3
"""Extract NR / P1 / P2 / avg-turns from BIRD-Interact result directories.

The pipeline writes `evaluation_results_metrics.json` next to
`evaluation_results.jsonl` when it finishes. This script walks one or more
result directories, parses those metrics files, computes the normalized
reward using the canonical formula, and prints a comparison table.

Usage:
  python scripts/extract_results.py results/
  python scripts/extract_results.py results/qwen-baseline/ results/qwen-union/

The pipeline's own `total_reward` / `avg_reward` fields use `last_reward`,
which only counts the final submit() call. The CORRECT a-Interact formula
sums per-phase rewards:

    NR = (P1_completed * 0.7 + P2_completed * 0.3) / total_samples

This script uses that formula; the pipeline's `avg_reward` is shown for
comparison only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def find_metrics(root: Path):
    """Yield every evaluation_results_metrics.json under `root`."""
    if root.is_file() and root.name == "evaluation_results_metrics.json":
        yield root
        return
    yield from root.rglob("evaluation_results_metrics.json")


def compute(metrics_path: Path) -> dict | None:
    try:
        with metrics_path.open() as f:
            m = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  skip {metrics_path}: {e}", file=sys.stderr)
        return None

    n = m.get("total_samples", 0)
    if not n:
        return None

    p1 = m.get("phase1_completed", 0)
    p2 = m.get("phase2_completed", 0)
    nr = (p1 * 0.7 + p2 * 0.3) / n

    return {
        "run": metrics_path.parent.name,
        "n": n,
        "p1": p1,
        "p1_rate": p1 / n,
        "p2": p2,
        "p2_rate": p2 / n,
        "nr": nr,
        "avg_turns": m.get("avg_turns", 0.0),
        "pipeline_reward": m.get("avg_reward", 0.0),  # the buggy last_reward avg
    }


def main():
    roots = [Path(p) for p in sys.argv[1:]] or [Path("results")]
    rows = []
    for r in roots:
        if not r.exists():
            print(f"  WARN: {r} does not exist", file=sys.stderr)
            continue
        for mp in find_metrics(r):
            row = compute(mp)
            if row:
                rows.append(row)

    if not rows:
        print("No evaluation_results_metrics.json files found.")
        sys.exit(1)

    # Sort by run name for stable display.
    rows.sort(key=lambda r: r["run"])

    # ─── tabulate ─────────────────────────────────────────────────────
    headers = ["run", "N", "P1", "P1%", "P2", "P2%", "NR", "avg_turns"]
    widths = [max(28, max(len(r["run"]) for r in rows)), 4, 4, 7, 4, 7, 8, 9]

    def fmt_row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))
    for r in rows:
        print(fmt_row([
            r["run"],
            r["n"],
            r["p1"],
            f"{r['p1_rate']*100:.2f}%",
            r["p2"],
            f"{r['p2_rate']*100:.2f}%",
            f"{r['nr']*100:.2f}%",
            f"{r['avg_turns']:.2f}",
        ]))

    print()
    print(f"NR = (P1 * 0.7 + P2 * 0.3) / N   (canonical a-Interact formula)")


if __name__ == "__main__":
    main()
