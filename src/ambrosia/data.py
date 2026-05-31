"""AMBROSIA dataset loader + per-sample helpers.

The dataset ships as a single ``ambrosia.csv`` plus per-domain SQLite files
(``data/<ambig_type>/<domain>/<question>/<question>.sqlite``). This module
parses the CSV, materialises the per-record fields the rest of the pipeline
needs, and resolves the bundled ``db_file`` field to an absolute SQLite path.
"""

from __future__ import annotations

import ast as _ast
import csv
import os
import random
from collections import Counter
from pathlib import Path


def _extract_ddl_from_dump(dump: str) -> str:
    """Pick out CREATE TABLE lines from a full SQL dump (DDL + INSERTs)."""
    if not dump:
        return ""
    out: list[str] = []
    in_create = False
    for line in dump.split("\n"):
        s = line.strip()
        if s.upper().startswith(("BEGIN", "COMMIT", "INSERT INTO")):
            continue
        if s.upper().startswith("CREATE TABLE") or s.upper().startswith("CREATE  TABLE"):
            in_create = True
        if in_create:
            out.append(line)
            if ";" in line:
                in_create = False
    return "\n".join(out).strip()


def load_ambrosia(dataset_dir: str, split: str = "test") -> list[dict]:
    """Load AMBROSIA samples and convert to the internal record schema.

    Args:
        dataset_dir: Path to a directory containing ``ambrosia.csv``.
        split: ``"test"`` (default, 3,819 records) or ``"few_shot_examples"``
               (423 records, the ICL pool the paper reserves).

    Returns:
        List of per-sample dicts with keys consumed by the runners:
        ``instance_id``, ``question``, ``is_ambiguous``, ``ambig_type``,
        ``interpretation_type``, ``domain``, ``question_type``, ``db_file``,
        ``schema_ddl``, ``_raw_db_dump`` (full dump for SQLite eval),
        ``gt_sqls``, ``ambig_sqls``.
    """
    csv_path = Path(dataset_dir) / "ambrosia.csv"
    samples: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if row.get("split") != split:
                continue

            ddl = _extract_ddl_from_dump(row.get("db_dump", ""))

            gold = (row.get("gold_queries") or "").strip()
            gt_sqls = [gold] if gold else []

            ambig_sqls: list[str] = []
            raw_ambig = (row.get("ambig_queries") or "").strip()
            if raw_ambig:
                try:
                    parsed = _ast.literal_eval(raw_ambig)
                    ambig_sqls = parsed if isinstance(parsed, list) else [parsed]
                except (ValueError, SyntaxError):
                    ambig_sqls = [s.strip() for s in raw_ambig.split("\n\n") if s.strip()]

            samples.append({
                "instance_id": f"ambrosia_{idx}",
                "question": row.get("question", ""),
                "schema_ddl": ddl,
                "_raw_db_dump": row.get("db_dump", ""),
                "is_ambiguous": row.get("is_ambiguous", "False") == "True",
                "ambig_type": row.get("ambig_type", ""),
                "interpretation_type": row.get("interpretation_type", ""),
                "domain": row.get("domain", ""),
                "question_type": row.get("question_type", ""),
                "db_file": row.get("db_file", ""),
                "gt_sqls": gt_sqls,
                "ambig_sqls": ambig_sqls,
            })

    type_counts = Counter(s["ambig_type"] for s in samples)
    n_ambig = sum(1 for s in samples if s["is_ambiguous"])
    print(f"Loaded {len(samples)} AMBROSIA samples (split={split})")
    print(f"  Ambiguous: {n_ambig}, Unambiguous: {len(samples) - n_ambig}")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    return samples


def stratified_subsample(samples: list[dict], max_samples: int, seed: int = 42) -> list[dict]:
    """Stratified subsample preserving (is_ambiguous, ambig_type) ratios."""
    rng = random.Random(seed)
    groups: dict[tuple, list] = {}
    for s in samples:
        key = (s["is_ambiguous"], s["ambig_type"])
        groups.setdefault(key, []).append(s)
    total = len(samples)
    picked: list[dict] = []
    for grp in groups.values():
        n = max(1, round(len(grp) / total * max_samples))
        n = min(n, len(grp))
        picked.extend(rng.sample(grp, n))
    if len(picked) > max_samples:
        picked = rng.sample(picked, max_samples)
    rng.shuffle(picked)
    return picked


def resolve_db_path(ambrosia_dir: str, db_file: str) -> str:
    """Map the CSV's ``db_file`` (``data/<...>.sqlite``) to a real path."""
    return os.path.join(ambrosia_dir, db_file.replace("data/", "", 1))
