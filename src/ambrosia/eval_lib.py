"""AMBROSIA coverage evaluator (execution equivalence).

For each ambiguous sample, AMBROSIA ships a list of ground-truth SQL queries —
one per disambiguated interpretation. Coverage on a sample is:

- **Recall**: fraction of GT SQLs matched by *some* unique predicted SQL.
- **Precision**: fraction of unique predicted SQLs that match *some* GT SQL.
- **AllFound**: ``True`` iff every GT SQL was matched. The headline metric.

Two SQLs match iff their execution results on the per-sample SQLite DB are
equal — Counter-multiset on cell values when neither has ``ORDER BY``,
row-by-row when either does. This is the AMBROSIA-paper protocol and what
our reported numbers use; pass ``--equivalence_threshold 1.01`` (the
default) to keep the AST fallback unreachable.

AST fallback is intentionally a no-op in this standalone package — the
paper-canonical protocol does not use it. If you want an AST fallback for
debugging, substitute your own ``structural_similarity`` implementation
(any sqlglot-based scorer works) by monkey-patching this module before
calling the evaluator.
"""

from __future__ import annotations

import sqlite3
from collections import Counter


def structural_similarity(_a: str, _b: str) -> float:
    """AST-similarity stub — always returns 0.0 in the standalone package.

    Paper numbers use execution equivalence only. See the module docstring
    for how to wire in a real implementation if you want the fallback.
    """
    return 0.0


def execute_sql(db_path: str, sql: str) -> tuple[list | None, str | None]:
    """Execute one SQL against a SQLite DB. Returns ``(rows, error)``."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows, None
    except Exception as e:
        return None, str(e)


def _flatten_to_counter(rows: list) -> Counter:
    """Collapse rows to a multiset of cell values for order-agnostic compare."""
    cells = []
    for row in rows:
        for cell in row:
            cells.append(cell)
    return Counter(cells)


def compare_query_results(
    rows_a: list | None,
    rows_b: list | None,
    sql_a: str = "",
    sql_b: str = "",
) -> bool:
    """AMBROSIA's native result-set comparison.

    ORDER BY anywhere → exact row-by-row equality. Otherwise → multiset
    equality on flattened cell values. Mirrors the AMBROSIA paper exactly.
    """
    if rows_a is None or rows_b is None:
        return False
    has_order = any("order by" in s.lower() for s in (sql_a, sql_b) if s)
    if has_order:
        return rows_a == rows_b
    return _flatten_to_counter(rows_a) == _flatten_to_counter(rows_b)


def dedup_by_ast(sqls: list[str], threshold: float = 0.70) -> list[str]:
    """Cluster SQLs by AST similarity, keep first member per cluster."""
    if not sqls:
        return []
    reps = [sqls[0]]
    for s in sqls[1:]:
        if not any(structural_similarity(s, r) >= threshold for r in reps):
            reps.append(s)
    return reps


def evaluate_coverage_execution(
    db_path: str,
    sampled_sqls: list[str],
    gt_sqls: list[str],
    ast_threshold: float = 0.70,
    dedup_method: str = "execution",
) -> dict:
    """Per-sample AMBROSIA coverage on a SQLite DB.

    Args:
        db_path: per-sample SQLite file (from ``resolve_db_path``).
        sampled_sqls: predicted SQLs from the pipeline.
        gt_sqls: gold interpretations (``ambig_sqls`` on AMBROSIA records).
        ast_threshold: similarity ≥ θ → AST fallback match. Set ≥1.01 to
                       disable the fallback entirely (paper protocol).
        dedup_method: ``"execution"`` (default, AMBROSIA-native) or ``"ast"``.

    Returns:
        Dict with ``recall``, ``precision``, ``f1``, ``all_found``, plus
        per-GT and per-prediction match traces (``method`` ∈ ``{"execution",
        "ast_fallback", None}``).
    """
    valid = [s for s in sampled_sqls if s and s.strip()]
    if not gt_sqls or not valid:
        return {
            "recall": 0.0, "precision": 0.0, "f1": 0.0, "all_found": False,
            "num_gt": len(gt_sqls), "num_pred": len(valid),
            "gt_details": [], "pred_details": [], "execution_errors": 0,
        }

    gt_results = []
    for sql in gt_sqls:
        res, err = execute_sql(db_path, sql)
        gt_results.append({"sql": sql, "results": res, "error": err})

    if dedup_method == "ast":
        deduped = dedup_by_ast(valid, threshold=ast_threshold)
        pred_results = []
        for sql in deduped:
            res, err = execute_sql(db_path, sql)
            pred_results.append({"sql": sql, "results": res, "error": err, "duplicate": False})
        for sql in valid:
            if sql not in deduped:
                pred_results.append({"sql": sql, "results": None, "error": "ast_dup", "duplicate": True})
    else:
        pred_results = []
        seen: list = []
        for sql in valid:
            res, err = execute_sql(db_path, sql)
            is_dup = False
            if res is not None:
                for s in seen:
                    if compare_query_results(res, s, sql):
                        is_dup = True
                        break
                if not is_dup:
                    seen.append(res)
            pred_results.append({"sql": sql, "results": res, "error": err, "duplicate": is_dup})

    unique_preds = [p for p in pred_results if not p["duplicate"]]
    exec_errors = sum(1 for p in pred_results if p["error"] is not None)

    # Recall — for each GT, did SOME unique prediction match?
    gt_matched: list[dict] = []
    for gt in gt_results:
        matched, method = False, None
        if gt["results"] is not None:
            for p in unique_preds:
                if p["results"] is not None and compare_query_results(
                    gt["results"], p["results"], gt["sql"], p["sql"]
                ):
                    matched, method = True, "execution"
                    break
        if not matched:
            for p in unique_preds:
                if structural_similarity(p["sql"], gt["sql"]) >= ast_threshold:
                    matched, method = True, "ast_fallback"
                    break
        gt_matched.append({
            "sql": gt["sql"][:200], "matched": matched,
            "method": method, "exec_error": gt["error"],
        })

    # Precision — for each unique prediction, did it match SOME GT?
    pred_matched: list[dict] = []
    for p in unique_preds:
        matched, method = False, None
        if p["results"] is not None:
            for gt in gt_results:
                if gt["results"] is not None and compare_query_results(
                    p["results"], gt["results"], p["sql"], gt["sql"]
                ):
                    matched, method = True, "execution"
                    break
        if not matched:
            for gt in gt_results:
                if structural_similarity(p["sql"], gt["sql"]) >= ast_threshold:
                    matched, method = True, "ast_fallback"
                    break
        pred_matched.append({"sql": p["sql"][:200], "matched": matched, "method": method})

    rh = sum(1 for g in gt_matched if g["matched"])
    ph = sum(1 for p in pred_matched if p["matched"])
    n_gt = len(gt_matched)
    n_pred = len(pred_matched)
    recall = rh / n_gt if n_gt else 0.0
    precision = ph / n_pred if n_pred else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "all_found": rh == n_gt,
        "num_gt": n_gt,
        "num_pred": n_pred,
        "num_unique_pred": len(unique_preds),
        "recall_hits": rh,
        "precision_hits": ph,
        "gt_details": gt_matched,
        "pred_details": pred_matched,
        "execution_errors": exec_errors,
    }
