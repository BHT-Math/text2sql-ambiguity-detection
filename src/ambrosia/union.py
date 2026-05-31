"""Pool a Direct channel with a sampling channel, dedup by execution result.

The paper headline is ``Direct ∪ \\MultiGen`` — Direct's per-interpretation
SQLs ∪ SPMI's single-call multi-interpretation SQLs, deduplicated by their
result-set on the per-sample SQLite DB. ``Direct ∪ Mcs`` (10 independent
samples) is the secondary union arm.

Direct's SQLs are placed first in the pooled list so that when two SQLs from
opposite channels execute to the same result, Direct's textual representation
is kept (Direct's SQLs come from explicit interpretations, so the textual
form carries more meaning per SQL). SQLs that fail to execute are kept as
unique entries — the downstream evaluator's AST fallback can still match
them when the fallback is enabled, and the exec-only protocol treats them
as non-matches anyway.
"""

from __future__ import annotations

import logging
import os

from .eval_lib import compare_query_results, execute_sql
from .data import resolve_db_path

logger = logging.getLogger(__name__)


def dedup_by_execution(db_path: str, sqls: list[str]) -> tuple[list[str], list[dict]]:
    """Dedup SQLs by execution result. Identical to the research repo."""
    seen_results: list[dict] = []  # [{"results": ..., "sql": kept_sql}]
    seen_errors: set[str] = set()
    deduped: list[str] = []
    trace: list[dict] = []
    for sql in sqls:
        if not sql or not sql.strip():
            trace.append({"sql": sql, "kept": False, "reason": "empty"})
            continue
        res, err = execute_sql(db_path, sql)
        if res is not None:
            dup_of = None
            for s in seen_results:
                if compare_query_results(res, s["results"], sql, s["sql"]):
                    dup_of = s["sql"]
                    break
            if dup_of is None:
                seen_results.append({"results": res, "sql": sql})
                deduped.append(sql)
                trace.append({"sql": sql[:200], "kept": True, "exec_error": None})
            else:
                trace.append({"sql": sql[:200], "kept": False, "dup_of": dup_of[:200]})
        else:
            key = f"{err}|{sql}"
            if key in seen_errors:
                trace.append({"sql": sql[:200], "kept": False, "reason": "dup-err"})
                continue
            seen_errors.add(key)
            deduped.append(sql)
            trace.append({"sql": sql[:200], "kept": True, "exec_error": err})
    return deduped, trace


def pool_records(
    direct_records: list[dict],
    sampling_records: list[dict],
    ambrosia_dir: str,
    *,
    mode: str = "union",
) -> list[dict]:
    """Pool Direct + sampling channels and dedup by execution per sample.

    Args:
        direct_records: output of ``channels.run_direct_stage2``.
        sampling_records: output of ``channels.run_mcs`` or ``run_spmi``.
        ambrosia_dir: dir containing per-sample SQLite files.
        mode: ``"union"`` (default) or ``"mg_only"``. ``mg_only`` skips
              the Direct channel entirely (ablation arm).

    Returns:
        Per-sample records with ``predicted_sqls`` set to the deduped pool,
        ready for ``eval_lib.evaluate_coverage_execution``.
    """
    if mode not in ("union", "mg_only"):
        raise ValueError(f"unknown union mode: {mode!r}")

    direct_by_id = {r["instance_id"]: r for r in direct_records}
    samp_by_id = {r["instance_id"]: r for r in sampling_records}
    if mode == "union":
        ids = sorted(set(direct_by_id) & set(samp_by_id))
        only_d = set(direct_by_id) - set(samp_by_id)
        only_s = set(samp_by_id) - set(direct_by_id)
        if only_d or only_s:
            logger.warning(
                f"ID mismatch: {len(only_d)} only in direct, {len(only_s)} only in "
                f"sampling. Keeping intersection ({len(ids)}).")
    else:
        ids = sorted(samp_by_id)

    out: list[dict] = []
    for i, iid in enumerate(ids):
        s = samp_by_id[iid]
        d = direct_by_id.get(iid, {}) if mode == "union" else {}
        direct_sqls = list(d.get("predicted_sqls", []) or [])
        mg_sqls = list(s.get("mg_sqls", []) or [])
        pooled = direct_sqls + mg_sqls if mode == "union" else mg_sqls
        db_path = resolve_db_path(ambrosia_dir, s.get("db_file", ""))
        if not pooled:
            deduped, trace = [], []
        elif os.path.exists(db_path):
            deduped, trace = dedup_by_execution(db_path, pooled)
        else:
            deduped, trace = pooled, []
        out.append({
            "instance_id": iid,
            "question": s.get("question", ""),
            "is_ambiguous": s.get("is_ambiguous", False),
            "ambig_type": s.get("ambig_type", ""),
            "interpretation_type": s.get("interpretation_type", ""),
            "domain": s.get("domain", ""),
            "question_type": s.get("question_type", ""),
            "db_file": s.get("db_file", ""),
            "schema_ddl": s.get("schema_ddl", ""),
            "_raw_db_dump": s.get("_raw_db_dump", ""),
            "gt_sqls": s.get("gt_sqls", []),
            "ambig_sqls": s.get("ambig_sqls", []),
            "direct_sqls": direct_sqls,
            "mg_sqls": mg_sqls,
            "predicted_sqls": deduped,
            "num_direct": len(direct_sqls),
            "num_mg": len(mg_sqls),
            "num_pooled": len(pooled),
            "num_predicted": len(deduped),
            "dedup_trace": trace,
        })
        if (i + 1) % 50 == 0 or (i + 1) == len(ids):
            logger.info(
                f"  union {i+1}/{len(ids)} (latest: {iid}, "
                f"direct={len(direct_sqls)}, mg={len(mg_sqls)}, dedup={len(deduped)})"
            )
    return out
