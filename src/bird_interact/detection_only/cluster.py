"""SE+AST clustering helpers.

A tiny wrapper on top of `ast_clustering.are_structurally_equivalent` that
mirrors the paper-canonical greedy-assignment scheme:

  for sql in samples:
      placed = False
      for cluster in clusters:
          if are_structurally_equivalent(sql, cluster[0], θ):
              cluster.append(sql); placed = True; break
      if not placed:
          clusters.append([sql])

θ (the equivalence threshold) is exposed as a parameter — the seedvar runs
used **0.70** for SE+AST sampling and **0.60 (GLM) / 0.50 (MM, Qwen)** for
MGA's AST grouping. See FINDINGS §16.4 for the temperature-scheduled
calibration.

The diff helpers (`analyze_cluster_differences`, `format_diff_text`)
produce the "Structural Disagreements" block that the analysis prompt
references for SE+AST and MGA-with-AST.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from . import ast_clustering as _ast

logger = logging.getLogger(__name__)


def cluster_by_ast(
    sql_list: list[Optional[str]],
    threshold: float = 0.70,
) -> list[list[str]]:
    """Greedy sequential clustering by AST jaccard."""
    clusters: list[list[str]] = []
    for sql in (s for s in sql_list if s):
        placed = False
        for cluster in clusters:
            try:
                if _ast.are_structurally_equivalent(sql, cluster[0], threshold=threshold):
                    cluster.append(sql)
                    placed = True
                    break
            except BaseException as e:
                logger.warning(f"AST comparison failed, treating as different: {e}")
                break
        if not placed:
            clusters.append([sql])
    return clusters


def compute_entropy(clusters: list[list[str]]) -> float:
    """Shannon entropy over the cluster size distribution."""
    if not clusters:
        return 0.0
    total = sum(len(c) for c in clusters)
    if total == 0:
        return 0.0
    h = 0.0
    for c in clusters:
        p = len(c) / total
        if p > 0:
            h -= p * math.log2(p)
    return h


_SET_COMPONENT_TYPES = (
    "select_columns", "from_tables", "where_conditions",
    "group_by", "aggregations",
)


def analyze_cluster_differences(clusters: list[list[str]]) -> dict:
    """Extract decision variables (varying AST components) from cluster reps.

    Returns a dict with `component_summary` mapping each component type to
    {'common': sorted list, 'varying': sorted list}. Empty `component_summary`
    means we couldn't parse two or more clusters — typically because every
    sampled SQL was malformed.
    """
    if len(clusters) < 2:
        return {"component_summary": {}, "num_parseable": len(clusters)}

    cluster_components: list[dict | None] = []
    for cluster in clusters:
        ast = _ast.parse_and_normalize(cluster[0])
        cluster_components.append(_ast.extract_components(ast) if ast is not None else None)

    parseable = [c for c in cluster_components if c is not None]
    if len(parseable) < 2:
        return {"component_summary": {}, "num_parseable": len(parseable)}

    summary: dict[str, dict] = {}
    for comp in _SET_COMPONENT_TYPES:
        sets = [c[comp] for c in cluster_components if c is not None and comp in c]
        if not sets:
            continue
        common = set.intersection(*sets) if sets else set()
        union = set.union(*sets) if sets else set()
        varying = union - common
        summary[comp] = {"common": sorted(common), "varying": sorted(varying)}
    return {"component_summary": summary, "num_parseable": len(parseable)}


def format_diff_text(diff: dict | None) -> str:
    """Render `analyze_cluster_differences` output as a prompt-friendly block."""
    if not diff:
        return ""
    comp_summary = diff.get("component_summary", {})
    labels = {
        "from_tables": "Tables used",
        "select_columns": "Columns/metrics selected",
        "where_conditions": "Filter conditions",
        "group_by": "Grouping",
        "aggregations": "Aggregation functions",
    }
    lines: list[str] = []
    for comp_type, label in labels.items():
        s = comp_summary.get(comp_type, {})
        varying = s.get("varying", [])
        common = s.get("common", [])
        if not varying:
            continue
        lines.append(f"  {label}:")
        if common:
            lines.append(f"    All clusters agree on: {', '.join(sorted(common)[:6])}")
        lines.append(f"    Clusters DISAGREE on: {', '.join(sorted(varying)[:8])}")
    if not lines:
        return ""
    return "Detected disagreements between SQL interpretations:\n" + "\n".join(lines)
