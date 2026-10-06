"""Data loading + ground-truth extraction + question parsing.

The lite-300 dataset is shipped under `data/bird-interact-lite/` with one
JSONL of samples plus a per-DB directory holding schema + KB + column
meanings. The helpers below are deliberately small so a reader can audit
exactly what the detection pipeline sees.

GT term extraction matches the paper's encoder mechanism: a clarification
question receives credit if the encoder labels it with the term name. The
ground-truth pool is the union of
  user_query_ambiguity.{critical, non_critical} ∪ knowledge_ambiguity
which together total 1,550 terms across the 300 lite samples
(UQA=1,285, KA=265, disjoint).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def load_samples(data_path: str | Path) -> list[dict]:
    """Load BIRD-Interact-Lite samples (one per line)."""
    samples: list[dict] = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


MISSING_SOL_SQL_HINT = (
    "The public BIRD-Interact release leaves sol_sql and test_cases empty. "
    "Request the ground-truth file from the BIRD team and merge it with "
    "upstream's combine_public_with_gt.py; see data/README.md."
)


def count_missing_sol_sql(samples: Iterable[dict]) -> int:
    """Number of samples without a reference SQL (``sol_sql``)."""
    return sum(1 for s in samples if not s.get("sol_sql"))


def load_schema(db_name: str, data_dir: str | Path) -> str:
    p = Path(data_dir) / db_name / f"{db_name}_schema.txt"
    if not p.exists():
        logger.warning(f"Schema not found: {p}")
        return ""
    return p.read_text()


def load_kb(db_name: str, data_dir: str | Path) -> list[dict]:
    p = Path(data_dir) / db_name / f"{db_name}_kb.jsonl"
    if not p.exists():
        logger.warning(f"KB not found: {p}")
        return []
    entries: list[dict] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_column_meanings(db_name: str, data_dir: str | Path) -> str:
    """Format column meanings as `key: value` lines, matching the agent view."""
    p = Path(data_dir) / db_name / f"{db_name}_column_meaning_base.json"
    if not p.exists():
        logger.warning(f"Column meanings not found: {p}")
        return ""
    with open(p) as f:
        meanings = json.load(f)
    return "\n".join(f"{k}: {v}" for k, v in meanings.items())


def filter_kb_for_sample(all_kb: list[dict], sample: dict) -> list[dict]:
    """Remove KB entries that are masked for this sample.

    Each `knowledge_ambiguity` entry may carry a `deleted_knowledge` id
    referencing an entry in `<db>_kb.jsonl`; those entries are stripped from
    the prompt so the model has to ask. Entries without a matching id are
    left visible.
    """
    ambiguities = sample.get("knowledge_ambiguity", []) or []
    if not ambiguities:
        return all_kb
    deleted_ids = {a.get("deleted_knowledge") for a in ambiguities
                   if a.get("deleted_knowledge") is not None}
    return [e for e in all_kb if e.get("id") not in deleted_ids]


def get_gt_terms(sample: dict) -> list[dict]:
    """Return all ground-truth ambiguity terms for a sample.

    Fields: term, type, critical (bool), is_mask (bool). Includes both
    user_query_ambiguity (critical + non_critical) and knowledge_ambiguity.
    """
    gt: list[dict] = []
    uqa = sample.get("user_query_ambiguity", {}) or {}
    ka_list = sample.get("knowledge_ambiguity", []) or []
    ka_terms_lower = {(a.get("term", "") or "").lower() for a in ka_list
                      if a.get("is_mask", False)}
    for amb in uqa.get("critical_ambiguity", []) or []:
        gt.append({
            "term": amb.get("term", ""),
            "type": amb.get("type", "unknown"),
            "critical": True,
            "is_mask": (amb.get("term", "") or "").lower() in ka_terms_lower,
        })
    for amb in uqa.get("non_critical_ambiguity", []) or []:
        gt.append({
            "term": amb.get("term", ""),
            "type": amb.get("type", "unknown"),
            "critical": False,
            "is_mask": False,
        })
    for amb in ka_list:
        gt.append({
            "term": amb.get("term", ""),
            "type": amb.get("type", "knowledge_ambiguity"),
            "critical": True,
            "is_mask": bool(amb.get("is_mask", False)),
        })
    return gt


# ── Response parsing ────────────────────────────────────────────────────

_LEADING_LABEL_RE = re.compile(
    r"^\*+\s*(ambiguity|term|clarification|definition|item|topic)\s*:?\s*\*+\s*",
    re.IGNORECASE,
)
_LEADING_BARE_LABEL_RE = re.compile(
    r"^(ambiguity|term|clarification|definition|item|topic)\s*[:\-—]\s*",
    re.IGNORECASE,
)
_BOLD_RE = re.compile(r"\*+")
_DASH_LEADER_RE = re.compile(r"^[\-—–:]\s*")
_QUOTED_RE = re.compile(r"['\"](.+?)['\"]")
_WS_RE = re.compile(r"\s+")


def _dedup_signature(text: str) -> str:
    t = _LEADING_LABEL_RE.sub("", text).strip()
    t = _LEADING_BARE_LABEL_RE.sub("", t).strip()
    t = _BOLD_RE.sub("", t)
    t = _DASH_LEADER_RE.sub("", t)
    t = t.rstrip(".:!? ")
    quoted = _QUOTED_RE.findall(t)
    sig = " ".join(quoted[:2]).lower() if quoted else t.lower()
    return _WS_RE.sub(" ", sig).strip()[:80]


def extract_questions_from_response(content: str) -> list[dict]:
    """Pull clarification questions from a numbered list, deduped.

    Returns [{'question': str}, ...]. Each element is sent verbatim to the
    encoder for labeled()/unlabeled()/unanswerable() judgment.
    """
    candidates: list[str] = []
    for line in (content or "").strip().split("\n"):
        line = line.strip()
        m = re.match(r"^\d+[\.\)]\s*(.*)", line)
        if not m:
            continue
        text = m.group(1).strip()
        text = re.sub(r"^[\[\(]?C\d[\]\)]?\s*[:\-—]?\s*", "", text, flags=re.IGNORECASE).strip()
        if text:
            candidates.append(text)

    seen: dict[str, str] = {}
    for c in candidates:
        key = _dedup_signature(c)
        if not key:
            continue
        existing = seen.get(key)
        if existing is None:
            seen[key] = c
            continue
        if "?" in c and "?" not in existing:
            seen[key] = c
        elif ("?" in c) == ("?" in existing) and len(c) > len(existing):
            seen[key] = c
    return [{"question": q} for q in seen.values()]


def strip_think(text: str) -> str:
    """Strip <think>...</think> blocks (GLM/MM/Qwen reasoning leakage)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return text


def kb_as_markdown(kb_entries: Iterable[dict]) -> str:
    lines = []
    for e in kb_entries:
        if isinstance(e, dict) and e.get("definition"):
            lines.append(f"- {e.get('knowledge','')}: {e.get('definition','')}")
    return "\n".join(lines)


def kb_as_agent_json(kb_entries: Iterable[dict]) -> str:
    """KB as JSON with the fields the agent's get_all_knowledge_definitions() returns."""
    fields = ("id", "knowledge", "description", "definition")
    payload = [{k: e[k] for k in fields if k in e} for e in kb_entries if isinstance(e, dict)]
    return json.dumps(payload, indent=2) if payload else ""


def kb_as_json(kb_entries: Iterable[dict]) -> str:
    payload = [
        {"knowledge": e.get("knowledge", ""), "definition": e.get("definition", "")}
        for e in kb_entries if isinstance(e, dict) and e.get("definition")
    ]
    return json.dumps(payload, indent=2) if payload else "[]"
