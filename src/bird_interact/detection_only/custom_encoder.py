"""Multi-label custom encoder (paper-canonical since 2026-05-01).

Drop-in replacement for ``encoder.judge_with_encoder`` that lets ONE
clarification question credit MULTIPLE labeled ambiguity terms in the
same call. Without this, the standard single-label encoder picks one
term per question, and any sample where a single question semantically
addresses both a ``knowledge_linking_ambiguity`` (UQA, user-vernacular)
term AND its ``knowledge_ambiguity`` (KA, canonical masked-KB name) peer
gets only one row credit. That artificially floors per-row
``knowledge_ambiguity`` recall at ~20-26%, even though ~87% of those
terms have a peer the model successfully clarifies. The multi-label
encoder lifts that ceiling to ~73-80% (FINDINGS §13).

Key shape changes vs ``encoder.judge_with_encoder``:

- The encoder is prompted to emit ``labeled(primary, also=[...])`` when a
  question maps to multiple labeled terms; the standard
  ``labeled("X")`` form is still accepted and produces ``also=[]``.
- The returned dict adds three fields:

  * ``primary_term`` — same as the legacy ``matched_term`` (kept for
    drop-in compatibility);
  * ``also_terms``  — additional labeled terms the question covers;
  * ``matched_terms`` — primary + also, deduped, lowercase-normalised.
    The recall scorer iterates over this list and credits every term.

- The decoder hand-off remains ``labeled("primary")`` byte-for-byte, so
  no spoiler reaches the agent in a-Interact runs.

Compatibility: existing code that reads ``matched_term`` keeps working
because it is set to ``primary_term``. Upgraded scorers should iterate
over ``matched_terms`` instead.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from openai import OpenAI

from .custom_encoder_prompts import USER_SIMULATOR_CUSTOM_ENCODER

logger = logging.getLogger(__name__)


# ── prompt assembly ─────────────────────────────────────────────────────


def _segment_sql_safe(reference_sql) -> str:
    """Stringify a list/str of SQL — same convention as the legacy encoder."""
    if isinstance(reference_sql, list):
        return "\n".join(reference_sql)
    return reference_sql or ""


def build_custom_encoder_prompt(
    question: str,
    sample: dict,
    db_schema: str,
    user_sim_prompt_version: str = "v2",
) -> str:
    """Fill the multi-label encoder template from a sample dict."""
    template = USER_SIMULATOR_CUSTOM_ENCODER.get(user_sim_prompt_version)
    if template is None:
        raise ValueError(f"Unknown prompt version: {user_sim_prompt_version}")

    amb_payload = {
        "user_query_ambiguity": sample.get("user_query_ambiguity", {}) or {},
        "knowledge_ambiguity": sample.get("knowledge_ambiguity", []) or [],
    }
    reference_sql = sample.get("sol_sql", "")
    prompt = (
        template
        .replace("[[clarification_Q]]", question)
        .replace("[[amb_json]]", json.dumps(amb_payload, indent=2))
        .replace("[[SQL_Glot]]", _segment_sql_safe(reference_sql))
        .replace("[[DB_schema]]", db_schema)
    )
    return prompt


# ── response parsing ────────────────────────────────────────────────────

# labeled("X")  OR  labeled('X')  OR  labeled("X", also=["Y", "Z"])
# Tolerates whitespace, mixed quotes, optional trailing comma.
# Word-boundary lookbehind avoids matching the trailing 'labeled' in 'unlabeled'.
_LABELED_RE = re.compile(
    r"""(?<![A-Za-z_])labeled\s*\(
        \s*["']([^"']+)["']
        \s*
        (?:
            ,\s*also\s*=\s*\[
            ([^\]]*)
            \]\s*
        )?
        ,?\s*\)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_UNLABELED_RE = re.compile(r"""unlabeled\s*\(\s*["']([^"']+)["']\s*\)""", re.IGNORECASE)
_UNANSWERABLE_RE = re.compile(r"unanswerable\s*\(\s*\)", re.IGNORECASE)
_QUOTED_RE = re.compile(r"""["']([^"']+)["']""")


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return text


def _strip_s_tags(text: str) -> str:
    cut = text.find("</s>")
    if cut != -1:
        text = text[:cut].strip()
    if "<s>" in text:
        text = text.split("<s>", 1)[1].strip()
    return text


def parse_custom_encoder_response(raw: str) -> dict:
    """Parse the multi-label encoder's raw response into a structured judgment.

    Returns:
        {
          "classification":  "labeled" | "unlabeled" | "unanswerable" | "error",
          "primary_term":    str | None,
          "also_terms":      list[str],            # additional matched (labeled-only)
          "matched_terms":   list[str],            # primary + also, deduped
          "matched_term":    str | None,           # COMPAT alias = primary_term
          "encoder_parsed":  str,
          "encoder_raw":     str,
        }
    """
    out = {
        "classification": "unanswerable",
        "primary_term": None,
        "also_terms": [],
        "matched_terms": [],
        "matched_term": None,
        "encoder_parsed": "",
        "encoder_raw": raw,
    }
    cleaned = _strip_s_tags(_strip_think(raw or ""))
    out["encoder_parsed"] = cleaned

    m = _LABELED_RE.search(cleaned)
    if m:
        primary = (m.group(1) or "").strip()
        also_raw = m.group(2) or ""
        also = [s.strip() for s in _QUOTED_RE.findall(also_raw)]
        seen = {primary.lower()}
        also_clean: list[str] = []
        for t in also:
            if not t or not re.search(r"[A-Za-z0-9]", t):
                # skip empty quotes / punctuation-only matches
                continue
            lk = t.lower()
            if lk in seen:
                continue
            seen.add(lk)
            also_clean.append(t)
        out["classification"] = "labeled"
        out["primary_term"] = primary
        out["also_terms"] = also_clean
        out["matched_terms"] = [primary] + also_clean
        out["matched_term"] = primary
        return out

    m = _UNLABELED_RE.search(cleaned)
    if m:
        seg = (m.group(1) or "").strip()
        out["classification"] = "unlabeled"
        out["primary_term"] = seg
        out["matched_term"] = seg
        out["matched_terms"] = [seg] if seg else []
        return out

    if _UNANSWERABLE_RE.search(cleaned):
        out["classification"] = "unanswerable"
        return out

    return out


# ── live encoder call ───────────────────────────────────────────────────


def judge_with_custom_encoder(
    client: OpenAI,
    model_id: str,
    question_text: str,
    sample: dict,
    db_schema: str,
    user_sim_prompt_version: str = "v2",
    extra_body: Optional[dict] = None,
    max_tokens: int = 2048,
) -> dict:
    """Drop-in multi-label replacement for ``encoder.judge_with_encoder``.

    Same signature as the legacy version. Returns the same shape with three
    extra fields (``primary_term``, ``also_terms``, ``matched_terms``).
    Existing scorers reading ``matched_term`` keep working; upgraded
    scorers iterate over ``matched_terms`` to credit the full set.
    """
    prompt = build_custom_encoder_prompt(
        question_text, sample, db_schema,
        user_sim_prompt_version=user_sim_prompt_version,
    )
    try:
        kwargs = dict(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"Custom encoder call failed: {e}")
        return {
            "question": question_text,
            "encoder_raw": f"ERROR: {e}",
            "encoder_parsed": "",
            "classification": "error",
            "primary_term": None,
            "also_terms": [],
            "matched_terms": [],
            "matched_term": None,
        }

    parsed = parse_custom_encoder_response(raw)
    parsed["question"] = question_text
    return parsed
