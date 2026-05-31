"""Encoder judge: maps a clarification question → GT term (or unlabeled / unanswerable).

We use the **released BIRD-Interact `user_simulator_encoder_v2`** template
(under `bird_interact.user_simulator.prompts`) without going through the
agent-side `SampleStatus` machinery — the detection pipeline doesn't need
phase tracking or budgets, so we just fill the placeholders directly.

Encoder output format (paper-canonical):
  <think> ... </think>
  <s> labeled("term") | unlabeled("segment") | unanswerable() </s>

We parse out the action and classify as 'labeled' / 'unlabeled' / 'unanswerable'.
A 'labeled' result with a matched_term name credits one GT term in the
recall denominator (the aggregator does the per-sample matching).

By default we also force `chat_template_kwargs.enable_thinking = False` on
vLLM hosts. Encoder CoT is structured-output overhead at best, and on
Qwen3.5 it routinely truncates the action — about 37% of encoder calls hit
the 2K token cap in thinking mode (see FINDINGS).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from openai import OpenAI

from bird_interact.user_simulator.prompts import USER_SIMULATOR_ENCODER
from bird_interact.user_simulator.sql_parser import segment_sql

logger = logging.getLogger(__name__)


def _segment_sql_safe(sql) -> str:
    """Try to segment SQL into named clauses; fall back to the raw SQL on parse error."""
    if not sql:
        return ""
    sqls = [sql] if isinstance(sql, str) else list(sql)
    out_parts: list[str] = []
    for i, s in enumerate(sqls):
        if i > 0:
            out_parts.append("===")
        try:
            for clause, text in segment_sql(s):
                out_parts.append(f"{clause}:\n{text}\n")
        except Exception:
            out_parts.append(f"QUERY:\n{s}\n")
    return "\n".join(out_parts).strip()


def build_encoder_prompt(
    question_text: str,
    sample: dict,
    db_schema: str,
    user_sim_prompt_version: str = "v2",
) -> str:
    """Fill the released encoder template directly from a sample dict."""
    template = USER_SIMULATOR_ENCODER[user_sim_prompt_version]
    amb_json = {
        "user_query_ambiguity": sample.get("user_query_ambiguity", {}) or {},
        "knowledge_ambiguity": sample.get("knowledge_ambiguity", []) or [],
    }
    sql_segments = _segment_sql_safe(sample.get("sol_sql", ""))
    return (
        template
        .replace("[[clarification_Q]]", question_text)
        .replace("[[amb_json]]", json.dumps(amb_json, indent=2))
        .replace("[[SQL_Glot]]", sql_segments)
        .replace("[[DB_schema]]", db_schema)
    )


def parse_encoder_response(raw: str) -> str:
    """Strip <think>...</think> and <s>...</s> tags; return inner text."""
    text = raw or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    cut = text.find("</s>")
    if cut != -1:
        text = text[:cut]
    if "<s>" in text:
        text = text[text.find("<s>"):].replace("<s>", "")
    return text.strip()


def classify_encoder_action(parsed: str) -> tuple[str, Optional[str]]:
    """Return (classification, matched_term_or_None)."""
    lower = (parsed or "").lower()
    if "labeled(" in lower:
        m = re.search(r'labeled\(\s*["\'](.+?)["\']\s*\)', parsed, re.IGNORECASE)
        return "labeled", (m.group(1) if m else None)
    if "unlabeled(" in lower:
        m = re.search(r'unlabeled\(\s*["\'](.+?)["\']\s*\)', parsed, re.IGNORECASE)
        return "unlabeled", (m.group(1) if m else None)
    return "unanswerable", None


def judge_with_encoder(
    client: OpenAI,
    model_id: str,
    question_text: str,
    sample: dict,
    db_schema: str,
    user_sim_prompt_version: str = "v2",
    extra_body: dict | None = None,
    max_tokens: int = 2048,
) -> dict:
    """Run the encoder on ONE clarification question.

    Returns:
        {
          'question': str,
          'encoder_raw': str,
          'encoder_parsed': str,
          'classification': 'labeled' | 'unlabeled' | 'unanswerable' | 'error',
          'matched_term': Optional[str],
        }
    """
    prompt = build_encoder_prompt(question_text, sample, db_schema, user_sim_prompt_version)
    try:
        kwargs = dict(model=model_id, messages=[{"role": "user", "content": prompt}],
                      temperature=0.0, max_tokens=max_tokens)
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"Encoder call failed: {e}")
        return {
            "question": question_text,
            "encoder_raw": f"ERROR: {e}",
            "encoder_parsed": "",
            "classification": "error",
            "matched_term": None,
        }

    parsed = parse_encoder_response(raw)
    classification, matched_term = classify_encoder_action(parsed)
    return {
        "question": question_text,
        "encoder_raw": raw,
        "encoder_parsed": parsed,
        "classification": classification,
        "matched_term": matched_term,
    }


def default_encoder_extra_body(force_no_thinking: bool = True) -> dict | None:
    """vLLM extra_body to disable encoder-side CoT (recommended)."""
    if not force_no_thinking:
        return None
    return {"chat_template_kwargs": {"enable_thinking": False}}
