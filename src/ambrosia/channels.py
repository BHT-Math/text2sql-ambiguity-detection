"""LLM channels that produce candidate SQLs for AMBROSIA coverage scoring.

Four channels feed the same downstream ``predicted_sqls`` field consumed by
``union.py`` and ``eval_lib.evaluate_coverage_execution``:

| Channel    | Calls / sample | Temp        | Notes                                  |
|------------|:--------------:|:-----------:|----------------------------------------|
| ``baseline`` | 1            | 0           | AMBROSIA-native "write all SQL" prompt |
| ``direct``   | 1 + N        | 0           | Two-stage Self-Introspection (Sᵢ)      |
| ``mcs``      | 10           | 0.7         | Independent single-SQL samples         |
| ``spmi``     | 1            | 1.0 / 1.3   | One call, asks for up to 10 interps    |

The four channels never share their *parsing* logic explicitly because the
formats differ — ``baseline``/``spmi`` are multi-SQL responses (blank-line
separated), ``direct`` Stage 2 + ``mcs`` are single-SQL responses (fenced),
``direct`` Stage 1 is a numbered list of NL paraphrases — but they share
``_strip_think_tags`` and a couple of low-level utilities defined at the top.

All record dicts follow the same superset schema so ``union.py`` and
``eval_lib`` can consume any of them. The MCS / SPMI channels intentionally
write their output under ``mg_sqls`` (not ``predicted_sqls``) so that
``union.py`` can distinguish the "sampling" channel from the Direct channel
when pooling. ``run_baseline`` and the final Direct stage write
``predicted_sqls`` directly because they are terminal — no union step.

Prompts are verbatim from the paper's appendix table arms so this package
reproduces the appendix numbers without rerunning the prompt-design
experiments.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import random
import re

from openai import OpenAI

logger = logging.getLogger(__name__)


# ── shared parsing helpers ──────────────────────────────────────────────


def _strip_think_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def _parse_single_sql(raw: str) -> str:
    """Pull one SQL statement out of a fenced or unfenced response."""
    raw = _strip_think_tags(raw)
    m = re.search(r"```(?:sql)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    sql = m.group(1).strip() if m else raw.strip()
    sql = re.sub(r"^\s*sql\s*", "", sql, flags=re.IGNORECASE).strip()
    sql = sql.rstrip(";").strip()
    if not any(k in sql.upper() for k in ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE")):
        return ""
    return sql


def parse_multi_sql(raw: str) -> list[str]:
    """Multi-SQL parser used by baseline and SPMI (blank-line separated)."""
    raw = _strip_think_tags(raw)
    raw = re.sub(r"```sql\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    out: list[str] = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        block = block.strip()
        if not block:
            continue
        block = re.sub(r"^(?:\d+[\.\)]\s*|Query\s*\d+\s*:\s*)", "", block,
                       flags=re.IGNORECASE).strip()
        if not block:
            continue
        if any(kw in block.upper() for kw in ("SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE")):
            block = block.rstrip(";").strip()
            if block:
                out.append(block)
    return out


def _parse_numbered_list(raw: str) -> list[str]:
    """Direct Stage-1 parser: numbered list (``1.``, ``1)``, ``(1)``)."""
    raw = _strip_think_tags(raw)
    items: list[str] = []
    cur: list[str] = []
    for line in raw.splitlines():
        m = re.match(r"^\s*\(?(\d+)[\.\)]\s*(.*)$", line)
        if m:
            if cur:
                items.append(" ".join(cur).strip())
            cur = [m.group(2).strip()]
        elif cur and line.strip():
            cur.append(line.strip())
    if cur:
        items.append(" ".join(cur).strip())
    return [it for it in items if it]


# ── prompts (verbatim) ──────────────────────────────────────────────────


_BASELINE_SYSTEM = (
    "The task is to write SQL queries based on the provided questions in English. "
    "Questions can take the form of an instruction or command and can be ambiguous, "
    "meaning they can be interpreted in different ways. In such cases, write all "
    "possible SQL queries corresponding to different interpretations and separate "
    "each SQL query with an empty line."
)

_DIRECT_STAGE1_SYSTEM = (
    "The task is to analyse questions in English about a SQL database. "
    "Questions can take the form of an instruction or command and can be "
    "ambiguous, meaning they can be interpreted in different ways. In such "
    "cases, enumerate every distinct interpretation that would correspond "
    "to a different SQL query against the given schema.\n\n"
    "CRITICAL: do NOT collapse alternative readings into a single 'best' "
    "interpretation. If the question has multiple plausible readings — for "
    "example because of unclear modifier attachment (which item does a "
    "clause modify?), quantifier or conjunction scope (does 'all X and Y' "
    "mean each of both groups, or only their intersection?), vague entity "
    "reference (which column / table / value is meant?), or any other "
    "structural or lexical ambiguity — list EACH reading as a separate "
    "paraphrase. List them all, even if one feels more natural than "
    "the others.\n\n"
    "For each interpretation, write a short, unambiguous paraphrase of the "
    "question that resolves that one reading. Number paraphrases and "
    "separate them with an empty line.\n\n"
    "Output format (strict):\n"
    "1. <paraphrase 1>\n\n"
    "2. <paraphrase 2>\n\n"
    "...\n\n"
    "If the question is unambiguous (only one plausible reading against "
    "this schema), output exactly one item:\n"
    "1. <the obvious reading>"
)

_DIRECT_STAGE2_SYSTEM = (
    "You are a SQL expert. Given a database schema and an UNAMBIGUOUS "
    "question, write a single SQL query that answers it.\n\n"
    "Rules:\n"
    "- Output ONLY the SQL inside a ```sql code block.\n"
    "- One query only. No explanations.\n"
    "- Use the dialect of the provided schema (PostgreSQL or SQLite — match "
    "what the schema shows)."
)

_MCS_SYSTEM = (
    "Write a single SQL query that answers the question against the given "
    "schema. Output ONLY the SQL inside a ```sql code block — no commentary."
)

_SPMI_SYSTEM = (
    "The task is to write SQL queries based on the provided question in English. "
    "The question can be ambiguous, meaning it can be interpreted in different ways. "
    "Write up to 10 different valid SQL queries, each representing a distinct "
    "interpretation of the question. Separate each SQL query with an empty line. "
    "If the question is unambiguous, write just one SQL query. "
    "Output ONLY the SQL queries — no explanation, no markdown fences."
)


# ── per-channel runners ─────────────────────────────────────────────────


def _no_think_body(no_thinking: bool) -> dict | None:
    return {"chat_template_kwargs": {"enable_thinking": False}} if no_thinking else None


def _record_passthrough(sample: dict) -> dict:
    """Common fields written to every output record."""
    return {
        "instance_id": sample["instance_id"],
        "question": sample["question"],
        "is_ambiguous": sample["is_ambiguous"],
        "ambig_type": sample["ambig_type"],
        "interpretation_type": sample["interpretation_type"],
        "domain": sample["domain"],
        "question_type": sample["question_type"],
        "db_file": sample.get("db_file", ""),
        "schema_ddl": sample.get("schema_ddl", ""),
        "_raw_db_dump": sample.get("_raw_db_dump", ""),
        "gt_sqls": sample.get("gt_sqls", []),
        "ambig_sqls": sample.get("ambig_sqls", []),
    }


# Baseline ───────────────────────────────────────────────────────────────

def _run_baseline_one(
    client: OpenAI, model_id: str, sample: dict, extra_body: dict | None,
    max_tokens: int,
) -> dict:
    user_msg = (
        f"SQL database dump:\n{sample.get('_raw_db_dump') or sample.get('schema_ddl', '')}"
        f"\n\nQuestion: {sample['question']}"
    )
    raw = ""
    try:
        kwargs = dict(
            model=model_id,
            messages=[
                {"role": "system", "content": _BASELINE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"baseline LLM call failed for {sample['instance_id']}: {e}")
    sqls = parse_multi_sql(raw)
    return {
        **_record_passthrough(sample),
        "predicted_sqls": sqls,
        "num_predicted": len(sqls),
        "raw_output": raw[:4000],
    }


def run_baseline(
    client: OpenAI, model_id: str, samples: list[dict], *,
    no_thinking: bool, max_tokens: int = 2048, num_threads: int = 8,
) -> list[dict]:
    extra_body = _no_think_body(no_thinking)
    results: list[dict | None] = [None] * len(samples)
    with cf.ThreadPoolExecutor(max_workers=num_threads) as pool:
        fut = {pool.submit(_run_baseline_one, client, model_id, s, extra_body,
                           max_tokens): i for i, s in enumerate(samples)}
        done = 0
        for f in cf.as_completed(fut):
            i = fut[f]
            results[i] = f.result()
            done += 1
            if done % 25 == 0 or done == len(samples):
                logger.info(f"  baseline {done}/{len(samples)}")
    return [r for r in results if r]


# Direct stage 1 ─────────────────────────────────────────────────────────

def _run_stage1_one(
    client: OpenAI, model_id: str, sample: dict, extra_body: dict | None,
    max_tokens: int,
) -> dict:
    schema = sample.get("_raw_db_dump") or sample.get("schema_ddl", "")
    raw = ""
    try:
        kwargs = dict(
            model=model_id,
            messages=[
                {"role": "system", "content": _DIRECT_STAGE1_SYSTEM},
                {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {sample['question']}"},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"direct stage1 failed for {sample['instance_id']}: {e}")
    interps = _parse_numbered_list(raw)
    return {
        **_record_passthrough(sample),
        "interpretations": interps,
        "num_interpretations": len(interps),
        "raw_output": raw[:4000],
    }


def run_direct_stage1(
    client: OpenAI, model_id: str, samples: list[dict], *,
    no_thinking: bool, max_tokens: int = 2048, num_threads: int = 8,
) -> list[dict]:
    extra_body = _no_think_body(no_thinking)
    results: list[dict | None] = [None] * len(samples)
    with cf.ThreadPoolExecutor(max_workers=num_threads) as pool:
        fut = {pool.submit(_run_stage1_one, client, model_id, s, extra_body,
                           max_tokens): i for i, s in enumerate(samples)}
        done = 0
        for f in cf.as_completed(fut):
            i = fut[f]
            results[i] = f.result()
            done += 1
            if done % 25 == 0 or done == len(samples):
                logger.info(f"  direct.stage1 {done}/{len(samples)}")
    return [r for r in results if r]


# Direct stage 2 ─────────────────────────────────────────────────────────

def _gen_sql_for_interp(
    client: OpenAI, model_id: str, schema: str, interpretation: str,
    extra_body: dict | None, max_tokens: int,
) -> tuple[str, str]:
    try:
        kwargs = dict(
            model=model_id,
            messages=[
                {"role": "system", "content": _DIRECT_STAGE2_SYSTEM},
                {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {interpretation}"},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"direct stage2 call failed: {e}")
        return "", ""
    return _parse_single_sql(raw), raw


def _run_stage2_one(
    client: OpenAI, model_id: str, rec: dict, extra_body: dict | None,
    max_tokens: int,
) -> dict:
    schema = rec.get("_raw_db_dump") or rec.get("schema_ddl", "")
    sqls: list[str] = []
    per: list[dict] = []
    for idx, interp in enumerate(rec.get("interpretations", []) or []):
        sql, raw = _gen_sql_for_interp(client, model_id, schema, interp, extra_body,
                                       max_tokens)
        per.append({"index": idx, "interpretation": interp, "sql": sql, "raw": raw[:1500]})
        if sql:
            sqls.append(sql)
    out = dict(rec)
    out["predicted_sqls"] = sqls
    out["num_predicted"] = len(sqls)
    out["per_interpretation"] = per
    out.pop("raw_output", None)
    return out


def run_direct_stage2(
    client: OpenAI, model_id: str, stage1_records: list[dict], *,
    no_thinking: bool, max_tokens: int = 1024, num_threads: int = 8,
) -> list[dict]:
    extra_body = _no_think_body(no_thinking)
    results: list[dict | None] = [None] * len(stage1_records)
    with cf.ThreadPoolExecutor(max_workers=num_threads) as pool:
        fut = {pool.submit(_run_stage2_one, client, model_id, r, extra_body,
                           max_tokens): i for i, r in enumerate(stage1_records)}
        done = 0
        for f in cf.as_completed(fut):
            i = fut[f]
            results[i] = f.result()
            done += 1
            if done % 25 == 0 or done == len(stage1_records):
                logger.info(f"  direct.stage2 {done}/{len(stage1_records)}")
    return [r for r in results if r]


# MCS ────────────────────────────────────────────────────────────────────

def _mcs_one_call(
    client: OpenAI, model_id: str, schema: str, question: str,
    temperature: float, max_tokens: int, seed: int, extra_body: dict | None,
) -> tuple[str, str]:
    try:
        kwargs = dict(
            model=model_id,
            messages=[
                {"role": "system", "content": _MCS_SYSTEM},
                {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"mcs call failed (seed={seed}): {e}")
        return "", ""
    return _parse_single_sql(raw), raw


def _run_mcs_one(
    client: OpenAI, model_id: str, sample: dict, num_samples: int,
    temperature: float, max_tokens: int, base_seed: int, extra_body: dict | None,
) -> dict:
    schema = sample.get("_raw_db_dump") or sample.get("schema_ddl", "")
    sqls: list[str] = []
    raws: list[str] = []
    for k in range(num_samples):
        sql, raw = _mcs_one_call(client, model_id, schema, sample["question"],
                                 temperature, max_tokens, base_seed + k, extra_body)
        sqls.append(sql)
        raws.append(raw[:1500])
    valid = [s for s in sqls if s]
    return {
        **_record_passthrough(sample),
        "mg_sqls_raw": sqls,
        "mg_sqls": valid,
        "num_mg_sqls": len(valid),
        "mg_raws": raws,
    }


def run_mcs(
    client: OpenAI, model_id: str, samples: list[dict], *,
    no_thinking: bool, num_samples: int = 10, temperature: float = 0.7,
    max_tokens: int = 1024, num_threads: int = 8, seed: int = 42,
) -> list[dict]:
    extra_body = _no_think_body(no_thinking)
    rng = random.Random(seed)
    base_seeds = [rng.randint(0, 1_000_000) for _ in samples]
    results: list[dict | None] = [None] * len(samples)
    with cf.ThreadPoolExecutor(max_workers=num_threads) as pool:
        fut = {
            pool.submit(_run_mcs_one, client, model_id, s, num_samples, temperature,
                        max_tokens, base_seeds[i], extra_body): i
            for i, s in enumerate(samples)
        }
        done = 0
        for f in cf.as_completed(fut):
            i = fut[f]
            results[i] = f.result()
            done += 1
            if done % 25 == 0 or done == len(samples):
                logger.info(f"  mcs {done}/{len(samples)}")
    return [r for r in results if r]


# SPMI ───────────────────────────────────────────────────────────────────

def _run_spmi_one(
    client: OpenAI, model_id: str, sample: dict, temperature: float,
    max_tokens: int, seed: int, extra_body: dict | None,
) -> dict:
    schema = sample.get("_raw_db_dump") or sample.get("schema_ddl", "")
    raw = ""
    try:
        kwargs = dict(
            model=model_id,
            messages=[
                {"role": "system", "content": _SPMI_SYSTEM},
                {"role": "user", "content": f"SQL database dump:\n{schema}\n\nQuestion: {sample['question']}"},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"spmi call failed (seed={seed}): {e}")
    sqls = parse_multi_sql(raw)
    valid = [s for s in sqls if s and s.strip()]
    return {
        **_record_passthrough(sample),
        "mg_sqls": valid,
        "mg_sqls_raw": sqls,
        "num_mg_sqls": len(valid),
        "spmi_raw": raw[:4000],
    }


def run_spmi(
    client: OpenAI, model_id: str, samples: list[dict], *,
    no_thinking: bool, temperature: float = 1.3, max_tokens: int = 4096,
    num_threads: int = 8, seed: int = 42,
) -> list[dict]:
    extra_body = _no_think_body(no_thinking)
    rng = random.Random(seed)
    seeds = [rng.randint(0, 1_000_000) for _ in samples]
    results: list[dict | None] = [None] * len(samples)
    with cf.ThreadPoolExecutor(max_workers=num_threads) as pool:
        fut = {
            pool.submit(_run_spmi_one, client, model_id, s, temperature, max_tokens,
                        seeds[i], extra_body): i
            for i, s in enumerate(samples)
        }
        done = 0
        for f in cf.as_completed(fut):
            i = fut[f]
            results[i] = f.result()
            done += 1
            if done % 25 == 0 or done == len(samples):
                logger.info(f"  spmi {done}/{len(samples)}")
    return [r for r in results if r]
