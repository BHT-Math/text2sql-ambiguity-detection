"""Detection-only pipeline runner (Tables 5-7 of the paper).

Four methods (paper name → CLI flag): Si-1 (direct), Si-10 (direct_multi),
Mcs-1 (mga), Mcs-10 (se_ast). All backed by an OpenAI-compatible LLM
endpoint plus a frozen encoder. Pipeline per sample:

  1. Build context: (question, schema, filtered KB, column meanings).
  2. Generate clarification questions via the chosen method.
  3. For each question, call the encoder; classify labeled / unlabeled / unanswerable.
  4. Compute per-sample recall against the GT term pool.
  5. Append the record to the output JSON.

The encoder typically runs against a DIFFERENT model than the detection LLM
(the paper uses Qwen3.5-122B as the encoder for GLM/MM/Gemini detection,
MiniMax-M2.5 as the encoder for Qwen3.5 detection) to avoid self-judging.
That mapping is documented in `REPRODUCING.md`; set --encoder_base_url +
--encoder_model_id accordingly.

This runner does ONE seed. For the paper's mean±std cells the same script
is invoked multiple times with different --seed values and the per-record
outputs are then aggregated; a single seed is sufficient to reproduce the
paper's numbers within seed-variance (≈±1.5 pp on overall recall).

Usage::

    python -m bird_interact.detection_only.pipeline \
        --method se_ast \
        --data_path data/bird-interact-lite/bird_interact_data.jsonl \
        --data_dir  data/bird-interact-lite \
        --output    results/det_se_ast_qwen.json \
        --base_url            "$QWEN35_122B_BASE_URL" \
        --model_id            qwen3.5-122b \
        --encoder_base_url    "$MINIMAX_M25_BASE_URL" \
        --encoder_model_id    minimax-m2.5 \
        --no_thinking \
        --num_threads 8 --seed 64
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from .data import (
    MISSING_SOL_SQL_HINT,
    count_missing_sol_sql,
    extract_questions_from_response,
    filter_kb_for_sample,
    get_gt_terms,
    kb_as_agent_json,
    kb_as_json,
    kb_as_markdown,
    load_column_meanings,
    load_kb,
    load_samples,
    load_schema,
    strip_think,
)
from .prompts import (
    NEUTRAL_ANALYSIS_PROMPT,
    SE_SAMPLING_SYSTEM_PROMPT,
    mga_gen_system,
)
from . import cluster as _cluster
from .encoder import (
    default_encoder_extra_body,
    judge_with_encoder,
)
from .custom_encoder import judge_with_custom_encoder

logger = logging.getLogger("detection_only")


# ── Seeded LLM call ─────────────────────────────────────────────────────

_seed_base: int | None = None
_seed_counter = itertools.count(1)
_seed_lock = threading.Lock()


def _next_seed() -> int | None:
    if _seed_base is None:
        return None
    with _seed_lock:
        return _seed_base + next(_seed_counter)


def _llm_call(client: OpenAI, *, extra_body: dict | None = None, **kwargs):
    """`chat.completions.create` with per-call seed injection."""
    extra = dict(extra_body or {})
    s = _next_seed()
    if s is not None:
        extra["seed"] = s
    if extra:
        kwargs["extra_body"] = extra
    return client.chat.completions.create(**kwargs)


# ── Method-specific generators ──────────────────────────────────────────

def _extract_sql(text: str) -> str | None:
    """Pull a single SQL statement out of an LLM response."""
    if not text or not text.strip():
        return None
    fence = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        sql = fence.group(1).strip()
        if sql:
            return sql
    text = strip_think(text)
    if text and not text.upper().lstrip().startswith(("SELECT", "WITH", "CREATE", "INSERT", "UPDATE", "DELETE")):
        last = list(re.finditer(r"^((?:WITH|SELECT|CREATE|INSERT|UPDATE|DELETE)\b.*)$",
                                text, re.MULTILINE | re.IGNORECASE))
        if last:
            text = text[last[-1].start():].strip()
    m = re.search(r"((?:WITH|SELECT)\b.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip() if text else None


def _parse_mga_interpretations(content: str) -> list[str]:
    """Split MGA generation output into per-interpretation SQL blocks."""
    blocks = re.split(r"\[Interpretation\s+\d+\][\s:]*", content, flags=re.IGNORECASE)
    out: list[str] = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        fence = re.search(r"```(?:sql)?\s*\n?(.*?)```", b, re.DOTALL | re.IGNORECASE)
        if fence:
            out.append(fence.group(1).strip())
        elif re.search(r"\b(SELECT|WITH|CREATE|INSERT)\b", b, re.IGNORECASE):
            m = re.search(r"((?:WITH|SELECT|CREATE|INSERT)\b.*)", b, re.DOTALL | re.IGNORECASE)
            if m:
                out.append(m.group(1).strip())
    return out


def _build_se_sampling_messages(
    question: str, schema: str, kb_entries: list[dict], column_meanings: str,
) -> list[dict]:
    user_parts = [f"## Database Schema\n{schema}"]
    if column_meanings:
        user_parts.append(f"## Column Meanings\n{column_meanings}")
    kb_json = kb_as_agent_json(kb_entries)
    if kb_json:
        user_parts.append(f"## External Knowledge\n{kb_json}")
    user_parts.append(f"## Question\n{question}")
    return [
        {"role": "system", "content": SE_SAMPLING_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _direct_one_call(
    client: OpenAI, model_id: str, question: str, schema: str,
    kb_entries: list[dict], column_meanings: str, *,
    temperature: float, max_tokens: int, extra_body: dict | None,
) -> str:
    """One Direct introspection call. Returns the post-`strip_think` content.

    Uses the shared neutral analysis prompt (NEUTRAL_ANALYSIS_PROMPT + the
    "List every ambiguity you can identify." closing line), matching the
    paper's Direct / Direct-multi runs. No SQL is shown; the evidence is the
    schema, column meanings, and filtered KB.
    """
    kb_md = kb_as_markdown(kb_entries)
    user_parts = [f"## Database Schema\n{schema}"]
    if column_meanings:
        user_parts.append(f"## Column Meanings\n{column_meanings}")
    if kb_md:
        user_parts.append(f"## External Knowledge\n{kb_md}")
    user_parts.append(f"## Question\n{question}")
    user_parts.append("List every ambiguity you can identify.")
    resp = _llm_call(client,
                     model=model_id,
                     messages=[
                         {"role": "system", "content": NEUTRAL_ANALYSIS_PROMPT},
                         {"role": "user", "content": "\n\n".join(user_parts)},
                     ],
                     temperature=temperature,
                     max_tokens=max_tokens,
                     extra_body=extra_body)
    return strip_think(resp.choices[0].message.content or "")


def run_direct(
    client: OpenAI, model_id: str, sample: dict, schema: str, kb_entries: list[dict],
    column_meanings: str, *, max_tokens: int, extra_body: dict | None,
) -> dict:
    """\\Direct: one deterministic introspection call (T=0)."""
    question = sample.get("amb_user_query") or sample.get("user_query") or ""
    content = _direct_one_call(
        client, model_id, question, schema, kb_entries, column_meanings,
        temperature=0.0, max_tokens=max_tokens, extra_body=extra_body,
    )
    return {
        "method": "direct",
        "analysis_response": content,
        "extracted_detections": extract_questions_from_response(content),
        "question": question,
    }


def run_direct_multi(
    client: OpenAI, model_id: str, sample: dict, schema: str, kb_entries: list[dict],
    column_meanings: str, *,
    num_samples: int, temperature: float, max_tokens: int, extra_body: dict | None,
) -> dict:
    """\\DirectMulti (SSI): N calls of the Direct prompt at temperature T.

    Per-call extracted questions are concatenated into a single
    ``extracted_detections`` list with verbatim dedup (a question is dropped
    if its normalised text already appeared in an earlier call). The encoder
    then judges every unique question, so the recall ceiling rises with N at
    the cost of N× the LLM bill. Paper-canonical: N=10, T=0.7.
    """
    question = sample.get("amb_user_query") or sample.get("user_query") or ""

    per_call_responses: list[str] = []
    per_call_questions: list[list] = []
    merged: list = []
    seen: set[str] = set()

    def _key(q) -> str:
        # extract_questions_from_response can yield dicts or strings — normalise.
        if isinstance(q, dict):
            text = q.get("question") or q.get("text") or ""
        else:
            text = str(q)
        return " ".join(text.lower().split())

    for k in range(num_samples):
        try:
            content = _direct_one_call(
                client, model_id, question, schema, kb_entries, column_meanings,
                temperature=temperature, max_tokens=max_tokens, extra_body=extra_body,
            )
        except Exception as e:
            logger.warning(f"direct_multi call {k+1}/{num_samples} failed: {e}")
            per_call_responses.append("")
            per_call_questions.append([])
            continue
        per_call_responses.append(content)
        qs = extract_questions_from_response(content) or []
        per_call_questions.append(qs)
        for q in qs:
            kk = _key(q)
            if not kk or kk in seen:
                continue
            seen.add(kk)
            merged.append(q)

    return {
        "method": "direct_multi",
        "num_samples": num_samples,
        "temperature": temperature,
        "per_call_responses": per_call_responses,
        "per_call_questions": per_call_questions,
        "analysis_response": "\n\n----\n\n".join(
            f"[call {i+1}/{num_samples}]\n{r}" for i, r in enumerate(per_call_responses)
        ),
        "extracted_detections": merged,
        "question": question,
    }


def run_mga(
    client: OpenAI, model_id: str, sample: dict, schema: str, kb_entries: list[dict],
    column_meanings: str, *,
    temperature: float, num_interpretations: int, ast_threshold: float, no_ast: bool,
    max_tokens: int, extra_body: dict | None,
) -> dict:
    question = sample.get("amb_user_query") or sample.get("user_query") or ""
    # Call 1: generate N SQL in one pass.
    gen_user_parts = [f"## Database Schema\n{schema}"]
    if column_meanings:
        gen_user_parts.append(f"## Column Meanings\n{column_meanings}")
    kb_json = kb_as_json(kb_entries)
    if kb_json and kb_json != "[]":
        gen_user_parts.append(f"## External Knowledge\n{kb_json}")
    gen_user_parts.append(
        f"## Question\n{question}\n\nGenerate {num_interpretations} different SQL interpretations:"
    )
    gen_resp = _llm_call(client,
                         model=model_id,
                         messages=[
                             {"role": "system", "content": mga_gen_system(num_interpretations)},
                             {"role": "user", "content": "\n\n".join(gen_user_parts)},
                         ],
                         temperature=temperature,
                         max_tokens=max_tokens,
                         extra_body=extra_body)
    gen_content = strip_think(gen_resp.choices[0].message.content or "")
    sql_list = _parse_mga_interpretations(gen_content)

    # AST cluster (optional) → diff text.
    if no_ast or len(sql_list) < 2:
        clusters = [[s] for s in sql_list]
        diff = None
    else:
        clusters = _cluster.cluster_by_ast(sql_list, threshold=ast_threshold)
        diff = _cluster.analyze_cluster_differences(clusters) if len(clusters) >= 2 else None
    diff_text = _cluster.format_diff_text(diff)
    entropy = _cluster.compute_entropy(clusters)

    # Format interpretations for the analysis call.
    if no_ast or not clusters:
        sql_display = "\n\n".join(f"[Interpretation {i+1}]:\n{s}" for i, s in enumerate(sql_list))
        header = f"## SQL Interpretations ({len(sql_list)} generated in one pass)"
    else:
        group_lines: list[str] = []
        for ci, c in enumerate(clusters):
            group_lines.append(f"[Group {chr(65+ci)}] ({len(c)} interpretation{'s' if len(c) > 1 else ''}):")
            for s in c:
                group_lines.append(f"  {s}")
            group_lines.append("")
        sql_display = "\n".join(group_lines)
        header = (
            f"## SQL Interpretations ({len(sql_list)} generated, grouped into "
            f"{len(clusters)} structurally distinct clusters by sqlglot AST)"
        )

    # Call 2: analyze ambiguities (neutral prompt).
    user_parts = [f"## Original Question\n{question}"]
    if diff_text:
        user_parts.append(f"## Structural Disagreements (from sqlglot AST analysis)\n{diff_text}")
    user_parts.append(f"{header}\n\n{sql_display}")
    user_parts.append("List every ambiguity you can identify.")

    ana_resp = _llm_call(client,
                         model=model_id,
                         messages=[
                             {"role": "system", "content": NEUTRAL_ANALYSIS_PROMPT},
                             {"role": "user", "content": "\n\n".join(user_parts)},
                         ],
                         temperature=0.0,
                         max_tokens=max_tokens,
                         extra_body=extra_body)
    content = strip_think(ana_resp.choices[0].message.content or "")
    return {
        "method": "mga",
        "question": question,
        "sql_samples": sql_list,
        "clusters": [list(c) for c in clusters],
        "entropy": entropy,
        "num_clusters": len(clusters),
        "diff_analysis": diff,
        "diff_text_shown": diff_text,
        "gen_response_raw": gen_content,
        "analysis_response": content,
        "extracted_detections": extract_questions_from_response(content),
    }


def run_se_ast(
    client: OpenAI, model_id: str, sample: dict, schema: str, kb_entries: list[dict],
    column_meanings: str, *,
    temperature: float, num_samples: int, ast_threshold: float, no_ast: bool,
    max_tokens: int, extra_body: dict | None,
) -> dict:
    question = sample.get("amb_user_query") or sample.get("user_query") or ""
    sampling_messages = _build_se_sampling_messages(question, schema, kb_entries, column_meanings)

    sql_list: list[str | None] = []
    for _ in range(num_samples):
        try:
            r = _llm_call(client,
                          model=model_id,
                          messages=sampling_messages,
                          temperature=temperature,
                          max_tokens=max_tokens,
                          extra_body=extra_body)
            sql_list.append(_extract_sql(r.choices[0].message.content or ""))
        except Exception as e:
            logger.warning(f"[{sample.get('instance_id')}] sampling failed: {e}")
            sql_list.append(None)

    valid_sql = [s for s in sql_list if s]
    if no_ast or len(valid_sql) < 2:
        clusters = [[s] for s in valid_sql]
        diff = None
    else:
        clusters = _cluster.cluster_by_ast(valid_sql, threshold=ast_threshold)
        diff = _cluster.analyze_cluster_differences(clusters) if len(clusters) >= 2 else None
    diff_text = _cluster.format_diff_text(diff)
    entropy = _cluster.compute_entropy(clusters)

    if not clusters:
        return {
            "method": "se_ast",
            "question": question,
            "sql_samples": sql_list,
            "clusters": [],
            "entropy": 0.0,
            "num_clusters": 0,
            "diff_analysis": None,
            "diff_text_shown": "",
            "analysis_response": "",
            "extracted_detections": [],
        }

    cluster_reps = [c[0] for c in clusters]
    clusters_text = "\n\n".join(f"[Cluster {i}]:\n{rep}" for i, rep in enumerate(cluster_reps))

    user_parts = [f"## Original Question\n{question}"]
    if diff_text:
        user_parts.append(f"## Structural Disagreements\n{diff_text}")
    user_parts.append(
        f"## SQL Interpretations ({len(cluster_reps)} cluster representatives)\n{clusters_text}"
    )
    user_parts.append("List every ambiguity you can identify.")

    ana_resp = _llm_call(client,
                         model=model_id,
                         messages=[
                             {"role": "system", "content": NEUTRAL_ANALYSIS_PROMPT},
                             {"role": "user", "content": "\n\n".join(user_parts)},
                         ],
                         temperature=0.0,
                         max_tokens=max_tokens,
                         extra_body=extra_body)
    content = strip_think(ana_resp.choices[0].message.content or "")
    return {
        "method": "se_ast",
        "question": question,
        "sql_samples": sql_list,
        "clusters": [list(c) for c in clusters],
        "entropy": entropy,
        "num_clusters": len(clusters),
        "diff_analysis": diff,
        "diff_text_shown": diff_text,
        "analysis_response": content,
        "extracted_detections": extract_questions_from_response(content),
    }


# ── Per-sample driver ───────────────────────────────────────────────────

def _evaluate_detections(
    enc_client: OpenAI, enc_model_id: str,
    detections: list[dict], sample: dict, db_schema: str,
    *, encoder_extra_body: dict | None, user_sim_prompt_version: str,
    encoder_kind: str = "legacy",
) -> dict:
    """Encode each clarification question, classify, score against GT.

    ``encoder_kind`` selects the encoder semantics:

    - ``legacy`` (default) — single-label encoder: one question credits at
      most one GT term. This is the standard BIRD-Interact user-sim
      encoder that everyone in the literature is reproducing.
    - ``multi_label`` (opt-in) — one question can credit up to 3 GT terms
      via ``labeled(primary, also=[...])``. Reproduces the
      ``knowledge_ambiguity`` recall lift in FINDINGS §13. Reads
      ``matched_terms`` (list) per judgment.
    """
    if encoder_kind not in ("multi_label", "legacy"):
        raise ValueError(f"encoder_kind must be 'multi_label' or 'legacy', got {encoder_kind!r}")

    gt_terms = get_gt_terms(sample)
    labeled_matches: set[str] = set()
    valid = 0
    fp = 0
    encoder_errors = 0
    raw_judgments: list[dict] = []

    for det in detections:
        q = det.get("question", "")
        if encoder_kind == "multi_label":
            res = judge_with_custom_encoder(
                enc_client, enc_model_id, q, sample, db_schema,
                user_sim_prompt_version=user_sim_prompt_version,
                extra_body=encoder_extra_body,
            )
            credited = [t.lower().strip() for t in (res.get("matched_terms") or [])
                        if isinstance(t, str) and t.strip()]
        else:
            res = judge_with_encoder(
                enc_client, enc_model_id, q, sample, db_schema,
                user_sim_prompt_version=user_sim_prompt_version,
                extra_body=encoder_extra_body,
            )
            credited = []
            if res.get("matched_term"):
                credited = [res["matched_term"].lower().strip()]

        raw_judgments.append(res)
        if res["classification"] == "labeled" and credited:
            labeled_matches.update(credited)
            valid += 1
        elif res["classification"] == "unlabeled":
            valid += 1
        elif res["classification"] == "unanswerable":
            fp += 1
        elif res["classification"] == "error":
            encoder_errors += 1

    detected_gt = 0
    gt_details: list[dict] = []
    for gt in gt_terms:
        was = gt["term"].lower().strip() in labeled_matches
        if was:
            detected_gt += 1
        gt_details.append({**gt, "detected": was})

    return {
        "total_gt": len(gt_terms),
        "detected_gt": detected_gt,
        "recall": detected_gt / len(gt_terms) if gt_terms else 0.0,
        "valid": valid,
        "fp": fp,
        "encoder_errors": encoder_errors,
        "total_detections": len(detections),
        "labeled_matches": sorted(labeled_matches),
        "gt_terms": gt_details,
        "raw_judgments": raw_judgments,
    }


# Per-DB caches shared across worker threads
_schema_cache: dict[str, str] = {}
_kb_cache: dict[str, list[dict]] = {}
_cm_cache: dict[str, str] = {}
_cache_lock = threading.Lock()


def _load_db_context(db_name: str, data_dir: str) -> tuple[str, list[dict], str]:
    with _cache_lock:
        if db_name not in _schema_cache:
            _schema_cache[db_name] = load_schema(db_name, data_dir)
            _kb_cache[db_name] = load_kb(db_name, data_dir)
            _cm_cache[db_name] = load_column_meanings(db_name, data_dir)
    return _schema_cache[db_name], _kb_cache[db_name], _cm_cache[db_name]


def _process_sample(
    sample: dict, *, args, det_client: OpenAI, enc_client: OpenAI,
    det_extra_body: dict | None, enc_extra_body: dict | None,
) -> dict:
    iid = sample.get("instance_id", "?")
    db_name = sample.get("selected_database", "")
    schema, all_kb, column_meanings = _load_db_context(db_name, args.data_dir)
    kb_entries = filter_kb_for_sample(all_kb, sample)

    try:
        if args.method == "direct":
            base = run_direct(
                det_client, args.model_id, sample, schema, kb_entries, column_meanings,
                max_tokens=args.max_tokens, extra_body=det_extra_body,
            )
        elif args.method == "direct_multi":
            base = run_direct_multi(
                det_client, args.model_id, sample, schema, kb_entries, column_meanings,
                num_samples=args.direct_multi_num_samples,
                temperature=args.temperature,
                max_tokens=args.max_tokens, extra_body=det_extra_body,
            )
        elif args.method == "mga":
            base = run_mga(
                det_client, args.model_id, sample, schema, kb_entries, column_meanings,
                temperature=args.temperature, num_interpretations=args.num_interpretations,
                ast_threshold=args.ast_threshold, no_ast=args.no_ast,
                max_tokens=args.max_tokens, extra_body=det_extra_body,
            )
        elif args.method == "se_ast":
            base = run_se_ast(
                det_client, args.model_id, sample, schema, kb_entries, column_meanings,
                temperature=args.temperature, num_samples=args.num_samples,
                ast_threshold=args.ast_threshold, no_ast=args.no_ast,
                max_tokens=args.max_tokens, extra_body=det_extra_body,
            )
        else:
            raise ValueError(f"unknown method {args.method}")
    except Exception as e:
        logger.exception(f"[{iid}] generation failed: {e}")
        return {"instance_id": iid, "method": args.method, "error": f"generation: {e}"}

    detections = base.get("extracted_detections") or []
    if detections:
        try:
            base["detection"] = _evaluate_detections(
                enc_client, args.encoder_model_id, detections, sample, schema,
                encoder_extra_body=enc_extra_body,
                user_sim_prompt_version=args.user_sim_prompt_version,
                encoder_kind=args.encoder_kind,
            )
        except Exception as e:
            logger.warning(f"[{iid}] encoder eval failed: {e}")
            base["detection"] = {"error": str(e), "total_gt": len(get_gt_terms(sample)),
                                 "detected_gt": 0, "recall": 0.0,
                                 "total_detections": len(detections)}
            base["error"] = f"encoder: {e}"
        n_enc_err = base["detection"].get("encoder_errors", 0)
        if n_enc_err:
            # A failed judgment can't credit its GT term, so this record's recall
            # is understated. Flag it so --resume retries it.
            base["error"] = f"encoder: {n_enc_err}/{len(detections)} judgments failed"
    else:
        gt = get_gt_terms(sample)
        base["detection"] = {
            "total_gt": len(gt), "detected_gt": 0, "recall": 0.0,
            "valid": 0, "fp": 0, "total_detections": 0,
            "labeled_matches": [], "gt_terms": [{**g, "detected": False} for g in gt],
            "raw_judgments": [],
        }
    base["instance_id"] = iid
    base["is_ambiguous"] = bool(sample.get("knowledge_ambiguity"))
    return base


# ── CLI ─────────────────────────────────────────────────────────────────

# Settings that must match for --resume to reuse records from an existing output file.
_RESUME_KEYS = (
    "method", "model_id", "encoder_model_id", "encoder_kind", "seed",
    "temperature", "num_samples", "num_interpretations", "direct_multi_num_samples",
    "ast_threshold", "no_ast", "no_thinking", "user_sim_prompt_version", "analysis_prompt_file",
)


def _run_config(args) -> dict:
    return {k: getattr(args, k) for k in _RESUME_KEYS}


def _load_partial(path: str | None, config: dict) -> tuple[list[dict], set[str]]:
    """Records to keep from an existing output file when resuming.

    Exits if the file was written with different settings, so a new model or
    method never inherits another run's records. Failed records are dropped so
    they get retried.
    """
    if not path or not os.path.exists(path):
        return [], set()
    try:
        with open(path) as f:
            blob = json.load(f)
    except Exception as e:
        logger.warning(f"--resume: could not read {path} ({e}); starting fresh")
        return [], set()
    # Older outputs have no "config" block; compare the header fields they do have.
    saved = blob.get("config") or {
        k: blob[k] for k in ("method", "model_id", "encoder_model_id", "encoder_kind", "seed") if k in blob
    }
    diff = {k: (saved[k], config[k]) for k in saved if k in config and saved[k] != config[k]}
    if diff:
        details = "; ".join(f"{k} is {old!r} in the file but {new!r} now" for k, (old, new) in diff.items())
        raise SystemExit(f"--resume: {path} was written with different settings ({details}). "
                         f"Choose another --output, or delete the file to start over.")
    results = [r for r in (blob.get("results") or []) if not r.get("error")]
    return results, {r.get("instance_id", "") for r in results if r.get("instance_id")}


def _atomic_dump(path: str, blob: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(blob, f, indent=2)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="BIRD-Interact detection-only pipeline (Tables 5–7).")
    p.add_argument("--method", choices=("direct", "direct_multi", "mga", "se_ast"), required=True)

    p.add_argument("--data_path", required=True, help="bird_interact_data.jsonl")
    p.add_argument("--data_dir", required=True, help="dir with per-DB schema/KB/columns")
    p.add_argument("--output", required=True, help="output JSON path")
    p.add_argument("--limit", type=int, default=None, help="cap samples (for smoke tests)")
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--resume", action="store_true",
                   help="skip instance_ids already present in --output")

    # Detection LLM
    p.add_argument("--base_url", required=True)
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model_id", required=True)

    # Encoder LLM
    p.add_argument("--encoder_base_url", required=True)
    p.add_argument("--encoder_api_key", default=os.environ.get("ENCODER_API_KEY", "EMPTY"))
    p.add_argument("--encoder_model_id", required=True)
    p.add_argument("--user_sim_prompt_version", default="v2", choices=("v1", "v2"))
    p.add_argument("--encoder_no_thinking", action="store_true", default=True,
                   help="force chat_template_kwargs.enable_thinking=False on encoder (default ON)")
    p.add_argument("--encoder_kind", choices=("legacy", "multi_label"), default="legacy",
                   help=("legacy (default) = single-label user-sim encoder used by all "
                         "BIRD-Interact baselines. multi_label = opt-in extension where one "
                         "question can credit up to 3 GT terms via labeled(primary, also=[...]); "
                         "boosts knowledge-ambiguity recall."))

    # Method knobs
    p.add_argument("--temperature", type=float, default=None,
                   help="generation temperature (default: 0.7 for se_ast, 1.3 for mga; 0 for direct)")
    p.add_argument("--num_samples", type=int, default=10, help="SE+AST: # SQL samples")
    p.add_argument("--num_interpretations", type=int, default=10, help="MGA: # interpretations")
    p.add_argument("--direct_multi_num_samples", type=int, default=10,
                   help="Direct-Multi (SSI): # independent Direct calls at T>0 (paper-canonical: 10)")
    p.add_argument("--ast_threshold", type=float, default=0.70,
                   help="θ for AST equivalence; paper-canonical: 0.70 (SE+AST), 0.60/0.50 (MGA)")
    p.add_argument("--no_ast", action="store_true",
                   help="skip AST clustering (MGA-no-AST / SE-no-AST ablation)")
    p.add_argument("--max_tokens", type=int, default=16384)
    p.add_argument("--no_thinking", action="store_true",
                   help="force chat_template_kwargs.enable_thinking=False on detection calls")
    p.add_argument("--seed", type=int, default=None,
                   help="base seed (each LLM call uses base + monotonic counter)")
    p.add_argument("--log_level", default="INFO")
    p.add_argument("--analysis_prompt_file", default=None,
                   help=("Path to a UTF-8 text file whose contents REPLACE the shared "
                         "neutral analysis/introspection system prompt used by ALL "
                         "methods (direct, direct_multi, mga, se_ast). Swap in your own "
                         "prompt without editing source. Default: the built-in "
                         "NEUTRAL_ANALYSIS_PROMPT in detection_only/prompts.py."))

    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s")
    logger.info(f"Detection-only pipeline: method={args.method} "
                f"model={args.model_id} encoder={args.encoder_model_id}")

    # Optional: swap in a custom analysis/introspection system prompt. All four
    # methods read this module global as their system prompt, so overriding it
    # here changes every method. See REPRODUCING.md "Customizing prompts".
    if args.analysis_prompt_file:
        global NEUTRAL_ANALYSIS_PROMPT
        NEUTRAL_ANALYSIS_PROMPT = Path(args.analysis_prompt_file).read_text().strip()
        logger.info(f"Overriding analysis prompt from {args.analysis_prompt_file} "
                    f"({len(NEUTRAL_ANALYSIS_PROMPT)} chars)")

    # Per-method temperature defaults (paper-canonical).
    if args.temperature is None:
        args.temperature = {"direct": 0.0, "direct_multi": 0.7,
                            "mga": 1.3, "se_ast": 0.7}[args.method]

    global _seed_base, _seed_counter
    if args.seed is not None:
        _seed_base = int(args.seed)
        _seed_counter = itertools.count(1)

    # Build clients.
    det_client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # Encoder client. A Gemini encoder (model id beginning with "gemini") routes
    # through the Vertex Express wrapper in bird_interact.llm.gemini_client, which
    # exposes the same chat.completions.create() surface as the OpenAI client.
    # The Vertex key is read from NEW_GEMINI / new_gemini in the environment;
    # --encoder_base_url / --encoder_api_key are unused on this path.
    _enc_is_gemini = str(args.encoder_model_id).lower().startswith("gemini")
    if _enc_is_gemini:
        from bird_interact.llm.gemini_client import build_gemini_client
        enc_client = build_gemini_client(backend="vertex_express")
        logger.info(f"Encoder via Vertex Express Gemini: {args.encoder_model_id}")
    else:
        enc_client = OpenAI(api_key=args.encoder_api_key, base_url=args.encoder_base_url)

    det_extra: dict | None = None
    if args.no_thinking:
        det_extra = {"chat_template_kwargs": {"enable_thinking": False}}
    # Gemini ignores vLLM-only chat-template kwargs; keep enc_extra clean so the
    # saved metadata / replay logs aren't misleading.
    enc_extra = None if _enc_is_gemini else default_encoder_extra_body(
        force_no_thinking=args.encoder_no_thinking)

    # Load samples + resume state.
    all_samples = load_samples(args.data_path)
    if args.limit is not None:
        all_samples = all_samples[: args.limit]

    # The encoder prompt shows the reference SQL; without it the judge works
    # from the ambiguity annotations alone. Not fatal, but not the paper's setup.
    n_no_sql = count_missing_sol_sql(all_samples)
    no_sql_msg = (f"{n_no_sql} of {len(all_samples)} samples have no reference SQL (sol_sql), "
                  f"so the encoder prompt's SQL section is empty and its judgments will "
                  f"differ from the paper's setup. {MISSING_SOL_SQL_HINT}")
    if n_no_sql:
        logger.warning(no_sql_msg)

    config = _run_config(args)
    prior_results, done_ids = _load_partial(args.output if args.resume else None, config)
    results: list[dict] = list(prior_results)
    todo = [s for s in all_samples if s.get("instance_id") not in done_ids]
    logger.info(f"Loaded {len(all_samples)} samples; {len(done_ids)} already done; "
                f"processing {len(todo)}")

    # Smoke-test single-sample path is fine to keep parallel; ThreadPoolExecutor
    # tolerates max_workers=1.
    start = time.time()
    save_every = max(1, len(todo) // 20)
    processed = 0
    with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
        futs = {ex.submit(_process_sample, s,
                          args=args, det_client=det_client, enc_client=enc_client,
                          det_extra_body=det_extra, enc_extra_body=enc_extra): s
                for s in todo}
        for fut in as_completed(futs):
            try:
                entry = fut.result()
            except Exception as e:
                entry = {"instance_id": futs[fut].get("instance_id", "?"),
                         "method": args.method, "error": f"worker crashed: {e}"}
            results.append(entry)
            processed += 1
            if processed % save_every == 0:
                _atomic_dump(args.output, {
                    "method": args.method, "model_id": args.model_id,
                    "encoder_model_id": args.encoder_model_id,
                    "encoder_kind": args.encoder_kind,
                    "seed": args.seed, "n_total": len(all_samples),
                    "n_done": len(results),
                    "elapsed_sec": round(time.time() - start, 1),
                    "config": config,
                    "results": results,
                })
                logger.info(f"  Saved checkpoint at {processed}/{len(todo)} "
                            f"({len(results)} cumulative)")
    _atomic_dump(args.output, {
        "method": args.method, "model_id": args.model_id,
        "encoder_model_id": args.encoder_model_id,
        "encoder_kind": args.encoder_kind,
        "seed": args.seed, "n_total": len(all_samples), "n_done": len(results),
        "elapsed_sec": round(time.time() - start, 1),
        "config": config,
        "results": results,
    })
    logger.info(f"DONE — wrote {len(results)} records to {args.output} "
                f"in {time.time() - start:.1f}s")
    if n_no_sql:
        logger.warning(no_sql_msg)
    failed = [r for r in results if r.get("error")]
    if failed:
        logger.error(f"{len(failed)} of {len(results)} records failed during generation or "
                     f"encoding, so their recall is missing or understated. Rerun the same "
                     f"command with --resume to retry only those records.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
