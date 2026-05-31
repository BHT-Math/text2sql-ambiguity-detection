"""
Forced-scaffold detection wrapper around the a-Interact agent loop. Three
arms via `--da_method`:

  direct  — single LLM call: question + schema + KB → numbered clarification list.
  mga     — 1 multi-gen call (temp=`--da_mga_temperature`, default 1.3; use
            1.0 for GLM-4.5-Air, 1.3 for Qwen3.5/MiniMax — GLM collapses to a
            single cluster at 1.3). "Generate 10 different SQL interpretations
            with different filters/joins/aggregations" → AST cluster the 10
            SQLs → 1 analysis call asking what ambiguous terms produced the
            structural disagreements.
  union   — direct first; then mga with its ANALYSIS step conditioned on
            direct's questions (shown them, told to emit ONLY genuinely-new
            ambiguities — generative novelty, no post-hoc dedup judge). The
            two lists are then round-robin interleaved (D1, M1, D2, M2, ...)
            so a novel mga question can reach the budget-clamped top-K
            instead of being clipped behind all of direct.

Pipeline per sample (all three arms):
  Turn 0-3: forced retrieval — get_schema, get_all_column_meanings,
             get_all_external_knowledge_names, get_all_knowledge_definitions.
             ~3.5 coins charged uniformly.
  Turn 4:   detection step (1-2 LLM calls depending on method, 0 coins).
  Turn 5..K+4: forced ask() per detected question.
  Turn K+5+: standard a-Interact agent loop (submit / debug / etc.).

K is per-sample budget-aware (`--da_top_k auto`): K_max = min(8, floor((remaining
- agent_reserve) / 2)). For patience=14 + amb_count=N typical: K ≈ N (tracks the
harness's own ambiguity-budget allocation of 2 coins per ambiguity).

Implementation: monkey-patches `parse_agent_response` so the LOCAL `obj` / `action`
variables in main.py's loop both reflect the override.

Usage:
    python run_direct_agentic.py \
        --agent_model qwen35-122b-cluster --max_turns 60 \
        --user_patience_budget 14 \
        --user_sim_prompt_version v2 \
        --da_method direct        # or mga, or union
        --da_base_url "$QWEN35_122B_BASE_URL"   # OpenAI-compatible endpoint for the detection LLM
        --da_model_id qwen3.5-122b

(scripts/run.sh wires --da_base_url for you from the per-model *_BASE_URL env
var; the default below falls back to OPENAI_API_BASE for direct invocation.)
"""

import sys
import os
import json
import re
import inspect
import logging

# Path setup
sys.path.insert(0, os.path.dirname(__file__))
strategies_dir = os.path.dirname(os.path.dirname(__file__))
if strategies_dir not in sys.path:
    sys.path.insert(0, strategies_dir)

logger = logging.getLogger(__name__)

# ── Parse direct-agentic-specific args ──────────────────────────────

def _extract_da_args():
    da_args = {}
    remaining = []
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith('--da_'):
            if '=' in arg:
                key, val = arg.split('=', 1)
                da_args[key[5:]] = val
            elif i + 1 < len(sys.argv):
                da_args[arg[5:]] = sys.argv[i + 1]
                i += 1
        else:
            remaining.append(arg)
        i += 1
    sys.argv = remaining
    return da_args

_da_args = _extract_da_args()
DA_BASE_URL = _da_args.get('base_url', os.environ.get('OPENAI_API_BASE', 'http://localhost:8000/v1'))
DA_MODEL_ID = _da_args.get('model_id', 'qwen3.5-122b')
# top_k:
#   "auto" (default): K is sized per-sample from the remaining budget at detection
#   time so easier samples (less ambiguity → less budget) get fewer asks and
#   harder samples (more ambiguity → more budget) get more, while always
#   leaving the agent enough coins to actually solve the SQL.
#   <int>: fixed K for every sample (legacy / smoke-debug use).
DA_TOP_K = _da_args.get('top_k', 'auto')
# Reserve for the agent's submit + retry path after the forced asks.
# 1 execute (1) + 3 submit attempts (9) + 2 coin buffer = 12.
DA_AGENT_RESERVE = float(_da_args.get('agent_reserve', '12'))
# Hard cap to prevent pathological detection outputs from flooding the user sim.
DA_TOP_K_HARD_CAP = int(_da_args.get('top_k_hard_cap', '8'))
DA_DETECTION_TIMEOUT = float(_da_args.get('detection_timeout', '120'))
# Detection method: direct | mga | union
DA_METHOD = _da_args.get('method', 'direct').lower().strip()
if DA_METHOD not in ('direct', 'mga', 'union'):
    raise SystemExit(f"--da_method must be one of direct/mga/union, got {DA_METHOD!r}")
# MGA-specific: temperature for multi-gen generation step.
# Default 1.3 per FINDINGS §16.4 for Qwen3.5 / MiniMax (1.0 worked for GLM-4.5-Air
# but produced single-cluster outputs on Qwen3.5 in our 100-sample round).
DA_MGA_TEMPERATURE = float(_da_args.get('mga_temperature', '1.3'))
DA_MGA_NUM_INTERPRETATIONS = int(_da_args.get('mga_num_interpretations', '10'))
# Ablation: skip AST clustering + diff analysis. Show all 10 SQLs raw to the
# analyzer. Matches FINDINGS' "SE raw_samples" mode (Phase 2.6.1b). Also
# bypasses the entire `_format_diff_text` code path.
DA_NO_AST = _da_args.get('no_ast', 'false').lower() in ('true', '1', 'yes')

# Detection cache: a per-instance JSON written by direct/mga arms and read
# by the union arm. Eliminates the redundant detection LLM calls in union
# (3-5x speedup) AND guarantees union sees the SAME questions that direct
# and mga arms produced (deterministic comparison instead of re-sampling
# at temp=1.3 within union).
# Default cache dir: ./detection_cache/<dataset_name>/<instance>.json
# (run.sh sets a per-output-dir cache; passing --da_cache_dir overrides.)
DA_CACHE_DIR = _da_args.get('cache_dir', './detection_cache')
DA_USE_CACHE = _da_args.get('use_cache', 'true').lower() in ('true', '1', 'yes')
# Verbose questions: skip the last-? walk-back. Returns the WHOLE numbered
# item content (after stripping topic prefix and bold markdown). Useful for
# A/B-testing whether the trim mid-sentence reasoning helps or hurts the
# user simulator's answer quality.
DA_VERBOSE_QUESTIONS = _da_args.get('verbose_questions', 'false').lower() in ('true', '1', 'yes')

# Free-detection-only mode: keep forced retrieval (so schema/KB land in
# history), run detection (free LLM call), but DO NOT queue forced asks.
# Instead, append the detected questions to status.current_prompt as a
# "[SUGGESTED CLARIFICATIONS]" block; the agent freely chooses to ask
# any/none of them. Also patches ACTION_COSTS so ask=0 and get_*=0, so
# the agent's free-choice exploration is not coin-pressured (only execute
# and submit deplete budget). This is the apples-to-apples counterpart
# to the official baseline that controls for budget allocation while
# keeping the agent autonomous (no forced asks).
DA_FREE_DETECTION_ONLY = _da_args.get('free_detection_only', 'false').lower() in ('true', '1', 'yes')

# Hint-only mode: minimal change from the official baseline. Keeps the RAW
# ReAct template (no enhanced/scaffolded swap), keeps standard ACTION_COSTS
# (no patches), keeps forced retrieval (so detection has schema/KB context),
# but does NOT queue forced asks — the detection output is appended to
# status.current_prompt as a [SUGGESTED CLARIFICATIONS] block, and the agent
# chooses freely whether to ask any/all/none. Pair with --user_patience_budget
# matched to the comparison run (e.g. 34 to match the extra-coins baseline)
# so the only delta is the hint itself.
DA_HINT_ONLY = _da_args.get('hint_only', 'false').lower() in ('true', '1', 'yes')
# Clean-slate mode: forced retrieval + 1 analysis call run as a FREE pre-phase.
# Afterwards the retrieved context (and even its placeholders) are discarded,
# the budget is restored to full, and the agent is re-entered with ONLY the
# original question + a NEUTRAL hint block — as if it were starting from
# turn 1 with hints given. Isolates "do verbatim introspection hints help?"
# with the agent's economy/prompt otherwise identical to BASE.
DA_CLEAN_SLATE = _da_args.get('clean_slate', 'false').lower() in ('true', '1', 'yes')

# Phase-2 (follow-up / SR2) hint injection. When false, detection runs ONLY for
# Phase 1; the follow-up gets no hint pool. The BIRD-Interact follow-up is not
# annotated with ambiguities (0/600) and the user simulator is fed an empty
# ambiguity set in Phase 2, so SR2 hints are off-target and (for union) double
# the per-sample detection cost. Default OFF; pass --da_phase2_hints true to
# restore the old Phase-2 hint behavior.
DA_PHASE2_HINTS = _da_args.get('phase2_hints', 'false').lower() in ('true', '1', 'yes')

# ── Loud assurance: the K-cap is forced-ask-only ────────────────────
# In hint-only / free-detection-only mode the detected questions are
# injected as free TEXT (zero coin cost), so the budget-aware K-cap
# (--da_top_k) does NOT apply: _run_detection returns the FULL list.
# This cap once silently throttled hints to K≈1 at patience=6. Logged
# loudly so the smoke check can verify the cap is inert.
if DA_HINT_ONLY or DA_FREE_DETECTION_ONLY:
    logger.warning(
        "[DIRECT-AGENTIC] hint-mode active → K-CAP DISABLED: ALL detected "
        f"questions injected as hints (--da_top_k={DA_TOP_K!r} is IGNORED). "
        "Forced-ask budget cap does not apply to free-text hints."
    )

_OVERRIDE_AGENT_URL = _da_args.get('override_agent_url', '')
if _OVERRIDE_AGENT_URL:
    from bird_interact.llm.config import model_config
    if 'qwen35-122b-cluster' in model_config:
        model_config['qwen35-122b-cluster']['base_url'] = _OVERRIDE_AGENT_URL
        logger.warning(f"[DIRECT-AGENTIC] Overrode qwen35-122b-cluster base_url to {_OVERRIDE_AGENT_URL}")

# ── Monkey-patch 1: enhanced prompt template ────────────────────────
# We use the SCAFFOLDED variant by default — strips strategies that are
# redundant with our forced retrieval+detection+asks scaffold, adds 3 small
# scaffold-aware hints (pre-retrieved context, re-read clarifications with
# negation awareness, prioritize submission). Pass `--da_use_full_enhanced`
# to fall back to the original EnhancedTemplate (for ablation).

import bird_interact.agent.prompt_utils as prompt_utils
if DA_HINT_ONLY:
    # Hint-only mode: do NOT swap template. Use the upstream raw ReAct
    # template that prompt_utils initialised at import (matches official
    # baseline exactly except for the appended detection hint).
    logger.warning("[DIRECT-AGENTIC] hint-only mode: keeping raw ReAct template (no swap)")
elif DA_FREE_DETECTION_ONLY:
    from enhanced_prompts_detection_only import EnhancedDetectionOnlyTemplate
    prompt_utils.react_template = EnhancedDetectionOnlyTemplate("bird_interact_sql", "PostgreSQL Database")
    logger.warning("[DIRECT-AGENTIC] using EnhancedDetectionOnlyTemplate (free-detection-only mode)")
elif _da_args.get('use_full_enhanced', 'false').lower() in ('true', '1', 'yes'):
    from enhanced_prompts import EnhancedTemplate
    prompt_utils.react_template = EnhancedTemplate("bird_interact_sql", "PostgreSQL Database")
    logger.warning("[DIRECT-AGENTIC] using FULL EnhancedTemplate (ablation)")
else:
    from enhanced_prompts_scaffolded import EnhancedScaffoldedTemplate
    prompt_utils.react_template = EnhancedScaffoldedTemplate("bird_interact_sql", "PostgreSQL Database")
    logger.warning("[DIRECT-AGENTIC] using EnhancedScaffoldedTemplate (default for forced scaffold)")

# ── Patch ACTION_COSTS for free-detection-only mode ─────────────────
# Set ask=0 and get_*=0 so the agent can freely explore and clarify.
# Only execute (1) and submit (3) deplete budget. The amb_resolve_budget
# component of total_budget becomes 0 (amb_count × ask_cost = N × 0).
# Total budget per sample: env(3) + submit(3) + 0 + patience(14) = 20.
if DA_FREE_DETECTION_ONLY:
    import bird_interact.agent.main as _main_module_for_costs
    _free_keys = ['ask', 'get_schema', 'get_all_column_meanings',
                  'get_column_meaning', 'get_all_external_knowledge_names',
                  'get_knowledge_definition', 'get_all_knowledge_definitions']
    _orig_costs = dict(_main_module_for_costs.ACTION_COSTS)
    for _k in _free_keys:
        _main_module_for_costs.ACTION_COSTS[_k] = 0
    logger.warning(
        f"[DIRECT-AGENTIC] free-detection-only: patched ACTION_COSTS "
        f"{ {k: (_orig_costs.get(k), 0) for k in _free_keys} }"
    )

# ── Forced-retrieval queue ──────────────────────────────────────────

_FORCED_RETRIEVAL_ACTIONS = [
    "get_schema()",
    "get_all_column_meanings()",
    "get_all_external_knowledge_names()",
    "get_all_knowledge_definitions()",
]

# ── Direct-prompting detection ──────────────────────────────────────

# Intent-biased Direct system prompt ("ask about USER INTENT, not SQL syntax";
# focus on domain concepts). This is NOT neutral. It is used ONLY for the
# agentic hint-pool detection step (Table tab:agentic-nr-lite300), which was
# run end-to-end with this prompt, so it is kept here for faithful
# reproduction of those numbers. The detection-only tables (5-7) use the
# shared NEUTRAL_ANALYSIS_PROMPT instead (see detection_only/prompts.py). Do
# NOT flip this to neutral without re-running the agentic numbers.
DIRECT_SYSTEM_PROMPT = (
    "You are a PostgreSQL expert analyzing ambiguity in a database question. "
    "You will be shown a user's question along with the database schema, "
    "column meanings, and external knowledge base.\n\n"
    "Your task: identify every ambiguous term or phrase in the question that "
    "could lead to different SQL interpretations.\n\n"
    "For EACH ambiguity, write a short clarification question that would "
    "resolve it.\n\n"
    "Rules:\n"
    "- Questions should be about the USER'S INTENT, not about SQL syntax.\n"
    "- Focus on domain concepts: formulas, metrics, thresholds, scope, "
    "classifications.\n"
    "- Output a numbered list (1. 2. 3. etc.).\n"
    "- If nothing is ambiguous, say 'No ambiguities detected.'"
)

DETECTION_USER_TEMPLATE = """## Database Schema
{schema}

## Column Meanings
{column_meanings}

## External Knowledge
{kb_text}

## Question
{question}

List every ambiguity you can identify."""

# Patterns indicating chain-of-thought / verbose reasoning leaked into a question.
_REASONING_MARKERS = (
    'looking at the', 'the schema has', 'the schema shows', 'according to the schema',
    'however,', 'wait,', 'this means', 'this implies', 'it seems',
    'i think', 'the user asks', 'the user is asking', 'the user wants',
    'or perhaps', 'given that', 'given the context', 'note that',
    'not in the database', 'this suggests', 'looking more carefully',
)


def _is_clean_question(q: str) -> bool:
    """Reject questions that look like chain-of-thought, section headers,
    or are otherwise malformed."""
    if not q or not q.endswith('?'):
        return False
    # Verbose mode raises the upper cap because items legitimately can run
    # 300-500 chars when reasoning prose is included.
    upper = 600 if DA_VERBOSE_QUESTIONS else 250
    if len(q) < 15 or len(q) > upper:
        return False
    if '**' in q:
        return False
    # Section-header pattern: "Some Title:?" — reject. A real question never
    # has a colon immediately before the question mark.
    if q.endswith(':?') or q.endswith(': ?'):
        return False
    lower = q.lower()
    if any(m in lower for m in _REASONING_MARKERS):
        return False
    # Reject imperatives masquerading as questions: lines that start with a
    # procedural verb (Analyze X..., Identify X..., Draft X..., Review X...)
    # are nearly always section labels, not user-facing clarifications.
    if re.match(r'^(analyze|analyse|identify|draft|drafting|review|clarify|consider|examine|list|note|observe|outline|verify|determine)\b',
                lower):
        return False
    # Must contain at least one alphabetic word (not pure punctuation)
    if not re.search(r'[a-zA-Z]{4,}', q):
        return False
    return True


def _extract_question(text: str) -> str:
    """Pull a clarification question from one numbered list item.

    Two modes (toggled by DA_VERBOSE_QUESTIONS env-style flag):
      * Default (trimmed): find the LAST '?' and walk backward to the start
        of the sentence containing it. Sentence boundary = [.!?] + whitespace
        + capital letter (robust to abbreviations like 'e.g.,').
      * Verbose: return the WHOLE numbered item content after stripping the
        topic-prefix / bold markdown. Tests whether mid-item reasoning helps
        the user simulator answer correctly.

    Examples (trimmed mode):
      '"Topic": Reasoning here. Final question?'
        → 'Final question?'
      '**Component**: Different SQLs filtered events differently. What does interference level mean (e.g., signal strength, noise floor, derived score)?'
        → 'What does interference level mean (e.g., signal strength, noise floor, derived score)?'

    Same inputs in verbose mode return the entire stripped string.
    """
    if not text:
        return ''
    text = text.replace('**', '').replace('__', '').strip()
    # Drop a leading "topic": prefix (with or without quotes around topic)
    text = re.sub(r'^\s*"[^"]+"\s*:\s*', '', text)
    text = re.sub(r'^[A-Z][\w\s]{1,40}:\s*', '', text)
    text = text.strip()
    if not text:
        return ''

    # Verbose mode: return the WHOLE stripped item.
    if DA_VERBOSE_QUESTIONS:
        # Item must end with a question mark to be considered a real question
        if not text.endswith('?'):
            return ''
        if not _is_clean_question(text):
            return ''
        return text

    # Trimmed mode: find the LAST '?' and walk backward to sentence start.
    last_q = text.rfind('?')
    if last_q < 0:
        return ''

    head = text[:last_q + 1]  # everything up to and including the '?'

    # Sentence boundary = [.!?] (with optional closing quote) + whitespace +
    # CAPITAL LETTER. The optional ['"] handles `."` and `?"` ending forms.
    # This skips abbreviations (e.g., 'e.g.,' has comma after period).
    body = head[:-1]
    boundaries = [m.end() for m in re.finditer(r'[.!?]["\']?\s+(?=[A-Z])', body)]
    sentence_start = boundaries[-1] if boundaries else 0
    candidate = head[sentence_start:].strip()

    # Strip leading conjunctions (Or, And, But, So) — these usually appear
    # when an item lists alternatives ('Should X? Or should Y?').
    candidate = re.sub(r'^(Or|And|But|So)\s+', '', candidate, flags=re.IGNORECASE)
    candidate = candidate.strip()

    if not _is_clean_question(candidate):
        return ''
    return candidate


# ── Detection cache (eliminates redundant LLM calls in union arm) ───

def _cache_path(instance_id, kind):
    """kind: 'direct' or 'mga'"""
    if not instance_id or instance_id == '?':
        return None
    os.makedirs(DA_CACHE_DIR, exist_ok=True)
    return os.path.join(DA_CACHE_DIR, f"{instance_id}_{kind}.json")


def _cache_read(instance_id, kind):
    if not DA_USE_CACHE:
        return None
    p = _cache_path(instance_id, kind)
    if not p or not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            data = json.load(f)
        questions = data.get('questions') or []
        return questions
    except Exception as e:
        logger.warning(f"[CACHE] read failed for {p}: {e}")
        return None


def _cache_write(instance_id, kind, questions, extra=None):
    if not DA_USE_CACHE:
        return
    p = _cache_path(instance_id, kind)
    if not p:
        return
    try:
        data = {'instance_id': instance_id, 'kind': kind, 'questions': questions}
        if extra:
            data.update(extra)
        with open(p, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"[CACHE] write failed for {p}: {e}")


def _extract_retrieved_context(status):
    """Pull schema / KB / column meanings from interaction history."""
    schema = ''
    kb_text = ''
    column_meanings = ''
    for entry in status.interaction_history:
        act = (entry.get('action') or '').strip()
        obs = entry.get('observation', '')
        if not obs:
            continue
        if act == 'get_schema()':
            schema = obs
        elif act == 'get_all_column_meanings()':
            column_meanings = obs
        elif act == 'get_all_knowledge_definitions()':
            kb_text = obs
    return schema, column_meanings, kb_text


def _strip_think(raw):
    if not raw:
        return raw
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    if '<think>' in raw:
        if '</think>' in raw:
            raw = raw.split('</think>', 1)[1]
        else:
            raw = re.sub(r'<think>.*$', '', raw, flags=re.DOTALL)
    return raw.strip()


def _strip_markdown(q):
    """Best-effort cleanup of markdown / leading-trailing punctuation."""
    # Strip bold/italic markers
    q = q.replace('**', '').replace('__', '')
    # Strip backticks
    q = re.sub(r'`([^`]+)`', r'\1', q)
    # If the question starts with `"<term>"<colon>`, drop that prefix and keep the rest
    m = re.match(r'^\s*"[^"]+"\s*:\s*(.+)$', q)
    if m:
        q = m.group(1)
    # Strip leading/trailing junk
    q = q.strip().strip('"').strip("'").strip()
    return q


def _parse_numbered_questions(raw):
    """Extract clarification questions from a numbered list.

    Delegates to detection_only.extract_questions_from_response — the SAME
    extractor used for the paper's detection tables (5-7) — so the agentic
    detection and the detection-only pipeline can never diverge.

    History (fixed 2026-05-30): this previously used a strict
    `_extract_question`/`_is_clean_question` filter that required each item to
    end in '?' and rejected markdown '**'. Models frequently emit ambiguities
    as `1. **term**: description` (no trailing '?'), so that filter silently
    discarded ~16-18% of valid detections and zeroed out ~30 instances/model,
    starving the agentic hint pool. The detection-only pipeline used the lenient
    extractor below all along, which is why detection Tables 5-7 never showed
    the loss. `_extract_question`/`_is_clean_question` are retained only for the
    agent-side ask() cleanup, not for detection parsing.
    """
    from bird_interact.detection_only.data import extract_questions_from_response
    raw = _strip_think(raw)
    if not raw:
        return []
    return [d["question"] for d in extract_questions_from_response(raw)]


def _run_direct_detection(status):
    """One LLM call: question + schema + KB → numbered clarification questions.
    Cache: writes to <DA_CACHE_DIR>/<instance>_direct.json after computing.
    """
    from openai import OpenAI

    instance_id = status.original_data.get('instance_id', '?')

    # Phase-2 (follow-up) detection: use the follow-up question and BYPASS the
    # detection cache (which is keyed by instance only and holds Phase-1 Qs).
    _p2 = getattr(status, '_da_phase2_active', False)

    # Cache hit? Skip LLM call entirely. (Phase-1 only.)
    cached = None if _p2 else _cache_read(instance_id, 'direct')
    if cached is not None:
        logger.info(f"[{DA_METHOD.upper()}] {instance_id}: direct cache HIT ({len(cached)} questions)")
        return cached

    schema, column_meanings, kb_text = _extract_retrieved_context(status)
    user_question = ((status.original_data.get('follow_up') or {}).get('query', '')
                     if _p2 else status.original_data.get('amb_user_query', ''))

    if not user_question:
        logger.warning(f"[DIRECT-AGENTIC] {instance_id}: no "
                        f"{'follow_up.query' if _p2 else 'amb_user_query'}; skipping detection")
        return []
    if not schema:
        logger.warning(f"[DIRECT-AGENTIC] {instance_id}: no schema observation; detection will be schema-blind")

    user_prompt = DETECTION_USER_TEMPLATE.format(
        question=user_question,
        schema=schema or "(unavailable)",
        column_meanings=column_meanings or "(unavailable)",
        kb_text=kb_text or "(unavailable)",
    )

    client = OpenAI(api_key='EMPTY', base_url=DA_BASE_URL, timeout=DA_DETECTION_TIMEOUT)
    try:
        response = client.chat.completions.create(
            model=DA_MODEL_ID,
            messages=[
                {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content or ''
    except Exception as e:
        logger.error(f"[DIRECT-AGENTIC] {instance_id}: detection LLM call failed: {e}")
        return []

    questions = _parse_numbered_questions(raw)
    logger.info(f"[{DA_METHOD.upper()}] {instance_id}: direct produced {len(questions)} clean questions"
                f"{' (phase-2)' if _p2 else ''}")
    if not _p2:
        _cache_write(instance_id, 'direct', questions, extra={'raw_response': raw})
    return questions


# ── MGA detection ──────────────────────────────────────────────────

# Neutral MGA generation prompt (FINDINGS Bundle C / §16.4) — drops the
# dimensional taxonomy ("different formulas, columns, aggregations, filters,
# joins") while keeping the "may be ambiguous, generate N interpretations"
# framing that mga's mechanism depends on.
MGA_GEN_SYSTEM_PROMPT = (
    "You are a PostgreSQL expert. A user asked a database question that may be "
    "ambiguous — it could be interpreted in multiple valid ways.\n\n"
    "Generate exactly {N} DIFFERENT SQL interpretations of the question. Each "
    "interpretation should reflect a genuinely different reading of the "
    "question.\n\n"
    "Rules:\n"
    "- Each SQL must be valid PostgreSQL.\n"
    "- Label each as [Interpretation 1], [Interpretation 2], etc.\n"
    "- Do NOT generate trivial variations (alias changes, formatting). Each must "
    "represent a meaningfully different interpretation.\n"
    "- If the question is completely unambiguous, generate {N} identical SQL.\n"
    "- Output ONLY the labeled SQL queries, no explanations."
)

# Neutral analysis prompt (FINDINGS NEUTRAL_ANALYSIS_PROMPT) — drops
# the SQL-vs-NL taxonomy and just asks for "every ambiguity".
MGA_ANALYZE_SYSTEM_PROMPT = (
    "You are analyzing a database question and several SQL interpretations "
    "that were generated for it.\n\n"
    "Identify every ambiguity in the question — anything that could explain "
    "why these SQL queries are not identical, or anything in the question "
    "itself that is unclear.\n\n"
    "For each ambiguity, write one short clarification question that would "
    "resolve it.\n\n"
    "Output a numbered list (1. 2. 3. etc.). If you find no ambiguities, "
    "say 'No ambiguities detected.'"
)

# Optional prompt overrides — swap in your own prompts WITHOUT editing source
# (see REPRODUCING.md "Customizing prompts"):
#   --da_system_prompt_file PATH    replaces DIRECT_SYSTEM_PROMPT  (direct + union arms)
#   --da_analysis_prompt_file PATH  replaces MGA_ANALYZE_SYSTEM_PROMPT (mga/union analysis)
# e.g. point --da_system_prompt_file at a neutral prompt to make the agentic
# detection step neutral too, without touching this file.
if _da_args.get('system_prompt_file'):
    DIRECT_SYSTEM_PROMPT = open(_da_args['system_prompt_file'], encoding='utf-8').read().strip()
    print(f"[DA] DIRECT_SYSTEM_PROMPT <- {_da_args['system_prompt_file']}", file=sys.stderr)
if _da_args.get('analysis_prompt_file'):
    MGA_ANALYZE_SYSTEM_PROMPT = open(_da_args['analysis_prompt_file'], encoding='utf-8').read().strip()
    print(f"[DA] MGA_ANALYZE_SYSTEM_PROMPT <- {_da_args['analysis_prompt_file']}", file=sys.stderr)


def _format_diff_text(diff):
    """Best-effort renderer for cluster-diff output. Robust to both shapes:
       1. {component: {'common': [...], 'varying': [(elem, fraction), ...]}}
       2. {component: [list_of_varying_things]}
    Catches any unexpected shape and returns "" rather than crashing."""
    if not diff:
        return ""
    if not isinstance(diff, dict):
        return ""
    lines = []
    try:
        for component, info in diff.items():
            varying = None
            if isinstance(info, dict):
                varying = info.get('varying')
            elif isinstance(info, list):
                varying = info
            if not varying:
                continue
            lines.append(f"**{component}** disagree:")
            for elem in varying[:5]:
                if isinstance(elem, tuple) and len(elem) == 2:
                    e, frac = elem
                    try:
                        lines.append(f"  - {e!r} appears in {float(frac):.0%} of clusters")
                    except (TypeError, ValueError):
                        lines.append(f"  - {e!r}")
                else:
                    lines.append(f"  - {elem!r}")
    except Exception as e:
        logger.warning(f"[_format_diff_text] unexpected diff shape: {e}; returning empty")
        return ""
    return "\n".join(lines)


def _run_mga_detection(status, direct_hints=None):
    """MGA detection: 1 multi-gen call + (optional) AST cluster + 1 analysis call.

    direct_hints: if a non-empty list of Direct questions is passed (union
    arm), the ANALYSIS step is shown them and instructed to emit ONLY
    genuinely-new ambiguities (generative novelty — replaces the old
    post-hoc batched dedup judge). Direct-conditioned outputs are cached
    under a SEPARATE kind ('mga_dc') so they can never be confused with
    the unconditioned 'mga' pool.

    Cache: writes to <DA_CACHE_DIR>/<instance>_<kind>.json after computing.
    """
    from openai import OpenAI

    instance_id = status.original_data.get('instance_id', '?')
    cache_kind = 'mga_dc' if direct_hints else 'mga'

    _p2 = getattr(status, '_da_phase2_active', False)

    # Cache hit? Skip the 2-3 LLM calls entirely. (Phase-1 only.)
    cached = None if _p2 else _cache_read(instance_id, cache_kind)
    if cached is not None:
        logger.info(f"[{DA_METHOD.upper()}] {instance_id}: mga cache HIT ({len(cached)} questions)")
        return cached

    schema, column_meanings, kb_text = _extract_retrieved_context(status)
    user_question = ((status.original_data.get('follow_up') or {}).get('query', '')
                     if _p2 else status.original_data.get('amb_user_query', ''))

    if not user_question or not schema:
        logger.warning(f"[MGA] {instance_id}: missing "
                        f"{'follow_up.query' if _p2 else 'question'} or schema; skipping")
        return []

    client = OpenAI(api_key='EMPTY', base_url=DA_BASE_URL, timeout=DA_DETECTION_TIMEOUT)

    # Stage 1: multi-gen call — produce N diverse SQL interpretations
    gen_user_parts = [f"## Database Schema\n{schema}"]
    if column_meanings:
        gen_user_parts.append(f"## Column Meanings\n{column_meanings}")
    if kb_text:
        gen_user_parts.append(f"## External Knowledge\n{kb_text}")
    gen_user_parts.append(
        f"## Question\n{user_question}\n\n"
        f"Generate {DA_MGA_NUM_INTERPRETATIONS} different SQL interpretations:"
    )
    try:
        gen_resp = client.chat.completions.create(
            model=DA_MODEL_ID,
            messages=[
                {"role": "system", "content": MGA_GEN_SYSTEM_PROMPT.format(N=DA_MGA_NUM_INTERPRETATIONS)},
                {"role": "user", "content": "\n\n".join(gen_user_parts)},
            ],
            temperature=DA_MGA_TEMPERATURE,
            max_tokens=8192,
        )
        gen_content = _strip_think(gen_resp.choices[0].message.content or "")
    except Exception as e:
        logger.error(f"[MGA] {instance_id}: gen call failed: {e}")
        return []

    # Parse SQL interpretations
    sql_blocks = re.split(r'\[Interpretation\s+\d+\][\s:]*', gen_content, flags=re.IGNORECASE)
    sql_list = []
    for block in sql_blocks:
        block = block.strip()
        if not block:
            continue
        fence = re.search(r'```(?:sql)?\s*\n?(.*?)```', block, re.DOTALL | re.IGNORECASE)
        if fence:
            sql_list.append(fence.group(1).strip())
        elif re.search(r'\b(SELECT|WITH|CREATE|INSERT)\b', block, re.IGNORECASE):
            sql_match = re.search(r'((?:WITH|SELECT|CREATE|INSERT)\b.*)', block, re.DOTALL | re.IGNORECASE)
            if sql_match:
                sql_list.append(sql_match.group(1).strip())

    logger.info(f"[MGA] {instance_id}: parsed {len(sql_list)} SQL interpretations "
                f"(no_ast={DA_NO_AST})")
    if len(sql_list) < 2:
        logger.warning(f"[MGA] {instance_id}: <2 SQL parsed, falling back to direct detection")
        return _run_direct_detection(status)

    diff_text = ""
    if DA_NO_AST:
        # Ablation: skip clustering entirely. Show all SQLs raw to the analyzer.
        # Matches FINDINGS' "SE raw_samples" mode.
        sql_block_lines = []
        for i, s in enumerate(sql_list):
            sql_block_lines.append(f"[Interpretation {i+1}]:")
            sql_block_lines.append(s)
            sql_block_lines.append("")
        grouped_text = "\n".join(sql_block_lines)
        cluster_summary = f"({len(sql_list)} generated, no AST grouping — raw)"
    else:
        # Stage 2: AST cluster + diff (in-package sqlglot clustering).
        try:
            from bird_interact.detection_only import cluster as _se_cluster
            clusters = _se_cluster.cluster_by_ast(sql_list, threshold=0.70)
            diff = _se_cluster.analyze_cluster_differences(clusters) if len(clusters) >= 2 else None
        except Exception as e:
            logger.error(f"[MGA] {instance_id}: clustering failed: {e}")
            return []

        diff_text = _format_diff_text(diff) if diff else ""

        group_lines = []
        for ci, cluster in enumerate(clusters):
            group_lines.append(f"[Group {chr(65+ci)}] ({len(cluster)} interpretation"
                               f"{'s' if len(cluster) > 1 else ''}):")
            for s in cluster:
                group_lines.append(f"  {s}")
            group_lines.append("")
        grouped_text = "\n".join(group_lines)
        cluster_summary = f"({len(sql_list)} generated, grouped into {len(clusters)} clusters)"

    analyze_user_parts = [f"## Original Question\n{user_question}"]
    if diff_text:
        analyze_user_parts.append(f"## Structural Disagreements (sqlglot AST analysis)\n{diff_text}")
    analyze_user_parts.append(
        f"## SQL Interpretations {cluster_summary}\n\n{grouped_text}"
    )
    if direct_hints:
        _dh_block = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(direct_hints))
        analyze_user_parts.append(
            "## Clarification questions ALREADY identified (treat as covered)\n"
            "A separate analysis already produced the clarification questions "
            "below. A user will already be asked these, so the ambiguities "
            "they resolve are considered handled.\n" + _dh_block
        )
        analyze_user_parts.append(
            "List ONLY the genuinely NEW ambiguities — ones that would remain "
            "unresolved even after the user answers every already-identified "
            "question above. Do NOT repeat, rephrase, narrow, broaden, or "
            "split any question above. Each item you output must target a "
            "DISTINCT underlying ambiguity not covered by the list above. If "
            "every ambiguity is already covered, output exactly "
            "'No ambiguities detected.'"
        )
    else:
        analyze_user_parts.append("List every ambiguity you can identify.")
    try:
        analyze_resp = client.chat.completions.create(
            model=DA_MODEL_ID,
            messages=[
                {"role": "system", "content": MGA_ANALYZE_SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n".join(analyze_user_parts)},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        analyze_content = _strip_think(analyze_resp.choices[0].message.content or "")
    except Exception as e:
        logger.error(f"[MGA] {instance_id}: analyze call failed: {e}")
        return []

    questions = _parse_numbered_questions(analyze_content)
    logger.info(f"[{DA_METHOD.upper()}] {instance_id}: mga produced {len(questions)} clean questions "
                f"({cluster_summary}; direct_conditioned={bool(direct_hints)})")
    if not _p2:
        _cache_write(instance_id, cache_kind, questions,
                     extra={'cluster_summary': cluster_summary,
                            'raw_response': analyze_content,
                            'direct_conditioned': bool(direct_hints),
                            'n_direct_hints': len(direct_hints) if direct_hints else 0})
    return questions


# ── Union detection — strict superset of direct + LLM dedup ────────

# Single batched dedup call: shows direct + mga questions to the LLM and
# asks for a yes/no per mga question. Cost: 1 LLM call per sample
# (instead of |direct| × |mga| pairwise calls).
DEDUP_BATCH_PROMPT = """You are deciding which of the MGA questions are duplicates of the DIRECT questions. Two questions are duplicates if a user answering one would automatically address the other (same underlying ambiguity, different wording).

DIRECT questions:
{direct_block}

MGA questions:
{mga_block}

For EACH MGA question, output ONE line with format `M<i>: yes` (it duplicates some DIRECT question) or `M<i>: no` (it does not duplicate any DIRECT question).

Output exactly {n_mga} lines, one per MGA question, no extra text. Example:
M1: no
M2: yes
M3: no
"""


def _llm_batch_dedup(client, direct_qs, mga_qs):
    """Returns list of bool, len == len(mga_qs). True means mga[i] is a
    duplicate of some direct question and should be DROPPED.

    On failure, falls back to lowercase-prefix match."""
    if not direct_qs or not mga_qs:
        return [False] * len(mga_qs)
    direct_block = "\n".join(f"D{i+1}: {q}" for i, q in enumerate(direct_qs))
    mga_block = "\n".join(f"M{i+1}: {q}" for i, q in enumerate(mga_qs))
    prompt = DEDUP_BATCH_PROMPT.format(
        direct_block=direct_block, mga_block=mga_block, n_mga=len(mga_qs),
    )
    try:
        resp = client.chat.completions.create(
            model=DA_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw = _strip_think(resp.choices[0].message.content or "")
        # Parse "M<i>: yes/no" lines
        verdicts = [False] * len(mga_qs)
        for line in raw.split('\n'):
            m = re.match(r'\s*M\s*(\d+)\s*[:.]?\s*(yes|no)\b', line, re.IGNORECASE)
            if not m:
                continue
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(mga_qs):
                verdicts[idx] = m.group(2).lower() == 'yes'
        return verdicts
    except Exception as e:
        logger.warning(f"[UNION] batched dedup judge failed ({e}); falling back to prefix match")
        d_prefixes = {q.lower()[:80] for q in direct_qs}
        return [q.lower()[:80] in d_prefixes for q in mga_qs]


def _run_union_detection(status):
    """Direct ∪ Direct-conditioned-MGA, round-robin interleaved.

    Direct's questions are computed first and passed into MGA's analysis
    step (see _run_mga_detection's `direct_hints`), which is instructed to
    return ONLY genuinely-new ambiguities. MGA is therefore novel-by-
    construction — the old post-hoc batched dedup judge is gone. The two
    lists are then round-robin interleaved (D1, M1, D2, M2, ...) so a novel
    MGA question can reach the budget-clamped top-K instead of being
    clipped behind ALL of Direct (the §24.3 direct-first-concat pathology
    that forced UNION ≈ DIRECT at patience=6). Still a content superset:
    every Direct question is retained.
    """
    from itertools import zip_longest

    instance_id = status.original_data.get('instance_id', '?')
    direct_qs = _run_direct_detection(status)
    mga_qs = _run_mga_detection(status, direct_hints=direct_qs)

    if not direct_qs and not mga_qs:
        logger.info(f"[UNION] {instance_id}: direct=0 + mga=0 → 0 questions")
        return []
    if not mga_qs:
        logger.info(f"[UNION] {instance_id}: direct={len(direct_qs)} + mga=0 "
                     f"→ {len(direct_qs)} (direct only; mga empty/all-covered)")
        return list(direct_qs)
    if not direct_qs:
        logger.info(f"[UNION] {instance_id}: direct=0 + mga={len(mga_qs)} "
                     f"→ {len(mga_qs)} (mga only; direct empty)")
        return list(mga_qs)

    # Round-robin interleave: D1, M1, D2, M2, ... Direct still leads each
    # pair (priority on the very top slot), but the top novel MGA question
    # lands at rank 2 — inside a K≈1-2 budget clamp.
    out = []
    for d, m in zip_longest(direct_qs, mga_qs):
        if d is not None:
            out.append(d)
        if m is not None:
            out.append(m)

    logger.info(f"[UNION] {instance_id}: direct={len(direct_qs)} + "
                f"novel-mga={len(mga_qs)} → round-robin interleaved "
                f"→ total={len(out)} head={out[:4]}")
    return out


# ── Method dispatch + K sizing ────────────────────────────────────

def _run_detection(status):
    """Dispatch to the configured detection method, then size K from budget."""
    instance_id = status.original_data.get('instance_id', '?')

    if DA_METHOD == 'direct':
        questions = _run_direct_detection(status)
    elif DA_METHOD == 'mga':
        questions = _run_mga_detection(status)
    elif DA_METHOD == 'union':
        questions = _run_union_detection(status)
    else:
        questions = []

    # The budget-aware K-cap below is vestigial forced-ask plumbing: it was
    # designed for the ORIGINAL design where each returned question became a
    # coin-costing ask(). In hint-only / free-detection-only mode the
    # questions are injected as a TEXT block the agent may freely use or
    # ignore — they cost zero coins, so there is nothing to budget against.
    # Capping there silently throttled the hint list to K≈1 at patience=6.
    # Return the FULL detected list for hint modes; apply K-sizing ONLY on
    # the forced-ask path where it is correct and load-bearing.
    if DA_HINT_ONLY or DA_FREE_DETECTION_ONLY:
        logger.info(
            f"[{DA_METHOD.upper()}] {instance_id}: hint-mode → returning ALL "
            f"{len(questions)} detected questions (no K cap)"
        )
        return questions

    # Forced-ask path: size K — per-sample budget-aware sizing or fixed.
    if DA_TOP_K == 'auto':
        remaining = float(getattr(status, 'remaining_budget', 16.5))
        asks_budget = max(0.0, remaining - DA_AGENT_RESERVE)
        k_max = min(DA_TOP_K_HARD_CAP, max(1, int(asks_budget / 2)))
    else:
        k_max = int(DA_TOP_K)

    logger.info(
        f"[{DA_METHOD.upper()}] {instance_id}: total {len(questions)} → K={k_max} "
        f"(remaining_budget={getattr(status, 'remaining_budget', '?')})"
    )
    if questions:
        logger.info(f"[{DA_METHOD.upper()}] {instance_id}: top-{min(k_max, len(questions))} → {questions[:k_max]}")
    return questions[:k_max]


# ── Monkey-patched parse_agent_response ─────────────────────────────

_orig_parse_agent_response = prompt_utils.parse_agent_response


def _patched_parse_agent_response(response):
    """Wrapper that returns the original parse for natural turns, but
    substitutes a forced (thought, object, action) tuple for forced retrieval
    + detection-driven asks during phase 1.

    Uses inspect to retrieve the caller's `status` local var so we can keep
    per-sample state (queue, detection_done flag).
    """
    thought, obj, action = _orig_parse_agent_response(response)

    # Find the SampleStatus in the caller's frame
    caller_frame = inspect.currentframe().f_back
    status = None
    if caller_frame is not None:
        status = caller_frame.f_locals.get('status')

    if status is None:
        return thought, obj, action

    # Initialize per-status state on first parse
    if not hasattr(status, '_da_queue'):
        status._da_queue = list(_FORCED_RETRIEVAL_ACTIONS)
        status._da_detection_done = False

    # Phase 2: give the follow-up question the SAME neutral-hint treatment as
    # Phase 1 (clean-slate only; once). No restart / no budget change / no
    # history wipe — the follow-up builds on the Phase-1 result and budget is
    # continuous. _extract_retrieved_context still works (full log persists),
    # so detection reuses the schema/KB already fetched in Phase 1.
    if getattr(status, 'current_phase', 1) != 1:
        if (DA_CLEAN_SLATE
                and getattr(status, '_da_clean_slate_active', False)
                and not getattr(status, '_da_phase2_detection_done', False)):
            status._da_phase2_detection_done = True
            iid = status.original_data.get('instance_id', '?')

            # Always strip the now-stale Phase-1 [POTENTIALLY AMBIGUOUS] block on
            # Phase-2 entry: its ambiguities were resolved during P1 and are
            # irrelevant clutter in P2 (the P1 *resolutions* survive in the
            # post-restart history). The regex matches only the P1 block — the
            # P2 "— FOLLOW-UP" tag has no bare "]" after "AMBIGUOUS", so it is
            # left intact. This runs regardless of DA_PHASE2_HINTS, so when the
            # Phase-2 hint pool is off, Phase 2 carries NO ambiguity-hint block
            # at all (rather than leaking the stale Phase-1 one).
            before = status.current_prompt
            status.current_prompt = re.sub(
                r'\n\n\[POTENTIALLY AMBIGUOUS\]\n.*?\[/POTENTIALLY AMBIGUOUS\]\n',
                '', status.current_prompt, flags=re.DOTALL,
            )
            _p1_stripped = (status.current_prompt != before)

            if not DA_PHASE2_HINTS:
                # Phase-2 hint pool OFF. The follow-up sub-task is not annotated
                # with ambiguities (0/600 in GT) and the user simulator is fed an
                # empty ambiguity set in Phase 2, so a detection pass would
                # surface nothing real. Skip the LLM call; inject no hints.
                logger.warning(
                    f"[DIRECT-AGENTIC] {iid}: PHASE-2 hint pool DISABLED "
                    f"(--da_phase2_hints=false): no detection call, no follow-up "
                    f"hints; stale-P1-block stripped={_p1_stripped}"
                )
                return thought, obj, action

            fu = (status.original_data.get('follow_up') or {}).get('query', '')
            if fu:
                try:
                    status._da_phase2_active = True
                    qs = _run_detection(status)
                    if qs:
                        lines = [f"  {i+1}. {q}" for i, q in enumerate(qs)]
                        blk = (
                            "\n\n[POTENTIALLY AMBIGUOUS — FOLLOW-UP]\n"
                            "An automated analysis flagged the terms/questions below "
                            "as possibly ambiguous in the follow-up question. They are "
                            "optional context, not instructions — use any you find "
                            "useful, ignore the rest, and decide for yourself.\n"
                            + "\n".join(lines)
                            + "\n[/POTENTIALLY AMBIGUOUS — FOLLOW-UP]\n"
                        )
                        status.current_prompt += blk
                        # Hard leak check: the P1 opener "[POTENTIALLY
                        # AMBIGUOUS]\n" must NOT survive into the P2 prompt
                        # (the "— FOLLOW-UP]" variant is the legitimate P2
                        # block and is expected). Logged, not raised, so a
                        # single bad sample can't kill a 300-sample run —
                        # smoke greps this line to gate the full run.
                        _p1_leaked = "[POTENTIALLY AMBIGUOUS]\n" in status.current_prompt
                        logger.warning(
                            f"[DIRECT-AGENTIC] {iid}: PHASE-2 neutral hints "
                            f"injected ({len(qs)} questions); stale-P1-block "
                            f"stripped={_p1_stripped}; "
                            f"P2-LEAK-CHECK={'FAIL-P1-LEAKED' if _p1_leaked else 'ok'}"
                        )
                    else:
                        logger.warning(f"[DIRECT-AGENTIC] {iid}: phase-2 "
                                        f"detection returned 0 questions")
                except Exception as e:
                    logger.error(f"[DIRECT-AGENTIC] {iid}: phase-2 detection "
                                 f"failed: {e}", exc_info=True)
                finally:
                    status._da_phase2_active = False
        return thought, obj, action

    # Detection runs once when retrieval queue is exhausted
    if not status._da_detection_done and len(status._da_queue) == 0:
        status._da_detection_done = True
        try:
            questions = _run_detection(status)
            instance_id = status.original_data.get('instance_id', '?')
            if DA_FREE_DETECTION_ONLY or DA_HINT_ONLY:
                # Hint-style modes: do NOT force asks. Append the detected
                # questions to current_prompt as a [SUGGESTED CLARIFICATIONS]
                # block — visible in every subsequent turn's full prompt. The
                # agent decides which (if any) to ask. Two variants share this
                # logic:
                #   - DA_FREE_DETECTION_ONLY: ask/get costs are patched to 0
                #     (free askget) AND template is swapped to one that
                #     advertises those costs.
                #   - DA_HINT_ONLY: raw ReAct template, standard ACTION_COSTS;
                #     the hint is the ONLY delta from the baseline runner.
                if DA_CLEAN_SLATE:
                    # Neutral, non-directive hint block. Wording deliberately
                    # frames the list as optional context the agent may ignore.
                    if questions:
                        hint_lines = [f"  {i+1}. {q}" for i, q in enumerate(questions)]
                        hint_block = (
                            "\n\n[POTENTIALLY AMBIGUOUS]\n"
                            "An automated analysis flagged the terms/questions below as "
                            "possibly ambiguous in the request. They are optional context, "
                            "not instructions — use any you find useful, ignore the rest, "
                            "and decide entirely for yourself how to proceed.\n"
                            + "\n".join(hint_lines)
                            + "\n[/POTENTIALLY AMBIGUOUS]\n"
                        )
                        status.current_prompt += hint_block
                    else:
                        logger.warning(
                            f"[DIRECT-AGENTIC] {instance_id}: detection returned 0 "
                            f"questions; clean-slate restart with NO hints (≈ BASE)"
                        )
                    # Discard the forced-retrieval pre-phase entirely: wipe
                    # (no observations, no placeholders) and restore the full
                    # budget (forced retrieval was FREE). The full log is kept
                    # intact for scoring/debugging; get_agent_prompt_for_turn is
                    # patched to show the agent only post-restart turns, so from
                    # the next turn it sees "BASE prompt + neutral hints" and
                    # then its OWN subsequent actions/observations normally.
                    status.remaining_budget = status.total_budget
                    status._da_restart_turn = getattr(status, 'current_turn', 0)
                    status._da_clean_slate_active = True
                    # One harmless read-only transition action for THIS turn
                    # (the turn-N agent response was generated on the polluted
                    # prompt; we discard it). Its history entry has turn ==
                    # _da_restart_turn, so the post-restart filter excludes it
                    # from the agent's view. Net cost zero (budget restored).
                    status._da_queue.append("get_schema()")
                    logger.warning(
                        f"[DIRECT-AGENTIC] {instance_id}: CLEAN-SLATE restart "
                        f"({len(questions)} neutral hints, budget restored to "
                        f"{status.total_budget:.1f}, restart_turn="
                        f"{getattr(status, 'current_turn', 0)})"
                    )
                elif questions:
                    hint_lines = [f"  {i+1}. {q}" for i, q in enumerate(questions)]
                    hint_block = (
                        "\n\n[SUGGESTED CLARIFICATIONS]\n"
                        "A detection step produced the following suggested clarification "
                        "questions. These are SUGGESTIONS — the user has NOT answered them yet. "
                        "Decide which (if any) to ask via `ask(...)`. Skip any you've already "
                        "resolved from the retrieved context.\n"
                        + "\n".join(hint_lines)
                        + "\n[/SUGGESTED CLARIFICATIONS]\n"
                    )
                    status.current_prompt += hint_block
                    mode = "free-detection-only" if DA_FREE_DETECTION_ONLY else "hint-only"
                    logger.warning(
                        f"[DIRECT-AGENTIC] {instance_id}: appended {len(questions)} "
                        f"suggested clarifications to prompt ({mode})"
                    )
                else:
                    logger.warning(
                        f"[DIRECT-AGENTIC] {instance_id}: detection returned 0 questions; "
                        f"no hint appended"
                    )
            else:
                for q in questions:
                    status._da_queue.append(f'ask({json.dumps(q)})')
        except Exception as e:
            instance_id = status.original_data.get('instance_id', '?')
            logger.error(
                f"[{DA_METHOD.upper()}] {instance_id}: detection failed: {e}", exc_info=True
            )

    # Override
    if status._da_queue:
        forced = status._da_queue.pop(0)
        instance_id = status.original_data.get('instance_id', '?')
        logger.debug(f"[DIRECT-AGENTIC] {instance_id}: forced action: {forced[:80]}")
        if forced.startswith('ask(') or forced.startswith('submit('):
            return thought, "User", forced
        else:
            return thought, "Environment", forced

    # Queue exhausted → natural agent action
    return thought, obj, action


# Patch in both modules — prompt_utils (canonical) and main (already imported)
prompt_utils.parse_agent_response = _patched_parse_agent_response
import bird_interact.agent.main as _main_module
_main_module.parse_agent_response = _patched_parse_agent_response


# ── Clean-slate prompt override ─────────────────────────────────────
# Once the free forced-retrieval + analysis pre-phase completes, the agent
# re-enters on a pristine prompt: current_prompt (= BASE initial ReAct
# prompt + neutral hint block) PLUS only the history accrued AFTER the
# restart turn. Forced-retrieval turns (and the throwaway transition turn)
# are filtered out so the agent never sees them — but it DOES see its own
# post-restart actions/observations, so it can iterate normally.
#
# (Bug fixed 2026-05-16: the original patch returned current_prompt
# verbatim with NO history *ever*, so the agent never saw any feedback
# after the restart — it repeated one action until budget depletion and
# scored 0/300. The fix reconstructs post-restart history via the canonical
# builder by temporarily masking the pre-restart entries.)
_orig_get_agent_prompt_for_turn = prompt_utils.get_agent_prompt_for_turn


def _patched_get_agent_prompt_for_turn(status):
    if getattr(status, '_da_clean_slate_active', False):
        rt = getattr(status, '_da_restart_turn', 0)
        _full = status.interaction_history
        # Show only post-restart turns to the agent; keep the full log intact
        # for scoring/debugging by restoring it immediately after.
        status.interaction_history = [
            e for e in _full if e.get('turn', 0) > rt
        ]
        try:
            return _orig_get_agent_prompt_for_turn(status)
        finally:
            status.interaction_history = _full
    return _orig_get_agent_prompt_for_turn(status)


prompt_utils.get_agent_prompt_for_turn = _patched_get_agent_prompt_for_turn
_main_module.get_agent_prompt_for_turn = _patched_get_agent_prompt_for_turn

# ── Run the main pipeline ───────────────────────────────────────────

from bird_interact.agent.main import main

if __name__ == "__main__":
    main()
