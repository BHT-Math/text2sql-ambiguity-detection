#!/usr/bin/env bash
# AMBROSIA cross-benchmark pipeline runner (paper appendix table).
#
# Usage:
#   ./scripts/run_ambrosia.sh --method {baseline|direct|union_mcs|union_spmi} \
#                             --model  {glm|mm|qwen} \
#                             --output-dir <path> [--num-samples N] [--seed S]
#
# Examples (full AMBROSIA test split, paper-canonical):
#   ./scripts/run_ambrosia.sh --method baseline   --model qwen --output-dir results/amb_qwen_baseline
#   ./scripts/run_ambrosia.sh --method direct     --model qwen --output-dir results/amb_qwen_direct
#   ./scripts/run_ambrosia.sh --method union_spmi --model qwen --output-dir results/amb_qwen_union_spmi
#
# Smoke (5 samples, ~1–2 min):
#   ./scripts/run_ambrosia.sh --method baseline --model qwen --output-dir results/smoke_amb \
#                             --num-samples 5
#
# Methods (paper name → CLI flag):
#   Baseline                  baseline    AMBROSIA-native single call at T=0
#                                         ("write all interpretations").
#   Si-1                      direct      Two-stage Self-Introspection. Stage 1
#                                         enumerates paraphrases, Stage 2 writes
#                                         one SQL per paraphrase.
#   Union (Si-1 ∪ Mcs-10)     union_mcs   Si-1 ∪ Mcs-10 (10 independent single-SQL
#                                         samples at T=0.7).
#   Union (Si-1 ∪ Mcs-1)      union_spmi  Si-1 ∪ Mcs-1 (single-pass: one call asking
#                                         for up to 10 distinct interpretations at
#                                         high T). PAPER HEADLINE.
#
# Per-model knobs (set automatically by --model). The "Mcs-1 temperature" applies
# to the single-pass channel used by union_spmi:
#   model | Mcs-1 temperature | no-think | server endpoint env var
#   glm   | 1.0               | yes      | GLM45_AIR_BASE_URL
#   mm    | 1.3               | no       | MINIMAX_M25_BASE_URL
#   qwen  | 1.3               | yes      | QWEN35_122B_BASE_URL
#
# Prerequisites:
#   - AMBROSIA dataset on disk (set AMBROSIA_DIR or pass --ambrosia-dir).
#     Default: data/ambrosia (contains ambrosia.csv + per-domain
#     SQLite files). The AMBROSIA dataset is released by Saparina & Lapata
#     (NeurIPS 2024) — see README.md for the download instructions.
#   - An OpenAI-compatible endpoint for the chosen model.
#   - No Postgres / no user simulator needed. Coverage is scored against
#     per-sample SQLite databases shipped with AMBROSIA.

set -euo pipefail

usage() {
  sed -n '2,42p' "$0" >&2
  exit 1
}

METHOD=""
MODEL=""
OUTPUT_DIR=""
AMBROSIA_DIR="${AMBROSIA_DIR:-data/ambrosia}"
NUM_SAMPLES=""
SEED="42"
EXTRA_FLAGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)        METHOD="$2";        shift 2 ;;
    --model)         MODEL="$2";         shift 2 ;;
    --output-dir)    OUTPUT_DIR="$2";    shift 2 ;;
    --ambrosia-dir)  AMBROSIA_DIR="$2";  shift 2 ;;
    --num-samples)   NUM_SAMPLES="$2";   shift 2 ;;
    --seed)          SEED="$2";          shift 2 ;;
    -h|--help) usage ;;
    *) EXTRA_FLAGS="$EXTRA_FLAGS $1"; shift ;;
  esac
done

[[ -z "$METHOD" || -z "$MODEL" || -z "$OUTPUT_DIR" ]] && usage

case "$METHOD" in
  baseline|direct|union_mcs|union_spmi) ;;
  *) echo "ERROR: --method must be baseline|direct|union_mcs|union_spmi (got '$METHOD')" >&2; exit 1 ;;
esac

# Friendly --model → vLLM endpoint, model_id, per-model knobs.
case "$MODEL" in
  glm)
    BASE_URL="${GLM45_AIR_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8000/v1}}"
    MODEL_ID="glm-4.5-air"
    API_KEY="${GLM45_AIR_API_KEY:-EMPTY}"
    # GLM uses Mcs-1 temperature 1.0; at 1.3 the 10 interpretations collapse.
    SPMI_TEMP="1.0"
    NO_THINK="--no_thinking"
    ;;
  mm)
    BASE_URL="${MINIMAX_M25_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8000/v1}}"
    MODEL_ID="minimax-m2.5"
    API_KEY="${MINIMAX_M25_API_KEY:-EMPTY}"
    # MiniMax-M2.5 needs --reasoning-parser minimax_m2 on the vLLM server.
    # Do NOT pass --no_thinking — its <think> blocks are the reasoning,
    # which the server-side parser routes into reasoning_content.
    SPMI_TEMP="1.3"
    NO_THINK=""
    ;;
  qwen)
    BASE_URL="${QWEN35_122B_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8000/v1}}"
    MODEL_ID="qwen3.5-122b"
    API_KEY="${QWEN35_122B_API_KEY:-EMPTY}"
    SPMI_TEMP="1.3"
    NO_THINK="--no_thinking"
    ;;
  *) echo "ERROR: --model must be glm|mm|qwen (got '$MODEL')" >&2; exit 1 ;;
esac

mkdir -p "$OUTPUT_DIR"
SAMPLES_FLAG=""
[[ -n "$NUM_SAMPLES" ]] && SAMPLES_FLAG="--max_samples $NUM_SAMPLES"

# We always set OPENAI_API_KEY so the OpenAI client doesn't complain.
export OPENAI_API_KEY="$API_KEY"

echo "→ AMBROSIA | method=$METHOD | model=$MODEL ($MODEL_ID) → $OUTPUT_DIR"
echo "   base_url     = $BASE_URL"
echo "   ambrosia_dir = $AMBROSIA_DIR"
echo "   spmi_temp    = $SPMI_TEMP (used by --method union_spmi)"

exec python -m ambrosia.pipeline \
  --method            "$METHOD" \
  --base_url          "$BASE_URL" \
  --api_key           "$API_KEY" \
  --model_id          "$MODEL_ID" \
  --ambrosia_dir      "$AMBROSIA_DIR" \
  --output_dir        "$OUTPUT_DIR" \
  --spmi_temperature  "$SPMI_TEMP" \
  --num_threads       8 \
  --seed              "$SEED" \
  $NO_THINK \
  $SAMPLES_FLAG \
  $EXTRA_FLAGS
