#!/usr/bin/env bash
# Detection-only pipeline runner (Tables 5-7).
#
# Usage:
#   ./scripts/run_detection.sh --method {direct|direct_multi|mga|se_ast} \
#                              --model {glm|mm|qwen} \
#                              --output <path> [--num-samples N] [--seed S] [--no-ast] \
#                              [--encoder-model {glm|mm|qwen|self|gemini}] \
#                              [--encoder-kind {legacy|multi_label}] \
#                              [--base-url URL --model-id ID --api-key KEY] \
#                              [--encoder-base-url URL --encoder-model-id ID --encoder-api-key KEY]
#
# Examples (full lite-300, paper-canonical encoder mapping):
#   ./scripts/run_detection.sh --method direct       --model qwen --output results/det_qwen_direct.json
#   ./scripts/run_detection.sh --method direct_multi --model qwen --output results/det_qwen_direct_multi.json
#   ./scripts/run_detection.sh --method mga          --model qwen --output results/det_qwen_mga.json
#   ./scripts/run_detection.sh --method se_ast       --model qwen --output results/det_qwen_se_ast.json
#
# direct_multi = "\DirectMulti / SSI" in the paper: 10 calls of the Direct
# prompt at T=0.7, per-call extracted questions merged with verbatim dedup.
# It is the reference row (not the headline) in Tables 5–7.
#
# Smoke (5 samples, ~2-3 min):
#   ./scripts/run_detection.sh --method direct --model qwen --output results/smoke.json \
#                              --num-samples 5
#
# Bring your own model (any OpenAI-compatible endpoint). --model still selects
# the knob profile (MGA temp/θ, no_thinking); the overrides repoint the endpoint
# and served model id. Use --encoder-model self to judge on the same endpoint:
#   ./scripts/run_detection.sh --method direct --model qwen --output results/byo.json \
#       --base-url http://my-host:8000/v1 --model-id my-model \
#       --encoder-model self --num-samples 5
#
# Encoder routing (default = cross-model, paper-canonical, no self-judging):
#   detection model = GLM     → encoder = Qwen3.5-122B
#   detection model = MiniMax → encoder = Qwen3.5-122B
#   detection model = Qwen    → encoder = MiniMax-M2.5
#
# Override the encoder with --encoder-model when:
#   - You only have ONE vLLM endpoint running          → --encoder-model self
#     (uses the same endpoint + model_id for both detection and encoder)
#   - You want a specific cross-encoder pairing         → --encoder-model {glm|mm|qwen}
#     (e.g. --model mm --encoder-model qwen to mirror the paper exactly)
#   - You have no second vLLM endpoint but have a Vertex key → --encoder-model gemini
#     (routes the encoder to Vertex Express Gemini 3.1 Flash-Lite via NEW_GEMINI;
#      a valid cross-family judge, but NOT the paper-canonical encoder for the
#      Qwen detection row — that row used MiniMax-M2.5)
#
# Setting --encoder-model self is the cheapest way to get the pipeline
# running end-to-end when you only have one model deployed; the cross-
# encoding routing is only required for paper-canonical Tables 5–7.
#
# Prerequisites:
#   - One vLLM endpoint for the chosen detection model (set *_BASE_URL env var
#     as in REPRODUCING.md), AND one for the encoder.
#   - Postgres is NOT required: the detection-only pipeline never executes SQL.

set -euo pipefail

usage() {
  sed -n '2,40p' "$0" >&2
  exit 1
}

METHOD=""
MODEL=""
ENCODER_MODEL=""    # default empty = use paper-canonical cross-routing
OUTPUT=""
DATA_PATH="data/bird-interact-lite/bird_interact_data.jsonl"
DATA_DIR="data/bird-interact-lite"
NUM_SAMPLES=""
SEED="64"
EXTRA_FLAGS=""
# Bring-your-own-model overrides (empty = use the --model profile's defaults).
# --model still selects the knob profile (MGA temp/θ, no_thinking); these just
# repoint the endpoint + served model id so you can use ANY OpenAI-compatible model.
DET_BASE_URL_OVERRIDE=""; DET_MODEL_ID_OVERRIDE=""; DET_API_KEY_OVERRIDE=""
ENC_BASE_URL_OVERRIDE=""; ENC_MODEL_ID_OVERRIDE=""; ENC_API_KEY_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)         METHOD="$2"; shift 2 ;;
    --model)          MODEL="$2"; shift 2 ;;
    --encoder-model)  ENCODER_MODEL="$2"; shift 2 ;;
    --output)         OUTPUT="$2"; shift 2 ;;
    --data-path)      DATA_PATH="$2"; shift 2 ;;
    --data-dir)       DATA_DIR="$2"; shift 2 ;;
    --num-samples)    NUM_SAMPLES="$2"; shift 2 ;;
    --seed)           SEED="$2"; shift 2 ;;
    --no-ast)         EXTRA_FLAGS="$EXTRA_FLAGS --no_ast"; shift ;;
    --base-url)          DET_BASE_URL_OVERRIDE="$2"; shift 2 ;;
    --model-id)          DET_MODEL_ID_OVERRIDE="$2"; shift 2 ;;
    --api-key)           DET_API_KEY_OVERRIDE="$2"; shift 2 ;;
    --encoder-base-url)  ENC_BASE_URL_OVERRIDE="$2"; shift 2 ;;
    --encoder-model-id)  ENC_MODEL_ID_OVERRIDE="$2"; shift 2 ;;
    --encoder-api-key)   ENC_API_KEY_OVERRIDE="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) EXTRA_FLAGS="$EXTRA_FLAGS $1"; shift ;;
  esac
done

[[ -z "$METHOD" || -z "$MODEL" || -z "$OUTPUT" ]] && usage

case "$METHOD" in
  direct|direct_multi|mga|se_ast) ;;
  *) echo "ERROR: --method must be direct|direct_multi|mga|se_ast (got '$METHOD')" >&2; exit 1 ;;
esac

# Detection LLM endpoint per --model (matches scripts/run.sh).
case "$MODEL" in
  glm)
    DET_BASE_URL="${GLM45_AIR_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8000/v1}}"
    DET_MODEL_ID="glm-4.5-air"
    DET_API_KEY="${GLM45_AIR_API_KEY:-EMPTY}"
    # Paper: GLM detection → Qwen3.5-122B encoder
    ENC_BASE_URL="${QWEN35_122B_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8002/v1}}"
    ENC_MODEL_ID="qwen3.5-122b"
    ENC_API_KEY="${QWEN35_122B_API_KEY:-EMPTY}"
    # Per-model knobs (paper-canonical): GLM uses MGA temp 1.0 and AST θ 0.60
    METHOD_OVERRIDES=""
    [[ "$METHOD" == "mga" ]] && METHOD_OVERRIDES="--temperature 1.0 --ast_threshold 0.60"
    ;;
  mm)
    DET_BASE_URL="${MINIMAX_M25_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8001/v1}}"
    DET_MODEL_ID="minimax-m2.5"
    DET_API_KEY="${MINIMAX_M25_API_KEY:-EMPTY}"
    # Paper: MiniMax detection → Qwen3.5-122B encoder
    ENC_BASE_URL="${QWEN35_122B_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8002/v1}}"
    ENC_MODEL_ID="qwen3.5-122b"
    ENC_API_KEY="${QWEN35_122B_API_KEY:-EMPTY}"
    METHOD_OVERRIDES=""
    [[ "$METHOD" == "mga" ]] && METHOD_OVERRIDES="--temperature 1.3 --ast_threshold 0.50"
    ;;
  qwen)
    DET_BASE_URL="${QWEN35_122B_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8002/v1}}"
    DET_MODEL_ID="qwen3.5-122b"
    DET_API_KEY="${QWEN35_122B_API_KEY:-EMPTY}"
    # Paper: Qwen detection → MiniMax-M2.5 encoder (avoids Qwen→Qwen self-judge)
    ENC_BASE_URL="${MINIMAX_M25_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8001/v1}}"
    ENC_MODEL_ID="minimax-m2.5"
    ENC_API_KEY="${MINIMAX_M25_API_KEY:-EMPTY}"
    METHOD_OVERRIDES=""
    [[ "$METHOD" == "mga" ]] && METHOD_OVERRIDES="--temperature 1.3 --ast_threshold 0.50"
    ;;
  *) echo "ERROR: --model must be glm|mm|qwen (got '$MODEL')" >&2; exit 1 ;;
esac

# BYO-model: repoint the DETECTION endpoint/model id to anything served.
# Applied here (before --encoder-model) so `--encoder-model self` mirrors the
# overridden detection model rather than the --model profile's default.
[[ -n "$DET_BASE_URL_OVERRIDE" ]] && DET_BASE_URL="$DET_BASE_URL_OVERRIDE"
[[ -n "$DET_MODEL_ID_OVERRIDE" ]] && DET_MODEL_ID="$DET_MODEL_ID_OVERRIDE"
[[ -n "$DET_API_KEY_OVERRIDE"  ]] && DET_API_KEY="$DET_API_KEY_OVERRIDE"

# Encoder override (--encoder-model). Default is paper-canonical cross-routing
# set by the case statement above; the override is for reviewers who don't
# have all three models deployed.
if [[ -n "$ENCODER_MODEL" ]]; then
  case "$ENCODER_MODEL" in
    self)
      ENC_BASE_URL="$DET_BASE_URL"; ENC_MODEL_ID="$DET_MODEL_ID"; ENC_API_KEY="$DET_API_KEY" ;;
    glm)
      ENC_BASE_URL="${GLM45_AIR_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8000/v1}}"
      ENC_MODEL_ID="glm-4.5-air"
      ENC_API_KEY="${GLM45_AIR_API_KEY:-EMPTY}" ;;
    mm)
      ENC_BASE_URL="${MINIMAX_M25_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8001/v1}}"
      ENC_MODEL_ID="minimax-m2.5"
      ENC_API_KEY="${MINIMAX_M25_API_KEY:-EMPTY}" ;;
    qwen)
      ENC_BASE_URL="${QWEN35_122B_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8002/v1}}"
      ENC_MODEL_ID="qwen3.5-122b"
      ENC_API_KEY="${QWEN35_122B_API_KEY:-EMPTY}" ;;
    gemini)
      # Vertex Express Gemini 3.1 Flash-Lite (the paper's user-sim encoder model).
      # pipeline.py detects the "gemini" model-id prefix and routes via
      # bird_interact.llm.gemini_client (Vertex Express); the key comes from
      # NEW_GEMINI / new_gemini in the env. base_url/api_key are unused on this
      # path but kept as the real dev endpoint for clarity / openai_compat fallback.
      ENC_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
      ENC_MODEL_ID="gemini-3.1-flash-lite-preview"
      ENC_API_KEY="${GEMINI_API_KEY:-EMPTY}" ;;
    *) echo "ERROR: --encoder-model must be glm|mm|qwen|self|gemini (got '$ENCODER_MODEL')" >&2; exit 1 ;;
  esac
fi

# BYO-model: repoint the ENCODER endpoint/model id (highest precedence, applied
# after --encoder-model). For a single self-hosted endpoint use
# `--encoder-model self` together with the detection --base-url/--model-id.
[[ -n "$ENC_BASE_URL_OVERRIDE" ]] && ENC_BASE_URL="$ENC_BASE_URL_OVERRIDE"
[[ -n "$ENC_MODEL_ID_OVERRIDE" ]] && ENC_MODEL_ID="$ENC_MODEL_ID_OVERRIDE"
[[ -n "$ENC_API_KEY_OVERRIDE"  ]] && ENC_API_KEY="$ENC_API_KEY_OVERRIDE"

mkdir -p "$(dirname "$OUTPUT")"
NUMSAMP_FLAG=""
[[ -n "$NUM_SAMPLES" ]] && NUMSAMP_FLAG="--limit $NUM_SAMPLES"

# Qwen detection needs --no_thinking; MM/GLM keep CoT.
THINK_FLAG=""
[[ "$MODEL" == "qwen" ]] && THINK_FLAG="--no_thinking"

echo "→ detection-only | method=$METHOD | model=$MODEL | seed=$SEED → $OUTPUT"
echo "   det:  $DET_MODEL_ID @ $DET_BASE_URL"
echo "   enc:  $ENC_MODEL_ID @ $ENC_BASE_URL"

exec python -m bird_interact.detection_only.pipeline \
  --method "$METHOD" \
  --data_path "$DATA_PATH" \
  --data_dir  "$DATA_DIR" \
  --output    "$OUTPUT" \
  --base_url  "$DET_BASE_URL" \
  --model_id  "$DET_MODEL_ID" \
  --api_key   "$DET_API_KEY" \
  --encoder_base_url  "$ENC_BASE_URL" \
  --encoder_model_id  "$ENC_MODEL_ID" \
  --encoder_api_key   "$ENC_API_KEY" \
  --num_threads 8 \
  --seed "$SEED" \
  --resume \
  $NUMSAMP_FLAG \
  $THINK_FLAG \
  $METHOD_OVERRIDES \
  $EXTRA_FLAGS
