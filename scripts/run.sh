#!/usr/bin/env bash
# BIRD-Interact unified run script.
#
# Usage:
#   ./scripts/run.sh --method {baseline|direct|union} --model {glm|mm|qwen} \
#                    --output-dir <path> [--num-samples N] [--data-path <jsonl>]
#
# Examples (full lite-300):
#   ./scripts/run.sh --method baseline --model qwen --output-dir results/qwen-baseline
#   ./scripts/run.sh --method direct   --model qwen --output-dir results/qwen-direct
#   ./scripts/run.sh --method union    --model qwen --output-dir results/qwen-union
#
# Smoke (1 sample, ~5 min):
#   ./scripts/run.sh --method baseline --model qwen --output-dir results/smoke --num-samples 1
#
# Prerequisites:
#   - Postgres reachable at $POSTGRES_HOST:$POSTGRES_PORT with the bird-interact
#     lite database images loaded (use docker-compose stack in ../docker/).
#   - vLLM (or OpenAI-compatible) endpoint for the chosen model; set the
#     matching *_BASE_URL env var or rely on OPENAI_API_BASE.
#   - User-simulator encoder: Gemini Vertex Express via NEW_GEMINI (paid
#     bulk-safe), OR the free tier via GEMINI_API_KEY (rate-limited; OK for
#     smoke runs only). The Vertex Express path uses google-genai.
#
# The canonical hyperparameters that produced our 3×3 grid are baked in
# (max_turns=60, patience=6, max_tokens=24576, num_threads=8, …). Do not
# change them if you want to reproduce the published NRs.

set -euo pipefail

usage() {
  sed -n '2,28p' "$0" >&2
  exit 1
}

METHOD=""
MODEL=""
OUTPUT_DIR=""
DATA_PATH="data/bird-interact-lite/bird_interact_data.jsonl"
NUM_SAMPLES=""
EXTRA_FLAGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --model)  MODEL="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) EXTRA_FLAGS="$EXTRA_FLAGS $1"; shift ;;
  esac
done

[[ -z "$METHOD" || -z "$MODEL" || -z "$OUTPUT_DIR" ]] && usage

case "$METHOD" in
  baseline|direct|union) ;;
  *) echo "ERROR: --method must be baseline|direct|union (got '$METHOD')" >&2; exit 1 ;;
esac

# Friendly model name → internal key registered in src/bird_interact/llm/config.py.
case "$MODEL" in
  glm)  MODEL_KEY="glm-4.5-air-cluster";   MODEL_ID="glm-4.5-air"   ;;
  mm)   MODEL_KEY="minimax-m25-cluster";   MODEL_ID="minimax-m2.5"  ;;
  qwen) MODEL_KEY="qwen35-122b-cluster";   MODEL_ID="qwen3.5-122b"  ;;
  *) echo "ERROR: --model must be glm|mm|qwen (got '$MODEL')" >&2; exit 1 ;;
esac

mkdir -p "$OUTPUT_DIR"

# Compute base_url for the detection LLM (used by direct/union only).
# Reads the model-specific env var that the matching k8s/vllm_server_*.yaml or
# your local vLLM exposes; falls back to OPENAI_API_BASE.
case "$MODEL" in
  glm)  DA_BASE_URL="${GLM45_AIR_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8000/v1}}" ;;
  mm)   DA_BASE_URL="${MINIMAX_M25_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8000/v1}}" ;;
  qwen) DA_BASE_URL="${QWEN35_122B_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8000/v1}}" ;;
esac

NUMSAMP_FLAG=""
[[ -n "$NUM_SAMPLES" ]] && NUMSAMP_FLAG="--limit $NUM_SAMPLES"

# ─── canonical hyperparams (DO NOT CHANGE) ──────────────────────────────
COMMON_FLAGS=(
  --data_path                "$DATA_PATH"
  --output_path              "$OUTPUT_DIR/evaluation_results.jsonl"
  --agent_model              "$MODEL_KEY"
  --user_model               "$MODEL_KEY"
  --user_encoder_model       gemini-3-1-flash-lite
  --user_sim_mode            encoder_decoder
  --user_sim_prompt_version  v2
  --num_threads              8
  --user_num_threads         8
  --max_turns                60
  --user_patience_budget     6
  --max_tokens               24576
  --log_level                INFO
  --resume
)

# ────────────────────────────────────────────────────────────────────────
if [[ "$METHOD" == "baseline" ]]; then
  echo "→ baseline (raw ReAct) | model=$MODEL ($MODEL_KEY) → $OUTPUT_DIR"
  exec python -m bird_interact.agent.main \
    "${COMMON_FLAGS[@]}" \
    $NUMSAMP_FLAG \
    $EXTRA_FLAGS
fi

# direct / union
DA_FLAGS=(
  --da_method        "$METHOD"
  --da_hint_only     true
  --da_clean_slate   true
  --da_cache_dir     "$OUTPUT_DIR/detection_cache"
  --da_base_url      "$DA_BASE_URL"
  --da_model_id      "$MODEL_ID"
)
if [[ "$METHOD" == "union" ]]; then
  # MGA-channel temperature is per-model: GLM collapses to one cluster at 1.3,
  # so it uses 1.0; MM and Qwen need 1.3 to keep the 10 interpretations diverse.
  case "$MODEL" in
    glm)        MGA_TEMP="1.0" ;;
    mm|qwen)    MGA_TEMP="1.3" ;;
  esac
  DA_FLAGS+=(--da_mga_temperature "$MGA_TEMP" --da_no_ast true)
fi

echo "→ $METHOD (hint-only clean-slate) | model=$MODEL ($MODEL_KEY) → $OUTPUT_DIR"
echo "   da_base_url=$DA_BASE_URL"
exec python -m bird_interact.detection.run_direct_agentic \
  "${COMMON_FLAGS[@]}" \
  "${DA_FLAGS[@]}" \
  $NUMSAMP_FLAG \
  $EXTRA_FLAGS
