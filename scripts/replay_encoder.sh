#!/usr/bin/env bash
# Re-judge an existing detection JSON with a different encoder, cheaply.
#
# Usage:
#   ./scripts/replay_encoder.sh --input <existing.json> --output <new.json> \
#                               --model {glm|mm|qwen} \
#                               [--encoder-kind {legacy|multi_label}]
#
# Examples:
#
#   # Take an existing legacy-encoder Direct run and re-judge with the
#   # multi-label encoder (measures the knowledge-ambiguity recall lift
#   # without re-running the detection LLM).
#   ./scripts/replay_encoder.sh \
#       --input  results/detection/qwen_direct.json \
#       --output results/detection/qwen_direct_multi.json \
#       --model  qwen \
#       --encoder-kind multi_label
#
#   # Cross-encoder swap: re-judge a Qwen-detection run with MiniMax as
#   # the encoder (paper-canonical cross-routing for Tables 5-7).
#   ./scripts/replay_encoder.sh \
#       --input  results/detection/qwen_direct.json \
#       --output results/detection/qwen_direct_mm_encoder.json \
#       --model  mm
#
#   # Then aggregate the two side-by-side:
#   python -m bird_interact.detection_only.aggregate \
#       --input results/detection/qwen_direct.json:legacy \
#       --input results/detection/qwen_direct_multi.json:multi_label \
#       --markdown results/detection/encoder_ablation.md
#
# This replays only the encoder calls — the per-sample
# `extracted_detections` from the original run are reused as-is. For an
# SE+AST run that took 11 LLM calls per sample, this is roughly a 10x
# cost reduction.
#
# --model -> encoder endpoint + model id (same mapping as run_detection.sh):
#   glm  | GLM45_AIR_BASE_URL    + glm-4.5-air
#   mm   | MINIMAX_M25_BASE_URL  + minimax-m2.5
#   qwen | QWEN35_122B_BASE_URL  + qwen3.5-122b
#
# --encoder-kind:
#   legacy       (default) standard single-label BIRD-Interact user-sim encoder
#   multi_label             extension where one question can credit up to 3 GT
#                           terms via labeled(primary, also=[...])

set -euo pipefail

usage() {
  sed -n '2,42p' "$0" >&2
  exit 1
}

INPUT=""
OUTPUT=""
MODEL=""
ENCODER_KIND="legacy"
DATA_PATH="data/bird-interact-lite/bird_interact_data.jsonl"
DATA_DIR="data/bird-interact-lite"
EXTRA_FLAGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)         INPUT="$2";        shift 2 ;;
    --output)        OUTPUT="$2";       shift 2 ;;
    --model)         MODEL="$2";        shift 2 ;;
    --encoder-kind)  ENCODER_KIND="$2"; shift 2 ;;
    --data-path)     DATA_PATH="$2";    shift 2 ;;
    --data-dir)      DATA_DIR="$2";     shift 2 ;;
    -h|--help) usage ;;
    *) EXTRA_FLAGS="$EXTRA_FLAGS $1"; shift ;;
  esac
done

[[ -z "$INPUT" || -z "$OUTPUT" || -z "$MODEL" ]] && usage

case "$ENCODER_KIND" in
  legacy|multi_label) ;;
  *) echo "ERROR: --encoder-kind must be legacy|multi_label (got '$ENCODER_KIND')" >&2; exit 1 ;;
esac

case "$MODEL" in
  glm)
    ENC_BASE_URL="${GLM45_AIR_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8000/v1}}"
    ENC_MODEL_ID="glm-4.5-air"
    ENC_API_KEY="${GLM45_AIR_API_KEY:-EMPTY}"
    ;;
  mm)
    ENC_BASE_URL="${MINIMAX_M25_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8001/v1}}"
    ENC_MODEL_ID="minimax-m2.5"
    ENC_API_KEY="${MINIMAX_M25_API_KEY:-EMPTY}"
    ;;
  qwen)
    ENC_BASE_URL="${QWEN35_122B_BASE_URL:-${OPENAI_API_BASE:-http://localhost:8002/v1}}"
    ENC_MODEL_ID="qwen3.5-122b"
    ENC_API_KEY="${QWEN35_122B_API_KEY:-EMPTY}"
    ;;
  *) echo "ERROR: --model must be glm|mm|qwen (got '$MODEL')" >&2; exit 1 ;;
esac

mkdir -p "$(dirname "$OUTPUT")"

echo "→ replay-encoder | encoder=$ENC_MODEL_ID | kind=$ENCODER_KIND"
echo "   input:  $INPUT"
echo "   output: $OUTPUT"
echo "   endpoint: $ENC_BASE_URL"

exec python -m bird_interact.detection_only.replay_encoder \
  --input         "$INPUT" \
  --output        "$OUTPUT" \
  --data_path     "$DATA_PATH" \
  --data_dir      "$DATA_DIR" \
  --base_url      "$ENC_BASE_URL" \
  --api_key       "$ENC_API_KEY" \
  --model_id      "$ENC_MODEL_ID" \
  --encoder_kind  "$ENCODER_KIND" \
  --num_threads   8 \
  $EXTRA_FLAGS
