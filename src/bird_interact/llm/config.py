# BIRD-Interact LLM endpoint registry.
#
# This dict maps a *model key* (the string passed to --agent_model /
# --user_model / --user_encoder_model in the run scripts) to an
# OpenAI-compatible endpoint config.
#
# Each entry reads base_url and api_key from environment variables FIRST,
# then falls back to a documented default. This lets a reviewer point at
# any vLLM (or OpenAI-compatible) endpoint without editing this file.
#
# Environment variables (all optional):
#   GLM45_AIR_BASE_URL       — endpoint for GLM-4.5-Air
#   GLM45_AIR_API_KEY        — auth key for GLM-4.5-Air (default "EMPTY")
#   MINIMAX_M25_BASE_URL     — endpoint for MiniMax-M2.5
#   MINIMAX_M25_API_KEY
#   QWEN35_122B_BASE_URL     — endpoint for Qwen3.5-122B
#   QWEN35_122B_API_KEY
#   GEMINI_BASE_URL          — endpoint for Gemini (user-sim encoder)
#   GEMINI_API_KEY
#
# `OPENAI_API_BASE` is honored as a universal fallback when no per-model
# base URL is set. Useful when all three models share one vLLM server (e.g.,
# a hot-swap setup) or when you only run one model at a time.

import os

_default_openai = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
_gemini_default = "https://generativelanguage.googleapis.com/v1beta/openai/"

model_config = {
    # ────────────────────────────────────────────────────────────────────
    # Agent / user-simulator backend models.
    # Keys here are what call_api_batch.py's generic OpenAI-compatible
    # branch looks up (via model_config.get(model_name)). Reviewer-facing
    # short names (--model glm | mm | qwen) live in scripts/run.sh.
    # ────────────────────────────────────────────────────────────────────

    "glm-4.5-air-cluster": {
        "api_key": os.environ.get("GLM45_AIR_API_KEY", "EMPTY"),
        "base_url": os.environ.get("GLM45_AIR_BASE_URL", _default_openai),
        "model_id": "glm-4.5-air",
    },
    "minimax-m25-cluster": {
        "api_key": os.environ.get("MINIMAX_M25_API_KEY", "EMPTY"),
        "base_url": os.environ.get("MINIMAX_M25_BASE_URL", _default_openai),
        "model_id": "minimax-m2.5",
    },
    "qwen35-122b-cluster": {
        "api_key": os.environ.get("QWEN35_122B_API_KEY", "EMPTY"),
        "base_url": os.environ.get("QWEN35_122B_BASE_URL", _default_openai),
        "model_id": "qwen3.5-122b",
    },

    # ────────────────────────────────────────────────────────────────────
    # User-simulator encoder (Gemini Vertex Express).
    # The experiments that produced our 3×3 grid used gemini-3-1-flash-lite
    # as the cross-model encoder for the user simulator. To swap in any
    # OpenAI-compatible endpoint, set GEMINI_BASE_URL + GEMINI_API_KEY.
    # Two Gemini backends exist: Vertex Express (paid, bulk-safe, via
    # google-genai with NEW_GEMINI) and the free developer endpoint (via
    # OpenAI-compat with GEMINI_API_KEY). Vertex Express is preferred for
    # multi-thousand-call encoder runs; the free tier is rate-limited.
    # ────────────────────────────────────────────────────────────────────

    "gemini-3-1-flash-lite": {
        "api_key": os.environ.get("GEMINI_API_KEY", "EMPTY"),
        "base_url": os.environ.get("GEMINI_BASE_URL", _gemini_default),
        "model_id": "gemini-3.1-flash-lite-preview",
    },

    # Any other model name passed to the runners is treated as a custom
    # OpenAI-compatible endpoint by call_api_batch.py: it reads OPENAI_API_BASE
    # / OPENAI_API_KEY and uses the model name verbatim as the served id. So a
    # reviewer can swap in their own model without editing this file.
}
