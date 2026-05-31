# Two Surfaces of Ambiguity: Complementary Detection for Text-to-SQL

**Reproduction package** for our paper. Standalone code for the three pipelines
it reports, evaluated on the BIRD-Interact and AMBROSIA benchmarks. The numbers
live in the paper; this package lets you reproduce them on your own infrastructure.

## What's in here

The paper names four detection methods, organized on two axes:

| | 1 call | 10 calls |
|---|---|---|
| **Introspection** (no SQL) | **Si-1** — Self-Introspection, single deterministic call | **Si-10** — 10× Si-1 at T=0.7 + dedup |
| **Sampling-based** (SQL) | **Mcs-1** — Multi-Candidate SQL, single-pass: one call yielding up to 10 interpretations | **Mcs-10** — 10 independent SQL samples + AST clustering |

Paper headline: **Si-1 ∪ Mcs-1** — per-term union of an Si-1 run and an Mcs-1 run.

The three pipelines:

| Pipeline | Module | What it does |
|---|---|---|
| **a-Interact** (end-to-end agentic) | `bird_interact.agent`, `bird_interact.detection` | Drives the full ReAct loop (`ask`, `execute_sql`, `get_schema`, `submit`) against Postgres + an LLM; reports normalized reward. Methods: `baseline` (raw ReAct), `direct` (Si-1 only), `union` (Si-1 ∪ Mcs-1). |
| **Detection-only** | `bird_interact.detection_only` | Generates clarification questions, routes them through a frozen encoder onto GT terms. Methods: `direct` (Si-1), `direct_multi` (Si-10), `mga` (Mcs-1), `se_ast` (Mcs-10). Per-term `Si-1 ∪ Mcs-1` union and an encoder-replay tool live in the aggregator. |
| **AMBROSIA cross-benchmark** | `ambrosia` (standalone package, separate from `bird_interact`) | Runs the four arms (`baseline`, `direct` = Si-1, `union_mcs` = Si-1 ∪ Mcs-10, `union_spmi` = Si-1 ∪ Mcs-1) against the AMBROSIA SQLite databases and scores by execution equivalence. |

CLI flags use the historic short names (`direct`, `union`, etc.); paper-table
rows use the `Si-N` / `Mcs-N` labels.

## Prerequisites

- **Python 3.10+.** Recommended install (pinned to the versions the paper ran
  on): `pip install -r requirements.txt && pip install -e . --no-deps`. A bare
  `pip install -e .` also works but pulls the latest compatible dependency releases.
- **A live LLM endpoint** that speaks the OpenAI-compatible API. We used
  vLLM with GLM-4.5-Air, MiniMax-M2.5, or Qwen3.5-122B (manifests under `k8s/`).
- **The lite-300 dataset** for the agentic and detection-only pipelines —
  not bundled, fetch from `https://huggingface.co/datasets/birdsql/bird-interact-lite`.
  See `data/README.md` for the expected layout.
- **A Postgres instance** for the agentic pipeline only — `./scripts/load_postgres.sh`
  pulls the upstream pre-built image.
- **The AMBROSIA dataset** for the AMBROSIA pipeline only — fetch from
  `https://github.com/saparina/AMBROSIA`, point `AMBROSIA_DIR` at it.
- **A user-sim encoder endpoint** for a-Interact only — we used Gemini Vertex
  Express (`gemini-3-1-flash-lite`); any OpenAI-compatible endpoint works.

## Quick start (smoke run)

```bash
# Pinned environment the paper ran on (recommended). A bare `pip install -e .`
# also works but resolves deps to the latest compatible releases.
pip install -r requirements.txt && pip install -e . --no-deps

# 1. Fetch the lite-300 dataset and place under data/bird-interact-lite/
#    (see data/README.md)

# 2. Point at your LLM endpoint
export QWEN35_122B_BASE_URL="http://your-vllm:8000/v1"

# 3. Run a 5-sample detection-only smoke (no Postgres needed).
#    --encoder-model self points the user-sim encoder at the same endpoint;
#    paper-canonical Tables 5-7 use a cross-encoder model — see REPRODUCING.md.
./scripts/run_detection.sh --method direct --model qwen \
  --encoder-model self \
  --output results/smoke.json --num-samples 5
python -m bird_interact.detection_only.aggregate --input results/smoke.json
```

`--model {glm|mm|qwen}` selects the per-model knob profile and assumes the
matching served model id. To run with **any other** OpenAI-compatible model,
add the override flags (and `--encoder-model self` to judge on the same endpoint):

```bash
./scripts/run_detection.sh --method direct --model qwen \
  --base-url "$YOUR_URL" --model-id "$YOUR_MODEL" \
  --encoder-model self --output results/smoke.json --num-samples 5
```

For the full per-pipeline procedure (including a-Interact's Postgres + Gemini
setup, vLLM flags per model, and the AMBROSIA fetch), see `REPRODUCING.md`.

## Expected wall-clock

Anchors so you can sanity-check before committing GPU time (the 10-call
detection methods are multi-hour). All numbers are for the full lite-300 /
AMBROSIA-1149-ambiguous splits on a single 2-GPU vLLM server matching the
recipes in `k8s/vllm_server_*.yaml`.

| Pipeline (full split) | Si-1 / direct | Si-10 / direct_multi | Mcs-1 / mga | Mcs-10 / se_ast | Si-1 ∪ Mcs-1 / union |
|---|---|---|---|---|---|
| a-Interact lite-300 (300 samples; baseline ≈1.5-2 h) | ≈2-3.5 h | — | — | — | ≈3 h |
| Detection-only lite-300 (300), from scratch | ~20-40 min | ~3-5 h | ~1-2 h | ~8-12 h | union is post-hoc (no LLM calls) |
| AMBROSIA (1,149 ambiguous of 3,819; baseline ≈20 min) | ≈1.5 h | — | — | ≈2.9 h (union_mcs) | ≈1.9 h (union_spmi) |

Detection-only figures are **from-scratch** runs (generation-dominated, scaling
with the per-method call count). The **replay** path — `scripts/replay_encoder.sh`
when the clarification questions already exist — is encoder-only and runs in
**minutes** (~10x cheaper for the 11-call methods). Anchored to GLM-4.5-Air with
reasoning; `--no_thinking` / faster endpoints are quicker. See `REPRODUCING.md` §2.

Quick sanity-check NRs (a-Interact lite-300, baseline → Si-1 → Si-1 ∪ Mcs-1):
GLM-4.5-Air 13.0 → 20.4 → 20.2; MiniMax-M2.5 18.2 → 24.6 → 25.2;
Qwen3.5-122B 24.3 → 26.4 → 27.1. Both detection arms beat the ReAct baseline on
all three models; if they don't, something is mis-configured.

To use your own prompts, pass `--analysis_prompt_file path/to/prompt.txt` to
`run_detection.sh` (replaces the shared analysis prompt for all four methods),
or see the prompt-location table in `REPRODUCING.md` (section "2c. Customizing
prompts").

## Layout

```
github-submission/
├── README.md                    ← this file
├── REPRODUCING.md               ← per-pipeline run procedure
├── pyproject.toml               ← pip install -e .
├── requirements.txt
├── data/                        ← mount point for upstream datasets (see data/README.md)
├── scripts/                     ← run.sh / run_detection.sh / run_ambrosia.sh / replay_encoder.sh / load_postgres.sh / extract_results.py
├── docker/                      ← docker-compose for the upstream Postgres image
├── k8s/                         ← vLLM server + eval-job manifests
└── src/
    ├── ambrosia/                ← AMBROSIA pipeline (top-level, independent of bird_interact)
    └── bird_interact/
        ├── agent/               ← raw-ReAct pipeline
        ├── detection/           ← Si-1 / Union forced-scaffold extension for a-Interact
        ├── detection_only/      ← standalone detection (legacy + multi-label encoders, replay tool)
        ├── env/                 ← SQL test-case execution + Postgres utilities (test_case_utils)
        ├── user_simulator/      ← user-simulator encoder/decoder prompt templates + SQL segmenter
        ├── llm/                 ← OpenAI-compatible client wrappers
        └── db_config/           ← Postgres connection config
```

## License

Apache-2.0 for the code in this repository. The datasets, model weights, and
Postgres image are governed by their respective upstream licenses.

## Citation

If you use this code, please cite our paper, *Two Surfaces of Ambiguity:
Complementary Detection for Text-to-SQL*, and the original BIRD-Interact
benchmark (`arxiv:2510.05318`).
