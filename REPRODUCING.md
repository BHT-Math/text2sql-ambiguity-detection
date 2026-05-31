# Reproducing the pipelines

Three independent pipelines: agentic (a-Interact), detection-only, and
AMBROSIA cross-benchmark. Each section below is self-contained — pick the
pipeline you want, install once, and follow it.

## Install

Install the pinned environment (the versions the paper ran on), then register
the package:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

A bare `pip install -e .` also works — it resolves dependencies from
`pyproject.toml` to the latest compatible releases, and the code imports and
runs on current versions — but the pinned set above is what produced the numbers.

## Bring your own data

This package does not ship any benchmark data. Fetch:

- **BIRD-Interact lite-300** — required by the agentic and detection-only
  pipelines. See `data/README.md`.
- **AMBROSIA** — required by the AMBROSIA pipeline only. See
  `data/README.md`.

The runners look for the data under `data/bird-interact-lite/` and
`data/ambrosia/` respectively; override with `--data_path` /
`--ambrosia_dir` if your layout differs.

## Bring your own LLM endpoints

Every pipeline talks to an OpenAI-compatible chat-completions endpoint.
Set one or more of these env vars to point at your endpoints:

```bash
export GLM45_AIR_BASE_URL="http://your-glm:8000/v1"
export MINIMAX_M25_BASE_URL="http://your-mm:8000/v1"
export QWEN35_122B_BASE_URL="http://your-qwen:8000/v1"
```

The friendly `--model {glm|mm|qwen}` flag in the run scripts reads the matching
variable: `glm` → `GLM45_AIR_BASE_URL`, `mm` → `MINIMAX_M25_BASE_URL`, `qwen` →
`QWEN35_122B_BASE_URL` (each falls back to `OPENAI_API_BASE`, then
`http://localhost:8000/v1`). So `./scripts/run.sh --model qwen …` (or
`run_detection.sh` / `run_ambrosia.sh`) talks to whatever `QWEN35_122B_BASE_URL`
points at — that one env var is how you say where your Qwen server lives.

Manifests for vLLM servers serving each of the three open-weight models we
used are under `k8s/`. Critical flags per model (see those files for the
full launch command):

| Model | Required vLLM flag | Why |
|---|---|---|
| GLM-4.5-Air | `--reasoning-parser glm45` | else `<think>` blocks leak into response and ~6 % of agent turns silently lose their action |
| MiniMax-M2.5 | `--reasoning-parser minimax_m2` | else `<think>` consumes the generation budget on hard samples |
| Qwen3.5-122B | `--chat-template <…>/qwen35_no_think_chat_template.jinja` | Qwen3.5 has no `<think>` parser; default-disable thinking via chat template |

---

## 1. Agentic (a-Interact)

End-to-end ReAct loop against Postgres + an LLM. Methods (paper Table
tab:agentic-nr-lite300, CLI flag in parens):
- **ReAct baseline** (`--method baseline`)
- **Si-1 only** (`--method direct`) — single Self-Introspection call wired in
  as a forced clarification pool.
- **Si-1 ∪ Mcs-1** (`--method union`) — Si-1 + a Multi-Candidate SQL call
  (single-pass, 10 interpretations from one call), round-robin interleaved.

Expected wall-clock on a 2-GPU vLLM matching `k8s/vllm_server_*.yaml`, lite-300:
baseline ≈1.5-2 h, `direct` ≈2-3.5 h, `union` ≈3 h per model (measured on our
GLM-4.5-Air / MiniMax-M2.5 / Qwen3.5-122B reruns; faster endpoints are quicker).

**Additional prerequisites:**

```bash
# Postgres image with the 20 lite-300 databases preloaded
./scripts/load_postgres.sh    # pulls docker.io/shawnxxh/bird-interact-postgresql:latest

# User-simulator encoder credentials. Vertex Express (paid) is what we used:
export NEW_GEMINI="<your-vertex-token>"
pip install google-genai
# Free Gemini tier also works for smoke runs:
export GEMINI_API_KEY="<your-key>"
```

Use Vertex Express (`NEW_GEMINI`) for full lite-300 runs — the free
`GEMINI_API_KEY` tier silently rate-limits after a few hundred encoder calls
and will corrupt a multi-thousand-call run.

**Run:**

```bash
./scripts/run.sh --method baseline --model qwen --output-dir results/qwen-baseline
./scripts/run.sh --method direct   --model qwen --output-dir results/qwen-direct
./scripts/run.sh --method union    --model qwen --output-dir results/qwen-union
```

Three ways to wire Postgres, in increasing complexity: (a) `./scripts/load_postgres.sh`
on the host machine → connect via the `localhost:5432` default; (b) the
`docker/docker-compose.yml` stack → it exports `POSTGRES_HOST=postgres` into
the eval container, which the runner reads automatically; (c) Kubernetes →
the shipped eval YAMLs export `POSTGRES_HOST=postgres` against an in-cluster
Postgres service. The runner picks `POSTGRES_HOST`/`POSTGRES_PORT` from the
env first, then `localhost:5432`; override on the CLI when neither fits:

```bash
./scripts/run.sh --method baseline --model qwen \
  --output-dir results/qwen-baseline \
  --db_host my-postgres-host.example.com --db_port 5432
```

Output: per-cell `evaluation_results.jsonl` (sample trajectories) +
`evaluation_results_metrics.json` (aggregates). Score with:

```bash
python scripts/extract_results.py results/
```

The script applies the canonical NR formula
`NR = (P1 * 0.7 + P2 * 0.3) / N`. The pipeline's own `avg_reward` field
holds the final-submit reward only and is not the NR; ignore it.

**Knobs that matter** (paper-canonical, baked into `scripts/run.sh`):
`--max_turns 60`, `--user_patience_budget 6`, `--max_tokens 24576`,
`--num_threads 8`, `--user_sim_prompt_version v2`,
`--user_encoder_model gemini-3-1-flash-lite`, `--resume`. Change only if
you know what you're doing.

For the `union` method the MGA-generation temperature is **model-specific**
(set automatically from `--model`): GLM-4.5-Air uses `--da_mga_temperature 1.0`
(it collapses to a single cluster at 1.3), MiniMax-M2.5 and Qwen3.5-122B use
1.3. Override on the CLI if needed; the value is forwarded and the last
`--da_mga_temperature` wins.

---

## 2. Detection-only

Generate clarification questions, route them through a frozen encoder
onto GT terms. No Postgres needed.

Methods (paper → CLI flag):

- **Si-1** = `--method direct` — Self-Introspection, single call at T=0.
- **Si-10** = `--method direct_multi` — 10 Si-1 calls at T=0.7, dedup.
- **Mcs-1** = `--method mga` — Multi-Candidate SQL, single-pass: one call
  producing up to 10 SQL interpretations (high T), AST-clustered.
- **Mcs-10** = `--method se_ast` — 10 independent SQL samples at T=0.7,
  AST-clustered (θ=0.70 by default).

Expected wall-clock, lite-300 (300 samples, 8 threads). Cost is dominated by
question GENERATION, so it scales with each method's call count (the encoder
step is the cheap part):

| Method | generation calls / sample      | full run (generate + encode) |
|--------|--------------------------------|------------------------------|
| Si-1  (direct)       | 1                        | ~20-40 min |
| Mcs-1 (mga)          | 2 (1 multi-gen + 1 analysis) | ~1-2 h |
| Si-10 (direct_multi) | 10 + 1 dedup             | ~3-5 h |
| Mcs-10 (se_ast)      | 10 SQL samples + 1 analysis | ~8-12 h (longest) |

(plus ~N encoder calls/sample, N = #questions). Anchored to GLM-4.5-Air with
reasoning on a 2-GPU vLLM; `--no_thinking` and faster/smaller endpoints cut
these substantially, and time scales inversely with endpoint concurrency.

Those are **from-scratch** (generate + encode) times. The minute-scale path is
**replay**: once a run's clarification questions already exist,
`scripts/replay_encoder.sh` re-judges them with **encoder-only** calls (no
generation) — roughly 10x cheaper for the 11-call methods (§ 2b). The post-hoc
Si-1 ∪ Mcs-1 union (§ 2a) is free (no LLM calls).

**Run:**

```bash
./scripts/run_detection.sh --method direct       --model qwen --output results/det_qwen_direct.json       # Si-1
./scripts/run_detection.sh --method direct_multi --model qwen --output results/det_qwen_direct_multi.json # Si-10
./scripts/run_detection.sh --method mga          --model qwen --output results/det_qwen_mga.json          # Mcs-1
./scripts/run_detection.sh --method se_ast       --model qwen --output results/det_qwen_se_ast.json      # Mcs-10
```

The wrapper applies the paper's cross-model encoder routing automatically:
GLM/MiniMax detection → Qwen encoder; Qwen detection → MiniMax encoder.
This is "no self-judging" — the encoder is a different model than the one
generating the clarifications. Both endpoints need to be reachable.

If you do not have all three models deployed, use `--encoder-model`:

```bash
# A) ONE vLLM endpoint, self-encoded (cheapest setup; deviates from paper):
./scripts/run_detection.sh --method direct --model qwen \
  --encoder-model self --output results/qwen_self.json

# B) A different cross-encoder pairing (e.g. MM detection → Qwen encoder,
#    overriding the default which would have been Qwen anyway here):
./scripts/run_detection.sh --method direct --model mm \
  --encoder-model qwen --output results/mm_via_qwen.json

# C) Bypass the wrapper entirely for full control:
python -m bird_interact.detection_only.pipeline \
  --method direct --base_url "$YOUR_DET_URL" --model_id your-det-model \
  --encoder_base_url "$YOUR_ENC_URL" --encoder_model_id your-enc-model \
  --data_path data/bird-interact-lite/bird_interact_data.jsonl \
  --data_dir  data/bird-interact-lite --output results/custom.json
```

`--encoder-model self` is the cheapest path to a working pipeline: it
points the encoder at the same endpoint and model as the detection LLM.
Paper-canonical Tables 5–7 require cross-encoding (default behaviour); the
self-encoded variant is for getting the package running end-to-end when
you only have one model deployed.

**Bring your own model (through the wrapper).** `--model` still selects the
knob profile (MGA temperature/θ, `no_thinking`) and the default cross-encoder,
but you can repoint the endpoints at any served OpenAI-compatible model:
`--base-url URL --model-id ID [--api-key KEY]` for the detection LLM, and
`--encoder-base-url/--encoder-model-id/--encoder-api-key` for the encoder (or
`--encoder-model self` to judge on the same single endpoint). This is option C
through the friendly wrapper, so you keep the per-model knobs:

```bash
./scripts/run_detection.sh --method direct --model qwen \
  --base-url "$YOUR_URL" --model-id my-model \
  --encoder-model self --output results/byo.json
```

Method knobs:

| Paper name | CLI `--method` | Default temperature | AST θ | Notes |
|---|---|:---:|:---:|---|
| Si-1 | `direct` | 0 | — | Single deterministic call. |
| Si-10 | `direct_multi` | 0.7 | — | N independent Si-1 calls; default N=10. Override with `--direct_multi_num_samples`. |
| Mcs-1 (GLM) | `mga` | 1.0 | 0.60 | GLM collapses to one cluster at 1.3. |
| Mcs-1 (MM, Qwen) | `mga` | 1.3 | 0.50 | Higher T + looser θ to keep diversity. |
| Mcs-10 (all) | `se_ast` | 0.7 | 0.70 | 10 SQL samples, AST jaccard θ=0.70. |

Pass `--no-ast` for the no-AST ablation rows.

**Score:**

```bash
python -m bird_interact.detection_only.aggregate \
  --input results/det_qwen_direct.json:direct \
  --input results/det_qwen_mga.json:mga \
  --input results/det_qwen_se_ast.json:se_ast \
  --markdown results/det_summary.md
```

### 2a. Per-term Si-1 ∪ Mcs-1 union (paper headline)

The paper's recommended-for-deployment row is the per-term union of an
Si-1 run and an Mcs-1 run. The aggregator computes it post-hoc — no
extra LLM calls:

```bash
python -m bird_interact.detection_only.aggregate \
  --input results/det_qwen_direct.json:direct \
  --input results/det_qwen_mga.json:mga \
  --union direct+mga:union \
  --markdown results/det_union.md
```

A GT term is detected by the union iff *either* input detected it.

### 2b. Encoder kind (legacy vs multi-label)

The pipeline defaults to the standard single-label BIRD-Interact user-sim
encoder. To measure the multi-label-encoder effect (one question crediting
up to 3 GT terms via `labeled(primary, also=[...])`; see
`src/bird_interact/detection_only/custom_encoder.py`), pass
`--encoder_kind multi_label`.

To compare both on the **same** detection output without re-running the
detection LLM:

```bash
# 1. Run detection once (legacy encoder, default)
./scripts/run_detection.sh --method direct --model qwen \
  --output results/det_qwen_direct.json

# 2. Replay with the multi-label encoder
./scripts/replay_encoder.sh \
  --input  results/det_qwen_direct.json \
  --output results/det_qwen_direct_multi.json \
  --model  qwen \
  --encoder-kind multi_label

# 3. Side-by-side
python -m bird_interact.detection_only.aggregate \
  --input results/det_qwen_direct.json:legacy \
  --input results/det_qwen_direct_multi.json:multi_label \
  --markdown results/encoder_ablation.md
```

The replay reuses each record's existing `extracted_detections`, so for
an SE+AST run that took 11 LLM calls per sample, replaying spends ~1 call
per question — roughly a 10× cost reduction.

### 2c. Customizing prompts (swap in your own)

All four detection methods share ONE analysis/introspection system prompt.
You can replace it without editing source:

```bash
./scripts/run_detection.sh --method direct --model qwen \
  --output results/my_prompt.json \
  --analysis_prompt_file path/to/my_prompt.txt
```

(The flag passes straight through to `python -m bird_interact.detection_only.pipeline
--analysis_prompt_file ...`.) The file's contents become the system prompt for
`direct`, `direct_multi`, `mga`, and `se_ast` alike; the default is the built-in
`NEUTRAL_ANALYSIS_PROMPT`.

Where each prompt lives, if you'd rather edit in place:

| Prompt | Used by | Location |
|---|---|---|
| Shared neutral analysis/introspection | all 4 detection methods (system prompt) | `src/bird_interact/detection_only/prompts.py` :: `NEUTRAL_ANALYSIS_PROMPT` (or `--analysis_prompt_file`) |
| Mcs-1 SQL-generation | `mga` (generation call) | `…/detection_only/prompts.py` :: `mga_gen_system()` |
| Mcs-10 SQL-sampling | `se_ast` (sampling call) | `…/detection_only/prompts.py` :: `SE_SAMPLING_SYSTEM_PROMPT` |
| Agentic Si-1 detection | `run.sh --method direct/union` | `…/detection/run_direct_agentic.py` :: `DIRECT_SYSTEM_PROMPT` (or `--da_system_prompt_file PATH`) |
| Agentic Mcs-1 analysis | `run.sh --method union` | `…/detection/run_direct_agentic.py` :: `MGA_ANALYZE_SYSTEM_PROMPT` (or `--da_analysis_prompt_file PATH`) |

The agentic file-overrides are passed through `run.sh`, e.g.
`./scripts/run.sh --method union --model qwen --output-dir out \
  --da_system_prompt_file path/to/my_prompt.txt`.

---

## 3. AMBROSIA cross-benchmark

Runs four arms against AMBROSIA's per-sample SQLite databases. No
Postgres, no user simulator. The `ambrosia` package is a top-level package
independent of `bird_interact`; it has its own AMBROSIA-dataset loader and
SQLite evaluator.

Method mapping (paper Table tab:ambrosia-main → CLI flag):

| Paper name | CLI `--method` | Notes |
|---|---|---|
| Baseline | `baseline` | AMBROSIA-native single call: "write SQL for every plausible interpretation". |
| Si-1 | `direct` | Two-stage Self-Introspection: enumerate paraphrases, one SQL per paraphrase. |
| Union (Si-1 ∪ Mcs-10) | `union_mcs` | Si-1 ∪ 10 independent SQL samples at T=0.7. |
| **Union (Si-1 ∪ Mcs-1)** | `union_spmi` | **Paper headline.** Si-1 ∪ single-pass Mcs-1 call (up to 10 interpretations from one call). |

Expected wall-clock on the n=1,149 ambiguous samples (of the 3,819-sample test
split): baseline ≈20 min, Si-1 ≈1.5 h, Union (Si-1 ∪ Mcs-10) ≈2.9 h, Union
(Si-1 ∪ Mcs-1) ≈1.9 h. (Estimated by scaling measured 300-sample timings by
~3.8x; AMBROSIA is per-sample parallel, so wall-clock scales ~linearly with
sample count and inversely with endpoint concurrency.)

**Run:**

```bash
export AMBROSIA_DIR=data/ambrosia    # or pass --ambrosia-dir
./scripts/run_ambrosia.sh --method baseline   --model qwen --output-dir results/amb_qwen_baseline
./scripts/run_ambrosia.sh --method direct     --model qwen --output-dir results/amb_qwen_direct
./scripts/run_ambrosia.sh --method union_mcs  --model qwen --output-dir results/amb_qwen_union_mcs
./scripts/run_ambrosia.sh --method union_spmi --model qwen --output-dir results/amb_qwen_union_spmi
```

Each run writes `baseline.json` / `stage1.json` / `stage2.json` /
`mcs.json` / `spmi.json` / `union.json` (whichever are applicable) plus
the final `eval.json` summary with the `{config, summary, per_type,
results}` schema.

Method knobs are paper-canonical and set by the wrapper from `--model`. The
"SPMI temperature" applies to the Mcs-1 channel (single-pass multi-interpretation):

| Model | Mcs-1 temperature | `--no_thinking` |
|---|:---:|:---:|
| `glm` | 1.0 | yes |
| `mm` | 1.3 | no (uses `<think>` via server-side parser) |
| `qwen` | 1.3 | yes |

**Evaluator protocol:** the default `--equivalence_threshold 1.01` keeps
the AST-similarity fallback unreachable, matching the AMBROSIA-paper
execution-equivalence protocol. The standalone `ambrosia` package does
not ship an AST scorer (paper-canonical numbers don't use it); if you
want the AST-augmented numbers as a recall upper bound, drop in your own
`structural_similarity` implementation (sqlglot-based) and pass
`--equivalence_threshold 0.70`.

---

## Kubernetes

Manifests under `k8s/` are what we used internally. `vllm_server_*.yaml`
launches one model; `k8s/examples/eval_job_*.yaml` shows one cluster job
per pipeline cell. The eval jobs assume the lite-300 / AMBROSIA datasets
are mounted at `/app/data/bird-interact-lite/` and `/app/data/ambrosia/`
respectively — either pre-populate a PVC and reference it as in the
example, or bake the data into the image. The image itself is your call;
the Dockerfile under `docker/` is what we used.
