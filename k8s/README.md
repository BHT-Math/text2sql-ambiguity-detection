# BIRD-Interact on Kubernetes

This directory contains the Kubernetes manifests we used to produce the 3×3
NR grid. They are templated to strip cluster-specific bits (private registry,
PVCs, OIDC, namespace) so a reviewer with their own GPU cluster can adapt
quickly.

## File layout

```
k8s/
├── README.md                            ← you are here
├── vllm_server_glm45_air.yaml           ← vLLM serving GLM-4.5-Air (FP8, 2×B200)
├── vllm_server_minimax_m25.yaml         ← vLLM serving MiniMax-M2.5 (FP8, 2×B200)
├── vllm_server_qwen35_122b.yaml         ← vLLM serving Qwen3.5-122B (2×H200)
├── qwen35_no_think_chat_template.jinja  ← chat template ConfigMap source for Qwen
└── examples/
    ├── eval_job_glm_baseline.yaml       ← 9 cells: model × method
    ├── eval_job_glm_direct.yaml
    ├── eval_job_glm_union.yaml
    ├── eval_job_mm_*.yaml               ← same pattern
    ├── eval_job_qwen_*.yaml
    └── …
```

## Required substitutions

Before `kubectl apply`, replace the following placeholders:

| Placeholder | Where | What |
|---|---|---|
| `<YOUR_BIRD_INTERACT_EVAL_IMAGE>` | `examples/eval_job_*.yaml` | Container image built from `../docker/Dockerfile`. Build and push to your registry, then paste the full image ref here. |
| `gemini-creds` Secret | (referenced by all eval jobs) | A Kubernetes Secret holding the keys `api_key` (free-tier Gemini) and `new_gemini` (Vertex Express). At least one must be valid for the user-simulator encoder to work. See `REPRODUCING.md` for setup. |

The vLLM server manifests use **public** Docker Hub images
(`vllm/vllm-openai:latest`) and `emptyDir` for the HF model cache, so they
apply unmodified. To accelerate cold start, swap `emptyDir` for a PVC.

## Bring-up order

```bash
# 1. (Qwen only, REQUIRED before applying its vLLM Pod) Create the chat-template
#    ConfigMap from the .jinja file in this directory:
kubectl create configmap qwen35-chat-template \
  --from-file=qwen35_no_think_chat_template.jinja=./qwen35_no_think_chat_template.jinja

# 2. Stand up one or more vLLM servers
kubectl apply -f vllm_server_qwen35_122b.yaml
kubectl apply -f vllm_server_glm45_air.yaml
kubectl apply -f vllm_server_minimax_m25.yaml

# 3. Wait until ready (HF model download dominates cold start — minutes; the
#    actual model-load step is ≈60-120 s after weights are cached)
kubectl logs vllm-qwen35-122b-server -f
# look for: "Application startup complete"

# 4. Apply eval jobs — one at a time per shared vLLM server
kubectl apply -f examples/eval_job_qwen_baseline.yaml   # ReAct baseline
# wait for completion, then:
kubectl apply -f examples/eval_job_qwen_direct.yaml     # Si-1 only
kubectl apply -f examples/eval_job_qwen_union.yaml      # Si-1 ∪ Mcs-1
```

Per-method wall-clock on a 2-GPU vLLM matching the manifests here: ≈1-1.5 h
each for the lite-300 agentic arms; AMBROSIA is faster (≈5-45 min depending
on arm; see `eval_job_ambrosia_qwen.yaml`).

The eval-job init container polls the vLLM service with `nc -zv` and won't
start until the endpoint is up. The shipped jobs hardcode `--num_threads 8`,
which sits at or below each model's vLLM `--max-num-seqs` (GLM=16, MM=8,
Qwen=8). To run multiple jobs against the same vLLM server in parallel, scale
`--num_threads` down by N so total concurrency stays ≤ `--max-num-seqs`;
otherwise excess requests queue and risk the 120 s OpenAI-client timeout.

## Critical hyperparameters baked into each YAML

These are NOT user-tunable; they're the values that produced the 3×3
NR grid reported in the paper. Changing any of them will shift the numbers.

| Flag | Value | Why |
|---|---|---|
| `--max_turns` | 60 | Paper-canonical; 20 causes 66 % budget exhaustion |
| `--user_patience_budget` | 6 | Paper-canonical patience budget |
| `--max_tokens` | 24576 | Must be explicit; runner default silently caps at 2048 |
| `--num_threads` | 8 | Must equal vLLM `--max-num-seqs`; over-subscription triggers 120 s OpenAI client timeout |
| `--user_sim_prompt_version` | v2 | More robust user simulation |
| `--user_encoder_model` | gemini-3-1-flash-lite | Cross-model encoder (not self-encoded) |
| `--da_no_ast` (Union only) | true | AST clustering disabled to match the runs that produced our table |
| `--da_clean_slate` + `--da_hint_only` (Direct/Union) | true | Hint-only one-shot injection; K-cap disabled in this mode |
| `--da_mga_temperature` (Union only) | 1.0 (GLM) / 1.3 (MM, Qwen) | GLM collapses to one cluster at 1.3 |

## vLLM-side critical flags

Repeated here because they're easy to miss:

- **MiniMax-M2.5** requires `--reasoning-parser minimax_m2`. Without it the
  `<think>` block consumes the entire generation budget and the model
  never emits `<action>`, silently degrading NR by several points.
- **GLM-4.5-Air** requires `--reasoning-parser glm45`. Even with the flag,
  a small fraction of responses still leak `<think>` into the response
  field due to parser non-determinism.
- **Qwen3.5-122B** is NOT a reasoning model; no parser flag. Use the
  `qwen35_no_think_chat_template.jinja` ConfigMap to default-disable
  thinking.

## Memory footnote

The eval jobs request 24 Gi and limit at 96 Gi. The cluster default of 16 Gi
OOMs the a-Interact pipeline around turn 15-16 (silent `exit 1` with no
Python traceback — Python catches the MemoryError, runs cleanup, then exits).
The pipeline holds all 300 `SampleStatus` histories plus per-turn raw
responses in memory, and accumulated context grows fast at low patience.
