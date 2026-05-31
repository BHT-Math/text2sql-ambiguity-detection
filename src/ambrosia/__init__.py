r"""AMBROSIA cross-benchmark validation pipeline.

Self-contained refactor of the AMBROSIA arms reported in the paper appendix
(`tab:ambrosia-cross-model`). Five channels feed one execution-equivalence
evaluator:

- ``baseline`` — AMBROSIA's native single-call prompt at T=0 (one call asks for
  all interpretations).
- ``direct`` (Sᵢ) — two-stage Self-Introspection: NL → enumerated paraphrases
  → one SQL per paraphrase at T=0.
- ``mcs`` — 10 independent single-SQL samples at T=0.7 ("Sampling-10",
  diversity channel; called ``Mcs`` in the paper).
- ``spmi`` — one call asks for up to 10 interpretations at T=1.3 (Qwen) or
  T=1.0 (GLM); this is the literal BIRD ``\MultiGen`` recipe.
- ``union`` — pool a Direct channel with a sampling channel (``mcs`` or
  ``spmi``) and deduplicate by per-sample execution result. The paper's
  ``Direct ∪ \MultiGen`` arm is ``--method union --sampling spmi``.

Evaluation is the official AMBROSIA execution-equivalence protocol: each
predicted SQL is executed against the per-sample SQLite DB and matched to
ground truth via multiset row equality (order-aware when ``ORDER BY`` is
present). AST similarity fallback is OFF for paper-reported numbers; set
``--equivalence_threshold 1.01`` to ensure the fallback is unreachable.
"""

__all__ = []
