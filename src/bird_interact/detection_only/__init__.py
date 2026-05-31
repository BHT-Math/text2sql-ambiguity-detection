"""Standalone detection-only pipeline.

This subpackage runs the §5 detection benchmark (Tables 5–7 of the paper) on
BIRD-Interact-Lite. It is independent of the a-Interact agent loop and does
not need Postgres: every method takes (question + schema + KB) → produces
clarification questions, which a frozen encoder routes to GT ambiguity terms.

Three methods (paper-canonical):
  - direct   : 1 LLM call at T=0
  - mga      : 1 high-T multi-interpretation call + 1 analysis call (+ optional AST clustering)
  - se_ast   : 10 T=0.7 samples + AST clustering + 1 diff-aware analysis call

The full a-Interact pipeline under bird_interact.agent + bird_interact.detection
is untouched by this module.
"""
