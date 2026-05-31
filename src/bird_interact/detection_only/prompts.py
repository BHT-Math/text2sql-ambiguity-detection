"""System prompts for the detection pipeline.

Paper-canonical wording, ported from the research repo so a re-run on the
released code matches what produced the published numbers (Tables 5-7).

All four detection methods (Direct, Direct-multi, MGA, SE+AST) use the same
neutral analysis instruction: the system prompt ``NEUTRAL_ANALYSIS_PROMPT``
plus the closing user line "List every ambiguity you can identify." The prompt
is deliberately untilted toward question-level vs. SQL-level ambiguities; an
intent-biased variant ("ask about USER INTENT, not SQL syntax; focus on domain
concepts") was tested in an internal control and rejected because it
suppressed implementation-level detections.

  - NEUTRAL_ANALYSIS_PROMPT  : shared neutral analysis prompt. System prompt
                               for Direct / Direct-multi (introspection, no SQL
                               shown) and for the MGA / SE+AST analysis step
                               (after SQL sampling + AST clustering).
  - mga_gen_system(n)        : MGA generation prompt (n SQL interpretations in
                               one call); neutral, no dimensional taxonomy.
  - SE_SAMPLING_SYSTEM_PROMPT: SE+AST single-SQL sampling prompt.
"""


NEUTRAL_ANALYSIS_PROMPT = (
    "You are analyzing a database question and several SQL interpretations "
    "that were generated for it.\n\n"
    "Identify every ambiguity in the question — anything that could explain "
    "why these SQL queries are not identical, or anything in the question "
    "itself that is unclear.\n\n"
    "For each ambiguity, write one short clarification question that would "
    "resolve it.\n\n"
    "Output a numbered list (1. 2. 3. etc.). If you find no ambiguities, "
    "say 'No ambiguities detected.'"
)


def mga_gen_system(num_interpretations: int) -> str:
    """Generation prompt for MGA / Direct-Multi (neutral variant)."""
    return (
        "You are a PostgreSQL expert. A user asked a database question that "
        "may be ambiguous — it could be interpreted in multiple valid ways.\n\n"
        f"Generate exactly {num_interpretations} DIFFERENT SQL interpretations "
        "of the question. Each interpretation should reflect a genuinely "
        "different reading of the question.\n\n"
        "Rules:\n"
        "- Each SQL must be valid PostgreSQL.\n"
        "- Label each as [Interpretation 1], [Interpretation 2], etc.\n"
        "- Do NOT generate trivial variations (alias changes, formatting). "
        "Each must represent a meaningfully different interpretation.\n"
        f"- If the question is completely unambiguous, generate {num_interpretations} identical SQL.\n"
        "- Output ONLY the labeled SQL queries, no explanations."
    )


# Single-SQL sampling prompt used by SE+AST. Matches the "minimal" prompt
# from entropy_detector._build_minimal_prompt — small, KB-as-markdown.
SE_SAMPLING_SYSTEM_PROMPT = (
    "You are a PostgreSQL expert. Given a database schema, external knowledge, "
    "and a user question, generate a single SQL query that answers the question. "
    "Output ONLY the SQL query — no explanation, no markdown fences, no comments."
)
