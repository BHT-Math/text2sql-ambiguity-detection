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


# Single-SQL sampling prompt used by SE+AST. Matches the "realistic" prompt
# (entropy_detector._build_realistic_prompt) that the paper's SE+AST runs used:
# the user message carries the agent's view of the database — schema, column
# meanings, and the KB as JSON — followed by the question.
SE_SAMPLING_SYSTEM_PROMPT = (
    "You are a helpful PostgreSQL agent that interacts with a user and a database "
    "to solve the user's question. Given the database context below, generate a single "
    "PostgreSQL query that answers the question. Output ONLY the SQL query.\n\n"
    "# PostgreSQL Syntax Rules (MUST FOLLOW)\n"
    "Your SQL MUST be valid PostgreSQL. Common mistakes to avoid:\n"
    "- Use GREATEST(a, b, c) for multi-value maximum, NOT MAX(a, b, c). MAX() takes a single column.\n"
    "- Use LEAST(a, b, c) for multi-value minimum, NOT MIN(a, b, c).\n"
    "- For float division, cast with ::NUMERIC or divide by a decimal (x / 100.0 NOT x / 100). "
    "Integer division truncates silently in PostgreSQL.\n"
    "- Use standard double quotes for column aliases: \"column name\" — do NOT escape with backslashes.\n"
    "- Use PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col) for median, NOT MEDIAN().\n"
    "- Do NOT include literal \\n in SQL strings.\n"
    "- Return ONLY the columns the user asked for. Extra columns cause test failures."
)
