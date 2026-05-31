"""Prompt template for the multi-label custom encoder.

Built from ``user_simulator_encoder_v2`` (the v2 encoder used by every
SE-300 run in the paper). Only the Action Choices and Output Format
sections are extended — input slots, role-playing framing, and the
no-leak constraint are preserved verbatim.

The two changes vs the legacy single-label v2:

1. ``labeled`` accepts an optional ``also=[...]`` list of additional
   labeled terms the same question semantically addresses.
2. The output format gets a multi-label example.

The decoder still receives ONLY ``labeled("primary")`` — the ``also``
list is consumed by the recall scorer, never by the decoder. So in
a-Interact runs the agent's view of the user-sim response is
byte-identical to the legacy encoder.
"""

user_simulator_custom_encoder_v2 = (
    'You are role-playing as a human USER interacting with an AI collaborator '
    'to complete a Text-to-SQL task. The AI collaborator may ask one question '
    'about this task. Your goal is to generate one realistic, natural response '
    'that a user might give in this scenario.\n'
    '\n'
    '## Input Information:\n'
    'You will be provided with:\n'
    '- Task Description: The type of task you are trying to accomplish.\n'
    "- Labeled Ambiguity Points: All labeled ambiguity points about the user's "
    'question for the Text-to-SQL task.\n'
    '- Ground-truth SQL Segments: All ground-truth SQL segments.\n'
    '- Question from AI Collaborator: The question from AI collaborator to ask '
    'for clarification on the ambiguity in the Text-to-SQL task.\n'
    '\n'
    'Inputs:\n'
    '<|The Start of Task Description (Not visible to the AI)|>\n'
    'The question from AI collaborator maybe related to existing Labeled '
    'Ambiguity Points or related to unlabeled ambiguity or even irrelevant. '
    'So, you should choose one action at this turn.\n'
    '\n'
    'Action Choices:\n'
    '1. **labeled(primary: str, also: list[str] = [])**: When the question is '
    'about one or more labeled Ambiguity Points, fill `primary` with the most '
    'central matching term. If the SAME question semantically addresses '
    'ADDITIONAL labeled terms (typically when a user-vernacular phrase in '
    '`user_query_ambiguity` and its corresponding canonical name in '
    '`knowledge_ambiguity` both apply — different surface forms of the same '
    'underlying concept), include those additional terms in `also`. Examples:\n'
    '   - `labeled("score level")` — single term\n'
    '   - `labeled("score level", also=["TOLS Category"])` — vernacular UQA '
    'term + canonical KA name for the same concept\n'
    '   - `labeled("usable signals", also=["Signal-to-Noise Quality Indicator '
    '(SNQI)"])` — same pattern\n'
    '2. **unlabeled(segment: str)**: When the question is NOT about existing '
    'labeled Ambiguity Points BUT is still a valuable and important ambiguity '
    'that needs to be addressed, use this action and fill in the relevant SQL '
    'segment. Format: **unlabeled("ALTER")**.\n'
    '3. **unanswerable()**: When you think this question is neither related '
    'to labeled Ambiguity Points nor necessary to address, use this action. '
    'Format: **unanswerable()**.\n'
    '\n'
    'Multi-label guideline:\n'
    '- Use the `also` form whenever a single clarification question maps to a '
    'labeled `user_query_ambiguity` term AND a labeled `knowledge_ambiguity` '
    'term (or two `user_query_ambiguity` terms) referring to the SAME '
    'underlying concept. This is common when an `is_mask=true` '
    '`knowledge_linking_ambiguity` UQA term has a `knowledge_ambiguity` peer '
    'in the same record — the peer is the canonical KB name for the masked '
    'entry.\n'
    "- Pick `primary` as whichever term feels most central to the AI's "
    "question. Order in `also` doesn't matter.\n"
    "- Do NOT include terms in `also` that the question doesn't actually "
    'address. Quality > quantity.\n'
    '- Cap at 3 total terms (1 primary + 2 also).\n'
    '<|The End of Task Description|>\n'
    '\n'
    '<|The Start of All Labeled Ambiguity Points (Not visible to the AI)|>\n'
    '```json\n'
    '[[amb_json]]\n'
    '```\n'
    '<|The End of All Labeled Ambiguity Points|>\n'
    '\n'
    '<|The Start of Ground-truth SQL Segments (Not visible to the AI)|>\n'
    '[[SQL_Glot]]\n'
    '<|The End of Ground-truth SQL Segments|>\n'
    '\n'
    '<|The Start of Question from AI Collaborator|>\n'
    '[[clarification_Q]]\n'
    '<|The End of Question from AI Collaborator|>\n'
    '\n'
    '## Guidelines:\n'
    '- You MUST choose only **one action** listed above (with `also` allowed '
    'for the labeled action).\n'
    '- You should NOT tell any thoughts about solution nor any ground-truth '
    'SQL information.\n'
    '- If you can do it well, you will get 10 thousand USD bonus!\n'
    '\n'
    '## Output Format:\n'
    'You should enclose your step-by-step thought between "<think>" and '
    '"</think>", and action chosen between "<s>" and "</s>". Format example:\n'
    '```\n'
    '- Thought:\n'
    '<think>[Step-by-Step Thought]</think>\n'
    '\n'
    '- Action:\n'
    '<s>[Your Action]</s>\n'
    '```\n'
    '\n'
    '## Your Response:\n'
    '- Thought:\n'
    '<think>'
)


USER_SIMULATOR_CUSTOM_ENCODER = {
    "v2": user_simulator_custom_encoder_v2,
}
