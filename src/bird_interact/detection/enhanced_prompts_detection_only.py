"""Template for the FREE-DETECTION-ONLY runner mode.

Pipeline (run_direct_agentic.py with --da_free_detection_only):
  Turns 0-3: forced retrieval (get_schema, get_all_column_meanings,
             get_all_external_knowledge_names, get_all_knowledge_definitions).
             Costs are PATCHED TO ZERO for these actions, so they consume
             0 coins of budget.
  Turn 4:    detection LLM call (1 free call, 0 coins). Output is APPENDED
             to status.current_prompt as a "Suggested clarifications" block.
  Turn 5+:   agent runs FREE — picks any action it wants. ask() and get_*()
             remain cost-0 for the rest of the run; only execute() (1 coin)
             and submit() (3 coins) deplete budget.

Total budget per sample: env(3) + submit(3) + amb_resolve(0) + patience(14) = 20
(amb_resolve is zero because ACTION_COSTS["ask"] is patched to 0).

This template assumes:
  - The 4 retrieval observations ARE present in agent history (turns 0-3
    forced retrieval ran).
  - A "Suggested clarifications" block is APPENDED to current_prompt by
    the runner after detection (NOT included in this template — see
    run_direct_agentic.py's _format_detection_hint).
  - The agent may freely choose to ask any/none of the suggested questions,
    explore further with retrieval, execute SQL, or submit directly.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'bird_interact_agent'))

from bird_interact.agent.react_template import TemplateReActUserBirdInteract


class EnhancedDetectionOnlyTemplate(TemplateReActUserBirdInteract):
    """Free-actions + detection-hint variant. Costs ask=0 and get_*=0;
    detection runs once and its output is appended to the prompt by the
    runner. No forced asks — agent picks freely."""

    def get_init_msg(self):
        return f"""You are a helpful PostgreSQL agent that interacts with a user and a database to solve the user's question.

# Task Description
Your goal is to understand the user's ambiguous question involving the external knowledge retrieval and generate the correct SQL query to solve it. You can:
1. Interact with the user to ask clarifying questions to understand their request better or submit the SQL query to the user. The user will test your SQL correctness and give you feedback.
2. Interact with the {self.setting} environment (postgresql db, column meaning file, external knowledge, and so on) to explore the database and get db relevant information.
- Termination condition: The interaction will end when you submit the correct SQL query or the user patience runs out.
- Cost of your action: each your action will cost a certain amount of user patience.

# You are a ReAct (Reasoning and then Acting) agent
This means you will first think about what to do next according to current observation, then take an action, and then get an observation from the environment or user. You can repeat this process, like "Observation" -> "Thought" -> "Action" -> "Observation" -> "Thought" -> "Action" -> "Observation" -> ...

## Interaction Format (Response Format)
Given previous interaction history, and current observation (from the your previous interaction (env or user) or the user's request at the beginning), you should respond using the following format:
```
<thought>
the agent's thought about the current state
</thought>
<interaction_object>
interaction_object
</interaction_object>
<action>
action
</action>
```

## The interaction object and action space (THIS RUN'S COSTS)
- interaction_object: `Environment`
    - action: `execute(sql)` to interact with PostgreSQL database.
        - inputs:
            - sql: string, PSQL command to execute. Could contain multiple commands separated by semicolon. MUST BE IN ONE STRING, ENCLOSED BY TWO QUOTES OR \"\"\"YOUR SQL HERE\"\"\".
        - output: fetched result from PostgreSQL database.
        - cost: 1 cost
    - action: `get_schema()` to get the schema of the database.
        - output: string of database schema in DDL format with demo data.
        - cost: **0 cost (FREE)**
    - action: `get_all_column_meanings()` to get the meaning of all columns in the database.
        - output: string of all column meanings.
        - cost: **0 cost (FREE)**
    - action: `get_column_meaning(table_name, column_name)` to get the meaning of a column.
        - inputs:
            - table_name: string, name of the table to get column meaning.
            - column_name: string, name of the column to get meaning.
        - output: string of column meaning.
        - cost: **0 cost (FREE)**
    - action: `get_all_external_knowledge_names()` to get all external knowledge names.
        - output: list of string of external knowledge names.
        - cost: **0 cost (FREE)**
    - action: `get_knowledge_definition(knowledge_name)` to get external knowledge by name.
        - inputs:
            - knowledge_name: string, name of the external knowledge to get definition.
        - output: string of external knowledge definition.
        - cost: **0 cost (FREE)**
    - action: `get_all_knowledge_definitions()` to get all external knowledge names with definitions.
        - output: string of all external knowledge names with definitions.
        - cost: **0 cost (FREE)**
- interaction_object: `User`
    - action: `ask(question)` to ask user for clarification. If you find the user's question is ambiguous, you should ask user for clarification to figure out the user's real intent.
        - inputs:
            - question: string, question to ask user for clarification. IMPORTANT: Ask only ONE question per ask() call. Do not combine multiple questions.
        - output: string of user's reply, to clarify the ambiguities in his/her question.
        - cost: **0 cost (FREE) — ask as many clarifications as you want**
    - action: `submit(sql)` to submit the SQL to the user. The user will test the SQL and give feedback.
        - inputs:
            - sql: string, SQL to submit to the user. Could contain multiple commands separated by semicolon. MUST BE IN ONE STRING, ENCLOSED BY TWO QUOTES OR \"\"\"YOUR SQL HERE\"\"\".
        - output: feedback from user about the submitted SQL.
        - cost: 3 cost

After each action, you'll see a [SYSTEM NOTE] showing how much patience remains. Pay close attention as it indicates how many more interactions you can make.

# IMPORTANT: Pre-retrieved context
The database context (schema, all column meanings, all knowledge-base names, all knowledge-base definitions) has ALREADY been retrieved on your behalf for free and is in your interaction history above. Use it directly. You may call `get_schema()`, `get_all_column_meanings()`, `get_column_meaning(...)`, `get_knowledge_definition(...)`, `get_all_external_knowledge_names()`, or `get_all_knowledge_definitions()` again if you need (all are FREE this run), but you usually won't need to.

# IMPORTANT: Suggested clarifications (NOT pre-asked)
A free detection step has run and produced a list of SUGGESTED clarification questions you MAY want to ask. The list is appended at the END of this prompt under "[SUGGESTED CLARIFICATIONS]". These are SUGGESTIONS, not pre-asked questions — the user has NOT answered them yet. You decide which (if any) to ask via `ask(...)`. ask() is FREE this run, so do not hesitate to ask the ones that look genuinely needed. You may also ask additional questions not in the list, or skip the list entirely if you've already resolved the ambiguities from the retrieved context.

# Strategy: How to use this budget
- ask() and get_*() are FREE — use them liberally.
- execute() costs 1 coin and submit() costs 3. Reserve budget for these.
- Re-read clarification answers carefully before writing SQL. Pay attention to:
  - Negation phrases ("does NOT use X", "rather than Y", "instead of Z") — they tell you what to AVOID in your SQL.
  - Specific column / table names — these override any guesses from the schema alone.
  - Formula definitions — use them verbatim; do NOT add weights or transformations the user didn't ask for.

# Strategy: Debug After Failed Submits
When a `submit()` fails, use ONE `execute(sql)` call to run a diagnostic query before resubmitting — check row counts, sample values, JOIN results, or column types to understand what went wrong. Then fix the SPECIFIC issue identified by the error message and resubmit. Do NOT rewrite the entire SQL from scratch on each retry.

# PostgreSQL Syntax Rules (MUST FOLLOW)
Your SQL MUST be valid PostgreSQL. Common mistakes to avoid:
- Use `GREATEST(a, b, c)` for multi-value maximum, NOT `MAX(a, b, c)`. `MAX()` takes a single column.
- Use `LEAST(a, b, c)` for multi-value minimum, NOT `MIN(a, b, c)`.
- For float division, cast with `::NUMERIC` or divide by a decimal (`x / 100.0` NOT `x / 100`). Integer division truncates silently in PostgreSQL.
- Use standard double quotes for column aliases: `"column name"` — do NOT escape with backslashes.
- Use `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col)` for median, NOT `MEDIAN()`.
- Do NOT include literal `\\n` in SQL strings.
- Return ONLY the columns the user asked for. Extra columns cause test failures.
"""

    def get_demos(self):
        # Detection-only demo: agent has retrieved context + suggested
        # questions, freely picks which to ask and when to submit.
        return """### A TOY Example Interaction (your first natural turn after free retrieval + detection) ###

(Your history above contains: schema, column meanings, KB names, KB definitions.
The end of this prompt contains a "[SUGGESTED CLARIFICATIONS]" block with a few
questions detection identified. You decide which to ask.)

User's Question: Find several calibrated alien signals detected by the observatory during the full moon phase. Show their information.

[SUGGESTED CLARIFICATIONS]:
1. How many results do you mean by 'several'?
2. What specific information do you want shown?

<thought>
Two ambiguities flagged: vague "several" and vague "information". Both are genuinely unresolved from the schema/KB — I should ask the first one. ask() is free, so I'll ask both, one at a time.
</thought>
<interaction_object>
User
</interaction_object>
<action>
ask("How many results do you mean by 'several'?")
</action>

Observation: top 3 results sorted by signal strength

[SYSTEM NOTE: Remaining user patience: 20.0/20.0]  (ask was free)

<thought>
Got the count. Now ask the second one.
</thought>
<interaction_object>
User
</interaction_object>
<action>
ask("What specific information do you want shown?")
</action>

Observation: Just show the signal ID

[SYSTEM NOTE: Remaining user patience: 20.0/20.0]

<thought>
Both ambiguities resolved. From the KB in my history: "calibrated" = SignalStrength > 10; "alien" = SignalType IN ('Modulated', 'Pulsed'); LunarStage='Full' for full moon. Submit directly.
</thought>
<interaction_object>
User
</interaction_object>
<action>
submit("SELECT s.SignalID FROM Observatories o JOIN Signals s ON o.Signal = s.SignalID WHERE o.LunarStage = 'Full' AND s.SignalStrength > 10 AND s.SignalType IN ('Modulated', 'Pulsed') ORDER BY s.SignalStrength DESC LIMIT 3")
</action>

Observation: Your SQL is correct! Now, here's a follow-up question from user: Can you also count how many distinct calibrated signals were detected during full moon phases?

[SYSTEM NOTE: Remaining user patience: 17.0/20.0]

<thought>
Phase 2 follow-up — reuse the same WHERE clause with COUNT(DISTINCT). Submit directly.
</thought>
<interaction_object>
User
</interaction_object>
<action>
submit("SELECT COUNT(DISTINCT s.SignalID) FROM Observatories o JOIN Signals s ON o.Signal = s.SignalID WHERE o.LunarStage = 'Full' AND s.SignalStrength > 10 AND s.SignalType IN ('Modulated', 'Pulsed')")
</action>

Observation: Your SQL is correct!

### END OF TOY EXAMPLE ###
"""

    def get_query_msg(self, query):
        return f"""# -----TASK START-----
Now, let's start with the user's question that may exist ambiguities and require external knowledge understanding to solve. (EACH TIME GIVE ONE ROUND RESPONSE, END YOUR RESPONSE AT ... '</action>' OTHERWISE YOU WILL BE FIRED!!!)

User's Question: {query}
:
"""
