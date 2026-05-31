"""
Enhanced prompt template for BIRD-Interact a-Interact evaluation.

Subclasses TemplateReActUserBirdInteract with 6 strategies for improved
ambiguity resolution and budget efficiency.
"""

import sys
import os

# Add bird_interact_agent to path so we can import the base class
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'bird_interact_agent'))

from bird_interact.agent.react_template import TemplateReActUserBirdInteract


class EnhancedTemplate(TemplateReActUserBirdInteract):
    """Enhanced prompt template v3 with structured exploration, ambiguity detection,
    and execute()-based debugging after failed submits."""

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

## The interaction object and action space with cost
- interaction_object: `Environment`
    - action: `execute(sql)` to interact with PostgreSQL database.
        - inputs:
            - sql: string, PSQL command to execute. Could contain multiple commands separated by semicolon. MUST BE IN ONE STRING, ENCLOSED BY TWO QUOTES OR \"\"\"YOUR SQL HERE\"\"\".
        - output: fetched result from PostgreSQL database.
        - cost: 1 cost
    - action: `get_schema()` to get the schema of the database.
        - output: string of database schema in DDL format with demo data.
        - cost: 1 cost
    - action: `get_all_column_meanings()` to get the meaning of all columns in the database.
        - output: string of all column meanings.
        - cost: 1 cost
    - action: `get_column_meaning(table_name, column_name)` to get the meaning of a column.
        - inputs:
            - table_name: string, name of the table to get column meaning.
            - column_name: string, name of the column to get meaning.
        - output: string of column meaning.
        - cost: 0.5 cost
    - action: `get_all_external_knowledge_names()` to get all external knowledge names.
        - output: list of string of external knowledge names.
        - cost: 0.5 cost
    - action: `get_knowledge_definition(knowledge_name)` to get external knowledge by name.
        - inputs:
            - knowledge_name: string, name of the external knowledge to get definition.
        - output: string of external knowledge definition.
        - cost: 0.5 cost
    - action: `get_all_knowledge_definitions()` to get all external knowledge names with definitions.
        - output: string of all external knowledge names with definitions.
        - cost: 1 cost
- interaction_object: `User`
    - action: `ask(question)` to ask user for clarification. If you find the user's question is ambiguous, you should ask user for clarification to figure out the user's real intent.
        - inputs:
            - question: string, question to ask user for clarification. IMPORTANT: Ask only ONE question per ask() call. Do not combine multiple questions.
        - output: string of user's reply, to clarify the ambiguities in his/her question.
        - cost: 2 cost
    - action: `submit(sql)` to submit the SQL to the user. The user will test the SQL and give feedback.
        - inputs:
            - sql: string, SQL to submit to the user. Could contain multiple commands separated by semicolon. MUST BE IN ONE STRING, ENCLOSED BY TWO QUOTES OR \"\"\"YOUR SQL HERE\"\"\".
        - output: feedback from user about the submitted SQL.
        - cost: 3 cost

After each action, you'll see a [SYSTEM NOTE] showing how much patience remains (e.g. "[SYSTEM NOTE: Remaining user patience: 7/10]"). Pay close attention to this note as it indicates how many more interactions you can make. If patience runs out, the task ends and you'll need to submit your final answer.

# Strategy: Structured Exploration Protocol
ALWAYS begin with these 4 steps in order to build full context before interacting with the user:
1. `get_schema()` (cost: 1) — understand the database structure
2. `get_all_external_knowledge_names()` (cost: 0.5) — see what domain knowledge is available
3. `get_all_knowledge_definitions()` (cost: 1) — read all knowledge definitions
4. `get_all_column_meanings()` (cost: 1) — get detailed column descriptions including full names, explanations, data types, possible categories, and example values
Total cost: 3.5 for complete context.

# Strategy: Ambiguity Detection
Before asking the user anything, systematically analyze the query:
- List every potentially vague or ambiguous term in the user's question.
- Cross-reference each term against the schema, column meanings, and external knowledge definitions.
- Only ask about terms you CANNOT resolve from schema, column meanings, or knowledge base alone.

# Strategy: Schema-Aware Questions
When asking clarifications, reference the specific tables, columns, or knowledge definitions you found. For example: "The database has columns 'order_date' and 'created_at' — which one corresponds to 'recent' in your question?"

# Strategy: Debug After Failed Submits
When a submit() fails, use `execute(sql)` (cost: 1) to run a diagnostic query before resubmitting. Check row counts, sample values, JOIN results, or column types to understand what went wrong. Then fix the issue and resubmit. This costs 4 coins (execute + submit) but is more effective than blindly resubmitting at 3 coins each.
Do NOT use execute() before your first submit — if you have resolved all ambiguities and are confident, submit directly.

# Strategy: Budget-Aware Planning
The budget is shared across BOTH phases. If you spend everything on Phase 1, you cannot attempt Phase 2 (losing 0.3 points). Budget allocation guide:
- Exploration: 3.5 coins (schema + KB + column meanings)
- Clarification (ask): 2-4 coins (1-2 asks)
- SQL verification (execute): 1-2 coins (only if unsure about data values)
- Submit Phase 1: 3 coins
- Reserve for Phase 2: at least 4-7 coins (execute + submit)
- Total budget is typically ~20-24. Never spend more than 3 coins on execute() per phase.

# Strategy: Knowledge-Gap Analysis
After retrieving knowledge definitions, compare them against every term in the user's query. If a query term seems related to a knowledge concept but the definition doesn't fully resolve the ambiguity, that's likely a hidden knowledge ambiguity — ask the user about it specifically.

# PostgreSQL Syntax Rules (MUST FOLLOW)
Your SQL MUST be valid PostgreSQL. Common mistakes to avoid:
- Use GREATEST(a, b, c) for multi-value maximum, NOT MAX(a, b, c). MAX() takes a single column.
- Use LEAST(a, b, c) for multi-value minimum, NOT MIN(a, b, c).
- For float division, cast with ::NUMERIC or divide by a decimal (x / 100.0 NOT x / 100). Integer division truncates silently in PostgreSQL.
- Use standard double quotes for column aliases: "column name" — do NOT escape with backslashes.
- Use PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col) for median, NOT MEDIAN().
- Do NOT include literal \\n in SQL strings.
- Return ONLY the columns the user asked for. Extra columns cause test failures.
"""

    def get_demos(self):
        return """### A TOY Example Interaction ###

User's Question: Find several calibrated alien signals detected by the observatory during the full moon phase. Show their information.

<thought>
Following the structured exploration protocol, I'll first get the database schema to understand available tables and columns.
</thought>
<interaction_object>
Environment
</interaction_object>
<action>
get_schema()
</action>

Observation: Database schema showing tables: Observatories(ObservID, ObservStation, ObservDate, LunarStage, Signal), Signals(SignalID, SignalType, SignalStrength, Frequency, EncodingMethod)...

[SYSTEM NOTE: Remaining user patience: 19.0/20.0]

<thought>
Schema retrieved. Next step in the structured exploration: get all external knowledge names to see what domain knowledge is available.
</thought>
<interaction_object>
Environment
</interaction_object>
<action>
get_all_external_knowledge_names()
</action>

Observation: ["Alien Signal Classification", "Lunar Phase Effects", "Calibrated Signal"]

[SYSTEM NOTE: Remaining user patience: 18.5/20.0]

<thought>
I see knowledge entries that look relevant. Let me get all definitions at once (cost 1) rather than fetching them one by one (cost 0.5 each x 3 = 1.5).
</thought>
<interaction_object>
Environment
</interaction_object>
<action>
get_all_knowledge_definitions()
</action>

Observation:
- Alien Signal Classification: Signals are classified by type: 'Modulated', 'Pulsed', 'Continuous'. Alien signals are those with SignalType = 'Modulated' or 'Pulsed'.
- Lunar Phase Effects: Observations during certain lunar phases may have noise interference.
- Calibrated Signal: A signal with SignalStrength > 10.

[SYSTEM NOTE: Remaining user patience: 17.5/20.0]

<thought>
Knowledge definitions retrieved. Final exploration step: get all column meanings for detailed column descriptions, full names, and possible categorical values.
</thought>
<interaction_object>
Environment
</interaction_object>
<action>
get_all_column_meanings()
</action>

Observation: Column meanings showing: ObservStation (Full name: 'Observatory Name', CHAR(60)), LunarStage (Full name: 'Lunar Phase', VARCHAR(25), Possible categories: First Quarter, Full, Last Quarter, New), SignalType (Full name: 'Signal Type', VARCHAR(20), Possible categories: Continuous, Modulated, Pulsed), SignalStrength (Full name: 'Signal Strength', DECIMAL(8,2))...

[SYSTEM NOTE: Remaining user patience: 16.5/20.0]

<thought>
Now I have full context. Let me analyze ambiguities in: "Find several calibrated alien signals detected by the observatory during the full moon phase. Show their information."

1. "calibrated" → RESOLVED from KB: SignalStrength > 10
2. "alien signals" → RESOLVED from KB: SignalType IN ('Modulated', 'Pulsed')
3. "full moon phase" → RESOLVED from column meanings: LunarStage = 'Full'
4. "several" → UNRESOLVED: how many?
5. "their information" → UNRESOLVED: which columns?

I need to ask about "several" and "their information". Starting with the most impactful.
</thought>
<interaction_object>
User
</interaction_object>
<action>
ask('How many results do you mean by "several"? For example, top 3, top 5, or top 10?')
</action>

Observation: User's Response: I want the top 3 results sorted by signal strength.

[SYSTEM NOTE: Remaining user patience: 14.5/20.0]

<thought>
"several" resolved: top 3 by signal strength. Now resolving "their information".
</thought>
<interaction_object>
User
</interaction_object>
<action>
ask('What specific information do you want shown — just signal IDs, or details like SignalType, SignalStrength, and Frequency?')
</action>

Observation: User's Response: Just show the signal ID.

[SYSTEM NOTE: Remaining user patience: 12.5/20.0]

<thought>
All ambiguities resolved. I know "calibrated" = SignalStrength > 10, "alien" = SignalType IN ('Modulated', 'Pulsed'), "full moon" = LunarStage = 'Full', "several" = top 3 by signal strength, "information" = just SignalID. I'm confident — submit directly.
</thought>
<interaction_object>
User
</interaction_object>
<action>
submit("SELECT s.SignalID FROM Observatories o JOIN Signals s ON o.Signal = s.SignalID WHERE o.LunarStage = 'Full' AND s.SignalStrength > 10 AND s.SignalType IN ('Modulated', 'Pulsed') ORDER BY s.SignalStrength DESC LIMIT 3")
</action>

Observation: Your SQL is correct! Now, here's a follow-up question from user: Can you also count how many distinct calibrated signals were detected during full moon phases?

[SYSTEM NOTE: Remaining user patience: 9.5/20.0]

<thought>
Phase 1 complete! Phase 2 follow-up is straightforward — reuse the same filters with COUNT(DISTINCT). Budget: 9.5 - 3 = 6.5 remaining. Submit directly since I'm confident.
</thought>
<interaction_object>
User
</interaction_object>
<action>
submit("SELECT COUNT(DISTINCT s.SignalID) FROM Observatories o JOIN Signals s ON o.Signal = s.SignalID WHERE o.LunarStage = 'Full' AND s.SignalStrength > 10 AND s.SignalType IN ('Modulated', 'Pulsed')")
</action>

Observation: Your SQL is correct!

### END OF TOY EXAMPLE INTERACTION ###
"""

    def get_query_msg(self, query):
        return f"""# -----TASK START-----
Now, let's start with the user's question that may exist ambiguities and require external knowledge understanding to solve. (EACH TIME GIVE ONE ROUND RESPONSE, END YOUR RESPONSE AT ... '</action>' OTHERWISE YOU WILL BE FIRED!!!)

Begin with the structured exploration protocol: get_schema() → get_all_external_knowledge_names() → get_all_knowledge_definitions() → get_all_column_meanings(). Then analyze ambiguities before asking the user.

User's Question: {query}
:
"""


class EnhancedTemplateV2(TemplateReActUserBirdInteract):
    """Enhanced prompt template v2 with formula elicitation, PostgreSQL syntax rules,
    submit/execute clarification, and updated demo example."""

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

## The interaction object and action space with cost
- interaction_object: `Environment`
    - action: `execute(sql)` to interact with PostgreSQL database.
        - inputs:
            - sql: string, PSQL command to execute. Could contain multiple commands separated by semicolon. MUST BE IN ONE STRING, ENCLOSED BY TWO QUOTES OR \"\"\"YOUR SQL HERE\"\"\".
        - output: fetched result from PostgreSQL database.
        - cost: 1 cost
    - action: `get_schema()` to get the schema of the database.
        - output: string of database schema in DDL format with demo data.
        - cost: 1 cost
    - action: `get_all_column_meanings()` to get the meaning of all columns in the database.
        - output: string of all column meanings.
        - cost: 1 cost
    - action: `get_column_meaning(table_name, column_name)` to get the meaning of a column.
        - inputs:
            - table_name: string, name of the table to get column meaning.
            - column_name: string, name of the column to get meaning.
        - output: string of column meaning.
        - cost: 0.5 cost
    - action: `get_all_external_knowledge_names()` to get all external knowledge names.
        - output: list of string of external knowledge names.
        - cost: 0.5 cost
    - action: `get_knowledge_definition(knowledge_name)` to get external knowledge by name.
        - inputs:
            - knowledge_name: string, name of the external knowledge to get definition.
        - output: string of external knowledge definition.
        - cost: 0.5 cost
    - action: `get_all_knowledge_definitions()` to get all external knowledge names with definitions.
        - output: string of all external knowledge names with definitions.
        - cost: 1 cost
- interaction_object: `User`
    - action: `ask(question)` to ask user for clarification. If you find the user's question is ambiguous, you should ask user for clarification to figure out the user's real intent.
        - inputs:
            - question: string, question to ask user for clarification. IMPORTANT: Ask only ONE question per ask() call. Do not combine multiple questions.
        - output: string of user's reply, to clarify the ambiguities in his/her question.
        - cost: 2 cost
    - action: `submit(sql)` to submit the SQL to the user. The user will test the SQL and give feedback.
        - inputs:
            - sql: string, SQL to submit to the user. Could contain multiple commands separated by semicolon. MUST BE IN ONE STRING, ENCLOSED BY TWO QUOTES OR \"\"\"YOUR SQL HERE\"\"\".
        - output: feedback from user about the submitted SQL.
        - cost: 3 cost

After each action, you'll see a [SYSTEM NOTE] showing how much patience remains (e.g. "[SYSTEM NOTE: Remaining user patience: 7/10]"). Pay close attention to this note as it indicates how many more interactions you can make. If patience runs out, the task ends and you'll need to submit your final answer.

# Strategy: Structured Exploration Protocol
ALWAYS begin with these 4 steps in order to build full context before interacting with the user:
1. `get_schema()` (cost: 1) — understand the database structure
2. `get_all_external_knowledge_names()` (cost: 0.5) — see what domain knowledge is available
3. `get_all_knowledge_definitions()` (cost: 1) — read all knowledge definitions
4. `get_all_column_meanings()` (cost: 1) — get detailed column descriptions including full names, explanations, data types, possible categories, and example values
Total cost: 3.5 for complete context. This investment pays off by enabling precise questions and correct SQL. Column meanings are especially valuable: they reveal human-readable names for cryptic column identifiers, enumerate categorical values (e.g., 'Clear', 'Cloudy', 'Partially Cloudy'), and explain what each column represents — all critical for resolving ambiguities.

# Strategy: Ambiguity Detection
Before asking the user anything, systematically analyze the query:
- List every potentially vague or ambiguous term in the user's question.
- Cross-reference each term against the schema (table/column names, data types), column meanings (full names, explanations, categories), and external knowledge definitions.
- Only ask about terms you CANNOT resolve from schema, column meanings, or knowledge base alone.

# Strategy: Schema-Aware Questions
When asking clarifications, always reference the specific tables, columns, column meanings, or knowledge definitions you found. For example: "The database has columns 'order_date' and 'created_at' — which one corresponds to 'recent' in your question?" This helps the user give precise answers.

# Strategy: Formula Elicitation
When a query term appears to be a computed metric, score, index, ratio, factor, or coefficient and its exact definition is NOT fully provided in the external knowledge:
- Ask specifically for the formula: "What is the exact formula for computing [term]? Which database columns does it use and what arithmetic operations (addition, multiplication, etc.)?"
- Do NOT ask vague questions like "What do you mean by [term]?" — this produces unhelpful answers.
- If the user's response lists ingredients but not the formula structure, follow up with: "How exactly are [col1], [col2], and [col3] combined? Is it a sum, product, ratio, or something else?"

# Strategy: Hypothesis-Driven SQL Testing (USE SPARINGLY)
Only use `execute(sql)` to verify specific data values exist (e.g., checking if a column contains an expected value like 'Full' vs 'full_moon'). Do NOT use execute() to iteratively debug SQL logic — if your query is wrong after 2 attempts, reconsider your approach or ask the user.
Limit: at most 2-3 execute() calls per phase. If you are confident in your SQL after resolving all ambiguities, submit directly (cost: 3) instead of execute-then-submit (cost: 4).

CRITICAL: NEVER use submit() for exploratory or diagnostic queries (e.g., SELECT COUNT(*), SELECT DISTINCT ...). submit() costs 3 coins AND runs test comparison — exploratory queries will always fail. Use execute() (cost 1) for exploration, submit() (cost 3) only for your final answer SQL.

# Strategy: Budget-Aware Planning
Before each action, mentally calculate: remaining_budget - 3 (reserve for final submit). Plan your remaining actions to fit within that. If budget is tight, prioritize ambiguities that change the SQL structure (which tables, columns, formulas, or filter conditions to use) over cosmetic details.

CRITICAL: The budget is shared across BOTH phases. If you spend everything on Phase 1, you cannot attempt Phase 2 at all (losing 0.3 points). Budget allocation guide:
- Exploration: 3.5 coins (schema + KB + column meanings)
- Clarification (ask): 2-4 coins (1-2 asks)
- SQL verification (execute): 1-2 coins (only if unsure)
- Submit Phase 1: 3 coins
- Reserve for Phase 2: at least 4-7 coins (execute + submit)
- Total budget is typically ~20-24. Never spend more than 3 coins on execute() per phase.

# Strategy: Knowledge-Gap Analysis
After retrieving knowledge definitions, compare them against every term in the user's query. If a query term seems related to a knowledge concept but the definition doesn't fully resolve the ambiguity, that's likely a hidden knowledge ambiguity — ask the user about it specifically.

# PostgreSQL Syntax Rules (MUST FOLLOW)
Your SQL MUST be valid PostgreSQL. Common mistakes to avoid:
- Use GREATEST(a, b, c) for multi-value maximum, NOT MAX(a, b, c). MAX() takes a single column.
- Use LEAST(a, b, c) for multi-value minimum, NOT MIN(a, b, c).
- For float division, cast with ::NUMERIC or divide by a decimal (x / 100.0 NOT x / 100). Integer division truncates silently in PostgreSQL.
- Use standard double quotes for column aliases: "column name" — do NOT escape with backslashes.
- Use PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col) for median, NOT MEDIAN().
- Do NOT include literal \\n in SQL strings.
- Return ONLY the columns the user asked for. Extra columns cause test failures.
"""

    def get_demos(self):
        return """### A TOY Example Interaction ###

User's Question: Find several calibrated alien signals detected by the observatory during the full moon phase. Show their information.

<thought>
Following the structured exploration protocol, I'll first get the database schema to understand available tables and columns.
</thought>
<interaction_object>
Environment
</interaction_object>
<action>
get_schema()
</action>

Observation: Database schema showing tables: Observatories(ObservID, ObservStation, ObservDate, LunarStage, Signal), Signals(SignalID, SignalType, SignalStrength, Frequency, EncodingMethod)...

[SYSTEM NOTE: Remaining user patience: 19.0/20.0]

<thought>
Schema retrieved. Next step in the structured exploration: get all external knowledge names to see what domain knowledge is available.
</thought>
<interaction_object>
Environment
</interaction_object>
<action>
get_all_external_knowledge_names()
</action>

Observation: ["Alien Signal Classification", "Lunar Phase Effects", "Calibrated Signal"]

[SYSTEM NOTE: Remaining user patience: 18.5/20.0]

<thought>
I see knowledge entries that look relevant. Let me get all definitions at once (cost 1) rather than fetching them one by one (cost 0.5 each x 3 = 1.5).
</thought>
<interaction_object>
Environment
</interaction_object>
<action>
get_all_knowledge_definitions()
</action>

Observation:
- Alien Signal Classification: Signals are classified by type: 'Modulated', 'Pulsed', 'Continuous'. Alien signals are those with SignalType = 'Modulated' or 'Pulsed'.
- Lunar Phase Effects: Observations during certain lunar phases may have noise interference.
- Calibrated Signal: [No definition available]

[SYSTEM NOTE: Remaining user patience: 17.5/20.0]

<thought>
Knowledge definitions retrieved. Final exploration step: get all column meanings for detailed column descriptions, full names, and possible categorical values.
</thought>
<interaction_object>
Environment
</interaction_object>
<action>
get_all_column_meanings()
</action>

Observation: Column meanings showing: ObservStation (Full name: 'Observatory Name', CHAR(60)), LunarStage (Full name: 'Lunar Phase', VARCHAR(25), Possible categories: First Quarter, Full, Last Quarter, New), SignalType (Full name: 'Signal Type', VARCHAR(20), Possible categories: Continuous, Modulated, Pulsed), SignalStrength (Full name: 'Signal Strength', DECIMAL(8,2))...

[SYSTEM NOTE: Remaining user patience: 16.5/20.0]

<thought>
Now I have full context (schema + KB + column meanings). Let me do ambiguity detection on the query "Find several calibrated alien signals detected by the observatory during the full moon phase. Show their information."

Ambiguous terms:
1. "alien signals" → RESOLVED from KB: SignalType IN ('Modulated', 'Pulsed')
2. "full moon phase" → RESOLVED from schema + column meanings: LunarStage = 'Full' (confirmed by column meanings showing categories: First Quarter, Full, Last Quarter, New)
3. "calibrated" → UNRESOLVED: KB entry "Calibrated Signal" exists but has no definition. This looks like a metric/threshold — I need the exact formula.
4. "several" → UNRESOLVED: how many? 3? 5? 10?
5. "their information" → UNRESOLVED: which columns? All? Just ID and strength?

I'll start with the most impactful ambiguity — "calibrated" — since it affects the WHERE clause. I need to ask for the exact formula, not a vague question.
</thought>
<interaction_object>
User
</interaction_object>
<action>
ask('The knowledge base mentions "Calibrated Signal" but does not give the exact formula. What is the exact formula for determining if a signal is calibrated? Which columns and what threshold are involved?')
</action>

Observation: User's Response: A calibrated signal is one where the SignalStrength is greater than 10.

[SYSTEM NOTE: Remaining user patience: 14.5/20.0]

<thought>
"calibrated" resolved: SignalStrength > 10. Now I need to resolve "several" — how many results. Budget: 14.5 - 2 (ask) - 3 (submit) = 9.5 available. Still plenty for Phase 2.
</thought>
<interaction_object>
User
</interaction_object>
<action>
ask('How many results do you mean by "several"? For example, top 3, top 5, or top 10?')
</action>

Observation: User's Response: I want the top 3 results sorted by signal strength.

[SYSTEM NOTE: Remaining user patience: 12.5/20.0]

<thought>
"several" resolved: top 3 by signal strength. For "their information" I'll just return the signal ID since the user didn't specify details. Budget check: 12.5 - 3 (submit) = 9.5 available. I need to reserve at least 4-7 coins for Phase 2 (follow-up question). I'm confident in my SQL since all terms are resolved from schema, column meanings, and KB, so I'll submit directly.
</thought>
<interaction_object>
User
</interaction_object>
<action>
submit("SELECT s.SignalID FROM Observatories o JOIN Signals s ON o.Signal = s.SignalID WHERE o.LunarStage = 'Full' AND s.SignalStrength > 10 AND s.SignalType IN ('Modulated', 'Pulsed') ORDER BY s.SignalStrength DESC LIMIT 3")
</action>

Observation: Your SQL is correct! Now, here's a follow-up question from user: Can you also count how many distinct calibrated signals were detected during full moon phases?

[SYSTEM NOTE: Remaining user patience: 9.5/20.0]

<thought>
Phase 1 complete! Budget: 9.5 remaining. This follow-up is straightforward — I already know the schema, KB definitions, column meanings, and filters from Phase 1. I can construct the SQL directly without any additional execute() or ask() calls. Budget: 9.5 - 3 (submit) = 6.5 remaining. Submit directly.
</thought>
<interaction_object>
User
</interaction_object>
<action>
submit("SELECT COUNT(DISTINCT s.SignalID) FROM Observatories o JOIN Signals s ON o.Signal = s.SignalID WHERE o.LunarStage = 'Full' AND s.SignalStrength > 10 AND s.SignalType IN ('Modulated', 'Pulsed')")
</action>

Observation: Your SQL is correct!

### END OF TOY EXAMPLE INTERACTION ###
"""

    def get_query_msg(self, query):
        return f"""# -----TASK START-----
Now, let's start with the user's question that may exist ambiguities and require external knowledge understanding to solve. (EACH TIME GIVE ONE ROUND RESPONSE, END YOUR RESPONSE AT ... '</action>' OTHERWISE YOU WILL BE FIRED!!!)

Begin with the structured exploration protocol: get_schema() → get_all_external_knowledge_names() → get_all_knowledge_definitions() → get_all_column_meanings(). Then analyze ambiguities before asking the user.

User's Question: {query}
:
"""
