from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class SampleStatus:
    """Holds the status and interaction history for a single sample."""
    idx: int
    original_data: Dict[str, Any]
    current_prompt: str = ""
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    remaining_budget: float = 0.0
    total_budget: float = 0.0
    phase1_completed: bool = False
    phase2_completed: bool = False
    task_finished: bool = False
    current_turn: int = 0
    current_phase: int = 1 # 1 or 2
    # Fields to store temporary results between steps
    last_agent_response: Optional[str] = None
    parsed_action_object: Optional[str] = None
    parsed_action: Optional[str] = None
    parsed_thought: Optional[str] = None
    last_observation: Optional[str] = None
    last_reward: Optional[float] = None
    last_user_response: Optional[str] = None
    force_submit: bool = False # Flag if budget runs out
    successful_phase1_sql: Optional[str] = None # Added field

    # Structured ambiguity detection logging
    detected_ambiguities: Optional[Dict[str, Any]] = None
    ambiguity_log_injected: bool = False

    # Context engineering (scratchpad) fields — opt-in via --enable_scratchpad
    enable_scratchpad: bool = False
    compress_exploration: bool = True  # Set False via --no_scratchpad_compression
    scratchpad: Dict[str, Any] = field(default_factory=dict)

    # Add fields for budget tracking categories if needed
    # env_interact_used: float = 0.0
    # submit_used: float = 0.0
    # user_patience_used: float = 0.0

    # You might add methods here for updating status, budget, etc.
    def add_turn_log(self, thought: str, interaction_object: str, action: str, observation: str, reward: float, budget_info: Dict):
        """Adds a log entry for the completed turn."""
        self.interaction_history.append({
            "turn": self.current_turn,
            "phase": self.current_phase,
            "thought": thought,
            "interaction_object": interaction_object,
            "action": action,
            "observation": observation,
            "reward": reward, # Reward *received* in this turn (usually 0 unless it's the final submit)
            "budget_after_action": budget_info
        })

    # Actions whose observations get compressed when scratchpad is active
    _EXPLORATION_ACTIONS = frozenset([
        'get_schema()', 'get_all_external_knowledge_names()',
        'get_all_knowledge_definitions()', 'get_all_column_meanings()',
    ])

    def _is_exploration_action(self, action: str) -> bool:
        """Check if this action's observation should be compressed."""
        return action.strip() in self._EXPLORATION_ACTIONS or \
               action.strip().startswith('get_column_meaning(') or \
               action.strip().startswith('get_knowledge_definition(')

    def _build_context_summary(self) -> str:
        """Build the [CONTEXT SUMMARY] section from scratchpad data."""
        parts = []
        if self.scratchpad.get('schema'):
            parts.append(f"## Database Schema\n{self.scratchpad['schema']}")
        if self.scratchpad.get('kb_names'):
            parts.append(f"## External Knowledge Names\n{self.scratchpad['kb_names']}")
        if self.scratchpad.get('kb_definitions'):
            parts.append(f"## External Knowledge Definitions\n{self.scratchpad['kb_definitions']}")
        if self.scratchpad.get('column_meanings'):
            parts.append(f"## Column Meanings\n{self.scratchpad['column_meanings']}")
        if self.scratchpad.get('ambiguity_table'):
            parts.append(f"## Ambiguity Analysis\n{self.scratchpad['ambiguity_table']}")
        if self.scratchpad.get('knowledge_gaps'):
            parts.append(f"## Knowledge Gap Analysis\n{self.scratchpad['knowledge_gaps']}")
        if self.scratchpad.get('db_probing_results'):
            parts.append(f"## Database Probing Results\n{self.scratchpad['db_probing_results']}")
        if self.scratchpad.get('ambisql_results'):
            r = self.scratchpad['ambisql_results']
            ambisql_text = f"Refined question: {r.get('refined_question', '')}\nEvidence:\n{r.get('evidence', '')}"
            parts.append(f"## AmbiSQL Clarification Results\n{ambisql_text}")
        if self.scratchpad.get('dfpl_results'):
            r = self.scratchpad['dfpl_results']
            parts_list = []
            if r.get('selected_interpretation'):
                parts_list.append(f"Selected interpretation: {r['selected_interpretation']}")
            if r.get('user_response'):
                parts_list.append(f"User response: {r['user_response']}")
            parts_list.append(f"Distinct groups: {r.get('num_distinct_groups', 0)}")
            parts.append(f"## Disambiguation Analysis\n" + "\n".join(parts_list))

        if not parts:
            return ""
        return "[CONTEXT SUMMARY]\n" + "\n\n".join(parts) + "\n[/CONTEXT SUMMARY]\n\n"

    def _compress_observation(self, action: str) -> str:
        """Return a short placeholder for an exploration observation."""
        action_stripped = action.strip()
        if action_stripped == 'get_schema()':
            return "[Schema retrieved — see CONTEXT SUMMARY above]"
        elif action_stripped == 'get_all_external_knowledge_names()':
            return "[KB names retrieved — see CONTEXT SUMMARY above]"
        elif action_stripped == 'get_all_knowledge_definitions()':
            return "[KB definitions retrieved — see CONTEXT SUMMARY above]"
        elif action_stripped == 'get_all_column_meanings()':
            return "[Column meanings retrieved — see CONTEXT SUMMARY above]"
        elif action_stripped.startswith('get_column_meaning('):
            return "[Column meaning retrieved — see CONTEXT SUMMARY above]"
        elif action_stripped.startswith('get_knowledge_definition('):
            return "[KB definition retrieved — see CONTEXT SUMMARY above]"
        return "[Exploration data — see CONTEXT SUMMARY above]"

    def get_full_interaction_prompt(self) -> str:
        """Constructs the full prompt history for the agent."""
        prompt = self.current_prompt # Initial query + budget info

        # If scratchpad is active and has exploration data, prepend context summary
        use_compression = (
            self.enable_scratchpad
            and self.scratchpad.get('exploration_complete', False)
            and self.compress_exploration
        )

        if use_compression:
            prompt += self._build_context_summary()

        for turn_log in self.interaction_history:
            observation = turn_log['observation']
            # Compress exploration observations if scratchpad is active
            if use_compression and self._is_exploration_action(turn_log['action']):
                # Keep the budget system note at the end of the observation
                budget_note = ""
                if "[SYSTEM NOTE:" in observation:
                    note_idx = observation.rfind("[SYSTEM NOTE:")
                    budget_note = "\n\n" + observation[note_idx:]
                observation = self._compress_observation(turn_log['action']) + budget_note

            # Strip the agent's past <think> reasoning from history reconstruction.
            # The agent does not need to read its own past reasoning verbatim — the
            # action it took and the observation it received together contain all
            # the information needed to plan the next step. Past thoughts can be
            # 1500–16000 chars each; dropping them saves substantial prompt tokens
            # (~25% reduction at turn 30) without removing useful state.
            prompt += f"""<think>[past reasoning truncated]</think>
<interaction_object>{turn_log['interaction_object']}</interaction_object>
<action>{turn_log['action']}</action>

Observation: {observation}

"""
        return prompt