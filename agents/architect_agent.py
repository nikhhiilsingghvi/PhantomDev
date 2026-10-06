"""
agents/architect_agent.py

Architect Agent. Reviews PM subtasks and codebase RAG context,
then produces a system design and API contracts for the engineers to follow.
"""

from __future__ import annotations

import logging
import re

from agents.base_agent import PhantomBaseAgent, normalize_content
from orchestrator.state import TaskState, TaskStatus

logger = logging.getLogger(__name__)

ARCHITECT_SYSTEM_PROMPT = """
You are the Architect Agent in PhantomDev.

Your job:
1. Review the subtasks from PMAgent.
2. Search the existing codebase for relevant patterns using rag_search().
3. Define the system design: folder structure, interfaces, data models, and patterns to follow.
4. Write API contracts (function signatures or OpenAPI snippets) for each subtask.
5. Call out key tech decisions and explain why.

Structure your response like this:
## Architecture Notes
[folder conventions, patterns, shared utilities]

## API Contracts
[function signatures or pseudo-code for each subtask's public interface]

## Tech Decisions
- [Decision]: [Rationale]

End with:
"ArchitectAgent done. EngineerAgents, begin implementation now."

Notes:
- Always call rag_search() first to find existing patterns before designing anything new.
- Match the code style already in the codebase.
- Keep the architecture notes concise — engineers need to act on them immediately.
- If the codebase is empty, set up clean FastAPI + SQLAlchemy patterns.

Available tool:
  rag_search(query: str, n_results: int = 5) -> str
    Searches the indexed codebase and returns relevant snippets.
"""


def build_architect_agent(llm_config: dict, state: TaskState) -> PhantomBaseAgent:
    agent = PhantomBaseAgent(
        name="ArchitectAgent",
        system_message=ARCHITECT_SYSTEM_PROMPT,
        llm_config=llm_config,
        state=state,
    )

    original_generate = agent.generate_reply

    def generate_with_persistence(messages=None, sender=None, **kwargs):
        reply = original_generate(messages=messages, sender=sender, **kwargs)
        if reply:
            text = normalize_content(reply)
            if text.strip():
                success, err = _parse_and_persist(text, state)
                if not success:
                    return f"VALIDATION_FAILED: {err}\nPlease fix your response and try again."
        return reply

    agent.generate_reply = generate_with_persistence
    return agent


def _parse_and_persist(reply: str, state: TaskState) -> tuple[bool, str]:
    """Extract architecture notes and tech decisions from architect reply."""
    # Pull architecture notes
    arch_match = re.search(r"## Architecture Notes\s*(.*?)(?=##|$)", reply, re.DOTALL)
    if arch_match:
        state.architecture_notes = arch_match.group(1).strip()
    else:
        return False, "Missing '## Architecture Notes' section."

    # Pull API contracts
    api_match = re.search(r"## API Contracts\s*(.*?)(?=##|$)", reply, re.DOTALL)
    if api_match:
        state.api_contracts = api_match.group(1).strip()
    else:
        return False, "Missing '## API Contracts' section."

    # Pull tech decisions
    td_match = re.search(
        r"## Tech Decisions\s*(.*?)(?=##|ArchitectAgent done|$)", reply, re.DOTALL
    )
    if td_match:
        decisions = []
        for line in td_match.group(1).strip().splitlines():
            line = line.strip("- ").strip()
            if line:
                decisions.append(line)
        state.tech_decisions = decisions
    else:
        return False, "Missing '## Tech Decisions' section."

    state.set_status(TaskStatus.CODING)
    state.add_message(
        "ArchitectAgent",
        f"✅ Architecture defined: {len(state.tech_decisions)} decisions",
    )
    logger.info(f"ArchitectAgent persisted: {len(state.tech_decisions)} tech decisions")
    return True, ""
