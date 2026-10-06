"""
agents/pm_agent.py

Product Manager Agent. Takes a GitHub Issue and produces a structured
requirements breakdown with subtasks for the engineering agents to pick up.
Results are written directly into the shared TaskState.
"""

from __future__ import annotations

import json
import logging
import re

from agents.base_agent import PhantomBaseAgent, normalize_content
from orchestrator.state import SubTask, TaskState, TaskStatus

logger = logging.getLogger(__name__)

PM_SYSTEM_PROMPT = """
You are the Product Manager Agent in PhantomDev, an autonomous software engineering team.

Your first step is to UNDERSTAND the codebase. You MUST use rag_search() and list_files() to find existing files relevant to the issue.

After researching, produce:
1. A list of clear, unambiguous requirements (functional and non-functional).
2. Specific, testable acceptance criteria.
3. A breakdown of work into subtasks — one subtask = one file to create or modify.

End your final response with a JSON block in exactly this shape:

```json
{
  "requirements": [
    "The system must ...",
    "Users should be able to ..."
  ],
  "acceptance_criteria": [
    "Given ... when ... then ...",
    "All endpoints return HTTP 200 for valid input"
  ],
  "subtasks": [
    {
      "title": "Modify user model",
      "description": "SQLAlchemy model for User - add phone_number",
      "file_path": "app/models/user.py"
    }
  ]
}
```

Rules:
- PREFER MODIFYING EXISTING FILES OVER CREATING NEW ONES. Search the codebase first.
- DO NOT create new files if the functionality belongs in an existing file.
- One subtask per file — never group multiple files into one subtask.
- file_path must be relative (e.g. app/api/users.py).
- Each subtask description must be self-contained.
- After the JSON, say: "PMAgent done. ArchitectAgent, please proceed."

Available tools:
  rag_search(query: str) -> str
  read_file(relative_path: str) -> str
  list_files() -> str
"""


def build_pm_agent(llm_config: dict, state: TaskState) -> PhantomBaseAgent:
    agent = PhantomBaseAgent(
        name="PMAgent",
        system_message=PM_SYSTEM_PROMPT,
        llm_config=llm_config,
        state=state,
    )

    # Intercept replies to parse the JSON block and persist into TaskState
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
    """Extract JSON from PM Agent reply and write into TaskState."""
    match = re.search(r"```json\s*(.*?)\s*```", reply, re.DOTALL)
    if not match:
        logger.warning("PMAgent: no JSON block found in reply")
        return False, "No JSON block found. Please wrap your output in ```json ... ```"

    try:
        data = json.loads(match.group(1))
        state.requirements = data.get("requirements", [])
        state.acceptance_criteria = data.get("acceptance_criteria", [])

        subtasks = []
        for raw in data.get("subtasks", []):
            subtasks.append(
                SubTask(
                    title=raw.get("title", ""),
                    description=raw.get("description", ""),
                    file_path=raw.get("file_path", ""),
                )
            )
        
        if not subtasks:
            return False, "No subtasks were generated. You must create at least one subtask."
            
        state.subtasks = subtasks
        state.set_status(TaskStatus.ARCHITECTING)

        logger.info(
            f"PMAgent persisted: {len(state.requirements)} requirements, "
            f"{len(state.subtasks)} subtasks"
        )
        state.add_message("PMAgent", f"✅ {len(state.subtasks)} subtasks created")
        return True, ""
    except Exception as e:
        logger.error(f"PMAgent JSON parse error: {e}\nRaw: {match.group(1)[:500]}")
        state.errors.append(f"PM parse error: {e}")
        return False, f"JSON parsing error: {e}"
