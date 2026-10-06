"""
agents/engineer_agent.py

Engineer Agents. Each one claims a subtask from the PM's list, generates
production code and unit tests, and writes both to the workspace.
Multiple instances run sequentially in the AutoGen GroupChat.
"""

from __future__ import annotations

import logging
import re

from agents.base_agent import PhantomBaseAgent, rag_search, write_workspace_file
from orchestrator.state import TaskState, TaskStatus

logger = logging.getLogger(__name__)


def _extract_text_local(content) -> str:
    """Normalize any AutoGen/Anthropic content shape to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        inner = content.get("content") or content.get("text", "")
        return _extract_text_local(inner)
    if isinstance(content, list):
        parts = [_extract_text_local(block) for block in content]
        return "\n".join(p for p in parts if p)
    if content is None:
        return ""
    return str(content)


ENGINEER_SYSTEM_PROMPT = """
You are EngineerAgent_{idx} in PhantomDev, an autonomous software engineering team.

Your job:
1. Find the next PENDING subtask from the list below.
2. Search the codebase for relevant patterns with rag_search().
3. If the subtask modifies an existing file, YOU MUST use read_file() to read its contents first!
4. Write production-quality code for the subtask file.
5. Write unit tests for the code (pytest style).
6. Save both files using write_file() (Note: write_file completely overwrites the file, so provide the full updated file).

Subtasks:
{subtasks_summary}

Architecture context:
{architecture_notes}

API contracts:
{api_contracts}

Expected output structure:

## Subtask Claimed
subtask_id: <id>
file_path: <path>

## Implementation
```python
# <file_path>
<your complete code here>
```

## Tests
```python
# tests/test_<filename>.py
<your complete pytest tests here>
```

After writing files, say:
"EngineerAgent_{idx} done with subtask <id>. Next engineer please proceed."

If QA or Security has reported a failure, your job is to FIX the bugs. 
Read the test results or security findings from the chat history, then output the corrected code blocks using the same format.

Coding standards:
1. Write COMPLETE files — never use placeholders like "# TODO" or "# implement later"
2. Add proper docstrings to every function and class
3. Use type hints throughout
4. Handle errors explicitly — raise specific exceptions, not bare Exception
5. Follow existing codebase patterns (check rag_search results)
6. Tests must cover: happy path, edge cases, error cases
7. Always call write_file() to save — do not just print the code

Available tools:
  rag_search(query: str) -> str
  read_file(relative_path: str) -> str
  write_file(relative_path: str, content: str) -> str
  list_files() -> str
"""


def build_engineer_agents(
    llm_config: dict,
    state: TaskState,
    count: int = 3,
) -> list[PhantomBaseAgent]:
    """Build N engineer agents, each aware of all subtasks."""
    agents = []
    for idx in range(count):
        subtasks_summary = _format_subtasks(state)
        system_msg = ENGINEER_SYSTEM_PROMPT.format(
            idx=idx,
            subtasks_summary=subtasks_summary,
            architecture_notes=state.architecture_notes or "Not yet defined",
            api_contracts=state.api_contracts or "Not yet defined",
        )

        agent = PhantomBaseAgent(
            name=f"EngineerAgent_{idx}",
            system_message=system_msg,
            llm_config=llm_config,
            state=state,
        )

        # Register file and search tools for this engineer
        agent.register_function(
            function_map={
                "rag_search": rag_search,
                "write_file": lambda path, content, _state=state: _write_and_persist(
                    path, content, _state
                ),
                "read_file": lambda path: _safe_read(path),
                "list_files": lambda: _safe_list_files(),
            }
        )

        original_generate = agent.generate_reply

        def make_reply_fn(ag, _idx=idx, _orig=original_generate):
            def generate_with_persistence(messages=None, sender=None, **kwargs):
                reply = _orig(messages=messages, sender=sender, **kwargs)
                if reply:
                    # Normalize Anthropic dict/list content to plain string
                    text = (
                        reply
                        if isinstance(reply, str)
                        else (_extract_text_local(reply))
                    )
                    if text.strip():
                        success, err = _parse_and_persist(text, state, _idx)
                        if not success:
                            return f"VALIDATION_FAILED: {err}\nPlease fix your response and try again."
                return reply

            return generate_with_persistence

        agent.generate_reply = make_reply_fn(agent)
        agents.append(agent)

    return agents


def _format_subtasks(state: TaskState) -> str:
    lines = []
    for st in state.subtasks:
        status_icon = {
            "pending": "⏳",
            "in_progress": "🔨",
            "done": "✅",
            "failed": "❌",
        }.get(st.status, "?")
        lines.append(f"{status_icon} [{st.id}] {st.title} → {st.file_path}")
        if st.status == "pending":
            lines.append(f"   DESC: {st.description}")
    return "\n".join(lines) if lines else "No subtasks defined yet — wait for PMAgent."


def _write_and_persist(path: str, content: str, state: TaskState) -> str:
    """Write file to workspace AND record in TaskState."""
    result = write_workspace_file(path, content)
    # Track in state
    if "test_" in path.split("/")[-1] or "/tests/" in path:
        state.test_files[path] = content
    else:
        state.generated_files[path] = content
    # Mark subtask done
    for st in state.subtasks:
        if st.file_path == path and st.status != "done":
            st.status = "done"
            st.code = content
            state.add_message("EngineerAgent", f"✅ Completed: {path}")
            break
    return result


def _safe_read(path: str) -> str:
    from agents.base_agent import read_workspace_file

    return read_workspace_file(path)


def _safe_list_files() -> str:
    from agents.base_agent import list_workspace_files

    return list_workspace_files()


def _parse_and_persist(reply: str, state: TaskState, agent_idx: int) -> tuple[bool, str]:
    """Parse engineer reply and save code blocks to workspace."""
    pending = [s for s in state.subtasks if s.status == "pending"]
    
    # Find all code blocks
    code_blocks = re.findall(r"```python\s*\n# ([\w/._-]+)\n(.*?)```", reply, re.DOTALL)

    if not code_blocks and pending:
        if "done" not in reply.lower():
            return False, "You must output at least one code block using the format ```python\n# file_path\n<code>\n```"

    for file_path, code in code_blocks:
        file_path = file_path.strip()
        code = code.strip()
        if file_path and code:
            _write_and_persist(file_path, code, state)
            logger.info(
                f"EngineerAgent_{agent_idx}: wrote {file_path} ({len(code)} chars)"
            )

    # Update status if all subtasks done
    pending_after = [s for s in state.subtasks if s.status == "pending"]
    if not pending_after:
        state.set_status(TaskStatus.TESTING)

    return True, ""
