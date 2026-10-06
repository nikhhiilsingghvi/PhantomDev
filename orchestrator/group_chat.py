"""
orchestrator/group_chat.py

Runs the AutoGen GroupChat pipeline. Uses a file-based state bridge
so state updates work reliably on Windows where asyncio event loop
access from thread executor context is restricted.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from pathlib import Path

import autogen
import openai
from autogen import GroupChat, GroupChatManager
from langsmith import wrappers

# Patch OpenAI clients so LangSmith tracing picks up AutoGen calls automatically.
_orig_sync_init = openai.OpenAI.__init__


def _patched_sync_init(self, *args, **kwargs):
    _orig_sync_init(self, *args, **kwargs)
    wrappers.wrap_openai(self)


openai.OpenAI.__init__ = _patched_sync_init

_orig_async_init = openai.AsyncOpenAI.__init__


def _patched_async_init(self, *args, **kwargs):
    _orig_async_init(self, *args, **kwargs)
    wrappers.wrap_openai(self)


openai.AsyncOpenAI.__init__ = _patched_async_init

from agents.architect_agent import build_architect_agent
from agents.engineer_agent import build_engineer_agents
from agents.pm_agent import build_pm_agent
from agents.pr_agent import build_pr_agent
from agents.qa_agent import build_qa_agent
from agents.security_agent import build_security_agent
from agents.writer_agent import build_writer_agent
from orchestrator.state import TaskState, TaskStatus

logger = logging.getLogger(__name__)

# Live task state is written here after each agent turn.
# The frontend polls GET /tasks/{id}, which reads from this directory.
STATE_DIR = Path(os.getenv("WORKSPACE_DIR", "./workspace")) / ".state"


def get_llm_config() -> dict:
    # Priority order: Anthropic → Groq → OpenAI → Ollama (local)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

    if anthropic_key:
        anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        logger.info(f"Using Anthropic: {anthropic_model}")
        return {
            "config_list": [
                {
                    "model": anthropic_model,
                    "api_key": anthropic_key,
                    "api_type": "anthropic",
                }
            ],
            "temperature": 0.1,
            "timeout": 120,
            "cache_seed": None,
        }

    if groq_key:
        # Groq recently decommissioned all LLaMA 3 models. Use the new standard.
        groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        logger.info(f"Using Groq: {groq_model}")
        return {
            "config_list": [
                {
                    "model": groq_model,
                    "api_key": groq_key,
                    "base_url": "https://api.groq.com/openai/v1",
                    "max_retries": 10,
                }
            ],
            "temperature": 0.1,
            "timeout": 120,
            "max_retries": 10,
            "cache_seed": None,
        }

    if openai_key:
        logger.info("Using OpenAI: gpt-4o-mini")
        return {
            "config_list": [
                {
                    "model": "gpt-4o-mini",
                    "api_key": openai_key,
                }
            ],
            "temperature": 0.1,
            "timeout": 60,
            "cache_seed": None,
        }

    logger.info(f"Using Ollama: {ollama_model}")
    return {
        "config_list": [
            {
                "model": ollama_model,
                "base_url": f"{ollama_url}/v1",
                "api_key": "ollama",
            }
        ],
        "temperature": 0.1,
        "timeout": int(os.getenv("AGENT_TIMEOUT", 180)),
        "cache_seed": None,
    }


def _extract_text(content) -> str:
    """Normalize any AutoGen/Anthropic content shape to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        # AutoGen message dict: {"content": "...", "role": "assistant", ...}
        # or Anthropic content block: {"type": "text", "text": "..."}
        inner = content.get("content") or content.get("text", "")
        return _extract_text(inner)
    if isinstance(content, list):
        # Anthropic list-of-blocks: [{"type": "text", "text": "..."}, ...]
        parts = [_extract_text(block) for block in content]
        return "\n".join(p for p in parts if p)
    if content is None:
        return ""
    return str(content)


def is_termination_msg(msg: dict) -> bool:
    content = _extract_text(msg.get("content", ""))
    return any(
        m in content
        for m in ["PHANTOMDEV_COMPLETE", "PHANTOMDEV_FAILED", "HUMAN_APPROVAL_REQUIRED"]
    )


# Fixed agent execution order — baseline flow.
AGENT_ORDER = [
    "PMAgent",
    "ArchitectAgent",
    "EngineerAgent_0",
    "EngineerAgent_1",
    "EngineerAgent_2",
    "QAAgent",
    "SecurityAgent",
    "WriterAgent",
    "PRAgent",
]


def _save_state_sync(state: TaskState) -> None:
    """
    Write task state to a JSON file synchronously.
    Safe to call from any thread context — no asyncio dependency.
    Uses an atomic write (temp file + rename) to avoid partial reads.
    """
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        path = STATE_DIR / f"{state.task_id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(state.model_dump_json(), encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        logger.warning(f"State save failed: {e}")


def load_state_from_file(task_id: str) -> TaskState | None:
    """Load task state from the file-based store. Used by the API to get the freshest state."""
    try:
        path = STATE_DIR / f"{task_id}.json"
        if path.exists():
            return TaskState.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"State load failed for {task_id}: {e}")
    return None


class PhantomDevOrchestrator:
    def __init__(self, on_update: Callable | None = None):
        self._on_update_cb = on_update
        self.llm_config = get_llm_config()
        self.loop_counts = {"qa": 0, "security": 0, "validation": 0}

    def _fire_update(self, state: TaskState) -> None:
        """
        Persist state after each agent turn.
        File save always runs (works from any thread).
        The async callback is attempted if an event loop is available.
        """
        _save_state_sync(state)

        if self._on_update_cb is not None:
            try:
                import inspect

                if inspect.iscoroutinefunction(self._on_update_cb):
                    try:
                        loop = asyncio.get_running_loop()
                        asyncio.run_coroutine_threadsafe(
                            self._on_update_cb(state), loop
                        )
                    except RuntimeError:
                        pass  # no running loop — file save is sufficient
                else:
                    self._on_update_cb(state)
            except Exception as e:
                logger.debug(f"on_update callback skipped: {e}")

    async def run(self, state: TaskState) -> TaskState:
        logger.info(f"Starting task {state.task_id}")
        state.set_status(TaskStatus.PLANNING)
        state.add_message(
            "PhantomDev", f"🚀 Pipeline starting for: {state.github_issue_title}"
        )
        _save_state_sync(state)

        try:
            pm_agent = build_pm_agent(self.llm_config, state)
            architect_agent = build_architect_agent(self.llm_config, state)
            engineer_agents = build_engineer_agents(
                self.llm_config, state, count=int(os.getenv("MAX_ENGINEERS", 3))
            )
            qa_agent = build_qa_agent(self.llm_config, state)
            security_agent = build_security_agent(self.llm_config, state)
            writer_agent = build_writer_agent(self.llm_config, state)
            pr_agent = build_pr_agent(self.llm_config, state)

            all_agents = [
                pm_agent,
                architect_agent,
                *engineer_agents,
                qa_agent,
                security_agent,
                writer_agent,
                pr_agent,
            ]

            for ag in all_agents:
                self._wrap_agent(ag, state)

            user_proxy = autogen.UserProxyAgent(
                name="HumanProxy",
                human_input_mode="NEVER",
                is_termination_msg=is_termination_msg,
                code_execution_config=False,
                max_consecutive_auto_reply=0,
            )

            def dynamic_speaker_selection(last_speaker, groupchat):
                agents_by_name = {a.name: a for a in groupchat.agents}
                last_msg = groupchat.messages[-1]["content"] if groupchat.messages else ""
                
                # 1. Validation Self-Correction
                if "VALIDATION_FAILED:" in last_msg:
                    self.loop_counts["validation"] += 1
                    if self.loop_counts["validation"] <= 3:
                        return last_speaker
                    else:
                        state.fail("Max validation retries exceeded. Aborting pipeline.")
                        return agents_by_name.get("PRAgent", groupchat.agents[-1])
                        
                # 2. QA Feedback Loop
                if last_speaker.name == "QAAgent" and "QAAgent BLOCKED" in last_msg:
                    self.loop_counts["qa"] += 1
                    if self.loop_counts["qa"] <= 2:
                        return agents_by_name.get("EngineerAgent_0")
                    else:
                        state.fail("Max QA fix retries exceeded. Proceeding blocked.")
                        return agents_by_name.get("SecurityAgent", groupchat.agents[-1])
                        
                # 3. Security Feedback Loop
                if last_speaker.name == "SecurityAgent" and "SecurityAgent BLOCKED" in last_msg:
                    self.loop_counts["security"] += 1
                    if self.loop_counts["security"] <= 2:
                        return agents_by_name.get("EngineerAgent_0")
                    else:
                        state.fail("Max Security fix retries exceeded. Proceeding blocked.")
                        return agents_by_name.get("WriterAgent", groupchat.agents[-1])

                if last_speaker.name not in AGENT_ORDER:
                    return agents_by_name.get("PMAgent", groupchat.agents[0])
                    
                idx = AGENT_ORDER.index(last_speaker.name)
                if idx >= len(AGENT_ORDER) - 1:
                    return agents_by_name.get("PRAgent", groupchat.agents[-1])
                
                next_name = AGENT_ORDER[idx + 1]
                return agents_by_name.get(next_name, groupchat.agents[0])

            groupchat = GroupChat(
                agents=[user_proxy] + all_agents,
                messages=[],
                max_round=int(os.getenv("MAX_ROUNDS", 80)),
                speaker_selection_method=dynamic_speaker_selection,
                allow_repeat_speaker=True,
            )

            manager = GroupChatManager(
                groupchat=groupchat,
                llm_config=self.llm_config,
                is_termination_msg=is_termination_msg,
            )

            state.add_message(
                "PhantomDev", "🤖 All agents ready. PMAgent analysing requirements…"
            )
            _save_state_sync(state)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: user_proxy.initiate_chat(
                    manager,
                    message=self._build_initial_message(state),
                    clear_history=True,
                ),
            )

            last_msgs = groupchat.messages[-5:] if groupchat.messages else []
            combined = " ".join(_extract_text(m.get("content", "")) for m in last_msgs)

            # Respect status already set by agents (e.g. PRAgent dry-run sets PR_OPEN)
            already_pr = state.status == TaskStatus.PR_OPEN

            if "PHANTOMDEV_FAILED" in combined:
                state.fail("Agent pipeline reported failure")
            elif "HUMAN_APPROVAL_REQUIRED" in combined or state.pr_url or already_pr:
                state.set_status(TaskStatus.PR_OPEN)
            else:
                state.fail("Pipeline ended without clear success signal")

        except Exception as exc:
            logger.exception(f"Orchestrator error: {exc}")
            state.fail(str(exc))

        _save_state_sync(state)
        logger.info(f"Task finished: {state.summary}")
        return state

    def _wrap_agent(self, agent, state: TaskState) -> None:
        """Intercept every agent reply, add it to the message log, and persist state."""
        original = agent.generate_reply
        orchestrator = self
        agent_name = agent.name

        def wrapped(messages=None, sender=None, **kwargs):
            # Normalize Anthropic content blocks in the INCOMING chat history.
            # Without this, AutoGen crashes internally when building the next LLM
            # request from a history that contains list-of-dict content blocks.
            if messages:
                normalized = []
                for m in messages:
                    content = m.get("content")
                    if isinstance(content, (list, dict)):
                        m = {**m, "content": _extract_text(content)}
                    normalized.append(m)
                messages = normalized

            try:
                reply = original(messages=messages, sender=sender, **kwargs)
            except Exception as exc:
                # An agent crash must NOT kill the whole pipeline.
                # Log it, record it in state, and return a neutral string so
                # the GroupChat can continue to the next agent.
                err_msg = f"⚠️ {agent_name} encountered an error: {exc}"
                logger.exception(f"{agent_name} generate_reply raised: {exc}")
                state.add_message(agent_name, err_msg)
                orchestrator._fire_update(state)
                return err_msg

            if reply is not None:
                reply_text = (
                    _extract_text(reply) if not isinstance(reply, str) else reply
                )
                if reply_text.strip():
                    last = state.agent_messages[-1] if state.agent_messages else {}
                    if (
                        last.get("content") != reply_text
                        or last.get("agent") != agent_name
                    ):
                        state.add_message(agent_name, reply_text[:3000])
                    orchestrator._fire_update(state)
                # Return normalized string so downstream regex in agent parsers works
                return reply_text
            return reply

        agent.generate_reply = wrapped

    def _build_initial_message(self, state: TaskState) -> str:
        return f"""New GitHub Issue assigned to PhantomDev autonomous engineering team.

ISSUE #{state.github_issue_number}: {state.github_issue_title}

DESCRIPTION:
{state.github_issue_body}

TARGET REPOSITORY: {state.target_repo}
BASE BRANCH: {state.base_branch}
TASK ID: {state.task_id}

PMAgent, please begin by extracting requirements and creating subtasks.""".strip()
