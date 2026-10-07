import pytest
import asyncio
from unittest.mock import patch, MagicMock

from orchestrator.group_chat import PhantomDevOrchestrator
from orchestrator.state import TaskState, TaskStatus

@pytest.fixture
def base_state():
    return TaskState(
        github_issue_number=1,
        github_issue_title="Test Fault Injection",
        github_issue_body="We need to test resilience.",
        target_repo="test/repo",
        base_branch="main",
        task_id="fault_test_123"
    )

@pytest.mark.asyncio
async def test_agent_exception_handled_gracefully(base_state):
    """
    Test that if an agent's generate_reply raises an unhandled exception,
    the _wrap_agent wrapper catches it and prevents the pipeline from crashing.
    """
    orchestrator = PhantomDevOrchestrator()
    
    mock_agent = MagicMock()
    mock_agent.name = "TestAgent"
    mock_agent.generate_reply.side_effect = RuntimeError("Simulated API Timeout")
    
    orchestrator._wrap_agent(mock_agent, base_state)
    
    # Execute the wrapped method
    reply = mock_agent.generate_reply(messages=[])
    
    assert "Simulated API Timeout" in reply
    assert "TestAgent encountered an error" in reply
    
    # State should have logged the error message
    assert len(base_state.agent_messages) == 1
    assert base_state.agent_messages[0]["agent"] == "TestAgent"
    assert "Simulated API Timeout" in base_state.agent_messages[0]["content"]


@pytest.mark.asyncio
@patch("orchestrator.group_chat.GroupChat")
@patch("orchestrator.group_chat.GroupChatManager")
@patch("autogen.UserProxyAgent")
async def test_speaker_selection_fault_loops(mock_user_proxy_cls, mock_gcm_cls, mock_gc_cls, base_state):
    """
    Test the dynamic_speaker_selection logic for handling retry limits
    when agents get stuck in validation, QA, or security loops.
    """
    orchestrator = PhantomDevOrchestrator()
    
    captured_selection_method = None
    
    def mock_gc_init(*args, **kwargs):
        nonlocal captured_selection_method
        captured_selection_method = kwargs.get("speaker_selection_method")
        return MagicMock()
        
    mock_gc_cls.side_effect = mock_gc_init
    
    # Mock user proxy to avoid actual LLM initiation
    mock_user_proxy = MagicMock()
    mock_user_proxy_cls.return_value = mock_user_proxy
    mock_user_proxy.initiate_chat.return_value = None
    
    # Mock all agent builders to just return dummy agents
    patchers = [
        patch("orchestrator.group_chat.build_pm_agent", return_value=MagicMock(name="PMAgent")),
        patch("orchestrator.group_chat.build_architect_agent", return_value=MagicMock(name="ArchitectAgent")),
        patch("orchestrator.group_chat.build_engineer_agents", return_value=[MagicMock(name="EngineerAgent_0")]),
        patch("orchestrator.group_chat.build_qa_agent", return_value=MagicMock(name="QAAgent")),
        patch("orchestrator.group_chat.build_security_agent", return_value=MagicMock(name="SecurityAgent")),
        patch("orchestrator.group_chat.build_writer_agent", return_value=MagicMock(name="WriterAgent")),
        patch("orchestrator.group_chat.build_pr_agent", return_value=MagicMock(name="PRAgent"))
    ]
    
    for p in patchers:
        p.start()
        
    try:
        # Run orchestrator to capture the internal dynamic_speaker_selection
        await orchestrator.run(base_state)
    finally:
        for p in patchers:
            p.stop()
            
    assert captured_selection_method is not None, "Failed to capture speaker_selection_method"
    
    # Setup mock groupchat state for testing the selection method
    mock_gc = MagicMock()
    
    # Create mock agents with name attribute explicitly set
    agents = {}
    for name in ["PMAgent", "ArchitectAgent", "EngineerAgent_0", "QAAgent", "SecurityAgent", "WriterAgent", "PRAgent"]:
        agent = MagicMock()
        agent.name = name
        agents[name] = agent
        
    mock_gc.agents = list(agents.values())
    
    # --- TEST 1: Validation Loop Limit (Max 3) ---
    mock_gc.messages = [{"content": "VALIDATION_FAILED: SyntaxError on line 1"}]
    
    # 1st fail
    next_agent = captured_selection_method(agents["EngineerAgent_0"], mock_gc)
    assert next_agent.name == "EngineerAgent_0"
    
    # 2nd fail
    next_agent = captured_selection_method(agents["EngineerAgent_0"], mock_gc)
    assert next_agent.name == "EngineerAgent_0"
    
    # 3rd fail
    next_agent = captured_selection_method(agents["EngineerAgent_0"], mock_gc)
    assert next_agent.name == "EngineerAgent_0"
    
    # 4th fail - should exceed limit and abort
    next_agent = captured_selection_method(agents["EngineerAgent_0"], mock_gc)
    assert next_agent.name == "PRAgent" # Jumps to PR agent on abort
    assert "Max validation retries exceeded" in base_state.errors[-1]
    
    
    # --- TEST 2: QA Loop Limit (Max 2) ---
    orchestrator.loop_counts["qa"] = 0 # Reset just in case
    base_state.errors = [] # Reset errors
    
    mock_gc.messages = [{"content": "QAAgent BLOCKED: 3 tests failing"}]
    
    # 1st fail
    next_agent = captured_selection_method(agents["QAAgent"], mock_gc)
    assert next_agent.name == "EngineerAgent_0"
    
    # 2nd fail
    next_agent = captured_selection_method(agents["QAAgent"], mock_gc)
    assert next_agent.name == "EngineerAgent_0"
    
    # 3rd fail - should exceed limit and bypass to SecurityAgent
    next_agent = captured_selection_method(agents["QAAgent"], mock_gc)
    assert next_agent.name == "SecurityAgent"
    assert "Max QA fix retries exceeded" in base_state.errors[-1]
    
    
    # --- TEST 3: Security Loop Limit (Max 2) ---
    orchestrator.loop_counts["security"] = 0
    base_state.errors = []
    
    mock_gc.messages = [{"content": "SecurityAgent BLOCKED: High severity vulnerability found"}]
    
    # 1st fail
    next_agent = captured_selection_method(agents["SecurityAgent"], mock_gc)
    assert next_agent.name == "EngineerAgent_0"
    
    # 2nd fail
    next_agent = captured_selection_method(agents["SecurityAgent"], mock_gc)
    assert next_agent.name == "EngineerAgent_0"
    
    # 3rd fail - should exceed limit and bypass to WriterAgent
    next_agent = captured_selection_method(agents["SecurityAgent"], mock_gc)
    assert next_agent.name == "WriterAgent"
    assert "Max Security fix retries exceeded" in base_state.errors[-1]

@pytest.mark.asyncio
@patch("orchestrator.group_chat.GroupChat")
@patch("orchestrator.group_chat.GroupChatManager")
@patch("autogen.UserProxyAgent")
async def test_orchestrator_run_exception(mock_user_proxy_cls, mock_gcm_cls, mock_gc_cls, base_state):
    """
    Test that an unexpected exception during orchestrator.run sets the state to FAILED.
    """
    orchestrator = PhantomDevOrchestrator()
    
    # Mock build_pm_agent to raise an exception
    with patch("orchestrator.group_chat.build_pm_agent", side_effect=ValueError("Failed to build agent")):
        await orchestrator.run(base_state)
        
    assert base_state.status == TaskStatus.FAILED
    assert "Failed to build agent" in base_state.errors[-1]

@pytest.mark.asyncio
@patch("orchestrator.group_chat.GroupChat")
@patch("orchestrator.group_chat.GroupChatManager")
@patch("autogen.UserProxyAgent")
async def test_orchestrator_failure_signal(mock_user_proxy_cls, mock_gcm_cls, mock_gc_cls, base_state):
    """
    Test that if agents report PHANTOMDEV_FAILED, the orchestrator updates the status.
    """
    orchestrator = PhantomDevOrchestrator()
    
    def mock_initiate_chat(*args, **kwargs):
        # We need to simulate that the last message contains PHANTOMDEV_FAILED
        # The orchestrator reads groupchat.messages[-5:]
        manager = args[0] if args else kwargs.get("manager")
        mock_gc_instance = manager.groupchat
        mock_gc_instance.messages = [{"content": "We cannot proceed. PHANTOMDEV_FAILED"}]
    
    mock_user_proxy_instance = MagicMock()
    mock_user_proxy_cls.return_value = mock_user_proxy_instance
    mock_user_proxy_instance.initiate_chat.side_effect = mock_initiate_chat
    
    mock_gc_instance = MagicMock()
    mock_gc_cls.return_value = mock_gc_instance
    
    mock_gcm_instance = MagicMock()
    mock_gcm_instance.groupchat = mock_gc_instance
    mock_gcm_cls.return_value = mock_gcm_instance
    
    patchers = [
        patch("orchestrator.group_chat.build_pm_agent", return_value=MagicMock()),
        patch("orchestrator.group_chat.build_architect_agent", return_value=MagicMock()),
        patch("orchestrator.group_chat.build_engineer_agents", return_value=[]),
        patch("orchestrator.group_chat.build_qa_agent", return_value=MagicMock()),
        patch("orchestrator.group_chat.build_security_agent", return_value=MagicMock()),
        patch("orchestrator.group_chat.build_writer_agent", return_value=MagicMock()),
        patch("orchestrator.group_chat.build_pr_agent", return_value=MagicMock())
    ]
    
    for p in patchers: p.start()
    try:
        await orchestrator.run(base_state)
    finally:
        for p in patchers: p.stop()
        
    assert base_state.status == TaskStatus.FAILED
    assert "Agent pipeline reported failure" in base_state.errors[-1]
