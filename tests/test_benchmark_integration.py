"""Real benchmark integration checks using offline ground-truth pipelines."""
from collections import Counter
from types import SimpleNamespace

import pytest

from autopi.adapters.base import AttackPoint


def test_agentdojo_uses_native_slots_and_native_execution_for_every_user_task():
    pytest.importorskip("agentdojo")
    from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
    from agentdojo.attacks.base_attacks import FixedJailbreakAttack
    from agentdojo.task_suite.load_suites import get_suites

    from autopi.adapters.agentdojo import AgentDojoAdapter

    checked = 0
    for suite_name, suite in get_suites("v1.2.2").items():
        injection = next(iter(suite.injection_tasks.values()))
        for user in suite.user_tasks.values():
            pipeline = GroundTruthPipeline(user)
            selector = FixedJailbreakAttack("{goal}", suite, pipeline)
            slots = selector.get_injection_candidates(user)
            adapter = AgentDojoAdapter.__new__(AgentDojoAdapter)
            adapter._pipeline = pipeline
            adapter._suites = {suite_name: suite}
            adapter._selectors = {suite_name: selector}
            adapter.journal = None
            point = AttackPoint(suite_name, user.ID, injection.ID, user.PROMPT, injection.GOAL,
                                {"_ut": user, "_it": injection})
            native = suite.run_task_with_pipeline(pipeline, user, injection, {slot: "TEST PAYLOAD" for slot in slots})
            result = adapter.run(point, "TEST PAYLOAD")
            assert (result.utility, result.security) == native
            assert result.info["injection_slots"] == slots
            checked += 1
    assert checked == 97


def test_agentdojo_provider_errors_cannot_be_scored_as_empty_answers():
    pytest.importorskip("agentdojo")
    from agentdojo.agent_pipeline.llms.local_llm import chat_completion_request

    def unavailable(**kwargs):
        raise ConnectionError("test transport unavailable")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=unavailable)))
    with pytest.raises(ConnectionError):
        chat_completion_request(client, "arbitrary-model", [{"role": "user", "content": "test"}])


def test_native_anthropic_message_conversion_works_with_current_sdk_types():
    pytest.importorskip("agentdojo")
    from agentdojo.agent_pipeline.llms.anthropic_llm import _conversation_to_anthropic

    system, messages = _conversation_to_anthropic([
        {"role": "system", "content": [{"type": "text", "content": "System message"}]},
        {"role": "user", "content": [{"type": "text", "content": "User message"}]},
    ])
    assert system == "System message"
    assert messages[0]["role"] == "user" and messages[0]["content"][0].text == "User message"


def test_all_ipi_scenarios_and_server_conversion_fixes_are_bundled():
    pytest.importorskip("ipi_arena_bench")
    from ipi_arena_bench.behavior import _detect_image_media_type, convert_anthropic_to_openai, load_behaviors

    behaviors = load_behaviors()
    assert Counter(behavior["category"] for behavior in behaviors.values()) == {"tool": 18, "coding": 15, "browser": 8}
    assert _detect_image_media_type("iVBORw0KGgoAAAANS", "image/jpeg") == "image/png"
    assert _detect_image_media_type("/9j/test", "image/png") == "image/jpeg"
    messages = [{"role": "assistant", "content": [
        {"type": "tool_use", "id": "already-done", "name": "done", "input": {}}]},
        {"role": "user", "content": "Next task"}]
    result = convert_anthropic_to_openai(messages)
    assert result[1] == {"role": "tool", "tool_call_id": "already-done", "content": ""}
    assert result[2]["content"] == "Next task"


def test_run_manifest_hashes_real_benchmark_data_for_both_benchmarks():
    pytest.importorskip("agentdojo")
    import hashlib
    import json

    from autopi.adapters.agentdojo import AgentDojoAdapter
    from autopi.harness.manifest import benchmark_data_digest

    empty = hashlib.sha256(b"{}").hexdigest()
    from agentdojo.task_suite.load_suites import get_suites

    suites = get_suites("v1.2.2")
    digests = []
    for chosen in (["banking"], ["slack", "travel"]):
        adapter = AgentDojoAdapter.__new__(AgentDojoAdapter)
        adapter.benchmark_version = "v1.2.2"
        adapter._suites = {name: suites[name] for name in chosen}
        digests.append(benchmark_data_digest(adapter))
    # A real digest that is the same for every worker of one benchmark version.
    assert digests[0] == digests[1] != empty
    # IPI Arena keeps hashing the scenario files it loaded.
    behaviors = {"scenario": {"category": "tool", "messages": []}}
    assert benchmark_data_digest(SimpleNamespace(_behaviors=behaviors)) == hashlib.sha256(
        json.dumps(behaviors, sort_keys=True).encode()).hexdigest()
