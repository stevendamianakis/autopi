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


def _anthropic_sse(text):
    """A complete streamed Messages API reply, as the Anthropic server sends it."""
    import json

    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_offline", "type": "message", "role": "assistant", "model": "any-model", "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 0}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 1}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def test_anthropic_target_sends_no_sampling_parameters_through_the_real_sdk(monkeypatch, tmp_path):
    """The request goes through the installed SDK's own stream call and HTTP layer, so a
    parameter the SDK or the API version does not accept fails here (AgentDojo always passes a
    temperature). The output cap is the alias's, else AgentDojo's 1,024, and the call is traced."""
    pytest.importorskip("agentdojo")
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")                   # the HTTP package of the anthropic SDK 2.x
    import json

    from agentdojo.agent_pipeline.llms.anthropic_llm import AnthropicLLM
    from agentdojo.functions_runtime import FunctionsRuntime

    from autopi.harness.journal import Journal
    from autopi.models.endpoints import ModelEndpoint
    from autopi.models.registry import build_target_pipeline

    sent = []

    def reply(request):
        sent.append(json.loads(request.content))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=_anthropic_sse("done"))

    class OfflineAsyncAnthropic(anthropic.AsyncAnthropic):
        def __init__(self, **kwargs):
            super().__init__(http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)), **kwargs)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", OfflineAsyncAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-test-value")
    journal = Journal(tmp_path)

    def request(**settings):
        endpoint = ModelEndpoint("anthropic", "any-model", **settings)
        pipeline = build_target_pipeline("anthropic:any-model", endpoint=endpoint, journal=journal)
        llm = next(element for element in pipeline.elements if isinstance(element, AnthropicLLM))
        _, _, _, messages, _ = llm.query("hi", FunctionsRuntime([]), messages=[
            {"role": "system", "content": [{"type": "text", "content": "System message"}]},
            {"role": "user", "content": [{"type": "text", "content": "hi"}]},
        ])
        assert messages[-1]["content"][0]["content"] == "done"
        return sent[-1]

    for body in (request(max_tokens=7777), request()):
        assert not {"temperature", "top_p", "top_k"} & set(body)
    assert (sent[0]["max_tokens"], sent[1]["max_tokens"]) == (7777, 1024)
    events = [json.loads(line)["event"] for line in journal.path.read_text().splitlines()]
    assert events.count("llm_request") == events.count("llm_response_original") == 2
    for setting in ({"temperature": 0.0}, {"top_p": 0.9}):
        with pytest.raises(ValueError, match="temperature or top_p"):
            ModelEndpoint("anthropic", "any-model", **setting)


def test_gemini_thought_signatures_go_back_with_their_tool_calls(monkeypatch):
    """Gemini 3 and later reject a follow-up request whose tool calls lack the thought
    signatures they came with. AgentDojo's own Gemini client rebuilds the calls without them."""
    pytest.importorskip("agentdojo")
    genai_models = pytest.importorskip("google.genai.models")
    from agentdojo.agent_pipeline.llms.google_llm import GoogleLLM
    from agentdojo.functions_runtime import FunctionsRuntime
    from google.genai import types as gt

    from autopi.models.endpoints import ModelEndpoint
    from autopi.models.registry import build_target_pipeline

    sent = []

    def generate_content(self, *, model, contents, config=None):
        sent.append(contents)
        part = (gt.Part(function_call=gt.FunctionCall(name="read_inbox", args={"folder": "work"}),
                        thought_signature=b"SIG-1") if len(sent) == 1 else gt.Part(text="done"))
        return gt.GenerateContentResponse(candidates=[gt.Candidate(content=gt.Content(role="model", parts=[part]))])

    monkeypatch.setattr(genai_models.Models, "generate_content", generate_content)
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test-value")
    pipeline = build_target_pipeline("google:any-model", endpoint=ModelEndpoint("google", "any-model"))
    llm = next(element for element in pipeline.elements if isinstance(element, GoogleLLM))
    messages = [{"role": "user", "content": [{"type": "text", "content": "check my inbox"}]}]
    _, _, _, messages, _ = llm.query("check my inbox", FunctionsRuntime([]), messages=messages)
    call = messages[-1]["tool_calls"][0]
    messages = [*messages, {"role": "tool", "content": [{"type": "text", "content": "3 emails"}],
                            "tool_call_id": call.id, "tool_call": call, "error": None}]
    llm.query("check my inbox", FunctionsRuntime([]), messages=messages)
    model_turn = next(content for content in sent[1] if content.role == "model")
    assert [part.thought_signature for part in model_turn.parts if part.function_call] == [b"SIG-1"]


def test_provider_errors_of_every_client_are_infrastructure_failures():
    pytest.importorskip("agentdojo")
    anthropic = pytest.importorskip("anthropic")
    errors = pytest.importorskip("google.genai.errors")
    openai = pytest.importorskip("openai")
    tenacity = pytest.importorskip("tenacity")
    httpx2 = pytest.importorskip("httpx2")
    import httpx

    from autopi.adapters.agentdojo import _classify_infra_error

    request = httpx2.Request("POST", "https://api.example.invalid/v1")       # anthropic SDK 2.x objects

    def status(code):
        return httpx2.Response(code, request=request, json={"error": {"message": "test"}})

    def wrapped(exc):  # what AgentDojo's Anthropic client raises after its three tries
        return tenacity.RetryError(tenacity.Future.construct(3, exc, True))

    provider_errors = [
        openai.APIConnectionError(request=httpx.Request("POST", "https://api.example.invalid/v1")),
        anthropic.APIConnectionError(request=request),
        anthropic.RateLimitError("rate", response=status(429), body=None),
        anthropic.InternalServerError("server", response=status(500), body=None),
        wrapped(anthropic.RateLimitError("rate", response=status(429), body=None)),
        errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}),
        errors.ServerError(503, {"error": {"code": 503, "message": "busy", "status": "UNAVAILABLE"}}),
    ]
    assert [_classify_infra_error(exc) for exc in provider_errors] == ["provider_error"] * len(provider_errors)
    assert _classify_infra_error(TypeError("a bug")) is None             # bugs still stop the run
    assert _classify_infra_error(wrapped(TypeError("a bug"))) is None


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
