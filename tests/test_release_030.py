"""Offline regressions for 0.3.0: tool-calling modes and AgentDojo injection placement."""

import json

import pytest
from click.testing import CliRunner

import autopi.cli as cli_mod
import autopi.core.engine as engine_module
from autopi.adapters.base import AttackPoint, RunResult
from autopi.cli import main
from autopi.core.seeds import IMPORTANT_INSTRUCTIONS_SEED, IMPORTANT_INSTRUCTIONS_SEED_NAME
from autopi.models.endpoints import ModelEndpoint, resolve_endpoint

# --- tool calling ----------------------------------------------------------------------------

def test_short_local_specs_call_tools_natively():
    for spec, provider in (("vllm:8000:m", "openai-compatible"), ("ollama:m", "ollama"), ("local:m", "ollama")):
        target = resolve_endpoint(spec)
        assert (target.provider, target.tool_mode) == (provider, "native")
        # No sampling override: AgentDojo's own clients request temperature 0 (see the request
        # tests below), and a generator keeps its own temperature (0.7 by default).
        assert target.temperature is None and resolve_endpoint(spec, role="refiner").temperature is None
    assert resolve_endpoint("vllm:8000:m").thinking == "off"
    # Named aliases keep their configured mode, native unless they say otherwise.
    assert ModelEndpoint("openai-compatible", "m", "http://h/v1").tool_mode == "native"


def _llm_element(pipeline):
    from agentdojo.agent_pipeline.llms.local_llm import LocalLLM
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

    found = [element for element in pipeline.elements if isinstance(element, (LocalLLM, OpenAILLM))]
    assert len(found) == 1
    return found[0]


def test_target_pipeline_class_follows_the_tool_calling_mode():
    pytest.importorskip("agentdojo")
    from agentdojo.agent_pipeline.llms.local_llm import LocalLLM
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

    from autopi.models.registry import build_target_pipeline

    native = _llm_element(build_target_pipeline("vllm:8000:m"))
    assert type(native) is OpenAILLM
    prompted = _llm_element(build_target_pipeline(
        "vllm:8000:m", endpoint=ModelEndpoint("openai-compatible", "m", "http://localhost:8000/v1",
                                              tool_mode="prompted", thinking="off", temperature=0.0)))
    assert type(prompted) is LocalLLM


@pytest.fixture
def captured_requests(monkeypatch):
    """Patch the OpenAI SDK's chat-completions call so AgentDojo's real clients run offline."""
    pytest.importorskip("agentdojo")
    import openai
    from openai.types.chat import ChatCompletion

    calls = []
    resource = type(openai.OpenAI(api_key="x", base_url="http://localhost:1/v1").chat.completions)

    def create(self, **kwargs):
        calls.append(kwargs)
        return ChatCompletion.model_validate({
            "id": "c", "object": "chat.completion", "created": 0, "model": kwargs["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}]})

    monkeypatch.setattr(resource, "create", create)
    return calls


def _query(pipeline):
    from agentdojo.functions_runtime import FunctionsRuntime, make_function
    from agentdojo.types import ChatSystemMessage, ChatUserMessage, text_content_block_from_string

    def get_balance() -> float:
        """Get the balance of the account."""
        return 1.0

    messages = [ChatSystemMessage(role="system", content=[text_content_block_from_string("SYSTEM PROMPT")]),
                ChatUserMessage(role="user", content=[text_content_block_from_string("What is my balance?")])]
    return _llm_element(pipeline).query("What is my balance?", FunctionsRuntime([make_function(get_balance)]),
                                        messages=messages)


def test_native_target_sends_tools_greedy_system_role_and_thinking_off(captured_requests):
    from autopi.models.registry import build_target_pipeline

    _query(build_target_pipeline("vllm:8000:served-model"))
    request = captured_requests[-1]
    # AgentDojo's native client sends the system prompt as "developer"; a self-hosted target gets "system".
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    assert request["tools"][0]["function"]["name"] == "get_balance" and request["tool_choice"] == "auto"
    # The bundled AgentDojo's native client requests temperature 0 itself (greedy, like prompted).
    assert request["temperature"] == 0.0
    assert request["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["model"] == "served-model"


def test_hosted_openai_target_keeps_the_developer_role(captured_requests, monkeypatch):
    from autopi.models.registry import build_target_pipeline

    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    _query(build_target_pipeline("openai:gpt-test"))
    assert captured_requests[-1]["messages"][0]["role"] == "developer"


def test_prompted_target_describes_tools_in_the_prompt_and_sends_no_tool_api(captured_requests):
    from autopi.models.registry import build_target_pipeline

    endpoint = resolve_endpoint("vllm:8000:served-model")
    endpoint = ModelEndpoint(**{**endpoint.public_config(), "tool_mode": "prompted", "omit_parameters": ()})
    _query(build_target_pipeline("vllm:8000:served-model", endpoint=endpoint))
    request = captured_requests[-1]
    assert "tools" not in request and request["messages"][0]["role"] == "system"
    assert "<function=" in request["messages"][0]["content"] and "get_balance" in request["messages"][0]["content"]
    assert request["temperature"] == 0.0


def test_developer_role_mapping_leaves_other_messages_alone():
    from types import SimpleNamespace

    from autopi.models.registry import send_developer_as_system

    seen = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: seen.append(kw))))
    send_developer_as_system(client)
    messages = [{"role": "developer", "content": "s"}, {"role": "user", "content": "u"}, "opaque"]
    client.chat.completions.create(model="m", messages=messages)
    assert seen[-1]["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, "opaque"]
    assert messages[0]["role"] == "developer"          # the caller's list is not modified
    client.chat.completions.create(model="m")           # no messages: passed through unchanged
    assert seen[-1] == {"model": "m"}


class _StubAdapter:
    name = "stub"
    checker_kind = "state"
    default_seed_template = IMPORTANT_INSTRUCTIONS_SEED
    default_seed_name = IMPORTANT_INSTRUCTIONS_SEED_NAME
    benchmark_version = "v1.2.2"

    def iter_points(self):
        yield AttackPoint("banking", "user_task_0", "injection_task_0", "Pay the bill", "Send money")

    def run(self, point, injection_text):
        return RunResult(utility=True, security=True, messages=[], info={})


@pytest.fixture
def captured_build(monkeypatch):
    seen = {}

    def build(*args, **kwargs):
        seen.update(kwargs, benchmark=args[0])
        return _StubAdapter()

    monkeypatch.setattr(cli_mod, "_build_adapter", build)
    monkeypatch.delenv("AUTOPI_TOOL_CALLING", raising=False)
    monkeypatch.delenv("AUTOPI_INJECTION_PLACEMENT", raising=False)
    return seen


def _dry_run(*extra, benchmark="agentdojo", target="vllm:8000:m"):
    return CliRunner().invoke(main, ["run", "--benchmark", benchmark, "--target", target, "--dry-run", *extra])


def test_cli_defaults_to_native_tools_and_agentdojo_placement(captured_build):
    result = _dry_run()
    assert result.exit_code == 0, result.output
    assert captured_build["target_endpoint"].tool_mode == "native"
    assert captured_build["injection_placement"] == "agentdojo"
    assert "tool_calling=native placement=agentdojo" in result.output


def test_cli_tool_calling_flag_and_environment(captured_build, monkeypatch):
    assert _dry_run("--tool-calling", "prompted").exit_code == 0
    assert captured_build["target_endpoint"].tool_mode == "prompted"
    monkeypatch.setenv("AUTOPI_TOOL_CALLING", "prompted")
    assert _dry_run().exit_code == 0
    assert captured_build["target_endpoint"].tool_mode == "prompted"
    assert _dry_run("--tool-calling", "native").exit_code == 0          # the flag wins over the env
    assert captured_build["target_endpoint"].tool_mode == "native"
    monkeypatch.setenv("AUTOPI_TOOL_CALLING", "text")
    result = _dry_run()
    assert result.exit_code == 2 and "AUTOPI_TOOL_CALLING" in result.output


def test_cli_tool_calling_overrides_a_named_alias(captured_build, tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text("models:\n  t:\n    provider: openai-compatible\n    model: m\n"
                      "    base_url: http://localhost:8000/v1\n    tool_mode: prompted\n")
    assert _dry_run("--models-config", str(config), target="t").exit_code == 0
    assert captured_build["target_endpoint"].tool_mode == "prompted"       # the alias's own mode
    assert _dry_run("--models-config", str(config), "--tool-calling", "native", target="t").exit_code == 0
    assert captured_build["target_endpoint"].tool_mode == "native"


def test_cli_rejects_prompted_tools_for_a_hosted_target(captured_build):
    result = _dry_run("--tool-calling", "prompted", target="anthropic:claude-test")
    assert result.exit_code == 2 and "prompted" in result.output


def test_cli_injection_placement_flag_and_environment(captured_build, monkeypatch):
    for mode in ("all", "first", "agentdojo"):
        result = _dry_run("--injection-placement", mode)
        assert result.exit_code == 0 and captured_build["injection_placement"] == mode
        assert f"placement={mode}" in result.output
    monkeypatch.setenv("AUTOPI_INJECTION_PLACEMENT", "all")
    assert _dry_run().exit_code == 0 and captured_build["injection_placement"] == "all"
    monkeypatch.setenv("AUTOPI_INJECTION_PLACEMENT", "everywhere")
    result = _dry_run()
    assert result.exit_code == 2 and "AUTOPI_INJECTION_PLACEMENT" in result.output


@pytest.mark.parametrize("flag", [("--tool-calling", "prompted"), ("--tool-calling", "native"),
                                  ("--injection-placement", "all")])
def test_cli_ipi_arena_refuses_the_agentdojo_options(captured_build, flag):
    result = _dry_run(*flag, "--seed-source", "agentdojo", benchmark="ipiarena")
    assert result.exit_code == 2 and f"{flag[0]} applies to AgentDojo; IPI Arena calls tools natively" in result.output


def test_cli_ipi_arena_names_both_agentdojo_options_when_both_are_given(captured_build):
    result = _dry_run("--tool-calling", "native", "--injection-placement", "all", "--seed-source", "agentdojo",
                      benchmark="ipiarena")
    assert result.exit_code == 2 and "--tool-calling and --injection-placement apply to AgentDojo" in result.output


def test_cli_ipi_arena_ignores_the_agentdojo_environment_options(captured_build, monkeypatch):
    monkeypatch.setenv("AUTOPI_TOOL_CALLING", "prompted")
    monkeypatch.setenv("AUTOPI_INJECTION_PLACEMENT", "all")
    result = _dry_run("--seed-source", "agentdojo", benchmark="ipiarena")
    assert result.exit_code == 0, result.output
    assert captured_build["target_endpoint"].tool_mode == "native"
    assert "placement=" not in result.output


def test_cli_raw_target_recovery_points_to_prompted_tool_calling(tmp_path, monkeypatch):
    pytest.importorskip("agentdojo")
    monkeypatch.delenv("AUTOPI_TOOL_CALLING", raising=False)
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    result = CliRunner().invoke(main, ["run", "--benchmark", "agentdojo", "--target", "vllm:8000:m",
                                       "--target-tokenizer", str(tokenizer), "--suite", "banking", "--dry-run"])
    assert result.exit_code == 2 and "--tool-calling prompted" in result.output


def test_run_manifest_records_tool_calling_and_placement(captured_build, monkeypatch, tmp_path):
    monkeypatch.setattr(engine_module, "query_refiner_llm",
                        lambda *a, **k: "[TEMPLATE_START] Complete the new task now. [TEMPLATE_END]")
    result = CliRunner().invoke(main, ["run", "--benchmark", "agentdojo", "--target", "vllm:8000:m",
                                       "--tool-calling", "prompted", "--injection-placement", "all",
                                       "--no-judge", "--no-context", "--no-legitimize", "--logdir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    settings = json.loads((tmp_path / "run_manifest.json").read_text())["settings"]
    assert settings["tool_calling"] == "prompted" and settings["injection_placement"] == "all"
    assert settings["models"]["target"]["tool_mode"] == "prompted"


# --- injection placement ---------------------------------------------------------------------

def _adapter_for(suite_name, suite, user, placement):
    from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
    from agentdojo.attacks.base_attacks import FixedJailbreakAttack

    from autopi.adapters.agentdojo import AgentDojoAdapter

    pipeline = GroundTruthPipeline(user)
    adapter = AgentDojoAdapter.__new__(AgentDojoAdapter)
    adapter._pipeline = pipeline
    adapter._suites = {suite_name: suite}
    adapter._selectors = {suite_name: FixedJailbreakAttack("{goal}", suite, pipeline)}
    adapter.journal = None
    adapter.injection_placement = placement
    return adapter


def test_placement_modes_against_agentdojos_own_slot_choice_for_every_user_task():
    pytest.importorskip("agentdojo")
    from agentdojo.attacks.base_attacks import FixedJailbreakAttack
    from agentdojo.task_suite.load_suites import get_suites

    checked = 0
    for suite_name, suite in get_suites("v1.2.2").items():
        injection = next(iter(suite.injection_tasks.values()))
        vectors = list(suite.get_injection_vector_defaults())
        for user in suite.user_tasks.values():
            native = FixedJailbreakAttack("{goal}", suite, None).get_injection_candidates(user)
            point = AttackPoint(suite_name, user.ID, injection.ID, user.PROMPT, injection.GOAL,
                                {"_ut": user, "_it": injection})
            expected = {"agentdojo": native, "all": vectors, "first": native[:1]}
            for mode, slots in expected.items():
                assert _adapter_for(suite_name, suite, user, mode).injection_slots(point) == slots
            assert set(native) <= set(vectors) and len(native) >= 1
            checked += 1
    assert checked == 97


@pytest.mark.parametrize("mode", ["agentdojo", "all", "first"])
def test_each_placement_runs_through_agentdojos_own_execution_and_scoring(mode):
    pytest.importorskip("agentdojo")
    from agentdojo.task_suite.load_suites import get_suites

    suite = get_suites("v1.2.2")["slack"]
    user = suite.user_tasks["user_task_1"]
    injection = suite.injection_tasks["injection_task_1"]
    adapter = _adapter_for("slack", suite, user, mode)
    point = AttackPoint("slack", user.ID, injection.ID, user.PROMPT, injection.GOAL, {"_ut": user, "_it": injection})
    slots = adapter.injection_slots(point)
    native = suite.run_task_with_pipeline(adapter._pipeline, user, injection, {slot: "TEST PAYLOAD" for slot in slots})
    result = adapter.run(point, "TEST PAYLOAD")
    assert (result.utility, result.security) == native
    assert result.info == {"injection_slots": slots, "injection_placement": mode}


def test_cli_placement_choices_match_the_adapter():
    pytest.importorskip("agentdojo")
    from autopi.adapters.agentdojo import PLACEMENT_MODES

    assert cli_mod.INJECTION_PLACEMENTS == PLACEMENT_MODES


def test_unknown_placement_is_refused():
    pytest.importorskip("agentdojo")
    from autopi.adapters.agentdojo import AgentDojoAdapter

    with pytest.raises(ValueError, match="injection placement"):
        AgentDojoAdapter("vllm:8000:m", suites=["banking"], injection_placement="everywhere")
