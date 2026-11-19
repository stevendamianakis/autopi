"""AgentDojo adapter using native task-visible slots, execution and scoring."""

from __future__ import annotations

from typing import Iterator

from autopi.adapters.base import AdapterInfraError, AttackPoint, RunResult
from autopi.core.seeds import IMPORTANT_INSTRUCTIONS_SEED, IMPORTANT_INSTRUCTIONS_SEED_NAME


def _to_plain_messages(messages) -> list[dict]:
    """Convert AgentDojo ChatMessages (TypedDicts with FunctionCall tool_calls) into the
    plain role/content/tool_calls dicts the engine's trace/context helpers expect."""
    out: list[dict] = []
    for m in messages or []:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        raw_tcs = (m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)) or []
        tcs = []
        for tc in raw_tcs:
            if isinstance(tc, dict):
                fn = tc.get("function")
                args = tc.get("args") or tc.get("arguments") or {}
            else:
                fn = getattr(tc, "function", None)
                args = getattr(tc, "args", None) or getattr(tc, "arguments", None) or {}
            tcs.append({"function": fn, "args": args})
        out.append({"role": role, "content": content, "tool_calls": tcs})
    return out


class AgentDojoAdapter:
    """Attack AgentDojo suites. ``suites`` selects which suites to attack (default: all in the
    benchmark version); ``injection_tasks`` / ``user_tasks`` optionally narrow the pairs."""

    checker_kind = "state"
    # AgentDojo ships the important-instructions attack; it is this benchmark's default seed
    # See autopi.core.seeds.
    default_seed_template = IMPORTANT_INSTRUCTIONS_SEED
    default_seed_name = IMPORTANT_INSTRUCTIONS_SEED_NAME

    def __init__(
        self,
        target: str,
        *,
        benchmark_version: str = "v1.2.2",
        suites: list[str] | None = None,
        user_tasks: list[str] | None = None,
        injection_tasks: list[str] | None = None,
        defense: str | None = None,
        name: str = "agentdojo",
        target_thinking: str | None = None,
        target_max_tokens: int | None = None,
        endpoint=None,
        journal=None,
    ) -> None:
        # Imported lazily so `import autopi` never hard-requires agentdojo.
        from agentdojo.task_suite.load_suites import get_suites

        from autopi.models.registry import build_target_pipeline

        self.name = name
        self.journal = journal
        self.benchmark_version = benchmark_version
        # Thinking mode and output cap of a locally served target (default: thinking off, no cap).
        self._pipeline = build_target_pipeline(
            target, defense=defense, target_thinking=target_thinking, target_max_tokens=target_max_tokens,
            endpoint=endpoint, journal=journal,
        )
        all_suites = get_suites(benchmark_version)
        chosen = suites or list(all_suites.keys())
        missing = [s for s in chosen if s not in all_suites]
        if missing:
            raise ValueError(f"unknown suite(s) {missing}; available: {sorted(all_suites)}")
        self._suites = {name: all_suites[name] for name in chosen}
        self._user_task_filter = set(user_tasks) if user_tasks else None
        self._injection_task_filter = set(injection_tasks) if injection_tasks else None
        for label, selected, known in [
            ("user task", self._user_task_filter, {key for suite in self._suites.values() for key in suite.user_tasks}),
            ("injection task", self._injection_task_filter, {key for suite in self._suites.values() for key in suite.injection_tasks}),
        ]:
            if selected and selected - known:
                raise ValueError(f"unknown {label} id(s): {sorted(selected - known)}")
        # Cache each suite's injection-vector slot ids.
        from agentdojo.attacks.base_attacks import FixedJailbreakAttack

        self._selectors = {name: FixedJailbreakAttack("{goal}", suite, self._pipeline)
                           for name, suite in self._suites.items()}

    def iter_points(self) -> Iterator[AttackPoint]:
        for sname, suite in self._suites.items():
            for ut_id, ut in suite.user_tasks.items():
                if self._user_task_filter and ut_id not in self._user_task_filter:
                    continue
                for it_id, it in suite.injection_tasks.items():
                    if self._injection_task_filter and it_id not in self._injection_task_filter:
                        continue
                    yield AttackPoint(
                        suite=sname,
                        user_task_id=ut_id,
                        injection_task_id=it_id,
                        user_task_prompt=getattr(ut, "PROMPT", "") or "",
                        injection_goal=getattr(it, "GOAL", "") or "",
                        extra={"_ut": ut, "_it": it},
                    )

    def run(self, point: AttackPoint, injection_text: str) -> RunResult:
        from agentdojo.agent_pipeline.errors import AbortAgentError
        from agentdojo.logging import Logger

        adapter = self

        class RecordingPipeline:
            messages = ()

            def query(self, *args, **kwargs):
                try:
                    result = adapter._pipeline.query(*args, **kwargs)
                except AbortAgentError as exc:
                    self.messages = exc.messages
                    raise
                self.messages = result[3]
                return result

        class TraceLogger(Logger):
            def log(self, messages, *args, **kwargs):
                if adapter.journal:
                    adapter.journal.emit("agent_messages", messages=messages)

            def log_error(self, *args, **kwargs):
                if adapter.journal:
                    adapter.journal.emit("agent_error", details=args)

        pipeline = RecordingPipeline()
        try:
            slots = list(self._selectors[point.suite].get_injection_candidates(point.extra["_ut"]))
            if not slots:
                raise AdapterInfraError("no_native_injection_slots")
            # AgentDojo substitutes each slot into a double-quoted YAML scalar
            # (``field: "{slot}"``) and then parses the result, so escape the payload's quotes
            # and backslashes. The target still sees the original text after YAML load; without
            # this a quote in a seed or generated line would break the environment parse.
            from autopi.core.prompts import yaml_escape

            injections = {slot: yaml_escape(injection_text) for slot in slots}
            if self.journal:
                self.journal.emit("injection_placement", slots=slots, injection_text=injection_text)
            with TraceLogger():
                utility, security = self._suites[point.suite].run_task_with_pipeline(
                    pipeline, point.extra["_ut"], point.extra["_it"], injections,
                )
        except Exception as exc:
            if getattr(exc, "fatal", False):
                raise
            kind = _classify_infra_error(exc)
            if kind is not None:
                raise AdapterInfraError(kind) from exc
            raise
        return RunResult(utility=bool(utility), security=bool(security),
                         messages=_to_plain_messages(pipeline.messages), info={"injection_slots": slots})


def _classify_infra_error(e: Exception) -> str | None:
    """Map provider/parse errors that abort before the agent fairly saw the injection to an
    infra-failure kind (counted against the budget but kept out of the refiner's history).
    Errors of every provider client count (OpenAI, Anthropic, Google), also when AgentDojo's
    Anthropic client hands one over wrapped in tenacity's ``RetryError`` after its retries."""
    import json

    import yaml

    if type(e).__module__.split(".")[0] == "tenacity":
        attempt = getattr(e, "last_attempt", None)
        inner = attempt.exception() if attempt is not None else None
        if isinstance(inner, Exception):
            return _classify_infra_error(inner)
    if isinstance(e, (TimeoutError, ConnectionError)):
        return "transport_error"
    if isinstance(e, json.JSONDecodeError):
        # A native tool call whose JSON arguments were cut off (e.g. a token cap) fails
        # AgentDojo's own parser. Treat it as a model crash, not a blocked attack: count it
        # against the budget and keep it out of the refiner's failed-template history.
        return "model_crash"
    module = type(e).__module__
    if module.split(".")[0] in {"openai", "anthropic", "httpx", "requests"} or \
            module.startswith(("google.genai", "google.api_core")):
        return "provider_error"
    msg = repr(e)
    if isinstance(e, yaml.YAMLError):
        return "yaml_error"
    low = msg.lower()
    if "context_length" in low or "max_tokens" in low or "max_new_tokens" in low or "reduce the length" in low:
        return "context_length"
    if "internal server error" in low or "server error" in low:
        return "server_error"
    if "invalidmodeloutput" in low or "model output" in low:
        return "model_crash"
    return None
