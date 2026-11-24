"""AgentDojo adapter: AgentDojo's own execution and scoring, with a selectable injection
placement whose default is AgentDojo's (the task-visible injection vectors)."""

from __future__ import annotations

import re
from typing import Iterator

from autopi.adapters.base import AdapterInfraError, AttackPoint, RunResult
from autopi.core.seeds import IMPORTANT_INSTRUCTIONS_SEED, IMPORTANT_INSTRUCTIONS_SEED_NAME

# Where the injection goes in a scenario's environment:
#   agentdojo  the injection vectors AgentDojo's own attacks fill: the ones the user task's
#              ground-truth run reads (``get_injection_candidates``). The default.
#   all        every injection vector of the suite, including ones the user task never reads.
#   first      only the first of AgentDojo's vectors, in the suite's vector order.
PLACEMENT_MODES = ("agentdojo", "all", "first")


# Content blocks that hold a target's reasoning (Anthropic's thinking and redacted thinking).
# The attacker observes the visible replies and tool calls only, so these blocks are left out
# of the transcript the engine's judge and context blending read. AgentDojo still scores the
# run on its own, unchanged messages.
_REASONING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})


def _visible_content(content):
    if isinstance(content, list):
        return [block for block in content
                if not (isinstance(block, dict) and block.get("type") in _REASONING_BLOCK_TYPES)]
    return content


def _to_plain_messages(messages) -> list[dict]:
    """Convert AgentDojo ChatMessages (TypedDicts with FunctionCall tool_calls) into the
    plain role/content/tool_calls dicts the engine's trace/context helpers expect, without
    reasoning blocks."""
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
        out.append({"role": role, "content": _visible_content(content), "tool_calls": tcs})
    return out


class AgentDojoAdapter:
    """Attack AgentDojo suites. ``suites`` selects which suites to attack (default: all in the
    benchmark version); ``injection_tasks`` / ``user_tasks`` optionally narrow the pairs."""

    checker_kind = "state"
    injection_placement = "agentdojo"
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
        injection_placement: str = "agentdojo",
    ) -> None:
        # Imported lazily so `import autopi` never hard-requires agentdojo.
        from agentdojo.task_suite.load_suites import get_suites

        from autopi.models.registry import build_target_pipeline

        if injection_placement not in PLACEMENT_MODES:
            raise ValueError(f"injection placement must be one of {', '.join(PLACEMENT_MODES)}; "
                             f"got {injection_placement!r}")
        self.name = name
        self.journal = journal
        self.benchmark_version = benchmark_version
        self.injection_placement = injection_placement
        # The target's thinking mode and output cap, when given, override the endpoint's own.
        self._pipeline = build_target_pipeline(
            target, defense=defense, target_thinking=target_thinking, target_max_tokens=target_max_tokens,
            endpoint=endpoint, journal=journal,
        )
        all_suites = get_suites(benchmark_version)
        if not all_suites:
            from agentdojo.task_suite import load_suites

            known = sorted(name for name, suites_of in getattr(load_suites, "_SUITES", {}).items() if suites_of)
            raise ValueError(f"AgentDojo has no suites for benchmark version {benchmark_version!r}"
                             + (f"; known versions: {', '.join(known)}" if known else ""))
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

    def injection_slots(self, point: AttackPoint) -> list[str]:
        """The injection vectors this scenario's payload is written into (see PLACEMENT_MODES)."""
        if self.injection_placement == "all":
            return list(self._suites[point.suite].get_injection_vector_defaults())
        slots = list(self._selectors[point.suite].get_injection_candidates(point.extra["_ut"]))
        return slots[:1] if self.injection_placement == "first" else slots

    def run(self, point: AttackPoint, injection_text: str) -> RunResult:
        from agentdojo.agent_pipeline.errors import AbortAgentError
        from agentdojo.logging import Logger

        adapter = self

        class RecordingPipeline:
            """Runs the configured pipeline and keeps the messages of every agent run. AgentDojo
            restarts an agent that ends without a final answer (up to three runs in one attempt),
            so the runs are kept in order. When the target fails during a run, the run is ended
            the way AgentDojo ends one that a defense aborts, so that AgentDojo scores the state
            the agent reached; :meth:`run` then decides how the attempt counts."""

            def __init__(self):
                self.runs: list[list] = []
                self.logged: list | None = None
                self.interrupted: tuple[str, Exception] | None = None

            @property
            def messages(self) -> list:
                return [message for run in self.runs for message in run]

            def query(self, *args, **kwargs):
                env = args[2] if len(args) > 2 else kwargs.get("env")
                self.logged = None
                try:
                    result = adapter._pipeline.query(*args, **kwargs)
                except AbortAgentError as exc:
                    self.runs.append(list(exc.messages))
                    raise
                except Exception as exc:
                    if getattr(exc, "fatal", False):
                        raise
                    _raise_if_rejected_configuration(exc)
                    kind = _classify_infra_error(exc)
                    if kind is None or env is None:
                        raise
                    # Keep what the agent did before the error. AgentDojo's abort appends a final
                    # reply holding its message; an empty message makes that reply empty, so the
                    # output AgentDojo scores is empty too.
                    abort = AbortAgentError("", list(self.logged or []), env)
                    self.runs.append(list(abort.messages))
                    self.interrupted = (kind, exc)
                    raise abort from exc
                self.runs.append(list(result[3]))
                return result

        class TraceLogger(Logger):
            def log(self, messages, *args, **kwargs):
                pipeline.logged = list(messages)
                if adapter.journal:
                    adapter.journal.emit("agent_messages", messages=messages)

            def log_error(self, *args, **kwargs):
                if adapter.journal:
                    adapter.journal.emit("agent_error", details=args)

        pipeline = RecordingPipeline()
        try:
            slots = self.injection_slots(point)
            if not slots:
                raise AdapterInfraError("no_native_injection_slots", endpoint_error=False)
            # AgentDojo substitutes each slot into a double-quoted YAML scalar
            # (``field: "{slot}"``) and then parses the result, so escape the payload's quotes
            # and backslashes. The target still sees the original text after YAML load; without
            # this a quote in a seed or generated line would break the environment parse.
            from autopi.core.prompts import yaml_escape

            injections = {slot: yaml_escape(injection_text) for slot in slots}
            if self.journal:
                self.journal.emit("injection_placement", placement=self.injection_placement, slots=slots,
                                  injection_text=injection_text)
            with TraceLogger():
                utility, security = self._suites[point.suite].run_task_with_pipeline(
                    pipeline, point.extra["_ut"], point.extra["_it"], injections,
                )
        except AdapterInfraError:
            raise
        except Exception as exc:
            if getattr(exc, "fatal", False):
                raise
            _raise_if_rejected_configuration(exc)
            kind = _classify_infra_error(exc)
            if kind is not None:
                raise AdapterInfraError(kind, endpoint_error=kind not in _SCENARIO_KINDS) from exc
            raise
        info = {"injection_slots": slots, "injection_placement": self.injection_placement}
        if pipeline.interrupted is not None:
            kind, exc = pipeline.interrupted
            error = f"{type(_unwrap_retry(exc)).__name__}: {str(_unwrap_retry(exc))[:200]}"
            if kind != "content_policy":
                # The run was cut short, so the attempt stays unmeasured, as in the paper's runs.
                # AgentDojo's verdict on the state the agent reached is recorded for analysis
                # and is not counted.
                raise AdapterInfraError(kind, endpoint_error=kind not in _SCENARIO_KINDS,
                                        info={"security_at_error": bool(security), "error": error}) from exc
            # The provider refused one of the run's requests on content-policy grounds, so the
            # injection did not get through that request. The attempt counts, scored by AgentDojo
            # on the state the agent had reached.
            info.update(target_stopped_by="content_policy", target_error=error)
        return RunResult(utility=bool(utility), security=bool(security),
                         messages=_to_plain_messages(pipeline.messages), info=info)


# Infrastructure kinds tied to one scenario or attempt, which another scenario would not hit:
# the target's context overflowed, its tool call was malformed, the environment or the
# injection slots could not be built. Every other kind is an error of the endpoint itself
# (unreachable, rate limited, failing, or rejecting the request) and counts toward the run's
# stop when the target keeps failing.
_SCENARIO_KINDS = frozenset({"context_length", "model_crash", "yaml_error", "no_native_injection_slots"})

_PROVIDER_MODULES = frozenset({"openai", "anthropic", "httpx", "httpx2", "requests"})

# A context overflow, worded as the providers and vLLM word it (and not a request that merely
# names max_tokens, such as "Unsupported parameter: 'max_tokens'").
_CONTEXT_OVERFLOW = re.compile(
    r"context_length_exceeded|context length|maximum context|reduce the length|too many tokens"
    r"|prompt is too long|input is too long|exceeds the (?:model'?s? )?(?:maximum|context)", re.IGNORECASE)
_INVALID_KEY = re.compile(r"API_KEY_INVALID|API key not valid|invalid[ _]api[ _]key|incorrect api key", re.IGNORECASE)


def _unwrap_retry(e: BaseException) -> BaseException:
    """The error inside tenacity's ``RetryError`` (AgentDojo's Anthropic client raises one after
    its retries), else ``e`` itself."""
    if type(e).__module__.split(".")[0] == "tenacity":
        attempt = getattr(e, "last_attempt", None)
        inner = attempt.exception() if attempt is not None else None
        if isinstance(inner, Exception):
            return _unwrap_retry(inner)
    return e


def _is_provider_error(e: BaseException) -> bool:
    module = type(e).__module__
    return module.split(".")[0] in _PROVIDER_MODULES or module.startswith(("google.genai", "google.api_core"))


def _raise_if_rejected_configuration(e: BaseException) -> None:
    """Stop the run when the target rejects its credentials, its account or its model id (HTTP
    401, 402, 403 other than a moderation block, 404, or an invalid API key, which Gemini
    answers with HTTP 400): it would fail the same way on every scenario."""
    from autopi.adapters.base import BenchmarkModelConfigError
    from autopi.models.clients import http_status, is_content_policy_rejection

    error = _unwrap_retry(e)
    if not _is_provider_error(error) or is_content_policy_rejection(error):
        return
    status = http_status(error)
    if status in (401, 402, 403, 404) or (status == 400 and _INVALID_KEY.search(str(error))):
        raise BenchmarkModelConfigError(
            f"the target rejected its credentials, account or model (HTTP {status}: {str(error)[:160]}). "
            "Fix the model id, the API key or the account, then rerun the same command with --resume") from e


def _classify_infra_error(e: BaseException) -> str | None:
    """Map an error that keeps an attempt from being scored fairly to an infrastructure-failure
    kind (counted against the budget, kept out of the generator's history), or to
    ``content_policy`` for a request the provider refused on content-policy grounds (scored as
    a defended attempt). Errors of every provider client count (OpenAI, Anthropic, Google), also
    when AgentDojo's Anthropic client hands one over wrapped in tenacity's ``RetryError``.
    ``None`` for anything else, which is not an infrastructure failure and propagates."""
    import json

    import yaml

    e = _unwrap_retry(e)
    if isinstance(e, (TimeoutError, ConnectionError)):
        return "transport_error"
    if isinstance(e, json.JSONDecodeError):
        # A native tool call whose JSON arguments were cut off (e.g. a token cap) fails
        # AgentDojo's own parser. Treat it as a model crash, not a blocked attack: count it
        # against the budget and keep it out of the generator's history of failed lines.
        return "model_crash"
    if type(e).__name__ == "ValidationError" and getattr(e, "title", None) == "FunctionCall":
        # Valid JSON arguments that are not an object ([], null, "text", 42): AgentDojo cannot
        # build a FunctionCall from them. The same kind of malformed model output as above.
        return "model_crash"
    from autopi.models.clients import http_status, is_content_policy_rejection

    low = repr(e).lower()
    context = bool(_CONTEXT_OVERFLOW.search(repr(e)))
    if _is_provider_error(e):
        if is_content_policy_rejection(e):
            return "content_policy"
        status = http_status(e)
        if status is None and any(word in type(e).__name__ for word in ("Connection", "Timeout", "Connect")):
            return "transport_error"
        if status is not None and (status == 429 or status >= 500):
            return "provider_unavailable"
        return "context_length" if context else "provider_error"
    if isinstance(e, yaml.YAMLError):
        return "yaml_error"
    if context:
        return "context_length"
    if "internal server error" in low or "server error" in low:
        return "server_error"
    if "invalidmodeloutput" in low or "model output" in low:
        return "model_crash"
    return None
