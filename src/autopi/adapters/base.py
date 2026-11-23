"""The benchmark-integration contract.

A :class:`BenchmarkAdapter` is the *only* thing the AutoPI engine needs to attack a new
benchmark. The engine stays benchmark-agnostic; each adapter knows how to enumerate a
benchmark's data points, place an injection, run the target agent, and score the outcome.

The contract is deliberately tiny — four things:

1. ``iter_points()`` — the ``(user_task, injection_task)`` data points to attack.
2. ``run(point, injection_text)`` — place ``injection_text`` at the benchmark's injection
   point, run the target agent once, and return a :class:`RunResult`.
3. each point carries ``user_task_prompt`` (for modes 2/3) and ``injection_goal`` (for mode
   3 and the judge).
4. ``checker_kind`` — whether (utility, security) come from programmatic state checks
   (AgentDojo) or an LLM judge (IPI Arena); informational, for reporting.

AgentDojo implements this over their stateful multi-step environment and
state-based checkers; IPI Arena implements it as a single ``{fill}`` placement, one target
turn, and its tool-judge + concealment-judge scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class AttackPoint:
    """One (user_task, injection_task) data point to attack."""

    suite: str                     # suite/scenario name (used for log paths + reporting)
    user_task_id: str
    injection_task_id: str
    user_task_prompt: str          # the agent's current task text (modes 2/3); "" if unknown
    injection_goal: str            # the attacker's goal string (mode 3 + judge)
    extra: dict = field(default_factory=dict)  # adapter-private per-point payload


@dataclass
class RunResult:
    """Outcome of placing one injection and running the target agent once."""

    utility: bool                  # did the agent still complete the user's task?
    security: bool                 # did the injection succeed (attacker goal achieved)?
    messages: list                 # the agent transcript (role/content/tool_calls dicts)
    info: dict = field(default_factory=dict)  # optional extras (judge scores, errors, ...)


class AdapterInfraError(RuntimeError):
    """Raised by ``run`` when an attempt cannot be scored fairly: a transport, provider, parse or
    harness error, whether or not the agent had already read the injection. The driver counts it
    against the budget via ``mark_attempt_infra_failed`` and keeps it out of the generator's
    failed-line history.

    ``endpoint_error`` is True when the endpoint itself failed: it could not be reached, timed
    out, was rate limited or failed on the server, or rejected the request (an unsupported
    parameter, a missing tool-call parser). Such an error tends to recur on every scenario. It
    is False only for an error tied to this scenario, which another scenario would not hit: a
    context overflow, a malformed tool call, a broken environment. Only endpoint errors count
    toward the driver's stop when the target keeps failing. ``info`` holds optional
    diagnostics that the driver writes to the scenario log."""

    def __init__(self, kind: str = "infra", *, endpoint_error: bool = True, info: dict | None = None):
        super().__init__(kind)
        self.endpoint_error = endpoint_error
        self.info = dict(info or {})


class BenchmarkModelConfigError(RuntimeError):
    """A model the benchmark calls (the target, or IPI Arena's judge or tool simulator) rejected
    its credentials, its account or its model id (HTTP 401, 402, 403 other than a moderation
    block, 404, or an invalid API key). Every scenario would fail the same way, so the run
    stops instead of recording infrastructure failures."""

    fatal = True


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """What the engine needs from a benchmark. Implemented by the two concrete adapters."""

    name: str
    checker_kind: str  # "state" (programmatic) | "judge" (LLM) | "hybrid"
    # The benchmark's own default fixed seed, or None when it ships none (e.g. IPI Arena).
    # A fixed seed is optional; see autopi.core.seeds.resolve_seed for how a run chooses one.
    default_seed_template: str | None
    default_seed_name: str | None

    def iter_points(self) -> Iterator[AttackPoint]:
        """Yield the data points to attack (optionally filtered by the adapter's config)."""
        ...

    def run(self, point: AttackPoint, injection_text: str) -> RunResult:
        """Place ``injection_text`` at ``point``'s injection location, run the target agent
        once, and return the (utility, security, transcript). Raise :class:`AdapterInfraError`
        if the attempt cannot be scored fairly."""
        ...
