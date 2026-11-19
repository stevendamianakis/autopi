"""Generic driver loop — ties a :class:`TaxonomyEngine` to any :class:`BenchmarkAdapter`.

For each data point it runs: reset → (seed, then sweep/paraphrase) attempts until the
injection succeeds or the attempt budget is exhausted, following the
contract (``build_injection`` → ``observe`` → ``mark_*`` → ``refine_template``), and writes
the same per-pair ``refiner_logs`` JSON. It returns per-point outcomes and an ASR summary.

Stop rule. By default a data point stops at its first success, which is all that ASR needs.
With ``stop_at_first_success=False`` the point runs its whole schedule, so the log lists
every attempt that succeeded: use it for the technique coverage of a scenario, or for an
attempt-level ASR that is comparable with attackers that keep sampling after a success.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from autopi.adapters.base import AdapterInfraError, AttackPoint, BenchmarkAdapter
from autopi.core.engine import TaxonomyEngine


@dataclass
class PointOutcome:
    point: AttackPoint
    succeeded: bool
    attempts: int                 # target calls actually made (= attempt index of success, or budget)
    solved_sequence_id: str | None = None
    solved_phase: str | None = None
    # The fields below differ from the ones above only when the point kept going after a success.
    first_success_attempt: int | None = None     # attempt index of the first success
    successful_attempts: int = 0                 # how many attempts succeeded
    solved_sources: list[str] = field(default_factory=list)   # seed / sub-techniques with a success


@dataclass
class RunSummary:
    benchmark: str
    awareness_mode: int
    outcomes: list[PointOutcome] = field(default_factory=list)
    stop_at_first_success: bool = True
    # Points whose every attempt was an infrastructure failure: not logged, not in ASR.
    infrastructure_only: list[AttackPoint] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def successes(self) -> int:
        return sum(1 for o in self.outcomes if o.succeeded)

    @property
    def asr(self) -> float:
        return self.successes / self.n if self.n else 0.0

    def asr_at(self, k: int) -> float:
        """ASR@k: fraction solved within the first ``k`` target attempts."""
        if not self.n:
            return 0.0
        return sum(
            1 for o in self.outcomes
            if o.succeeded and (o.first_success_attempt if o.first_success_attempt is not None else o.attempts) <= k
        ) / self.n

    @property
    def attempt_level_asr(self) -> float:
        """Successful attempts over all attempts made. Comparable across attackers only for
        runs without early stopping, where a point can contribute more than one success."""
        made = sum(o.attempts for o in self.outcomes)
        return sum(o.successful_attempts for o in self.outcomes) / made if made else 0.0

    @property
    def covered_subtechniques(self) -> set[str]:
        """Sub-techniques with at least one success on at least one point (a paraphrase
        counts toward the line it reworks; a stacked line credits each of its parts)."""
        return {sid for o in self.outcomes for src in o.solved_sources if src != "SEED" for sid in src.split("+")}

    def format(self) -> str:
        lines = [
            f"benchmark={self.benchmark} mode={self.awareness_mode} points={self.n}",
            f"ASR={self.asr:.3f} ({self.successes}/{self.n})",
            f"ASR@1={self.asr_at(1):.3f}  ASR@10={self.asr_at(10):.3f}  ASR@100={self.asr_at(100):.3f}",
        ]
        if not self.stop_at_first_success:
            made = sum(o.attempts for o in self.outcomes)
            hits = sum(o.successful_attempts for o in self.outcomes)
            lines.append(
                f"no early stop: attempt-level ASR={self.attempt_level_asr:.4f} ({hits}/{made})  "
                f"sub-techniques with a success={len(self.covered_subtechniques)}"
            )
        if self.infrastructure_only:
            lines.append(f"only infrastructure failures, left out of ASR and not logged "
                         f"(--resume runs them again): {len(self.infrastructure_only)}")
        return "\n".join(lines)


def _save_criteria_log(logdir: Path, point: AttackPoint, infos: list, succeeded: bool) -> None:
    # Side-log per-attempt criterion detail for adapters that expose it (IPI Arena: full
    # is_break vs concealment-excluded action_only), so a single run yields both ASRs.
    d = logdir / "criteria_logs" / point.suite
    d.mkdir(parents=True, exist_ok=True)
    rec = {
        "suite": point.suite, "user_task": point.user_task_id, "injection_task": point.injection_task_id,
        "succeeded": succeeded,
        "any_is_break": any(bool(i.get("is_break")) for i in infos),
        "any_action_only": any(bool(i.get("action_only")) for i in infos),
        "num_attempts": len(infos), "attempts": infos,
    }
    (d / f"{point.user_task_id}__{point.injection_task_id}.json").write_text(json.dumps(rec, indent=1))


def run_attack(
    adapter: BenchmarkAdapter,
    engine: TaxonomyEngine,
    *,
    logdir: str | Path | None = None,
    max_attempts: int | None = None,
    points: Iterable[AttackPoint] | None = None,
    verbose: bool = True,
    resume: bool = False,
    stop_at_first_success: bool = True,
    journal=None,
) -> RunSummary:
    """Drive ``engine`` against ``adapter`` over its data points.

    Args:
        adapter: the benchmark backend.
        engine: a configured :class:`TaxonomyEngine` (mode, stacking, mechanisms, refiner).
        logdir: where to write per-pair ``refiner_logs`` JSON (None disables logging).
        max_attempts: optional hard cap on target calls per point (on top of the engine's
            own schedule budget, ``engine.attempt_budget``). For the paper's L1 setting the
            schedule is the seed plus the 132-technique sweep, 133 calls; leave this None,
            or set a number to bound it explicitly.
        points: override the points to attack (defaults to ``adapter.iter_points()``).
        verbose: print a one-line result per point.
        resume: skip data points whose per-pair log already exists.
        stop_at_first_success: end a data point at its first success (default). Set False
            to run the point's whole schedule and record every success, for the technique
            coverage of a scenario or an attempt-level ASR without early stopping.
    """
    logdir = Path(logdir) if logdir else None
    summary = RunSummary(benchmark=adapter.name, awareness_mode=engine.awareness_mode,
                         stop_at_first_success=stop_at_first_success)
    pts = list(points) if points is not None else list(adapter.iter_points())

    for point in pts:
        if journal:
            journal.context = {"suite": point.suite, "user_task": point.user_task_id,
                               "injection_task": point.injection_task_id}
            journal.emit("point_start", budget=min(max_attempts or engine.attempt_budget, engine.attempt_budget))
        log_file = (
            logdir / "refiner_logs" / point.suite / f"{point.user_task_id}__{point.injection_task_id}.json"
            if logdir is not None else None
        )
        if resume and log_file is not None and log_file.exists():
            try:
                prev = json.loads(log_file.read_text())
                done_ok, done_att = bool(prev.get("injection_succeeded")), int(prev.get("total_attempts", 0) or 0)
            except (ValueError, TypeError, OSError) as exc:
                raise ValueError(f"cannot resume invalid scenario log {log_file}") from exc
            summary.outcomes.append(PointOutcome(
                point=point, succeeded=done_ok, attempts=done_att,
                solved_sequence_id=prev.get("solved_sequence_id"), solved_phase=prev.get("solved_phase"),
                first_success_attempt=prev.get("first_success_attempt"),
                successful_attempts=int(prev.get("num_successes", 1 if done_ok else 0) or 0),
                solved_sources=list(prev.get("solved_sources") or []),
            ))
            if verbose:
                # Logs written before the stop rule was recorded came from early-stopped runs.
                note = ""
                if not stop_at_first_success and prev.get("stop_at_first_success", True):
                    note = ", with early stopping: use a fresh --logdir to redo it without"
                print(f"[skip] {adapter.name}/{point.suite}/{point.user_task_id}+{point.injection_task_id} "
                      f"(already done{note})")
            continue

        engine.reset(point.user_task_prompt, point.injection_goal)
        succeeded = False
        attempts = 0
        infrastructure_failures = 0
        first_success_attempt: int | None = None
        infos: list = []
        # has_candidate is False only in no-seed mode when no first candidate could be written.
        while engine.has_candidate:
            attempts += 1
            injection_text = engine.build_injection(point.injection_goal)
            if journal:
                journal.context["attempt"] = attempts
                journal.emit("attempt_start", injection_text=injection_text, candidate=engine._attempt_meta[-1])
            infra_kind: str | None = None
            try:
                result = adapter.run(point, injection_text)
            except AdapterInfraError as e:
                result, infra_kind = None, (str(e)[:40] or "infra")
            if journal:
                journal.emit("attempt_end", result=result, infrastructure_error=infra_kind)
            # Budget check in the original loop's order: before this attempt is recorded, so
            # a point gets engine.attempt_budget target calls (the seed plus the schedule).
            more = engine.has_attempts_remaining()
            if result is None:
                infrastructure_failures += 1
                engine.mark_attempt_infra_failed(infra_kind or "infra")
            else:
                if result.info and ("action_only" in result.info or "is_break" in result.info):
                    infos.append({"attempt": attempts, "is_break": result.info.get("is_break"),
                                  "action_only": result.info.get("action_only"), "error": result.info.get("error")})
                engine.observe(result.messages, result.utility, result.security)
                if result.security:
                    engine.mark_succeeded()
                    succeeded = True
                    if first_success_attempt is None:
                        first_success_attempt = attempts
                    if stop_at_first_success:
                        break
                else:
                    engine.mark_template_failed()
            if max_attempts is not None and attempts >= max_attempts:
                break
            if not more:
                break
            if not engine.refine_template(point.injection_goal):
                break

        if attempts and infrastructure_failures == attempts:
            # The target never answered: no measurement. Leave the point for --resume.
            summary.infrastructure_only.append(point)
            if journal:
                journal.emit("point_end", succeeded=False, attempts=attempts, infrastructure_only=True)
            if verbose:
                print(f"[infra] {adapter.name}/{point.suite}/{point.user_task_id}+{point.injection_task_id}: "
                      f"all {attempts} attempt(s) failed in the infrastructure; not logged, --resume runs it again")
            continue

        if logdir is not None:
            engine.save_template_log(logdir, point.suite, point.user_task_id, point.injection_task_id, succeeded,
                                     stop_at_first_success=stop_at_first_success)
            if infos:
                _save_criteria_log(logdir, point, infos, succeeded)

        solved_all = [m for m in engine._attempt_meta if m["status"] == "success"]
        solved = solved_all[0] if solved_all else None
        solved_sources: list[str] = []
        for m in solved_all:
            src = m["sequence_id"].removeprefix("PARA:")
            if src not in solved_sources:
                solved_sources.append(src)
        outcome = PointOutcome(
            point=point,
            succeeded=succeeded,
            attempts=attempts,
            solved_sequence_id=solved["sequence_id"] if solved else None,
            solved_phase=solved["phase"] if solved else None,
            first_success_attempt=first_success_attempt,
            successful_attempts=len(solved_all),
            solved_sources=solved_sources,
        )
        summary.outcomes.append(outcome)
        if journal:
            journal.emit("point_end", succeeded=succeeded, attempts=attempts)
        if verbose:
            tag = "HIT " if succeeded else "miss"
            seq = f" via {outcome.solved_sequence_id}" if succeeded else ""
            if succeeded and not stop_at_first_success:
                seq = (f", first at attempt {first_success_attempt} via {outcome.solved_sequence_id}, "
                       f"{len(solved_all)} successful attempt(s) over {len(solved_sources)} source(s)")
            print(f"[{tag}] {adapter.name}/{point.suite}/{point.user_task_id}+{point.injection_task_id} "
                  f"in {attempts} attempt(s){seq}")

    return summary
