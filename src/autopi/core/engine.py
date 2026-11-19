"""AutoPI explore/exploit state machine — :class:`TaxonomyEngine`.

The engine is decoupled from any benchmark and drives one through an adapter. Schedule:
seed → seed-paraphrase → L1..Lk taxonomy sweep → sequence-local paraphrase, with three
awareness modes, the judge/context/legitimize adaptive mechanisms, stacking, and
resume/strict toggles.

* :meth:`build_injection` (goal string in,
  ready-to-place injection text out) — the goal is substituted into the ``<task_goes_here>``
  placeholder for modes 1/2 and is a no-op for mode 3 lines that already carry it;
* the agent's current task is passed as a plain string (``reset``), not a task object.

A generic driver (``autopi.harness.runner``) calls, per data point:
``reset(user_task_prompt, injection_goal)`` → loop: ``build_injection(goal)`` → run via the
adapter → ``observe(messages, utility, security)`` → stop on success, else
``mark_template_failed()`` + ``refine_template(goal)`` until exhausted → ``save_template_log``.

Search strategy. The default schedule above is one choice. Each phase can be sized or
switched off, and a few switches change what is searched and in what order:

* ``seed_only`` paraphrases the seed and never sweeps; ``use_seed=False`` starts from the
  sweep with no seed at all;
* ``sweep_before_seed_para`` runs the sweep before the seed paraphrases;
* ``techniques`` / ``max_sequences`` restrict the sweep to part of the taxonomy, ``order``
  sets the sweep order (priority, canonical, random), and the ``stack_*`` knobs choose how
  many sub-techniques one injection combines;
* the driver may keep going after a success (``run_attack(stop_at_first_success=False)``):
  :meth:`mark_succeeded` then counts the attempt against the budget, a successful sweep
  candidate is not paraphrased (only failed ones are), and the log lists every success.

"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

from autopi.core.prompts import (
    _ANTI_SUSPICION_NOTE,
    _LEGITIMIZE_NOTE,
    _MODE_PHRASE,
    _NEW_TASK_OUT_OF_PLACE_NOTE,
    _build_judge_prompt,
    _build_paraphrase_prompt,
    _build_taxonomy_prompt,
    _extract_context,
    _extract_template_seq,
    _format_signals,
    _mask_targets,
    _parse_judge,
    _scrub_guidance,
)
from autopi.core.refiner import RefinerConfig, query_refiner_llm
from autopi.core.seeds import (
    IMPORTANT_INSTRUCTIONS_SEED,
    IMPORTANT_INSTRUCTIONS_SEED_NAME,
    SEED_PLACEHOLDER,
)
from autopi.core.taxonomy import (
    build_combo_schedule,
    load_sequences,
    order_sequences,
    select_sequences,
)
from autopi.core.taxonomy import (
    combo_allows_unicode as _combo_allows_unicode,
)
from autopi.core.taxonomy import (
    combo_id as _combo_id,
)
from autopi.core.taxonomy import (
    combo_name as _combo_name,
)

# The default fixed seed is AgentDojo's important-instructions attack. A run can instead pass
# its own ``seed_template`` (a benchmark's own seed, or a researcher-written one), or skip the
# seed entirely (``use_seed=False``). The goal is substituted into ``<task_goes_here>`` at build
# time; for mode 3 the sweep/paraphrase candidates instead carry the goal directly and this
# placeholder seed is still used for the free attempt-1 baseline.
_INITIAL_TEMPLATE = IMPORTANT_INSTRUCTIONS_SEED  # backward-compatible module default


class GeneratorUnavailableError(RuntimeError):
    """The generator (or its diagnostic judge) could not be reached after the endpoint's own
    retries. The run stops before the scenario is logged, so ``--resume`` repeats it."""

    fatal = True


def _generator_unreachable(exc: BaseException) -> bool:
    """A connection failure, timeout, rate limit or server error: the generator is down, as
    opposed to an answer it returned that cannot be used (which is skipped)."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    name = type(exc).__name__
    if "Connection" in name or "Timeout" in name:
        return True
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "code", None)
    return isinstance(status, int) and (status == 429 or status >= 500)


class TaxonomyEngine:
    """Benchmark-agnostic PIAtlas taxonomy explore/exploit attacker.

    All knobs mirror the original constructor and its ``TAX_*`` environment overrides, so
    existing run scripts/presets behave identically. ``awareness_mode`` (or
    ``TAX_AWARENESS_MODE``) selects context-blind mode 1, original-task-aware mode 2, or
    new-task-aware mode 3 (writes the goal directly, no placeholder).
    """

    name = "taxonomy_refined"
    is_dos_attack = False

    def __init__(
        self,
        *,
        refiner_model: str = "",
        paraphrase_reps: int = 1,
        taxonomy_path: str | Path | None = None,
        use_judge: bool | None = None,
        use_context: bool | None = None,
        use_legitimize: bool | None = None,
        stack_depth: int | None = None,
        stack_pool: int | None = None,
        seed_only: bool | None = None,
        stack_random: bool | None = None,
        stack_seed: int | None = None,
        stack_only: int | None = None,
        seed_paraphrase_reps: int | None = None,
        usertask_aware: bool | None = None,
        awareness_mode: int | None = None,
        refiner_config: RefinerConfig | None = None,
        use_seed: bool | None = None,
        sweep_before_seed_para: bool | None = None,
        order: str | None = None,
        order_seed: int | None = None,
        max_sequences: int | None = None,
        techniques: Iterable[str] | str | None = None,
        seed_template: str | None = None,
        seed_name: str | None = None,
    ) -> None:
        if not refiner_model:
            raise ValueError("refiner_model must be explicitly configured")
        self._refiner_config = refiner_config
        seqs = load_sequences(taxonomy_path)
        # ── which sub-techniques the sweep covers, and in what order ──
        # 1) optional subset by family / technique / sub-technique id, 2) sweep order,
        # 3) optional cap to the first N of that order. All default to "everything, priority".
        if techniques is None:
            techniques = os.environ.get("TAX_TECHNIQUES") or None
        if isinstance(techniques, str):
            techniques = techniques.split(",")
        self.techniques = [t.strip().upper() for t in (techniques or []) if t and t.strip()]
        seqs = select_sequences(seqs, self.techniques)
        self.order = (order if order is not None else (os.environ.get("TAX_ORDER") or "priority")).lower()
        self.order_seed = int(order_seed if order_seed is not None else os.environ.get("TAX_ORDER_SEED", "0"))
        seqs = order_sequences(seqs, self.order, self.order_seed)
        _cap = max_sequences if max_sequences is not None else (os.environ.get("TAX_MAX_SEQUENCES") or None)
        self.max_sequences = int(_cap) if _cap is not None else None
        if self.max_sequences is not None:
            seqs = seqs[: self.max_sequences]
        self._sequences = seqs
        self._seq_by_id = {s["id"]: s for s in self._sequences}
        if paraphrase_reps < 0 or (self.max_sequences is not None and self.max_sequences < 1):
            raise ValueError("paraphrase reps must be nonnegative and max_sequences positive")
        self.paraphrase_reps = paraphrase_reps
        # ── stacking ──
        self.stack_depth = int(stack_depth if stack_depth is not None else os.environ.get("TAX_STACK_DEPTH", "1"))
        _pool_env = os.environ.get("TAX_STACK_POOL")
        self.stack_pool = stack_pool if stack_pool is not None else (int(_pool_env) if _pool_env else None)
        self.stack_random = stack_random if stack_random is not None else (os.environ.get("TAX_STACK_RANDOM", "0") != "0")
        self.stack_seed = int(stack_seed if stack_seed is not None else os.environ.get("TAX_STACK_SEED", "0"))
        _only_env = os.environ.get("TAX_STACK_ONLY")
        self.stack_only = int(stack_only) if stack_only is not None else (int(_only_env) if _only_env else 0)
        self.stack_pool_ids: list[str] = []
        self.seed_only = seed_only if seed_only is not None else (os.environ.get("TAX_SEED_ONLY", "0") != "0")
        self.seed_paraphrase_reps = int(
            seed_paraphrase_reps if seed_paraphrase_reps is not None
            else os.environ.get("TAX_SEED_PARA_REPS", "0")
        )
        if self.seed_paraphrase_reps < 0:
            raise ValueError("seed_paraphrase_reps must be nonnegative")
        self.strict_refiner = os.environ.get("TAX_REFINER_STRICT", "0") != "0"
        # ── phase switches ──
        # use_seed=False: no seed attempt, the first candidate of the schedule goes first.
        self.use_seed = use_seed if use_seed is not None else (os.environ.get("TAX_NO_SEED", "0") == "0")
        # Optional order: run the taxonomy sweep BEFORE the seed-paraphrase phase
        # (default: seed paraphrases first).
        self.sweep_before_seed_para = (
            sweep_before_seed_para if sweep_before_seed_para is not None
            else (os.environ.get("TAX_SWEEP_BEFORE_SEEDPARA") == "1")
        )
        if not self.use_seed and (self.seed_only or self.seed_paraphrase_reps > 0):
            raise ValueError(
                "use_seed=False leaves no seed to paraphrase: it cannot be combined with "
                "seed_only or seed_paraphrase_reps > 0"
            )
        # The fixed seed: the benchmark's own, a researcher-written one, or the default
        # AgentDojo important-instructions seed. It is unused when use_seed=False.
        self.seed_template = seed_template if seed_template is not None else _INITIAL_TEMPLATE
        self.seed_name = seed_name or IMPORTANT_INSTRUCTIONS_SEED_NAME
        if self.use_seed and SEED_PLACEHOLDER not in self.seed_template:
            raise ValueError(
                f"the seed template must contain the {SEED_PLACEHOLDER!r} placeholder "
                "(the injection goal is substituted into it); got a template without it"
            )
        if self.seed_only:
            self._combos: list[list[dict]] = []
            self._combo_by_id: dict[str, list[dict]] = {}
            max_attempts = paraphrase_reps
        else:
            self._combos = self._build_combo_schedule()
            self._combo_by_id = {}  # Filled only for candidates actually generated.
            if self.stack_depth >= 2 and self.stack_pool is None and self.num_combos > 5000:
                print(
                    f"[TAXREFINER] WARNING: stack_depth={self.stack_depth} with no stack_pool yields "
                    f"{self.num_combos} combinations/pair -- likely infeasible. Set TAX_STACK_POOL to bound it."
                )
            if self.stack_random and self.stack_pool_ids:
                print(f"[TAXREFINER] RANDOM stacking pool (seed={self.stack_seed}, n={len(self.stack_pool_ids)}): "
                      f"{self.stack_pool_ids}")
            if self.stack_only:
                print(f"[TAXREFINER] STACK-ONLY L{self.stack_only}: sweeping only the {self.num_combos} "
                      f"{self.stack_only}-way combinations of the pool (no singles / other depths)")
            max_attempts = self.seed_paraphrase_reps + self.num_combos * (1 + paraphrase_reps)

        # ── inlined RefinedInstructionsAttack base state ──
        self.refiner_model = refiner_model
        # max_attempts keeps its original meaning: the size of the schedule that follows the
        # seed. attempt_budget is the number of target calls one data point may use in total.
        self.max_attempts = max_attempts
        self.attempt_budget = max_attempts + (1 if self.use_seed else 0)
        self._current_template: str = self.seed_template if self.use_seed else ""
        self._failed_templates: list[str] = []
        self._infra_failures: list[str] = []
        # Successful attempts. More than one only when the driver keeps going after a success.
        self._succeeded_templates: list[str] = []
        # False only in no-seed mode when no first candidate could be generated.
        self.has_candidate = True

        self._combo_idx = 0
        self._phase = "taxonomy"
        self._para_sources: list[dict] | None = None
        self._para_src_idx = 0
        self._para_try = 0
        self._cur_para_failed: list[str] = []
        self._seed_para_try = 0
        self._seed_para_failed: list[str] = []
        self._seed_para_signals: list[str] = []
        self._attempt_meta: list[dict] = []

        def _flag(arg: bool | None, env: str) -> bool:
            return arg if arg is not None else (os.environ.get(env, "1") != "0")

        self.use_judge = _flag(use_judge, "TAX_JUDGE")
        self.use_context = _flag(use_context, "TAX_CONTEXT")
        self.use_legitimize = _flag(use_legitimize, "TAX_LEGITIMIZE")
        legacy_usertask_aware = (
            usertask_aware if usertask_aware is not None
            else (os.environ.get("TAX_USERTASK_AWARE", "0") != "0")
        )
        mode_env = os.environ.get("TAX_AWARENESS_MODE")
        selected_mode = (
            awareness_mode
            if awareness_mode is not None
            else (mode_env if mode_env and mode_env.strip() else (2 if legacy_usertask_aware else 1))
        )
        try:
            self.awareness_mode = int(selected_mode)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"TAX_AWARENESS_MODE/awareness_mode must be 1, 2, or 3; got {selected_mode!r}"
            ) from e
        if self.awareness_mode not in {1, 2, 3}:
            raise ValueError(
                f"TAX_AWARENESS_MODE/awareness_mode must be 1, 2, or 3; got {self.awareness_mode}"
            )
        self.usertask_aware = self.awareness_mode >= 2
        self.newtask_aware = self.awareness_mode == 3
        # per-pair adaptive state
        self._last_messages: list = []
        self._last_utility: bool | None = None
        self._last_security: bool | None = None
        # Template of the attempt that produced _last_messages (None before the first run).
        self._last_observed_template: str | None = None
        self._user_task_prompt: str = ""
        self._judge_mode: str | None = None
        self._adaptive_block: str = ""
        self._para_reasonings: str = ""
        self._last_para_raw: str = ""
        self._seq_transcript: dict[str, list] = {}
        self._cur_seq_signals: list[str] = []
        self._resume_after_attempt = 0
        self._resume_pending = False

    # ── properties ──
    @property
    def current_template(self) -> str:
        return self._current_template

    @property
    def num_sequences(self) -> int:
        return len(self._sequences)

    @property
    def num_combos(self) -> int:
        return self._combos.size if hasattr(self._combos, "size") else len(self._combos)

    def _query(self, prompt: str) -> str:
        config = self._refiner_config
        if getattr(self, "_querying_judge", False):
            config = replace(config, role="diagnostic_judge") if config else RefinerConfig(role="diagnostic_judge")
        return query_refiner_llm(self.refiner_model, prompt, config)

    def _build_combo_schedule(self) -> list[list[dict]]:
        combos, pool_ids = build_combo_schedule(
            self._sequences,
            stack_depth=self.stack_depth,
            stack_pool=self.stack_pool,
            stack_random=self.stack_random,
            stack_seed=self.stack_seed,
            stack_only=self.stack_only,
        )
        self.stack_pool_ids = pool_ids
        return combos

    # ── inlined base refiner-state methods ──
    def mark_template_failed(self) -> None:
        self._failed_templates.append(self._current_template)
        self._set_last_status_if_pending("failed")
        last = self._attempt_meta[-1] if self._attempt_meta else None
        if last and last["phase"] == "paraphrase" and last["template"]:
            self._cur_para_failed.append(last["template"])
        elif last and last["phase"] == "seed_paraphrase" and last["template"]:
            self._seed_para_failed.append(last["template"])

    def mark_attempt_infra_failed(self, failure_kind: str) -> None:
        self._infra_failures.append(failure_kind)
        self._set_last_status_if_pending(f"infra_{failure_kind}")

    def mark_succeeded(self) -> None:
        """Mark the current (just-succeeded) attempt as a solving one, so the trace and
        summary record which sequence/phase solved the pair even before the log is written.
        The attempt also counts against the budget, which matters only when the driver keeps
        going after a success (``stop_at_first_success=False``)."""
        self._succeeded_templates.append(self._current_template)
        self._set_last_status_if_pending("success")

    def _attempts_recorded(self) -> int:
        return len(self._failed_templates) + len(self._infra_failures) + len(self._succeeded_templates)

    def get_attempt_number(self) -> int:
        return self._attempts_recorded() + 1

    def has_attempts_remaining(self) -> bool:
        """True if one more target attempt fits in the budget.

        Contract of the original driver loop: call this BEFORE recording the outcome of the
        attempt that just ran (``mark_template_failed`` / ``mark_attempt_infra_failed`` /
        ``mark_succeeded``). A data point then gets ``attempt_budget`` target calls in total,
        the seed plus ``max_attempts`` refinements. Calling it after recording stops one
        attempt short."""
        return self._attempts_recorded() < self.attempt_budget - 1

    # ── driver-facing API ──
    def reset(self, user_task_prompt: str = "", injection_goal: str = "") -> None:
        """Start a new (user_task, injection_task) pair. ``user_task_prompt`` is the agent's
        current task text (used by modes 2/3); ``injection_goal`` is needed when a resume is
        pending (``TAX_RESUME_AFTER_ATTEMPT``) and in no-seed mode, where the first candidate
        is generated here. Check :attr:`has_candidate` before the first target call."""
        self._user_task_prompt = user_task_prompt or ""
        self.reset_for_new_task()
        if self._resume_pending:
            if not self.use_seed or self.sweep_before_seed_para:
                raise NotImplementedError(
                    "TAX_RESUME_AFTER_ATTEMPT assumes the default schedule (seed, seed paraphrases, "
                    "sweep); it cannot be combined with use_seed=False or sweep_before_seed_para"
                )
            self._restore_sweep_position(self._resume_after_attempt)
            self._resume_pending = False
            if not self.refine_template(injection_goal):
                raise RuntimeError(f"Could not resume taxonomy sweep after attempt {self._resume_after_attempt}")
        elif not self.use_seed:
            # No seed: the first candidate of the schedule is the first attempt. It is written
            # without context blending, because the engine has not seen a target run yet.
            self.has_candidate = self.refine_template(injection_goal)

    def build_injection(self, injection_goal: str) -> str:
        """Return the ready-to-place injection text for the current template: the goal is
        substituted into ``<task_goes_here>`` (modes 1/2); a no-op for mode-3 lines that
        already contain the goal. The adapter places this at the benchmark's injection point(s)."""
        return self._current_template.replace("<task_goes_here>", f"\n\n{injection_goal}\n\n")

    def observe(self, messages: list, utility: bool, security: bool) -> None:
        """Feed back the agent's transcript + outcome so the judge (①) and context-blending
        (②) can inspect what the agent actually did on the last attempt."""
        self._last_messages = messages or []
        self._last_utility = utility
        self._last_security = security
        # The injection these messages contain. An attempt that aborts on an infrastructure
        # error is never observed, so this can differ from the current template.
        self._last_observed_template = self._current_template
        # Per-attempt outcome in the trace, so a run that keeps going after a success can be
        # read attempt by attempt (which attempts, and so which techniques, succeeded).
        if self._attempt_meta:
            self._attempt_meta[-1]["security"] = bool(security)

    # legacy alias (matches the original benchmark loop's method name)
    observe_last_attempt = observe

    def reset_for_new_task(self) -> None:
        self._current_template = self.seed_template if self.use_seed else ""
        self._failed_templates = []
        self._infra_failures = []
        self._succeeded_templates = []
        self.has_candidate = True
        self._combo_idx = 0
        self._phase = "seed_paraphrase" if (self.seed_paraphrase_reps > 0 and not self.seed_only) else "taxonomy"
        self._para_sources = None
        self._para_src_idx = 0
        self._para_try = 0
        self._cur_para_failed = []
        self._seed_para_try = 0
        self._seed_para_failed = []
        self._seed_para_signals = []
        self._last_messages = []
        self._last_utility = None
        self._last_security = None
        self._last_observed_template = None
        self._judge_mode = None
        self._adaptive_block = ""
        self._para_reasonings = ""
        self._last_para_raw = ""
        self._seq_transcript = {}
        self._cur_seq_signals = []
        self._attempt_meta = [
            {
                "attempt": 1,
                "phase": "seed",
                "sequence_id": "SEED",
                "sequence_name": self.seed_name,
                "stack_level": 0,
                "template": self._current_template,
                "status": "pending",
            }
        ] if self.use_seed else []
        self._resume_after_attempt = max(0, int(os.environ.get("TAX_RESUME_AFTER_ATTEMPT", "0")))
        self._resume_pending = self._resume_after_attempt > 0

    # ── adaptive mechanisms ──
    def _usertask_prompt_text(self) -> str:
        if not self.usertask_aware:
            return ""
        return self._user_task_prompt or ""

    def _context_legit_block(self, injection_goal: str) -> str:
        parts: list[str] = []
        if self.use_context and self._last_messages:
            # Mask the injection that the last observed run actually contained.
            legacy = os.environ.get("TAX_CONTEXT_LEGACY_MASK", "0") != "0"
            shown = self._current_template if (legacy or self._last_observed_template is None) \
                else self._last_observed_template
            ctx = _extract_context(self._last_messages, injection_goal, shown)
            if ctx:
                parts.append("\nMatch my agent's actual situation:\n" + ctx + "\n")
        if self.use_legitimize:
            parts.append("\n" + _LEGITIMIZE_NOTE)
        return "".join(parts)

    def _check_refiner_error(self, e: Exception) -> None:
        """Stop on a generator that cannot be reached, in a strict run, or on a fatal error.
        Any other failed request is skipped like an unusable answer."""
        if _generator_unreachable(e):
            raise GeneratorUnavailableError(
                f"the generator could not be reached ({type(e).__name__}: {str(e)[:160]})") from e
        if self.strict_refiner or getattr(e, "fatal", False):
            raise RuntimeError("Taxonomy judge/refiner request failed") from e

    def _judge_text(self, messages: list, goal: str = "") -> str | None:
        self._judge_mode = None
        if not (self.use_judge and messages):
            return None
        try:
            self._querying_judge = True
            resp = self._query(_build_judge_prompt(messages, goal))
            mode, reason, _guidance = _parse_judge(resp)
        except Exception as e:
            print(f"[TAXREFINER] judge error: {e}")
            self._check_refiner_error(e)
            return None
        finally:
            self._querying_judge = False
        self._judge_mode = mode
        reason = _scrub_guidance(_mask_targets(reason))
        if not (mode or reason):
            return None
        note = f"my agent {_MODE_PHRASE.get(mode or '', 'did not act on the line')}."
        if reason:
            note += f" ({reason})"
        return note

    def _last_failed_transcript(self) -> list | None:
        """Transcript of the last attempt, for the judge to diagnose in a paraphrase chain.
        None when that attempt succeeded, which can happen only when the driver keeps going
        after a success: a success is not a failure to explain."""
        return None if self._last_security else self._last_messages

    def _stash_last_transcript(self) -> None:
        if not (self._attempt_meta and self._last_messages):
            return
        last = self._attempt_meta[-1]
        if last["template"] is not None and last["status"] == "failed":
            if last["phase"] == "taxonomy":
                self._seq_transcript[last["sequence_id"]] = self._last_messages
            elif last["phase"] == "seed":
                self._seq_transcript["SEED"] = self._last_messages

    def _set_last_status_if_pending(self, status: str) -> None:
        if self._attempt_meta and self._attempt_meta[-1]["status"] == "pending":
            self._attempt_meta[-1]["status"] = status

    def _record(self, seq_id: str, seq_name: str, phase: str, template: str | None, status: str,
                raw: str | None = None, stack_level: int = 1) -> None:
        entry = {
            "attempt": self.get_attempt_number(),
            "phase": phase,
            "sequence_id": seq_id,
            "sequence_name": seq_name,
            "stack_level": stack_level,
            "template": template,
            "status": status,
        }
        if raw is not None:
            entry["raw"] = raw
        self._attempt_meta.append(entry)

    # ── generation helpers ──
    def _generate_from_combo(self, combo: list[dict], injection_goal: str) -> str | None:
        cid, cname, lvl = _combo_id(combo), _combo_name(combo), len(combo)
        self._combo_by_id[cid] = combo
        last_raw = ""
        tiers = [self._adaptive_block, ""] if self._adaptive_block else [""]
        for adaptive in tiers:
            prompt = _build_taxonomy_prompt(combo, injection_goal, adaptive=adaptive,
                                            user_task_prompt=self._usertask_prompt_text(),
                                            awareness_mode=self.awareness_mode)
            try:
                response = self._query(prompt)
            except Exception as e:
                print(f"[TAXREFINER] {cid} refiner error: {e}")
                self._check_refiner_error(e)
                self._record(cid, cname, "taxonomy", None, "refiner_error", raw=str(e)[:200], stack_level=lvl)
                return None
            template = _extract_template_seq(
                response,
                _combo_allows_unicode(combo),
                require_placeholder=not self.newtask_aware,
            )
            if template is not None:
                return template
            last_raw = response
        self._record(cid, cname, "taxonomy", None, "unparseable", raw=last_raw[:300], stack_level=lvl)
        return None

    def _generate_paraphrase(
        self, failed_template: str, injection_goal: str, seqs: dict | list[dict] | None, prior: list[str] | None = None
    ) -> str | None:
        self._last_para_raw = ""
        reasonings = self._para_reasonings
        tiers = (
            [(self._adaptive_block, reasonings), ("", "")]
            if (self._adaptive_block or reasonings)
            else [("", "")]
        )
        for adaptive, cur_reasonings in tiers:
            prompt = _build_paraphrase_prompt(
                failed_template, injection_goal, seqs, prior_paraphrases=prior,
                adaptive=adaptive, failure_reasonings=cur_reasonings,
                user_task_prompt=self._usertask_prompt_text(),
                awareness_mode=self.awareness_mode,
            )
            try:
                response = self._query(prompt)
            except Exception as e:
                print(f"[TAXREFINER] paraphrase refiner error: {e}")
                self._check_refiner_error(e)
                self._last_para_raw = str(e)[:200]
                return None
            template = _extract_template_seq(
                response,
                _combo_allows_unicode(seqs) if seqs else False,
                require_placeholder=not self.newtask_aware,
            )
            if template is not None:
                return template
            self._last_para_raw = response
        return None

    def _build_paraphrase_sources(self) -> list[dict]:
        if self.seed_only:
            seed = self._attempt_meta[0]["template"] if self._attempt_meta else self._current_template
            return [{"template": seed, "combo": [], "src": "SEED"}]
        seen: dict[str, str] = {}
        for m in self._attempt_meta:
            if m["phase"] == "taxonomy" and m["template"] is not None and m["status"] == "failed":
                seen[m["sequence_id"]] = m["template"]
        return [{"template": t, "combo": self._combo_by_id.get(cid, []), "src": cid} for cid, t in seen.items()]

    # ── phase driver ──
    def refine_template(self, injection_goal: str) -> bool:
        """Produce the next template. Skips refused/unparseable candidates inline. Returns
        False only when no usable template remains (sweep exhausted AND all paraphrase
        targets tried)."""
        self._stash_last_transcript()
        cl_block = self._context_legit_block(injection_goal)

        # ── SEED-PARAPHRASE phase ── (default order: before the sweep)
        if not self.sweep_before_seed_para and self._next_seed_paraphrase(injection_goal, cl_block):
            return True

        # ── EXPLORE / broad sweep ──
        while self._combo_idx < self.num_combos:
            combo = self._combos[self._combo_idx]
            self._combo_idx += 1
            print(f"[TAXREFINER] attempt {self.get_attempt_number()} taxonomy L{len(combo)} {_combo_id(combo)}")
            self._adaptive_block = cl_block
            template = self._generate_from_combo(combo, injection_goal)
            if template is not None:
                self._current_template = template
                self._record(_combo_id(combo), _combo_name(combo), "taxonomy", template, "pending",
                             stack_level=len(combo))
                return True

        # ── SEED-PARAPHRASE phase with sweep_before_seed_para ── it starts once the sweep is
        #    exhausted, also when the sweep's last candidates could not be written, so the order
        #    becomes seed, sweep, seed paraphrases, exploit
        if self.sweep_before_seed_para and self._next_seed_paraphrase(injection_goal, cl_block):
            return True

        # ── EXPLOIT / paraphrase ──
        return self._next_exploit(injection_goal, cl_block)

    def _next_seed_paraphrase(self, injection_goal: str, cl_block: str) -> bool:
        """Write the next seed paraphrase. Skips refused/unparseable candidates inline and
        returns False when the seed-paraphrase phase has nothing left."""
        while (not self.seed_only) and self._seed_para_try < self.seed_paraphrase_reps:
            seed_tmpl = self._attempt_meta[0]["template"]
            sig = self._judge_text(
                self._seq_transcript.get("SEED") if self._seed_para_try == 0 else self._last_failed_transcript(),
                injection_goal,
            )
            if sig and (not self._seed_para_signals or sig != self._seed_para_signals[-1]):
                self._seed_para_signals.append(sig)
            self._seed_para_try += 1
            out_of_place_note = _NEW_TASK_OUT_OF_PLACE_NOTE if self.newtask_aware else _ANTI_SUSPICION_NOTE
            style_addendum = ("\n" + out_of_place_note) if self._judge_mode == "flagged_as_suspicious" else ""
            self._adaptive_block = cl_block + style_addendum
            self._para_reasonings = _format_signals(self._seed_para_signals)
            print(
                f"[TAXREFINER] attempt {self.get_attempt_number()} seed-paraphrase "
                f"try {self._seed_para_try}/{self.seed_paraphrase_reps} "
                f"({len(self._seed_para_signals)} judge signal(s))"
            )
            template = self._generate_paraphrase(seed_tmpl, injection_goal, None, prior=self._seed_para_failed)
            if template is not None:
                self._current_template = template
                self._record("PARA:SEED", "paraphrase_of_seed", "seed_paraphrase", template, "pending", stack_level=0)
                return True
            self._record("PARA:SEED", "paraphrase_of_seed", "seed_paraphrase", None, "unparseable",
                         raw=(getattr(self, "_last_para_raw", "") or "")[:300], stack_level=0)
        return False

    def _next_exploit(self, injection_goal: str, cl_block: str) -> bool:
        """Write the next paraphrase of a failed sweep candidate (of the seed in seed-only
        mode). Returns False when every paraphrase target has been tried."""
        self._phase = "paraphrase"
        if self._para_sources is None:
            self._para_sources = self._build_paraphrase_sources()
            self._para_src_idx = 0
            self._para_try = 0
            self._cur_para_failed = []
            self._cur_seq_signals = []
            print(
                f"[TAXREFINER] entering paraphrase phase: {len(self._para_sources)} failed instructions "
                f"x {self.paraphrase_reps} paraphrase attempts each"
            )
        while self._para_src_idx < len(self._para_sources):
            if self._para_try >= self.paraphrase_reps:
                self._para_src_idx += 1
                self._para_try = 0
                self._cur_para_failed = []
                self._cur_seq_signals = []
                continue
            src = self._para_sources[self._para_src_idx]
            sig = self._judge_text(
                self._seq_transcript.get(src["src"]) if self._para_try == 0 else self._last_failed_transcript(),
                injection_goal,
            )
            if sig and (not self._cur_seq_signals or sig != self._cur_seq_signals[-1]):
                self._cur_seq_signals.append(sig)
            self._para_try += 1
            out_of_place_note = _NEW_TASK_OUT_OF_PLACE_NOTE if self.newtask_aware else _ANTI_SUSPICION_NOTE
            style_addendum = ("\n" + out_of_place_note) if self._judge_mode == "flagged_as_suspicious" else ""
            self._adaptive_block = cl_block + style_addendum
            self._para_reasonings = _format_signals(self._cur_seq_signals)
            print(
                f"[TAXREFINER] attempt {self.get_attempt_number()} paraphrase src "
                f"{self._para_src_idx + 1}/{len(self._para_sources)} try {self._para_try}/{self.paraphrase_reps} "
                f"of {src['src']} ({len(self._cur_seq_signals)} judge signal(s))"
            )
            _lvl = len(src["combo"]) or 1
            template = self._generate_paraphrase(
                src["template"], injection_goal, src["combo"], prior=self._cur_para_failed
            )
            if template is not None:
                self._current_template = template
                self._record(f"PARA:{src['src']}", f"paraphrase_of_{src['src']}", "paraphrase", template, "pending",
                             stack_level=_lvl)
                return True
            self._record(f"PARA:{src['src']}", f"paraphrase_of_{src['src']}", "paraphrase", None, "unparseable",
                         raw=(getattr(self, "_last_para_raw", "") or "")[:300], stack_level=_lvl)
        return False

    def _restore_sweep_position(self, completed_attempts: int) -> None:
        """Resume the judge-free taxonomy sweep after a transport-aborted process: the seed
        and seed-paraphrase phases are assumed already evaluated; placeholder history
        preserves the budget and accounting without repeating paid target calls."""
        completed_attempts = min(completed_attempts, self.max_attempts)
        seed_paras = min(self.seed_paraphrase_reps, max(0, completed_attempts - 1))
        taxonomy_count = min(self.num_combos, max(0, completed_attempts - 1 - seed_paras))
        self._seed_para_try = seed_paras
        self._combo_idx = taxonomy_count
        self._phase = "taxonomy"
        self._failed_templates = [
            f"<resumed failed target attempt {attempt}>" for attempt in range(1, completed_attempts + 1)
        ]
        self._attempt_meta = [
            {"attempt": 1, "phase": "seed", "sequence_id": "SEED", "sequence_name": self.seed_name,
             "stack_level": 0, "template": None, "status": "failed_resumed"}
        ]
        for offset in range(seed_paras):
            self._attempt_meta.append(
                {"attempt": offset + 2, "phase": "seed_paraphrase", "sequence_id": "PARA:SEED",
                 "sequence_name": "paraphrase_of_seed", "stack_level": 0, "template": None, "status": "failed_resumed"}
            )
        for offset, combo in enumerate(self._combos[:taxonomy_count]):
            self._attempt_meta.append(
                {"attempt": seed_paras + offset + 2, "phase": "taxonomy", "sequence_id": _combo_id(combo),
                 "sequence_name": _combo_name(combo), "stack_level": len(combo), "template": None,
                 "status": "failed_resumed"}
            )
        print(
            f"[TAXREFINER] resuming after target attempt {completed_attempts}: "
            f"seed paraphrases={seed_paras}, taxonomy candidates={taxonomy_count}"
        )

    # ── logging ──
    def save_template_log(
        self,
        logdir: Path | None,
        suite_name: str,
        user_task_id: str,
        injection_task_id: str,
        final_security: bool,
        *,
        stop_at_first_success: bool = True,
    ) -> None:
        """Write the per-pair trace. ``final_security`` is whether the pair was broken at
        all. ``stop_at_first_success`` records the driver's stop rule: when False, the pair
        ran its whole schedule and the log may hold several successes."""
        if logdir is None:
            return
        logdir = Path(logdir)
        # An attempt the driver never marked takes its own observed outcome when there is
        # one, so "broken at some point" is not mistaken for "this last attempt succeeded".
        last_ok = self._attempt_meta[-1].get("security", final_security) if self._attempt_meta else final_security
        self._set_last_status_if_pending("success" if last_ok else "failed")
        successes = [m for m in self._attempt_meta if m["status"] == "success"]
        solved = successes[0] if successes else None
        # Sources credited with a success: a paraphrase counts toward the line it reworks.
        solved_sources: list[str] = []
        for m in successes:
            src = m["sequence_id"].removeprefix("PARA:")
            if src not in solved_sources:
                solved_sources.append(src)
        covered = sorted({sid for src in solved_sources if src != "SEED" for sid in src.split("+")})
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "suite": suite_name,
            "user_task": user_task_id,
            "injection_task": injection_task_id,
            "attack": self.name,
            "refiner_model": self.refiner_model,
            "num_sequences": len(self._sequences),
            "num_combos": self.num_combos,
            "stack_depth": self.stack_depth,
            "stack_pool": self.stack_pool,
            "stack_random": self.stack_random,
            "stack_only": self.stack_only,
            "stack_pool_ids": self.stack_pool_ids,
            "paraphrase_reps": self.paraphrase_reps,
            "seed_paraphrase_reps": self.seed_paraphrase_reps,
            "seed_only": self.seed_only,
            "use_seed": self.use_seed,
            "sweep_before_seed_para": self.sweep_before_seed_para,
            "order": self.order,
            "order_seed": self.order_seed,
            "max_sequences": self.max_sequences,
            "techniques": self.techniques,
            "stop_at_first_success": stop_at_first_success,
            "mechanisms": {"judge": self.use_judge, "context": self.use_context, "legitimize": self.use_legitimize},
            "awareness_mode": self.awareness_mode,
            "usertask_aware": self.usertask_aware,
            "newtask_aware": self.newtask_aware,
            "resumed_after_attempt": self._resume_after_attempt or None,
            "max_attempts": self.max_attempts,
            "attempt_budget": self.attempt_budget,
            "total_attempts": self._attempts_recorded(),
            "total_candidates": len(self._attempt_meta),
            "strategy": getattr(self, "strategy_config", None),
            "injection_succeeded": final_security,
            "solved_phase": solved["phase"] if solved else None,
            "solved_sequence_id": solved["sequence_id"] if solved else None,
            "solved_stack_level": solved.get("stack_level") if solved else None,
            # Every success, not only the first (more than one only without early stopping).
            "num_successes": len(successes),
            "first_success_attempt": solved["attempt"] if solved else None,
            "solved_sources": solved_sources,
            "covered_subtechniques": covered,
            "successful_templates": [
                {"attempt": m["attempt"], "phase": m["phase"], "sequence_id": m["sequence_id"],
                 "stack_level": m.get("stack_level"), "template": m["template"]}
                for m in successes
            ],
            "infra_failures": list(self._infra_failures),
            "failed_templates": [
                {"attempt": m["attempt"], "sequence_id": m["sequence_id"], "template": m["template"], "status": m["status"]}
                for m in self._attempt_meta
                if m["template"] is not None and m["status"] != "success"
            ],
            # The first solving template when there is one (the current template when the run
            # stopped at its first success), else the last template tried.
            "final_template": {
                "template": solved["template"] if solved else self._current_template,
                "status": "SUCCESS" if final_security else "FAILED",
            },
            "attempts": self._attempt_meta,
        }
        refiner_log_dir = logdir / "refiner_logs" / suite_name
        refiner_log_dir.mkdir(parents=True, exist_ok=True)
        log_file = refiner_log_dir / f"{user_task_id}__{injection_task_id}.json"
        tmp_file = log_file.with_name(f".{log_file.name}.tmp.{os.getpid()}")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, log_file)
        finally:
            if tmp_file.exists():
                tmp_file.unlink()
        print(f"[TAXREFINER] template log saved to {log_file}")
