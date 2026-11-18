"""Composable strategies from Section 4 and Appendix G of the AutoPI paper.

Blocks are an ordered finite list, not a fixed pipeline. They may be omitted,
reordered or repeated. Combinations are generated lazily, including higher L_n.
"""
from __future__ import annotations

from itertools import islice
from pathlib import Path

import yaml

from autopi.core.engine import TaxonomyEngine
from autopi.core.prompts import _ANTI_SUSPICION_NOTE, _NEW_TASK_OUT_OF_PLACE_NOTE, _format_signals
from autopi.core.taxonomy import (
    build_combo_schedule,
    combo_id,
    combo_name,
    load_sequences,
    order_sequences,
    select_sequences,
)


def load_strategy(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return validate_strategy(data)


def validate_strategy(data: dict) -> dict:
    if not isinstance(data, dict) or set(data) - {"version", "budget", "stop", "blocks"}:
        raise ValueError("strategy keys are version, budget, stop, blocks")
    if data.get("version", 1) != 1:
        raise ValueError("unsupported strategy version")
    if type(data.get("budget")) is not int or data["budget"] < 1:
        raise ValueError("strategy budget must be a positive integer")
    if data.get("stop", "first_success") not in {"first_success", "budget"}:
        raise ValueError("strategy stop must be first_success or budget")
    blocks = data.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("strategy needs a nonempty blocks list")
    ids, sweeps, has_seed, normalized = set(), set(), False, []
    fields = {
        "seed": {"kind", "id"},
        "seed_exploit": {"kind", "id", "reps"},
        "sweep": {"kind", "id", "level", "techniques", "order", "order_seed", "order_ids", "limit", "pool_size"},
        "sweep_exploit": {"kind", "id", "reps", "source"},
    }
    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or block.get("kind") not in fields:
            raise ValueError(f"block {index + 1}: use seed, sweep, seed_exploit or sweep_exploit")
        block = dict(block)
        kind = block["kind"]
        if set(block) - fields[kind]:
            raise ValueError(f"unknown {kind} block keys: {sorted(set(block) - fields[kind])}")
        block.setdefault("id", f"block_{index + 1}")
        if not isinstance(block["id"], str) or not block["id"] or block["id"] in ids:
            raise ValueError("each block id must be a unique nonempty string")
        ids.add(block["id"])
        if kind.endswith("exploit"):
            if type(block.get("reps")) is not int or block["reps"] < 0:
                raise ValueError("exploit reps must be a nonnegative integer")
        if kind == "seed_exploit" and not has_seed:
            raise ValueError("seed_exploit needs an earlier seed block")
        if kind == "sweep_exploit":
            if not sweeps or (block.get("source") is not None and block["source"] not in sweeps):
                raise ValueError("sweep_exploit source must name an earlier sweep; omit source to use all earlier sweeps")
        if kind == "sweep":
            for field in ("level", "limit", "pool_size"):
                if field in block and (type(block[field]) is not int or block[field] < 1):
                    raise ValueError(f"sweep {field} must be a positive integer")
            if "techniques" in block and not isinstance(block["techniques"], (str, list)):
                raise ValueError("sweep techniques must be a list or comma-separated ids")
            sweeps.add(block["id"])
        has_seed |= kind == "seed"
        normalized.append(block)
    return {"version": 1, "budget": data["budget"], "stop": data.get("stop", "first_success"), "blocks": normalized}


class StrategyEngine(TaxonomyEngine):
    """The existing prompt/adaptation machinery driven by an arbitrary block list."""

    def __init__(self, strategy: dict, **kwargs):
        self.strategy_config = validate_strategy(strategy)
        # Schedule-related TAX_* variables must not silently modify an explicit strategy.
        super().__init__(**kwargs, use_seed=False, seed_only=False, paraphrase_reps=0,
                         seed_paraphrase_reps=0, stack_depth=1, stack_pool=None,
                         stack_only=0, stack_random=False, stack_seed=0, sweep_before_seed_para=False,
                         techniques=[], max_sequences=len(load_sequences(kwargs.get("taxonomy_path"))), order="canonical", order_seed=0)
        self._plans = {}
        for block in self.strategy_config["blocks"]:
            if block["kind"] != "sweep":
                continue
            selectors = block.get("techniques")
            if isinstance(selectors, str):
                selectors = selectors.split(",")
            seqs = select_sequences(self._sequences, selectors)
            seqs = order_sequences(seqs, block.get("order", "priority"), block.get("order_seed", 0))
            if block.get("order_ids"):
                preferred = block["order_ids"]
                if not isinstance(preferred, list) or len(preferred) != len(set(preferred)):
                    raise ValueError("order_ids must be a list of distinct sub-technique ids")
                known = {seq["id"] for seq in seqs}
                if set(preferred) - known:
                    raise ValueError(f"unknown order_ids: {sorted(set(preferred) - known)}")
                rank = {sid: index for index, sid in enumerate(preferred)}
                seqs.sort(key=lambda seq: rank.get(seq["id"], len(rank)))
            level = block.get("level", 1)
            schedule, _ = build_combo_schedule(seqs, stack_only=level, stack_pool=block.get("pool_size"))
            self._plans[block["id"]] = (schedule, min(schedule.size, block.get("limit", schedule.size)))
        self.use_seed = any(block["kind"] == "seed" for block in self.strategy_config["blocks"])
        if self.use_seed and "<task_goes_here>" not in self.seed_template:
            raise ValueError("the seed template must contain <task_goes_here>")
        self.attempt_budget = self.strategy_config["budget"]
        self.max_attempts = self.attempt_budget
        self._block_id = None

    @property
    def num_combos(self):
        if not hasattr(self, "_plans"):
            return super().num_combos
        return sum(count for _, count in self._plans.values())

    def _record(self, *args, **kwargs):
        super()._record(*args, **kwargs)
        self._attempt_meta[-1]["block_id"] = self._block_id

    def reset(self, user_task_prompt="", injection_goal=""):
        self._user_task_prompt = user_task_prompt or ""
        self.reset_for_new_task()
        if self._resume_pending:
            raise ValueError("TAX_RESUME_AFTER_ATTEMPT cannot reconstruct arbitrary strategy history; use --resume for completed points")
        self._attempt_meta = []
        self._sources = []
        self._seed_source = None
        self._iterator = self._candidates(injection_goal)
        self.has_candidate = next(self._iterator, False)

    def refine_template(self, injection_goal):
        return next(self._iterator, False)

    def _new_source(self, template, combo, block, record):
        source = {"template": template, "combo": combo, "block": block, "record": record,
                  "messages": [], "last_security": None, "prior": [], "signals": []}
        self._sources.append(source)
        return source

    def _finish(self, source, record, *, variant=False):
        if record["status"] in {"success", "failed"}:
            source["messages"] = self._last_messages
            source["last_security"] = record["status"] == "success"
            if variant and record["status"] == "failed":
                source["prior"].append(record["template"])

    def _exploit(self, source, reps, goal, seed=False):
        for _ in range(reps):
            feedback = self._judge_text(source["messages"] if source["last_security"] is False else [], goal)
            if feedback and (not source["signals"] or feedback != source["signals"][-1]):
                source["signals"].append(feedback)
            self._adaptive_block = self._context_legit_block(goal)
            if self._judge_mode == "flagged_as_suspicious":
                self._adaptive_block += "\n" + (_NEW_TASK_OUT_OF_PLACE_NOTE if self.newtask_aware else _ANTI_SUSPICION_NOTE)
            self._para_reasonings = _format_signals(source["signals"])
            template = self._generate_paraphrase(source["template"], goal, source["combo"] or None,
                                                   prior=source["prior"])
            sid = "SEED" if seed else combo_id(source["combo"])
            phase = "seed_paraphrase" if seed else "paraphrase"
            self._record(f"PARA:{sid}", f"paraphrase_of_{sid}", phase, template,
                         "pending" if template is not None else "unparseable",
                         raw=self._last_para_raw[:300] if template is None else None,
                         stack_level=len(source["combo"]))
            record = self._attempt_meta[-1]
            record["source_attempt"] = source["record"]["attempt"]
            record["source_block"] = source["block"]
            if template is not None:
                self._current_template = template
                yield True
                self._finish(source, record, variant=True)

    def _candidates(self, goal):
        for block in self.strategy_config["blocks"]:
            self._block_id = block["id"]
            kind = block["kind"]
            if kind == "seed":
                self._current_template = self.seed_template
                self._record("SEED", self.seed_name, "seed", self.seed_template, "pending", stack_level=0)
                record = self._attempt_meta[-1]
                self._seed_source = self._new_source(self.seed_template, [], block["id"], record)
                yield True
                self._finish(self._seed_source, record)
            elif kind == "sweep":
                schedule, count = self._plans[block["id"]]
                for combo in islice(schedule, count):
                    self._adaptive_block = self._context_legit_block(goal)
                    template = self._generate_from_combo(combo, goal)
                    if template is None:
                        continue
                    self._current_template = template
                    self._record(combo_id(combo), combo_name(combo), "taxonomy", template, "pending", stack_level=len(combo))
                    record = self._attempt_meta[-1]
                    source = self._new_source(template, combo, block["id"], record)
                    yield True
                    self._finish(source, record)
            elif kind == "seed_exploit":
                if self._seed_source["record"]["status"] == "failed":
                    yield from self._exploit(self._seed_source, block["reps"], goal, seed=True)
            elif kind == "sweep_exploit":
                sources = [source for source in self._sources if source["combo"]
                           and source["record"]["status"] == "failed"
                           and (not block.get("source") or source["block"] == block["source"])]
                for source in sources:
                    yield from self._exploit(source, block["reps"], goal)
