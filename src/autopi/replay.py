"""CSV replay — re-run previously harvested successful injections, no refiner.

A benchmark-agnostic replay path. Two CSV shapes are supported, auto-detected by column:

* ``template_instruction`` — a placeholder template carrying ``<task_goes_here>`` (the
  ``autopi_mode2_csv`` harvest); the goal is substituted at replay time.
* ``whole_instruction`` — a concrete, goal-embedded instruction with no placeholder (the
  ``autopi_union_csv`` harvest, the replay analog of mode 3); used verbatim.

Both are keyed by ``(suite, user_task, injection_task)`` with task ids normalised so the
compact (``ut0``/``it0``) and canonical (``user_task_0``/``injection_task_0``) forms match.
This is a cheap, deterministic baseline (and a contamination/regression set) that needs no
LLM refiner — only the target benchmark.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from autopi.adapters.base import AdapterInfraError, BenchmarkAdapter
from autopi.harness.runner import PointOutcome, RunSummary

_PLACEHOLDER = "<task_goes_here>"


def _norm_task_id(value: str, canonical_prefix: str, compact_prefix: str) -> str:
    """Normalise e.g. 'ut0' or 'user_task_0' → 'user_task_0'."""
    v = value.strip()
    m = re.fullmatch(rf"(?:{canonical_prefix}|{compact_prefix})(\d+)", v)
    if m:
        return f"{canonical_prefix}{m.group(1)}"
    return v


def load_replay_map(csv_path: str | Path) -> dict[tuple[str, str, str], list[str]]:
    """Load a replay CSV into ``{(suite, user_task, injection_task): [instruction, ...]}``."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        return {}
    col = "whole_instruction" if "whole_instruction" in rows[0] else "template_instruction"
    out: dict[tuple[str, str, str], list[str]] = {}
    for r in rows:
        key = (
            (r.get("suite") or "").strip(),
            _norm_task_id(r.get("user_task", ""), "user_task_", "ut"),
            _norm_task_id(r.get("injection_task", ""), "injection_task_", "it"),
        )
        text = (r.get(col) or "").strip()
        if text:
            out.setdefault(key, []).append(text)
    return out


def run_replay(
    adapter: BenchmarkAdapter,
    csv_path: str | Path,
    *,
    max_candidates: int | None = None,
    verbose: bool = True,
) -> RunSummary:
    """Replay harvested instructions for each of ``adapter``'s points. For each point, try its
    harvested instruction(s) in order (substituting the goal into any placeholder) until one
    succeeds or the candidates run out."""
    replay_map = load_replay_map(csv_path)
    summary = RunSummary(benchmark=f"{adapter.name}+replay", awareness_mode=0)
    for point in adapter.iter_points():
        candidates = replay_map.get((point.suite, point.user_task_id, point.injection_task_id), [])
        if max_candidates is not None:
            candidates = candidates[:max_candidates]
        succeeded = False
        attempts = 0
        for text in candidates:
            attempts += 1
            injection = text.replace(_PLACEHOLDER, f"\n\n{point.injection_goal}\n\n") if _PLACEHOLDER in text else text
            try:
                result = adapter.run(point, injection)
            except AdapterInfraError:
                continue
            if result.security:
                succeeded = True
                break
        summary.outcomes.append(PointOutcome(point=point, succeeded=succeeded, attempts=attempts))
        if verbose:
            tag = "HIT " if succeeded else "miss"
            print(f"[{tag}] replay {adapter.name}/{point.suite}/{point.user_task_id}+{point.injection_task_id} "
                  f"({attempts} candidate(s))")
    return summary
