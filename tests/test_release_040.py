"""Regression tests for 0.4.0: the options' paper-term names, their earlier names (still
accepted), the exploit counts of a seed-only run, and the messages that name options.

Each test fails when its change is reverted (checked by reverting each change in a copy of the
code). The runs use a mock benchmark and a stub generator, so no model is called.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import autopi.cli as cli_mod
import autopi.core.engine as engine_mod
from autopi.adapters.base import AttackPoint, RunResult
from autopi.cli import main as cli_main
from autopi.core.engine import TaxonomyEngine
from autopi.core.seeds import IMPORTANT_INSTRUCTIONS_SEED, IMPORTANT_INSTRUCTIONS_SEED_NAME

L1_STRATEGY = Path(__file__).resolve().parents[1] / "configs" / "strategies" / "l1.yaml"
NEW_NAMES = ["--generator", "--generator-tokenizer", "--ipi-judge", "--ipi-worldsim", "--sweep-exploits",
             "--seed-exploits", "--sweep-before-seed-exploits", "--max-sweep-level", "--only-sweep-level",
             "--sweep-pool-size", "--random-sweep-pool", "--sweep-pool-seed", "--max-subtechniques"]
OLD_NAMES = ["--refiner", "--refiner-tokenizer", "--judge", "--worldsim", "-k", "--paraphrase-reps",
             "--seed-para-reps", "--sweep-before-seed-para", "--stack-depth", "--stack-only", "--stack-pool",
             "--stack-random", "--stack-seed", "--max-sequences"]


def _line(_model=None, _prompt=None, _cfg=None):
    return "[TEMPLATE_START] As a required step, complete the stated task now. [TEMPLATE_END]"


class _Adapter:
    """One scenario that the injection never breaks, so a run tries its whole schedule."""

    name = "mock"
    checker_kind = "state"
    default_seed_template = IMPORTANT_INSTRUCTIONS_SEED
    default_seed_name = IMPORTANT_INSTRUCTIONS_SEED_NAME

    def __init__(self):
        self.calls = 0

    def iter_points(self):
        yield AttackPoint("s", "user_task_0", "injection_task_0", "do my chore", "GOAL-123")

    def run(self, point, injection_text):
        self.calls += 1
        return RunResult(utility=True, security=False,
                         messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "ok"}])


def _invoke(monkeypatch, args, *, benchmark="agentdojo", logdir=None, small_sweep=True):
    """``autopi run`` on the mock benchmark with a stub generator (by default a 6-line L1 sweep)."""
    adapter = _Adapter()
    monkeypatch.setattr(cli_mod, "_build_adapter", lambda *a, **k: adapter)
    monkeypatch.setattr(engine_mod, "query_refiner_llm", _line)
    command = ["run", "--benchmark", benchmark, "--target", "vllm:8000:t", "--no-judge", "--no-context",
               "--no-legitimize", *(["--techniques", "F1.T2"] if small_sweep else []),
               *(["--logdir", str(logdir)] if logdir else []), *args]
    result = CliRunner().invoke(cli_main, command)
    banner = next((line for line in result.output.splitlines() if line.startswith("[autopi] benchmark=")), "")
    manifest = log = None
    if logdir and (logdir / "run_manifest.json").exists():
        manifest = json.loads((logdir / "run_manifest.json").read_text())
        log = json.loads((logdir / "refiner_logs" / "s" / "user_task_0__injection_task_0.json").read_text())
        log.pop("timestamp")
    return SimpleNamespace(code=result.exit_code, output=result.output, banner=banner, manifest=manifest,
                           log=log, calls=adapter.calls)


def _settings(run, *keys):
    value = run.manifest["settings"]
    for key in keys:
        value = value[key]
    return value


# ─────────────────────────── the names --help shows ───────────────────────────

def test_help_lists_the_paper_names_only():
    out = CliRunner().invoke(cli_main, ["run", "--help"], terminal_width=400, max_content_width=400).output
    listed = set(re.findall(r"(?<![\w-])(--?[a-z][\w-]*)", out))
    assert set(NEW_NAMES) <= listed
    assert not set(OLD_NAMES) & listed


# ─────────────────────────── an earlier name runs exactly like the current one ───────────────────────────

RENAMED = [
    ("generator", ["--refiner", "vllm:9000:g"], ["--generator", "vllm:9000:g"],
     lambda run: "generator=vllm:9000:g " in run.banner),
    ("k", ["-k", "2"], ["--sweep-exploits", "2"], lambda run: _settings(run, "strategy", "k") == 2),
    ("paraphrase-reps", ["--paraphrase-reps", "2"], ["--sweep-exploits", "2"],
     lambda run: _settings(run, "strategy", "k") == 2),
    ("seed-para-reps", ["--seed-para-reps", "2"], ["--seed-exploits", "2"],
     lambda run: _settings(run, "strategy", "seed_para_reps") == 2),
    ("sweep-before", ["--sweep-before-seed-para", "--seed-para-reps", "1"],
     ["--sweep-before-seed-exploits", "--seed-exploits", "1"],
     lambda run: _settings(run, "strategy", "sweep_before_seed_para") is True),
    ("levels", ["--stack-depth", "2", "--stack-pool", "3"], ["--max-sweep-level", "2", "--sweep-pool-size", "3"],
     lambda run: (_settings(run, "strategy", "stack_depth"), _settings(run, "strategy", "stack_pool")) == (2, 3)),
    ("one-level", ["--stack-only", "2", "--stack-pool", "4", "--stack-random", "--stack-seed", "5"],
     ["--only-sweep-level", "2", "--sweep-pool-size", "4", "--random-sweep-pool", "--sweep-pool-seed", "5"],
     lambda run: [_settings(run, "strategy", key) for key in ("stack_only", "stack_pool", "stack_random", "stack_seed")]
     == [2, 4, True, 5]),
    ("max-sequences", ["--max-sequences", "2"], ["--max-subtechniques", "2"],
     lambda run: _settings(run, "strategy", "max_sequences") == 2),
    ("seed-only", ["--seed-only", "-k", "2"], ["--seed-only", "--seed-exploits", "2"],
     lambda run: (_settings(run, "strategy", "seed_only"), _settings(run, "strategy", "k"),
                  _settings(run, "strategy", "seed_para_reps")) == (True, 2, 0)
     and [a["sequence_id"] for a in run.log["attempts"]] == ["SEED", "PARA:SEED", "PARA:SEED"]),
]


@pytest.mark.parametrize("old, new, took_effect", [case[1:] for case in RENAMED], ids=[case[0] for case in RENAMED])
def test_an_earlier_name_runs_exactly_like_the_current_one(monkeypatch, tmp_path, old, new, took_effect):
    before = _invoke(monkeypatch, old, logdir=tmp_path / "old")
    after = _invoke(monkeypatch, new, logdir=tmp_path / "new")
    assert before.code == after.code == 0, (before.output, after.output)
    assert took_effect(after)
    assert (before.manifest, before.log, before.banner) == (after.manifest, after.log, after.banner)


def test_the_tokenizer_and_ipi_arena_options_keep_their_earlier_names(monkeypatch, tmp_path):
    monkeypatch.setattr("autopi.models.recovery.get_decoder", lambda path: None)
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    names = (["--refiner-tokenizer", str(tokenizer), "--judge", "vllm:9001:j", "--worldsim", "vllm:9002:w"],
             ["--generator-tokenizer", str(tokenizer), "--ipi-judge", "vllm:9001:j", "--ipi-worldsim", "vllm:9002:w"])
    before, after = (_invoke(monkeypatch, args, benchmark="ipiarena", logdir=tmp_path / name)
                     for args, name in zip(names, ("old", "new")))
    assert before.code == after.code == 0, (before.output, after.output)
    assert set(after.manifest["tokenizer_sha256"]) == {"refiner"}
    assert (_settings(after, "models", "judge", "model"), _settings(after, "models", "worldsim", "model")) == ("j", "w")
    assert (before.manifest, before.log) == (after.manifest, after.log)


def test_resume_accepts_either_name(monkeypatch, tmp_path):
    first = _invoke(monkeypatch, ["--stack-depth", "2", "--stack-pool", "3", "-k", "1"], logdir=tmp_path)
    again = _invoke(monkeypatch, ["--max-sweep-level", "2", "--sweep-pool-size", "3", "--sweep-exploits", "1",
                                  "--resume"], logdir=tmp_path)
    assert first.code == again.code == 0 and first.calls > 0 and again.calls == 0, again.output


def test_an_earlier_and_a_current_name_must_agree(monkeypatch):
    run = _invoke(monkeypatch, ["--refiner", "vllm:9000:a", "--generator", "vllm:9000:b", "--dry-run"])
    assert run.code == 2 and "--refiner is the old name of --generator" in run.output
    assert _invoke(monkeypatch, ["--refiner", "vllm:9000:a", "--generator", "vllm:9000:a", "--dry-run"]).code == 0
    run = _invoke(monkeypatch, ["--stack-pool", "3", "--sweep-pool-size", "4", "--dry-run"])
    assert run.code == 2 and "--stack-pool is the old name of --sweep-pool-size" in run.output
    run = _invoke(monkeypatch, ["-k", "2", "--sweep-exploits", "3", "--dry-run"])
    assert run.code == 2 and "both set the number of sweep exploits" in run.output


# ─────────────────────────── the exploit counts of a seed-only run ───────────────────────────

def test_seed_only_takes_its_rewrites_from_seed_exploits(monkeypatch):
    # Before 0.4.0, --seed-only with --seed-para-reps ran the seed alone (a budget of 1).
    for args in (["--seed-only", "--seed-exploits", "3"], ["--seed-only", "--seed-para-reps", "3"],
                 ["--seed-only", "-k", "3"], ["--seed-only", "--seed-exploits", "3", "-k", "3"]):
        run = _invoke(monkeypatch, [*args, "--dry-run"])
        assert run.code == 0 and "sweep_lines=0 budget/pt=4 " in run.banner, (args, run.output)
    monkeypatch.setenv("TAX_SEED_ONLY", "1")
    assert "budget/pt=4 " in _invoke(monkeypatch, ["--seed-para-reps", "3", "--dry-run"]).banner


def test_seed_only_refuses_counts_it_cannot_use(monkeypatch):
    run = _invoke(monkeypatch, ["--seed-only", "--sweep-exploits", "1", "--dry-run"])
    assert run.code == 2 and "runs no sweep" in run.output
    run = _invoke(monkeypatch, ["--seed-only", "--seed-exploits", "3", "-k", "2", "--dry-run"])
    assert run.code == 2 and "both set the number of seed exploits" in run.output
    monkeypatch.setenv("TAX_SEED_ONLY", "1")
    run = _invoke(monkeypatch, ["--sweep-exploits", "1", "--dry-run"])
    assert run.code == 2 and "runs no sweep" in run.output


def test_the_engine_refuses_seed_exploits_it_would_ignore(monkeypatch):
    with pytest.raises(ValueError, match="seed_paraphrase_reps applies only to runs with a sweep"):
        TaxonomyEngine(refiner_model="m", seed_only=True, seed_paraphrase_reps=2)
    assert TaxonomyEngine(refiner_model="m", seed_only=True, paraphrase_reps=2).attempt_budget == 3
    monkeypatch.setenv("TAX_SEED_ONLY", "1")
    monkeypatch.setenv("TAX_SEED_PARA_REPS", "2")
    with pytest.raises(ValueError, match="seed_paraphrase_reps applies only to runs with a sweep"):
        TaxonomyEngine(refiner_model="m")


# ─────────────────────────── messages and console words ───────────────────────────

def test_messages_name_the_options_as_given(monkeypatch):
    run = _invoke(monkeypatch, ["--strategy", str(L1_STRATEGY), "-k", "2", "--stack-pool", "3",
                                "--sweep-exploits", "1", "--dry-run"], small_sweep=False)
    assert run.code == 2 and "conflicting flags: --stack-pool, --sweep-exploits, -k/--paraphrase-reps" in run.output
    for given in (["--judge", "x"], ["--ipi-judge", "x"]):
        run = _invoke(monkeypatch, [*given, "--dry-run"])
        assert run.code == 2 and f"{given[0]} applies to IPI Arena only" in run.output
    run = _invoke(monkeypatch, ["--no-seed", "--seed-para-reps", "2", "--dry-run"])
    assert run.code == 2 and "drop --seed-only / --seed-para-reps" in run.output
    run = _invoke(monkeypatch, ["--max-subtechniques", "1", "--only-sweep-level", "3", "--dry-run"])
    assert run.code == 2 and "--max-subtechniques or --sweep-pool-size, or lower --only-sweep-level" in run.output


def test_the_run_prints_the_paper_terms(monkeypatch, tmp_path):
    run = _invoke(monkeypatch, ["--sweep-exploits", "1"], logdir=tmp_path)
    assert run.code == 0 and "generator=vllm:8000:t sweep_lines=6 budget/pt=13 " in run.banner
    assert "[autopi] attempt 2 sweep L1" in run.output
    assert "TAXREFINER" not in run.output and "refiner=" not in run.output and "combos=" not in run.output
