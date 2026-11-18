"""Behavioral regressions for the paper's general strategy interface."""
import itertools
import json
import math
from pathlib import Path

import pytest
from click.testing import CliRunner

import autopi.core.engine as engine_module
from autopi.adapters.base import AdapterInfraError, AttackPoint, RunResult
from autopi.cli import main
from autopi.core.strategy import StrategyEngine, load_strategy
from autopi.core.taxonomy import build_combo_schedule, load_sequences
from autopi.harness.runner import run_attack

ROOT = Path(__file__).resolve().parents[1]


class Adapter:
    name = "offline"
    checker_kind = "state"
    default_seed_template = "Seed: <task_goes_here>"
    default_seed_name = "test-seed"

    def __init__(self, hits=(), infrastructure=()):
        self.calls = []
        self.hits = set(hits)
        self.infrastructure = set(infrastructure)

    def iter_points(self):
        yield AttackPoint("test", "u", "i", "Summarize a document", "TEST-GOAL")

    def run(self, point, injection_text):
        self.calls.append(injection_text)
        if len(self.calls) in self.infrastructure:
            raise AdapterInfraError("transport")
        return RunResult(True, len(self.calls) in self.hits,
                         [{"role": "tool", "content": f"Document {injection_text} end"}])


def make_engine(monkeypatch, blocks, budget=30, **kwargs):
    counter = itertools.count()
    monkeypatch.setattr(engine_module, "query_refiner_llm",
                        lambda *args: f"[TEMPLATE_START] TEST-GOAL variant {next(counter)} [TEMPLATE_END]")
    return StrategyEngine({"budget": budget, "blocks": blocks}, refiner_model="test-model",
                          awareness_mode=3, use_judge=False, use_context=False, use_legitimize=False, **kwargs)


def test_blocks_can_repeat_and_exploit_only_the_named_sweep(monkeypatch):
    engine = make_engine(monkeypatch, [
        {"kind": "sweep", "id": "one", "limit": 2},
        {"kind": "sweep_exploit", "source": "one", "reps": 1},
        {"kind": "seed"},
        {"kind": "seed_exploit", "reps": 1},
        {"kind": "sweep", "id": "two", "level": 2, "pool_size": 3},
        {"kind": "sweep_exploit", "source": "two", "reps": 1},
        {"kind": "sweep_exploit", "source": "one", "reps": 1},
    ])
    result = run_attack(Adapter(), engine, verbose=False)
    assert result.outcomes[0].attempts == 14
    assert [record["phase"] for record in engine._attempt_meta] == (
        ["taxonomy"] * 2 + ["paraphrase"] * 2 + ["seed", "seed_paraphrase"]
        + ["taxonomy"] * 3 + ["paraphrase"] * 5)
    sources = [record["source_block"] for record in engine._attempt_meta if record["phase"] == "paraphrase"]
    assert sources == ["one", "one", "two", "two", "two", "one", "one"]


@pytest.mark.parametrize("file,budget", [("l1.yaml", 133), ("full.yaml", 529), ("seed-exploit.yaml", 601)])
def test_paper_strategy_budgets(monkeypatch, file, budget):
    monkeypatch.setattr(engine_module, "query_refiner_llm",
                        lambda *args: "[TEMPLATE_START] TEST-GOAL [TEMPLATE_END]")
    engine = StrategyEngine(load_strategy(ROOT / "configs/strategies" / file), refiner_model="test-model",
                            awareness_mode=3, use_judge=False, use_context=False, use_legitimize=False)
    result = run_attack(Adapter(), engine, verbose=False)
    assert engine.attempt_budget == result.outcomes[0].attempts == budget


def test_infrastructure_failures_are_not_exploit_sources_and_successes_are_not_failures(monkeypatch):
    engine = make_engine(monkeypatch, [{"kind": "sweep", "id": "s", "limit": 3},
                                       {"kind": "sweep_exploit", "source": "s", "reps": 2}])
    result = run_attack(Adapter(hits={1}, infrastructure={2}), engine,
                        verbose=False, stop_at_first_success=False)
    assert result.outcomes[0].attempts == 5
    variants = [record for record in engine._attempt_meta if record["phase"] == "paraphrase"]
    assert {record["source_attempt"] for record in variants} == {3}
    assert len(engine._failed_templates) == 3
    assert engine._infra_failures == ["transport"]


def test_invalid_candidates_do_not_consume_target_attempts_or_corrupt_resume_counts(monkeypatch, tmp_path):
    engine = make_engine(monkeypatch, [{"kind": "sweep", "limit": 3}])
    answers = iter(["invalid", "[TEMPLATE_START] TEST-GOAL [TEMPLATE_END]",
                    "[TEMPLATE_START] TEST-GOAL [TEMPLATE_END]"])
    monkeypatch.setattr(engine_module, "query_refiner_llm", lambda *args: next(answers))
    result = run_attack(Adapter(), engine, verbose=False, logdir=tmp_path)
    assert result.outcomes[0].attempts == 2
    log = json.loads((tmp_path / "refiner_logs/test/u__i.json").read_text())
    assert log["total_attempts"] == 2 and log["total_candidates"] == 3
    assert run_attack(Adapter(), engine, verbose=False, logdir=tmp_path, resume=True).outcomes[0].attempts == 2


def test_large_combinations_are_lazy_and_indexed_correctly():
    sequences = load_sequences()
    schedule, _ = build_combo_schedule(sequences, stack_only=6)
    assert len(schedule) == math.comb(132, 6)
    assert schedule[0] == sequences[:6]
    assert schedule[-1] == sequences[-6:]
    small, _ = build_combo_schedule(sequences[:7], stack_depth=3, stack_pool=5)
    expected = [[sequence] for sequence in sequences[:7]]
    expected += [list(combo) for level in (2, 3) for combo in itertools.combinations(sequences[:5], level)]
    assert list(small) == expected
    assert all(small[index] == combo for index, combo in enumerate(expected))


def test_a_bounded_strategy_can_search_even_larger_combination_spaces(monkeypatch):
    engine = make_engine(monkeypatch, [{"kind": "sweep", "level": 20, "limit": 2}])
    result = run_attack(Adapter(), engine, verbose=False)
    assert engine.num_combos == result.outcomes[0].attempts == 2
    assert all(record["stack_level"] == 20 for record in engine._attempt_meta)


def test_strategy_ignores_legacy_schedule_environment(monkeypatch):
    monkeypatch.setenv("TAX_MAX_SEQUENCES", "1")
    monkeypatch.setenv("TAX_TECHNIQUES", "F1")
    monkeypatch.setenv("TAX_STACK_ONLY", "4")
    engine = make_engine(monkeypatch, [{"kind": "sweep", "limit": 4}])
    assert engine.num_combos == 4
    assert run_attack(Adapter(), engine, verbose=False).outcomes[0].attempts == 4


def test_custom_taxonomy_supports_techniques_without_subunits_and_unicode(tmp_path):
    from autopi.core.taxonomy import seq_allows_unicode

    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps({"taxonomy": {"families": [{"id": "CUSTOM", "name": "Custom",
                    "techniques": [{"id": "CUSTOM.T1", "name": "Custom construction",
                                    "definition": "A researcher-defined construction", "allow_unicode": True}]}]}}))
    sequences = load_sequences(path)
    assert len(sequences) == 1 and sequences[0]["id"] == "CUSTOM.T1"
    assert seq_allows_unicode(sequences[0]) is True


def test_cli_uses_strategy_stop_and_rejects_mixed_schedule_flags(monkeypatch, tmp_path):
    import autopi.cli as cli

    monkeypatch.setattr(cli, "_build_adapter", lambda *args, **kwargs: Adapter(hits={1}))
    monkeypatch.setattr(engine_module, "query_refiner_llm",
                        lambda *args: "[TEMPLATE_START] TEST-GOAL [TEMPLATE_END]")
    path = tmp_path / "strategy.yaml"
    path.write_text("budget: 3\nstop: budget\nblocks:\n  - kind: seed\n  - kind: sweep\n    limit: 2\n")
    args = ["run", "--benchmark", "agentdojo", "--target", "vllm:1234:any-model", "--strategy", str(path)]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "in 3 attempt(s)" in result.output
    result = CliRunner().invoke(main, args + ["-k", "5"])
    assert result.exit_code == 2 and "conflicting flags" in result.output
