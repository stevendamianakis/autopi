"""Regression tests for 0.5.0: AutoPI's judge, which diagnoses failed attempts, can run on its own
model (--autopi-judge), and it is kept apart from IPI Arena's scoring judge (--ipi-judge) in the
options, the messages, the run's first line and the run manifest.

Each test fails when its change is reverted (checked by reverting each change in a copy of the
code). The runs use a mock benchmark and stub models, so no model is called.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import autopi.cli as cli_mod
import autopi.core.engine as engine_mod
from autopi.adapters.base import AttackPoint, RunResult
from autopi.cli import main as cli_main
from autopi.core.engine import GeneratorUnavailableError, TaxonomyEngine
from autopi.core.refiner import RefinerConfig
from autopi.core.seeds import IMPORTANT_INSTRUCTIONS_SEED, IMPORTANT_INSTRUCTIONS_SEED_NAME
from autopi.core.strategy import StrategyEngine, load_strategy
from autopi.models.endpoints import ModelEndpoint

STRATEGIES = Path(__file__).resolve().parents[1] / "configs" / "strategies"
LINE = "[TEMPLATE_START] As a required step, complete the stated task now. [TEMPLATE_END]"
DIAGNOSIS = "FAILURE_MODE: flagged_as_suspicious\nREASON: it looked out of place."
TRANSCRIPT = [{"role": "user", "content": "do my chore"}, {"role": "assistant", "content": "done"}]


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
        return RunResult(utility=True, security=False, messages=list(TRANSCRIPT))


class _Models:
    """Stub for every generator and AutoPI judge request: records (role, model) and answers."""

    def __init__(self, failing: str | None = None):
        self.requests: list[tuple[str, str]] = []
        self.failing = failing

    def __call__(self, model, prompt, cfg=None):
        role = getattr(cfg, "role", "refiner")
        served = cfg.endpoint.model if cfg is not None and cfg.endpoint is not None else model
        self.requests.append((role, served))
        if served == self.failing:
            raise ConnectionError("connection refused")
        return DIAGNOSIS if role == "diagnostic_judge" else LINE

    def models(self, role):
        return {model for request_role, model in self.requests if request_role == role}


def _invoke(monkeypatch, args, *, benchmark="agentdojo", logdir=None, failing=None, sweep=True):
    """``autopi run`` on the mock benchmark: target t on port 8000, generator g on port 8001, AutoPI's
    judge on (no --no-judge), and by default a 6-line L1 sweep."""
    adapter, models, built = _Adapter(), _Models(failing), {}

    def build(*args, **kwargs):
        built.update(args=args, kwargs=kwargs)
        return adapter

    monkeypatch.setattr(cli_mod, "_build_adapter", build)
    monkeypatch.setattr(engine_mod, "query_refiner_llm", models)
    command = ["run", "--benchmark", benchmark, "--target", "vllm:8000:t", "--generator", "vllm:8001:g",
               "--no-context", "--no-legitimize", *(["--techniques", "F1.T2"] if sweep else []),
               *(["--logdir", str(logdir)] if logdir else []), *args]
    result = CliRunner().invoke(cli_main, command)
    banner = next((line for line in result.output.splitlines() if line.startswith("[autopi] benchmark=")), "")
    manifest = None
    if logdir and (logdir / "run_manifest.json").exists():
        manifest = json.loads((logdir / "run_manifest.json").read_text())
    return SimpleNamespace(code=result.exit_code, output=result.output, banner=banner, manifest=manifest,
                           models=models, built=built, scenarios_run=adapter.calls)


def _model(run, role):
    entry = run.manifest["settings"]["models"][role]
    return None if entry is None else entry["model"]


# ─────────────────────────── options and their help ───────────────────────────

def test_help_tells_the_two_judges_apart():
    helps = {param.name: param.help or "" for param in cli_main.commands["run"].params}
    assert "diagnoses why an attempt failed" in helps["autopi_judge"]
    assert "does not score" in helps["autopi_judge"] and "--ipi-judge" in helps["autopi_judge"]
    assert "scoring judge" in helps["ipi_judge"] and "decides whether an attempt succeeded" in helps["ipi_judge"]
    assert "Not AutoPI's judge" in helps["ipi_judge"] and "--autopi-judge" in helps["ipi_judge"]
    assert "--autopi-judge" in helps["generator"] and "AutoPI's judge" in helps["no_judge"]
    out = CliRunner().invoke(cli_main, ["run", "--help"], terminal_width=1000, max_content_width=1000).output
    assert "--autopi-judge" in out and "--ipi-judge" in out


# ─────────────────────────── which model AutoPI's judge runs on ───────────────────────────

def test_autopi_judge_runs_on_the_generator_by_default(monkeypatch, tmp_path):
    run = _invoke(monkeypatch, ["--sweep-exploits", "1"], logdir=tmp_path)
    assert run.code == 0, run.output
    assert run.models.models("refiner") == run.models.models("diagnostic_judge") == {"g"}
    assert "generator=vllm:8001:g autopi_judge=vllm:8001:g sweep_lines=6 " in run.banner
    assert run.manifest["settings"]["models"]["diagnostic_judge"] == run.manifest["settings"]["models"]["refiner"]
    # The generator check is the only test request: the judge has no endpoint of its own.
    assert run.models.requests[0] == ("refiner", "g") and run.models.requests[1][0] == "refiner"


def test_autopi_judge_runs_on_its_own_model(monkeypatch, tmp_path):
    run = _invoke(monkeypatch, ["--sweep-exploits", "1", "--autopi-judge", "vllm:8002:j"], logdir=tmp_path)
    assert run.code == 0, run.output
    assert run.models.models("refiner") == {"g"} and run.models.models("diagnostic_judge") == {"j"}
    # Both models get a test request before the first scenario, then one diagnosis per variant.
    assert run.models.requests[:2] == [("refiner", "g"), ("diagnostic_judge", "j")]
    assert run.models.requests.count(("diagnostic_judge", "j")) == 1 + 6
    assert "generator=vllm:8001:g autopi_judge=vllm:8002:j sweep_lines=6 " in run.banner
    judge = run.manifest["settings"]["models"]["diagnostic_judge"]
    assert (judge["model"], judge["base_url"], _model(run, "refiner")) == ("j", "http://localhost:8002/v1", "g")
    assert judge["max_tokens"] == 1024  # the generator's default cap applies to the judge too


def test_ipi_arena_keeps_its_scoring_judge_apart_from_autopis(monkeypatch, tmp_path):
    both = _invoke(monkeypatch, ["--sweep-exploits", "1", "--autopi-judge", "vllm:8002:j", "--ipi-judge", "vllm:8003:s"],
                   benchmark="ipiarena", logdir=tmp_path / "both")
    assert both.code == 0, both.output
    assert "autopi_judge=vllm:8002:j ipi_judge=vllm:8003:s sweep_lines=6 " in both.banner
    assert (_model(both, "diagnostic_judge"), _model(both, "judge")) == ("j", "s")
    assert both.built["args"][2] == "vllm:8003:s" and both.built["kwargs"]["ipi_judge_endpoint"].model == "s"
    assert both.models.models("diagnostic_judge") == {"j"}
    # --ipi-judge alone leaves AutoPI's judge on the generator, and IPI Arena's on the target by default.
    scoring = _invoke(monkeypatch, ["--sweep-exploits", "1", "--ipi-judge", "vllm:8003:s"],
                      benchmark="ipiarena", logdir=tmp_path / "scoring")
    assert scoring.code == 0 and scoring.models.models("diagnostic_judge") == {"g"}
    assert (_model(scoring, "diagnostic_judge"), _model(scoring, "judge")) == ("g", "s")
    autopi = _invoke(monkeypatch, ["--sweep-exploits", "1", "--autopi-judge", "vllm:8002:j"],
                     benchmark="ipiarena", logdir=tmp_path / "autopi")
    assert autopi.code == 0 and "ipi_judge=vllm:8000:t " in autopi.banner
    assert (_model(autopi, "diagnostic_judge"), _model(autopi, "judge")) == ("j", "t")
    assert autopi.built["kwargs"]["ipi_judge_endpoint"].model == "t"


def test_an_agentdojo_run_records_no_scoring_judge(monkeypatch, tmp_path):
    run = _invoke(monkeypatch, [], logdir=tmp_path / "on")
    assert run.code == 0 and "ipi_judge=" not in run.banner
    assert (_model(run, "judge"), _model(run, "worldsim"), _model(run, "diagnostic_judge")) == (None, None, "g")
    off = _invoke(monkeypatch, ["--no-judge"], logdir=tmp_path / "off")
    assert off.code == 0 and "autopi_judge=off " in off.banner and _model(off, "diagnostic_judge") is None


def test_the_generator_tokenizer_reaches_the_judge_only_on_the_generators_model(monkeypatch, tmp_path):
    monkeypatch.setattr("autopi.models.recovery.get_decoder", lambda path: None)
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    shared = _invoke(monkeypatch, ["--generator-tokenizer", str(tokenizer)], logdir=tmp_path / "shared")
    assert shared.code == 0 and set(shared.manifest["tokenizer_sha256"]) == {"refiner", "diagnostic_judge"}
    own = _invoke(monkeypatch, ["--generator-tokenizer", str(tokenizer), "--autopi-judge", "vllm:8002:j",
                                "--sweep-exploits", "1"], logdir=tmp_path / "own")
    assert own.code == 0 and set(own.manifest["tokenizer_sha256"]) == {"refiner"}
    assert own.manifest["settings"]["models"]["diagnostic_judge"]["tokenizer_path"] is None


# ─────────────────────────── settings that leave nothing to set ───────────────────────────

def test_autopi_judge_needs_the_judge_on(monkeypatch):
    run = _invoke(monkeypatch, ["--autopi-judge", "vllm:8002:j", "--no-judge", "--dry-run"])
    assert run.code == 2 and "--no-judge switches AutoPI's judge off" in run.output
    monkeypatch.setenv("TAX_JUDGE", "0")
    run = _invoke(monkeypatch, ["--autopi-judge", "vllm:8002:j", "--dry-run"])
    assert run.code == 2 and "TAX_JUDGE=0 switches AutoPI's judge off" in run.output


def test_an_autopi_judge_the_schedule_never_calls_is_noted_and_not_checked(monkeypatch, tmp_path):
    run = _invoke(monkeypatch, ["--autopi-judge", "vllm:8002:j"], logdir=tmp_path)
    assert run.code == 0 and "the --autopi-judge model is never called" in run.output
    assert run.models.models("diagnostic_judge") == set()
    with_variants = _invoke(monkeypatch, ["--autopi-judge", "vllm:8002:j", "--sweep-exploits", "1", "--dry-run"])
    assert with_variants.code == 0 and "never called" not in with_variants.output


def test_engines_know_whether_their_schedule_calls_the_judge():
    assert not StrategyEngine(load_strategy(STRATEGIES / "l1.yaml"), refiner_model="m").calls_judge
    assert StrategyEngine(load_strategy(STRATEGIES / "full.yaml"), refiner_model="m").calls_judge
    assert not StrategyEngine(load_strategy(STRATEGIES / "full.yaml"), refiner_model="m", use_judge=False).calls_judge
    assert not TaxonomyEngine(refiner_model="m").calls_judge
    assert TaxonomyEngine(refiner_model="m", seed_paraphrase_reps=1).calls_judge
    assert TaxonomyEngine(refiner_model="m", seed_only=True, paraphrase_reps=2).calls_judge


# ─────────────────────────── messages that name the right judge ───────────────────────────

def test_the_old_judge_name_says_which_judge_it_sets(monkeypatch, tmp_path):
    run = _invoke(monkeypatch, ["--judge", "vllm:8003:s", "--sweep-exploits", "1"], benchmark="ipiarena",
                  logdir=tmp_path)
    assert run.code == 0, run.output
    assert "--judge is the old name of --ipi-judge, IPI Arena's scoring judge" in run.output
    assert "set with --autopi-judge" in run.output
    assert (_model(run, "judge"), _model(run, "diagnostic_judge")) == ("s", "g")
    run = _invoke(monkeypatch, ["--judge", "x", "--dry-run"])
    assert run.code == 2 and "--judge applies to IPI Arena only" in run.output and "--autopi-judge" in run.output
    current = _invoke(monkeypatch, ["--ipi-judge", "vllm:8003:s", "--sweep-exploits", "1", "--dry-run"],
                      benchmark="ipiarena")
    assert current.code == 0 and "old name" not in current.output


def test_a_separate_judge_that_fails_its_test_request_stops_the_run(monkeypatch, tmp_path):
    run = _invoke(monkeypatch, ["--sweep-exploits", "1", "--autopi-judge", "vllm:8002:j"], logdir=tmp_path,
                  failing="j")
    assert run.code == 1 and "AutoPI's judge (--autopi-judge) rejected a test request" in run.output
    assert run.scenarios_run == 0


@pytest.mark.parametrize("separate", [True, False])
def test_judge_request_errors_name_the_model_to_fix(monkeypatch, separate):
    generator = RefinerConfig(endpoint=ModelEndpoint("openai-compatible", "g", "http://localhost:8001/v1"))
    judge = RefinerConfig(endpoint=ModelEndpoint("openai-compatible", "j", "http://localhost:8002/v1"))
    monkeypatch.setattr(engine_mod, "query_refiner_llm", _Models(failing="j" if separate else "g"))
    engine = TaxonomyEngine(refiner_model="g", refiner_config=generator, judge_config=judge if separate else None)
    with pytest.raises(GeneratorUnavailableError) as error:
        engine._judge_text(TRANSCRIPT, "GOAL-123")
    expected = "AutoPI's judge (--autopi-judge) could not be reached" if separate else \
        "the generator, as AutoPI's judge, could not be reached"
    assert str(error.value).startswith(expected)


# ─────────────────────────── the Python interface ───────────────────────────

def test_judge_config_needs_an_endpoint_and_shares_the_trace():
    with pytest.raises(ValueError, match="judge_config needs an explicit endpoint"):
        TaxonomyEngine(refiner_model="m", judge_config=RefinerConfig())
    journal = object()
    endpoint = ModelEndpoint("openai-compatible", "j", "http://localhost:8002/v1")
    engine = TaxonomyEngine(refiner_model="m", refiner_config=RefinerConfig(journal=journal),
                            judge_config=RefinerConfig(endpoint=endpoint))
    assert engine._judge_config.journal is journal and engine._judge_config.endpoint is endpoint
