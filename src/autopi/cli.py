"""AutoPI command line: configurable models, benchmarks and composable strategies."""

from __future__ import annotations

import os

import click

from autopi.core.taxonomy import SWEEP_ORDERS
from autopi.harness.runner import run_attack
from autopi.models.endpoints import TOOL_CALLING_MODES
from autopi.models.registry import (
    TARGET_THINKING_CHOICES,
    resolve_target_max_tokens,
    resolve_target_thinking,
)

# Mirrors autopi.adapters.agentdojo.PLACEMENT_MODES (kept here so --help needs no agentdojo import).
INJECTION_PLACEMENTS = ("agentdojo", "all", "first")


def _build_adapter(benchmark: str, target: str, ipi_judge: str | None, suites: list[str] | None,
                   user_tasks: list[str] | None, injection_tasks: list[str] | None,
                   benchmark_version: str | None, defense: str | None,
                   ipi_ignore_concealment: bool = False, target_thinking: str | None = None,
                   target_max_tokens: int | None = None, ipi_behaviors_path: str | None = None,
                   target_endpoint=None, ipi_judge_endpoint=None, ipi_worldsim=None, ipi_worldsim_endpoint=None,
                   journal=None, ipi_max_steps=None, ipi_harness_retries=2,
                   injection_placement: str = "agentdojo"):
    """The benchmark's adapter. The ``ipi_*`` judge and WorldSim are IPI Arena's own scoring
    models; AutoPI's judge, which diagnoses failed attempts, belongs to the engine instead."""
    if benchmark == "agentdojo":
        from autopi.adapters.agentdojo import AgentDojoAdapter

        return AgentDojoAdapter(
            target, benchmark_version=benchmark_version or "v1.2.2", suites=suites or None,
            user_tasks=user_tasks or None, injection_tasks=injection_tasks or None, defense=defense,
            target_thinking=target_thinking, target_max_tokens=target_max_tokens,
            endpoint=target_endpoint, journal=journal, injection_placement=injection_placement,
        )
    if benchmark == "ipiarena":
        from autopi.adapters.ipi_arena import IPIArenaAdapter

        return IPIArenaAdapter(target, judge=ipi_judge, categories=suites or None,
                               behaviors=user_tasks or None, behaviors_path=ipi_behaviors_path,
                               ignore_concealment=ipi_ignore_concealment, target_thinking=target_thinking,
                               target_endpoint=target_endpoint, judge_endpoint=ipi_judge_endpoint,
                               worldsim=ipi_worldsim, worldsim_endpoint=ipi_worldsim_endpoint, journal=journal,
                               max_steps=ipi_max_steps, harness_retries=ipi_harness_retries)
    raise click.BadParameter(f"unknown benchmark {benchmark!r}")


# Environment variables the provider SDKs read to reroute or switch their requests. A manifest
# records the ones that are set (a URL without any user:password part): the native SDK paths
# (AgentDojo targets, the Anthropic and Google generators) and the OpenAI generator honour them,
# while IPI Arena's clients use each endpoint's base_url or the provider's public URL.
_PROVIDER_ENVIRONMENT = ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "GOOGLE_GEMINI_BASE_URL",
                         "GOOGLE_VERTEX_BASE_URL", "GOOGLE_GENAI_USE_VERTEXAI", "OLLAMA_BASE_URL")


def _provider_environment() -> dict:
    import re

    return {name: re.sub(r"//[^/@]*@", "//", os.environ[name]) for name in _PROVIDER_ENVIRONMENT
            if os.environ.get(name)}


def _check_generator(engine, *, judge: bool = False) -> None:
    """Send the generator one short request through the engine's own request path, with the
    run's settings, and with ``judge`` also AutoPI's judge on its own endpoint. Raise a
    ClickException when one fails, unless the provider refused the text on content-policy
    grounds (the settings were accepted)."""
    from autopi.harness.journal import TraceWriteError
    from autopi.models.clients import is_content_policy_rejection

    for who, querying_judge in [("the generator", False)] + ([("AutoPI's judge (--autopi-judge)", True)] if judge else []):
        engine._querying_judge = querying_judge
        try:
            engine._query("Reply with the single word OK.")
        except TraceWriteError as exc:
            # The trace could not be written; that says nothing about the model.
            raise click.ClickException(str(exc)) from exc
        except Exception as exc:
            if is_content_policy_rejection(exc):
                continue
            raise click.ClickException(
                f"{who} rejected a test request with the run's settings ({type(exc).__name__}: "
                f"{str(exc)[:200]}). Check its model id, key and settings (for example temperature, "
                "max_tokens or max_tokens_field, thinking) before starting the run") from exc
        finally:
            engine._querying_judge = False


def _text_model_endpoint(spec: str, catalog, *, tokenizer_path=None, ollama_subject: str = "the Ollama generator"):
    """The endpoint of a role that only writes text, the generator or AutoPI's judge, with the run's
    defaults: an exact tokenizer checked at setup and a 1,024-token output cap unless set."""
    from autopi.models.endpoints import override_endpoint, resolve_endpoint

    endpoint = override_endpoint(resolve_endpoint(spec, catalog, role="refiner"), tokenizer_path=tokenizer_path)
    if endpoint.provider == "ollama" and endpoint.reasoning_effort:
        # The alias may serve as the target too, where Ollama's OpenAI-compatible API reads it.
        click.echo(f"[autopi] note: {ollama_subject} calls Ollama's native chat API, which takes no "
                   "reasoning_effort; it sends think from the alias's thinking setting. Put the native "
                   "API's own field in extra_body to set an effort there.", err=True)
    if endpoint.tokenizer_path:
        from autopi.models.recovery import get_decoder

        get_decoder(endpoint.tokenizer_path)
    if endpoint.max_tokens is None:
        endpoint = override_endpoint(endpoint, max_tokens=int(os.environ.get("VLLM_REFINER_MAX_TOKENS", "1024")))
    return endpoint


# Options renamed to the paper's terms in 0.4.0. Each old name still works: it is a hidden option
# of the same type, merged into the new one right after the command line is read, so commands and
# scripts written for earlier versions run unchanged. Saved files keep their field names
# (docs/glossary.md). -k/--paraphrase-reps, the old name of --sweep-exploits, is not listed here: in
# a --seed-only run it set the seed exploits, so it is resolved in _exploit_counts.
_OLD_NAMES = {
    "generator": ("--refiner",),
    "generator_tokenizer": ("--refiner-tokenizer",),
    "ipi_judge": ("--judge",),
    "ipi_worldsim": ("--worldsim",),
    "seed_exploits": ("--seed-para-reps",),
    "sweep_before_seed_exploits": ("--sweep-before-seed-para",),
    "max_sweep_level": ("--stack-depth",),
    "only_sweep_level": ("--stack-only",),
    "sweep_pool_size": ("--stack-pool",),
    "random_sweep_pool": ("--stack-random",),
    "sweep_pool_seed": ("--stack-seed",),
    "max_subtechniques": ("--max-sequences",),
}
_OLD_NAMES_USED = "autopi.old_names_used"
_COMMANDLINE = click.core.ParameterSource.COMMANDLINE


class _RunCommand(click.Command):
    """``autopi run``, which also accepts the old names of its renamed options (not in --help)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        options = {param.name: param for param in self.params}
        for name, old in _OLD_NAMES.items():
            new = options[name]
            kind = {"is_flag": True} if new.is_flag else {"type": new.type}
            self.params.append(click.Option([*old, f"old_{name}"], hidden=True, default=None, **kind))

    def parse_args(self, ctx, args):
        rest = super().parse_args(ctx, args)
        used = ctx.meta.setdefault(_OLD_NAMES_USED, {})
        for name, old in _OLD_NAMES.items():
            value = ctx.params.pop(f"old_{name}")
            if ctx.get_parameter_source(f"old_{name}") != _COMMANDLINE:
                continue
            if ctx.get_parameter_source(name) == _COMMANDLINE and ctx.params[name] != value:
                raise click.UsageError(f"{'/'.join(old)} is the old name of {_flag(ctx, name)}; "
                                       "give only one of them", ctx=ctx)
            ctx.params[name] = value
            ctx.set_parameter_source(name, _COMMANDLINE)
            used[name] = "/".join(old)
        return rest


def _flag(context: click.Context, name: str) -> str:
    """How the command line named option ``name``: by its old name when that one was used."""
    used = context.meta.get(_OLD_NAMES_USED, {})
    if name in used:
        return used[name]
    return "/".join(next(param.opts for param in context.command.params if param.name == name))


def _exploit_counts(context: click.Context, seed_only, sweep_exploits: int, seed_exploits: int,
                    old_k: int | None) -> tuple[int, int, bool]:
    """Return the engine's ``(paraphrase_reps, seed_paraphrase_reps)`` and whether the run is
    seed-only (the flag, else ``TAX_SEED_ONLY``, as the engine reads it).

    A --seed-only run has no sweep: its rewrites of the seed come from --seed-exploits and reach
    the engine as ``paraphrase_reps``, as ``-k`` gave them before 0.4.0, so both spellings write
    the same manifest and logs. ``-k``/``--paraphrase-reps`` still sets the sweep exploits, or with
    --seed-only the seed exploits."""
    if seed_only is None:
        seed_only = os.environ.get("TAX_SEED_ONLY", "0") != "0"
    if seed_only and context.get_parameter_source("sweep_exploits") == _COMMANDLINE:
        raise ValueError("--seed-only (or TAX_SEED_ONLY) runs no sweep, so --sweep-exploits has no failed "
                         "sweep line to vary; set the number of rewrites of the seed with --seed-exploits")
    if old_k is not None:
        name = "seed_exploits" if seed_only else "sweep_exploits"
        current = seed_exploits if seed_only else sweep_exploits
        if context.get_parameter_source(name) == _COMMANDLINE and current != old_k:
            raise ValueError(f"-k/--paraphrase-reps and {_flag(context, name)} both set the number of "
                             f"{'seed' if seed_only else 'sweep'} exploits; give only one of them")
        if seed_only:
            seed_exploits = old_k
        else:
            sweep_exploits = old_k
    if seed_only:
        return seed_exploits, 0, True
    return sweep_exploits, seed_exploits, False


@click.group()
@click.version_option(package_name="autopi", prog_name="autopi")
def main() -> None:
    """AutoPI — taxonomy-driven adaptive prompt-injection evaluation."""


@main.command(cls=_RunCommand)
@click.option("--benchmark", type=click.Choice(["agentdojo", "ipiarena"]), required=True)
@click.option("--mode", type=click.IntRange(1, 3), default=3, show_default=True, help="Awareness mode of the generator: 3 is the paper's scenario-aware setting (it sees the user task and the injection goal); 1 and 2 write goal-blind templates.")
@click.option("--target", required=True, help="Configured model alias, provider:MODEL_ID, or vllm:PORT:MODEL_ID.")
@click.option("--generator", default=None, help="Generator model spec: the LLM that writes the injections. Defaults to --target. AutoPI's judge runs on it too, unless --autopi-judge names another model.")
@click.option("--autopi-judge", default=None,
              help="AutoPI's judge: the LLM that diagnoses why an attempt failed, to guide the next variant (mechanism ①). "
                   "A model alias or spec, as for --generator; defaults to --generator, as in the paper. It does not "
                   "score attempts: on IPI Arena that is the scoring judge, --ipi-judge.")
@click.option("--ipi-judge", default=None, help="IPI Arena's scoring judge, part of the benchmark's own scoring: it decides whether an attempt succeeded. Defaults to --target. Not AutoPI's judge, which --autopi-judge sets.")
@click.option("--ipi-worldsim", default=None, help="IPI Arena's tool simulator, WorldSim (defaults to --ipi-judge or --target).")
@click.option("--models-config", type=click.Path(exists=True, dir_okay=False), default=None, help="YAML model alias definitions.")
@click.option("--strategy", type=click.Path(exists=True, dir_okay=False), default=None, help="Ordered/repeatable strategy blocks in YAML.")
@click.option("--target-tokenizer", type=click.Path(exists=True, dir_okay=False), default=None, help="Exact tokenizer.json for prompted-tool target recovery.")
@click.option("--generator-tokenizer", type=click.Path(exists=True, dir_okay=False), default=None, help="Exact tokenizer.json for raw-token recovery of generator replies, and of AutoPI's judge replies while the judge runs on the generator. A separate --autopi-judge uses its alias's tokenizer_path.")
@click.option("--trace-dir", type=click.Path(file_okay=False), default=None, help="Incremental traces (default: LOGDIR/traces).")
@click.option("--dry-run", is_flag=True, help="Validate settings and enumerate scenarios without calling models.")
@click.option("--ipi-max-steps", type=click.IntRange(1), default=None, help="IPI Arena target operation limit (benchmark default: 5).")
@click.option("--ipi-harness-retries", type=click.IntRange(0), default=2, help="Retries after IPI harness errors.")
@click.option("--suite", "suites", multiple=True, help="Restrict to these suites/categories (repeatable).")
@click.option("-ut", "--user-task", "user_tasks", multiple=True,
              help="Restrict to these user task ids (repeatable). IPI Arena: scenario ids.")
@click.option("-it", "--injection-task", "injection_tasks", multiple=True, help="Restrict to these injection task ids.")
@click.option("--benchmark-version", default=None, help="AgentDojo benchmark version.")
@click.option("--defense", default=None, help="AgentDojo defense to run under (default: none).")
@click.option("--sweep-exploits", type=click.IntRange(0), default=0, show_default=True,
              help="Sweep exploits: N variants of each failed sweep line (r in the paper).")
@click.option("-k", "--paraphrase-reps", "paraphrase_reps", type=click.IntRange(0), default=None, hidden=True)
@click.option("--max-sweep-level", default=1, show_default=True, help="Run Ln sweeps for n = 1 up to this level, one line per set of n sub-techniques (1 = the L1 sweep only).")
@click.option("--sweep-pool-size", default=None, type=int, help="Number of sub-techniques that --only-sweep-level and Ln sweeps with n >= 2 draw from: the first in the sweep order, or a random draw with --random-sweep-pool [default: all].")
@click.option("--seed-exploits", type=click.IntRange(0), default=0, show_default=True,
              help="Seed exploits: N rewrites of the seed (r₀ in the paper), tried before the sweep, or "
                   "after the seed alone with --seed-only.")
@click.option("--seed-only", is_flag=True, default=None,
              help="Try only the seed and its --seed-exploits rewrites, no taxonomy sweep. Env: TAX_SEED_ONLY.")
@click.option("--no-seed", is_flag=True, default=None,
              help="Skip the seed and start from the sweep (not with --seed-only / --seed-exploits). "
                   "Env: TAX_NO_SEED.")
@click.option("--sweep-before-seed-exploits", is_flag=True, default=None,
              help="Run the sweep before the seed exploits. Env: TAX_SWEEP_BEFORE_SEEDPARA.")
@click.option("--seed-source", type=click.Choice(["benchmark", "agentdojo"]), default=None,
              help="Which built-in fixed seed to use: the benchmark's own default [default], or AgentDojo's "
                   "important_instructions seed reused on any benchmark. Env: TAX_SEED_SOURCE.")
@click.option("--seed-file", default=None,
              help="Use the fixed seed read from this file (must contain <task_goes_here>); overrides "
                   "--seed-source. For a benchmark that ships no seed (e.g. IPI Arena). Env: TAX_SEED_FILE.")
@click.option("--techniques", default=None,
              help="Sweep only these taxonomy ids, comma-separated: classes (F2), techniques (F2.T5), "
                   "or sub-techniques (F2.T5.S2). Env: TAX_TECHNIQUES.")
@click.option("--max-subtechniques", default=None, type=int,
              help="Sweep only the first N sub-techniques of the sweep order. Env: TAX_MAX_SEQUENCES.")
@click.option("--order", default=None, type=click.Choice(list(SWEEP_ORDERS)),
              help="Sweep order [default: priority]. Env: TAX_ORDER.")
@click.option("--order-seed", default=None, type=int,
              help="Seed of the random sweep order [default: 0]. Env: TAX_ORDER_SEED.")
@click.option("--only-sweep-level", default=None, type=int,
              help="Restrict the sweep to the Ln sweep of the pool with n = N, one line per set of N sub-techniques (1 = an L1 sweep of the pool). Env: TAX_STACK_ONLY.")
@click.option("--random-sweep-pool", is_flag=True, default=None,
              help="Draw the pool at random instead of taking the first --sweep-pool-size sub-techniques. Env: TAX_STACK_RANDOM.")
@click.option("--sweep-pool-seed", default=None, type=int, help="Seed of the random pool. Env: TAX_STACK_SEED.")
@click.option("--no-early-stop", is_flag=True,
              help="Keep attacking a data point after a success, through the rest of its schedule (the "
                   "budget, or earlier if the schedule ends first). The log then lists every success "
                   "(technique coverage of a scenario, attempt-level ASR).")
@click.option("--max-attempts", default=None, type=click.IntRange(1), help="Hard cap on target calls per data point.")
@click.option("--no-judge", is_flag=True, help="Disable AutoPI's judge (①), which diagnoses failed attempts. IPI Arena's scoring judge is part of the benchmark and always runs.")
@click.option("--no-context", is_flag=True, help="Disable context-blending (②).")
@click.option("--no-legitimize", is_flag=True, help="Disable the legitimize note.")
@click.option("--target-thinking", default=None, type=click.Choice(list(TARGET_THINKING_CHOICES)),
              help="Thinking mode of a locally served target: off, on, or server (send nothing and let "
                   "the server decide). Overrides the target's own setting. A vllm:PORT:MODEL_ID target "
                   "defaults to off; any other target keeps its alias's or spec's value (server unless set). "
                   "Env: AUTOPI_TARGET_THINKING.")
@click.option("--target-max-tokens", default=None, type=int,
              help="Output cap of the target, on either benchmark and any provider (default: the alias's own, "
                   "else the benchmark client's: none for OpenAI-compatible AgentDojo targets, 1,024 for "
                   "Anthropic, 65,535 for Google, 4,096 on IPI Arena). Env: AUTOPI_TARGET_MAX_TOKENS.")
@click.option("--tool-calling", default=None, type=click.Choice(list(TOOL_CALLING_MODES)),
              help="How an AgentDojo target calls tools: native (default; the provider's or server's "
                   "tool-calling API, as AgentDojo runs models with tool support) or prompted (AgentDojo's "
                   "LocalLLM: tools listed in the system prompt, <function=...> text parsed from the reply, "
                   "tool results sent back as JSON). Overrides the target alias's tool_mode. "
                   "Env: AUTOPI_TOOL_CALLING.")
@click.option("--injection-placement", default=None, type=click.Choice(list(INJECTION_PLACEMENTS)),
              help="AgentDojo: where the injection goes. agentdojo (default): the injection vectors "
                   "AgentDojo's own attacks fill, the ones the user task reads; all: every injection vector "
                   "of the suite; first: only the first of AgentDojo's vectors. "
                   "Env: AUTOPI_INJECTION_PLACEMENT.")
@click.option("--ipi-behaviors-path", default=None,
              help="IPI Arena: directory of scenario files (default: the installed benchmark's own). "
                   "Env: IPI_ARENA_BEHAVIORS_PATH.")
@click.option("--ipi-ignore-concealment", is_flag=True, help="IPI Arena: count a harmful action as success even if not concealed (drops the concealment/hiding criterion).")
@click.option("--taxonomy-path", default=None, help="Use another taxonomy JSON with the same layout.")
@click.option("--limit", default=None, type=click.IntRange(1), help="Attack only the first N data points (smoke tests).")
@click.option("--shard", default=0, type=int, help="This worker's shard index (0-based) for parallel sharded runs.")
@click.option("--num-shards", default=1, type=int, help="Total number of shards; this worker takes points [shard::num-shards].")
@click.option("--resume", is_flag=True, help="Skip scenarios whose log already exists (resume an interrupted run); "
                                              "scenarios an earlier run lost to infrastructure errors run last.")
@click.option("--logdir", default=None, help="Output directory: the run manifest, one JSON log per scenario under refiner_logs/, and traces.")
def run(benchmark, mode, target, generator, autopi_judge, ipi_judge, suites, user_tasks, injection_tasks, benchmark_version,
        defense, sweep_exploits, paraphrase_reps, max_sweep_level, sweep_pool_size, seed_exploits, seed_only, no_seed,
        sweep_before_seed_exploits, seed_source, seed_file, techniques, max_subtechniques, order, order_seed,
        only_sweep_level, random_sweep_pool, sweep_pool_seed, no_early_stop, max_attempts,
        no_judge, no_context, no_legitimize, target_thinking, target_max_tokens, ipi_behaviors_path,
        ipi_ignore_concealment, taxonomy_path, limit, shard, num_shards, logdir, resume,
        ipi_worldsim, models_config, strategy, target_tokenizer, generator_tokenizer, trace_dir, dry_run,
        ipi_max_steps, ipi_harness_retries, tool_calling, injection_placement) -> None:
    """Run the taxonomy attack against a benchmark and print the ASR summary."""
    import hashlib
    import re
    from pathlib import Path

    from autopi.adapters.base import BenchmarkModelConfigError
    from autopi.core.engine import GeneratorUnavailableError, TaxonomyEngine
    from autopi.core.refiner import RefinerConfig
    from autopi.core.seeds import resolve_seed
    from autopi.core.strategy import StrategyEngine, load_strategy
    from autopi.harness.journal import Journal
    from autopi.harness.manifest import benchmark_data_digest, check_run_manifest, save_run_manifest
    from autopi.harness.runner import (
        CorruptScenarioLogError,
        TargetUnavailableError,
        check_adapter_points,
        pending_points,
    )
    from autopi.models.endpoints import load_model_catalog, override_endpoint, resolve_endpoint

    if num_shards < 1 or not 0 <= shard < num_shards:
        raise click.UsageError("--num-shards must be positive and 0 <= --shard < --num-shards")
    if resume and not logdir:
        raise click.UsageError("--resume reads each scenario's log, so it needs --logdir")
    if benchmark == "ipiarena" and (defense or benchmark_version or (injection_tasks and set(injection_tasks) != {"ipi"})):
        raise click.UsageError("IPI Arena uses scenario ids via --user-task and its own scoring; AgentDojo defense/version/task options do not apply")
    context = click.get_current_context()
    if benchmark == "ipiarena":
        # IPI Arena's own client always calls tools natively and places each scenario's injection
        # itself; the AgentDojo options are refused on the command line and ignored from the env.
        given = [_flag(context, name) for name in ("tool_calling", "injection_placement")
                 if context.get_parameter_source(name) == _COMMANDLINE]
        if given:
            raise click.UsageError(f"{' and '.join(given)} {'apply' if len(given) > 1 else 'applies'} to "
                                   "AgentDojo; IPI Arena calls tools natively and places the injection where "
                                   "each scenario defines it")
        tool_calling = injection_placement = None
        if context.meta.get(_OLD_NAMES_USED, {}).get("ipi_judge") == "--judge":
            click.echo("[autopi] note: --judge is the old name of --ipi-judge, IPI Arena's scoring judge. AutoPI's "
                       "own judge, which diagnoses failed attempts, is set with --autopi-judge.", err=True)
    else:
        # The IPI Arena options have no meaning on AgentDojo (its scoring is programmatic and it
        # simulates no tools), so they are refused rather than silently ignored.
        given = [_flag(context, name) for name in ("ipi_judge", "ipi_worldsim", "ipi_behaviors_path",
                                                   "ipi_ignore_concealment", "ipi_max_steps", "ipi_harness_retries")
                 if context.get_parameter_source(name) == _COMMANDLINE]
        if given:
            raise click.UsageError(f"{', '.join(given)} {'apply' if len(given) > 1 else 'applies'} to IPI Arena "
                                   "only; AgentDojo scores attempts with its own checks and simulates no tools"
                                   + (". AutoPI's own judge, which diagnoses failed attempts, is set with "
                                      "--autopi-judge" if context.get_parameter_source("ipi_judge") == _COMMANDLINE
                                      else ""))
        tool_calling = tool_calling or os.environ.get("AUTOPI_TOOL_CALLING") or None
        injection_placement = injection_placement or os.environ.get("AUTOPI_INJECTION_PLACEMENT") or "agentdojo"
        if tool_calling is not None and tool_calling not in TOOL_CALLING_MODES:
            raise click.UsageError(f"AUTOPI_TOOL_CALLING must be one of {', '.join(TOOL_CALLING_MODES)}; got {tool_calling!r}")
        if injection_placement not in INJECTION_PLACEMENTS:
            raise click.UsageError(f"AUTOPI_INJECTION_PLACEMENT must be one of {', '.join(INJECTION_PLACEMENTS)}; "
                                   f"got {injection_placement!r}")
    strategy_data = None
    try:
        if strategy:
            strategy_data = load_strategy(strategy)
            schedule_flags = ("sweep_exploits", "paraphrase_reps", "max_sweep_level", "sweep_pool_size",
                              "seed_exploits", "seed_only", "no_seed", "sweep_before_seed_exploits", "techniques",
                              "max_subtechniques", "order", "order_seed", "only_sweep_level", "random_sweep_pool",
                              "sweep_pool_seed")
            conflicts = sorted(_flag(context, name) for name in schedule_flags
                               if context.get_parameter_source(name) == _COMMANDLINE)
            if conflicts:
                raise ValueError("put schedule choices inside --strategy; conflicting flags: " + ", ".join(conflicts))
            no_seed = not any(block["kind"] == "seed" for block in strategy_data["blocks"])
            sweep_reps = seed_reps = 0  # a strategy sets its own exploit counts
        else:
            if no_seed is None:
                no_seed = os.environ.get("TAX_NO_SEED", "0") != "0"
            sweep_reps, seed_reps, seed_only_run = _exploit_counts(context, seed_only, sweep_exploits,
                                                                   seed_exploits, paraphrase_reps)
            if no_seed and (seed_only_run or seed_reps > 0):
                raise ValueError("--no-seed leaves no seed to rewrite: drop --seed-only / "
                                 + _flag(context, "seed_exploits"))
        generator = generator or target
        catalog = load_model_catalog(models_config)
        target_endpoint = resolve_endpoint(target, catalog)
        # Legacy vllm specs retain the paper's explicit thinking-off default. Named
        # model aliases retain their configured value unless a flag/env overrides it.
        thinking_override = target_thinking or (os.environ.get("AUTOPI_TARGET_THINKING") or None)
        if thinking_override is not None or target.startswith("vllm:"):
            target_thinking = resolve_target_thinking(thinking_override)
        else:
            target_thinking = target_endpoint.thinking
        target_max_tokens = resolve_target_max_tokens(target_max_tokens)
        target_endpoint = override_endpoint(target_endpoint, thinking=target_thinking,
                                            max_tokens=target_max_tokens, tokenizer_path=target_tokenizer,
                                            tool_mode=tool_calling)
        refiner_endpoint = _text_model_endpoint(generator, catalog, tokenizer_path=generator_tokenizer)
        # AutoPI's judge diagnoses failed attempts. It runs on the generator's endpoint unless
        # --autopi-judge names another model, and it never scores: IPI Arena's scoring judge
        # (--ipi-judge) is part of the benchmark.
        if autopi_judge is not None and no_judge:
            raise ValueError("--no-judge switches AutoPI's judge off, so --autopi-judge has nothing to set; "
                             "give only one of them")
        if (autopi_judge or generator) == generator:
            autopi_judge_endpoint = refiner_endpoint
        else:
            autopi_judge_endpoint = _text_model_endpoint(autopi_judge, catalog,
                                                         ollama_subject="AutoPI's judge, on Ollama,")
        if benchmark == "ipiarena":
            ipi_judge_endpoint = resolve_endpoint(ipi_judge or target, catalog)
            ipi_worldsim_endpoint = resolve_endpoint(ipi_worldsim or ipi_judge or target, catalog)
        else:
            # AgentDojo scores attempts with its own checks: it has no LLM judge and no simulator.
            ipi_judge_endpoint = ipi_worldsim_endpoint = None
        secret_envs = {ep.api_key_env for ep in (target_endpoint, refiner_endpoint, autopi_judge_endpoint,
                                                 ipi_judge_endpoint, ipi_worldsim_endpoint) if ep and ep.api_key_env}
        journal = (Journal(trace_dir or Path(logdir) / "traces", extra_secret_envs=secret_envs)
                   if (trace_dir or logdir) and not dry_run else None)
        adapter = _build_adapter(benchmark, target, ipi_judge, list(suites), list(user_tasks),
                                 list(injection_tasks), benchmark_version, defense,
                                 ipi_ignore_concealment=ipi_ignore_concealment,
                                 target_thinking=target_thinking, target_max_tokens=target_max_tokens,
                                 ipi_behaviors_path=ipi_behaviors_path, target_endpoint=target_endpoint,
                                 ipi_judge_endpoint=ipi_judge_endpoint, ipi_worldsim=ipi_worldsim,
                                 ipi_worldsim_endpoint=ipi_worldsim_endpoint,
                                 journal=journal, ipi_max_steps=ipi_max_steps, ipi_harness_retries=ipi_harness_retries,
                                 injection_placement=injection_placement or "agentdojo")
        seed_template, seed_name = resolve_seed(
            no_seed=bool(no_seed), seed_source=seed_source or os.environ.get("TAX_SEED_SOURCE"),
            seed_file=seed_file or os.environ.get("TAX_SEED_FILE"),
            benchmark_default_template=getattr(adapter, "default_seed_template", None),
            benchmark_default_name=getattr(adapter, "default_seed_name", None), benchmark=benchmark,
            no_seed_hint="use a strategy without a seed block" if strategy_data else None)
        cfg = RefinerConfig(endpoint=refiner_endpoint, journal=journal)
        judge_cfg = (None if autopi_judge_endpoint is refiner_endpoint
                     else RefinerConfig(endpoint=autopi_judge_endpoint, journal=journal))
        common = dict(refiner_model=refiner_endpoint.model, refiner_config=cfg, judge_config=judge_cfg, awareness_mode=mode,
                      taxonomy_path=taxonomy_path, use_judge=False if no_judge else None,
                      use_context=False if no_context else None, use_legitimize=False if no_legitimize else None,
                      seed_template=seed_template, seed_name=seed_name)
        if strategy_data:
            engine = StrategyEngine(strategy_data, **common)
        else:
            engine = TaxonomyEngine(
                **common, paraphrase_reps=sweep_reps, stack_depth=max_sweep_level, stack_pool=sweep_pool_size,
                seed_paraphrase_reps=seed_reps, seed_only=True if seed_only else None,
                use_seed=False if no_seed else None,
                sweep_before_seed_para=True if sweep_before_seed_exploits else None,
                techniques=techniques, max_sequences=max_subtechniques, order=order, order_seed=order_seed,
                stack_only=only_sweep_level, stack_random=True if random_sweep_pool else None,
                stack_seed=sweep_pool_seed)
        # AgentDojo embeds the injection in a double-quoted YAML scalar (quotes escaped at
        # placement); IPI Arena places it in JSON, so quotes/backslashes are kept as written.
        engine.yaml_safe_output = benchmark == "agentdojo"
        if autopi_judge is not None and not engine.use_judge:
            raise ValueError("TAX_JUDGE=0 switches AutoPI's judge off, so --autopi-judge has nothing to set; "
                             "unset one of them")
        if autopi_judge is not None and not engine.calls_judge:
            click.echo("[autopi] note: this run has no seed or sweep exploits, the only attempts that use "
                       "AutoPI's judge, so the --autopi-judge model is never called.", err=True)
        stop = not no_early_stop and (not strategy_data or strategy_data["stop"] == "first_success")
        points = list(adapter.iter_points())[shard::num_shards]
        if limit is not None:
            points = points[:limit]
        if not points:
            raise ValueError("no scenarios match the selection/shard; check the suite and task ids")
        # The adapter's setup requirements for the scenarios this run will attack, after --limit,
        # --shard and --resume (IPI Arena: a browser scenario needs playwright and its Chromium).
        check_adapter_points(adapter, pending_points(points, logdir, resume))
        click.echo(f"[autopi] benchmark={benchmark} mode={mode} target={target} generator={generator} "
                   f"autopi_judge={(autopi_judge or generator) if engine.use_judge else 'off'} "
                   + (f"ipi_judge={ipi_judge or target} " if benchmark == "ipiarena" else "")
                   + f"sweep_lines={engine.num_combos} budget/pt={min(max_attempts or engine.attempt_budget, engine.attempt_budget)} "
                   f"points={len(points)} target_thinking={target_thinking} "
                   + (f"tool_calling={target_endpoint.tool_mode} placement={injection_placement} "
                      if benchmark == "agentdojo" else "")
                   + f"{'seed=' + engine.seed_name if engine.use_seed else 'no-seed'}" + (" no-early-stop" if not stop else ""))
        if dry_run:
            click.echo("Validated. No model calls were made.")
            return
        settings = None
        if logdir:
            from autopi.core.taxonomy import _BUNDLED_TAXONOMY
            settings = {
                "benchmark": benchmark, "benchmark_version": benchmark_version or getattr(adapter, "benchmark_version", None),
                # "diagnostic_judge" is AutoPI's judge (null when it is off); "judge" and "worldsim"
                # are IPI Arena's scoring judge and tool simulator (null on AgentDojo).
                "models": {"target": target_endpoint.public_config(), "refiner": refiner_endpoint.public_config(),
                           "diagnostic_judge": autopi_judge_endpoint.public_config() if engine.use_judge else None,
                           "judge": ipi_judge_endpoint.public_config() if ipi_judge_endpoint else None,
                           "worldsim": ipi_worldsim_endpoint.public_config() if ipi_worldsim_endpoint else None},
                "strategy": strategy_data or {"seed_only": engine.seed_only, "use_seed": engine.use_seed,
                    "k": sweep_reps, "seed_para_reps": seed_reps, "stack_depth": engine.stack_depth,
                    "stack_pool": engine.stack_pool, "stack_only": engine.stack_only, "stack_random": engine.stack_random,
                    "stack_seed": engine.stack_seed, "techniques": engine.techniques, "max_sequences": engine.max_sequences,
                    "order": engine.order, "order_seed": engine.order_seed, "sweep_before_seed_para": engine.sweep_before_seed_para},
                "mode": mode, "mechanisms": [engine.use_judge, engine.use_context, engine.use_legitimize],
                "generator_limits": {"max_consecutive_refiner_errors": engine.max_consecutive_refiner_errors,
                                     "max_consecutive_refiner_400": engine.max_consecutive_refiner_400,
                                     "max_consecutive_unusable": engine.max_consecutive_unusable},
                "stop_at_first_success": stop, "max_attempts": max_attempts, "defense": defense,
                "tool_calling": target_endpoint.tool_mode if benchmark == "agentdojo" else "native",
                "injection_placement": injection_placement if benchmark == "agentdojo" else "scenario",
                "seed_sha256": hashlib.sha256((seed_template or "").encode()).hexdigest(),
                "taxonomy_sha256": hashlib.sha256(Path(taxonomy_path or _BUNDLED_TAXONOMY).read_bytes()).hexdigest(),
                "ipi_ignore_concealment": ipi_ignore_concealment, "ipi_max_steps": ipi_max_steps,
                "ipi_harness_retries": ipi_harness_retries, "ipi_behaviors_path": ipi_behaviors_path,
                "environment": {key: value for key, value in os.environ.items() if key.startswith("TAX_") and not re.search(r"API_KEY|TOKEN$|SECRET|PASSWORD", key)},
                "provider_environment": _provider_environment(),
                "benchmark_data_sha256": benchmark_data_digest(adapter),
            }
            # Refuse a folder that holds another run before any request, so that nothing (not even
            # a trace of the generator check) is written into it.
            check_run_manifest(Path(logdir), settings)
        # One short test request to the generator before the first scenario, with the settings the
        # run will use, so that a generator rejecting them (an unsupported temperature or max_tokens,
        # a wrong model id or key) stops the run now instead of turning it into seed-only results.
        # AutoPI's judge on its own endpoint gets one too when the schedule calls it. Skipped when
        # the run makes no generator request (only the seed is tried).
        budget = min(max_attempts or engine.attempt_budget, engine.attempt_budget)
        if not (engine.use_seed and budget <= 1):
            _check_generator(engine, judge=autopi_judge_endpoint is not refiner_endpoint and engine.calls_judge)
        if settings is not None:
            save_run_manifest(Path(logdir), settings)
    except (ValueError, OSError) as exc:
        # Configuration/setup problems are usage errors. A failure during the run itself
        # (below) is not caught here, so a mid-run OSError is not mislabelled as a usage error.
        raise click.UsageError(str(exc)) from exc
    try:
        summary = run_attack(adapter, engine, logdir=logdir, max_attempts=max_attempts, points=points, resume=resume,
                             stop_at_first_success=stop, journal=journal)
    except CorruptScenarioLogError as exc:
        raise click.UsageError(str(exc)) from exc
    except (GeneratorUnavailableError, TargetUnavailableError, BenchmarkModelConfigError) as exc:
        raise click.ClickException(f"{exc}.") from exc
    click.echo("\n" + summary.format())
    if not summary.n and summary.infrastructure_only:
        # Nothing was measured: exit with an error so a script does not take the run for a result.
        raise click.ClickException(
            "no scenario was scored: every attempt failed in the infrastructure (see the lines above). "
            "Check the target, then rerun the same command with --resume")


if __name__ == "__main__":
    main()
