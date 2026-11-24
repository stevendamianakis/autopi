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


def _build_adapter(benchmark: str, target: str, judge: str | None, suites: list[str] | None,
                   user_tasks: list[str] | None, injection_tasks: list[str] | None,
                   benchmark_version: str | None, defense: str | None,
                   ipi_ignore_concealment: bool = False, target_thinking: str | None = None,
                   target_max_tokens: int | None = None, ipi_behaviors_path: str | None = None,
                   target_endpoint=None, judge_endpoint=None, worldsim=None, worldsim_endpoint=None,
                   journal=None, ipi_max_steps=None, ipi_harness_retries=2,
                   injection_placement: str = "agentdojo"):
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

        return IPIArenaAdapter(target, judge=judge, categories=suites or None,
                               behaviors=user_tasks or None, behaviors_path=ipi_behaviors_path,
                               ignore_concealment=ipi_ignore_concealment, target_thinking=target_thinking,
                               target_endpoint=target_endpoint, judge_endpoint=judge_endpoint,
                               worldsim=worldsim, worldsim_endpoint=worldsim_endpoint, journal=journal,
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


def _check_generator(engine) -> None:
    """Send the generator one short request through the engine's own request path, with the
    run's settings. Raise a ClickException when it fails, unless the provider refused the text on
    content-policy grounds (the settings were accepted)."""
    from autopi.models.clients import is_content_policy_rejection

    try:
        engine._query("Reply with the single word OK.")
    except Exception as exc:
        if is_content_policy_rejection(exc):
            return
        raise click.ClickException(
            f"the generator rejected a test request with the run's settings ({type(exc).__name__}: "
            f"{str(exc)[:200]}). Check its model id, key and settings (for example temperature, "
            "max_tokens or max_tokens_field, thinking) before starting the run") from exc


@click.group()
@click.version_option(package_name="autopi", prog_name="autopi")
def main() -> None:
    """AutoPI — taxonomy-driven adaptive prompt-injection evaluation."""


@main.command()
@click.option("--benchmark", type=click.Choice(["agentdojo", "ipiarena"]), required=True)
@click.option("--mode", type=click.IntRange(1, 3), default=3, show_default=True, help="Awareness mode of the generator: 3 is the paper's scenario-aware setting (it sees the user task and the injection goal); 1 and 2 write goal-blind templates.")
@click.option("--target", required=True, help="Configured model alias, provider:MODEL_ID, or vllm:PORT:MODEL_ID.")
@click.option("--refiner", default=None, help="Generator model spec (the code calls the generator the refiner); the judge uses it too. Defaults to --target.")
@click.option("--judge", default=None, help="IPI Arena's scoring judge (defaults to --target). AutoPI's own judge, which diagnoses failed attempts, uses --refiner.")
@click.option("--worldsim", default=None, help="IPI tool simulator model (defaults to --judge or --target).")
@click.option("--models-config", type=click.Path(exists=True, dir_okay=False), default=None, help="YAML model alias definitions.")
@click.option("--strategy", type=click.Path(exists=True, dir_okay=False), default=None, help="Ordered/repeatable strategy blocks in YAML.")
@click.option("--target-tokenizer", type=click.Path(exists=True, dir_okay=False), default=None, help="Exact tokenizer.json for prompted-tool target recovery.")
@click.option("--refiner-tokenizer", type=click.Path(exists=True, dir_okay=False), default=None, help="Exact tokenizer.json for raw-token recovery of generator and judge replies.")
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
@click.option("-k", "--paraphrase-reps", default=0, show_default=True, help="Sweep exploits: variants of each failed sweep line. With --seed-only, rewrites of the seed (seed exploits).")
@click.option("--stack-depth", default=1, show_default=True, help="Run Ln sweeps up to this n, one line per set of n sub-techniques (1 = L1 sweep only).")
@click.option("--stack-pool", default=None, type=int, help="Number of sub-techniques that --stack-only and Ln sweeps with n >= 2 draw from: the first in the sweep order, or a random draw with --stack-random [default: all].")
@click.option("--seed-para-reps", default=0, show_default=True, help="Seed exploits (rewrites of the seed) before the sweep.")
@click.option("--seed-only", is_flag=True, default=None,
              help="Only rewrite the seed (-k seed exploits), no taxonomy sweep. Env: TAX_SEED_ONLY.")
@click.option("--no-seed", is_flag=True, default=None,
              help="Skip the seed and start from the sweep (not with --seed-only / --seed-para-reps). "
                   "Env: TAX_NO_SEED.")
@click.option("--sweep-before-seed-para", is_flag=True, default=None,
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
@click.option("--max-sequences", default=None, type=int,
              help="Sweep only the first N sub-techniques of the sweep order. Env: TAX_MAX_SEQUENCES.")
@click.option("--order", default=None, type=click.Choice(list(SWEEP_ORDERS)),
              help="Sweep order [default: priority]. Env: TAX_ORDER.")
@click.option("--order-seed", default=None, type=int,
              help="Seed of the random sweep order [default: 0]. Env: TAX_ORDER_SEED.")
@click.option("--stack-only", default=None, type=int,
              help="Restrict the sweep to an Ln sweep of the pool with n = N, one line per set of N sub-techniques (1 = an L1 sweep of the pool). Env: TAX_STACK_ONLY.")
@click.option("--stack-random", is_flag=True, default=None,
              help="Draw the Ln-sweep pool at random instead of taking the first --stack-pool sub-techniques. Env: TAX_STACK_RANDOM.")
@click.option("--stack-seed", default=None, type=int, help="Seed of the random Ln-sweep pool. Env: TAX_STACK_SEED.")
@click.option("--no-early-stop", is_flag=True,
              help="Keep attacking a data point after a success, through the rest of its schedule (the "
                   "budget, or earlier if the schedule ends first). The log then lists every success "
                   "(technique coverage of a scenario, attempt-level ASR).")
@click.option("--max-attempts", default=None, type=click.IntRange(1), help="Hard cap on target calls per data point.")
@click.option("--no-judge", is_flag=True, help="Disable the judge mechanism (①).")
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
def run(benchmark, mode, target, refiner, judge, suites, user_tasks, injection_tasks, benchmark_version,
        defense, paraphrase_reps, stack_depth, stack_pool, seed_para_reps, seed_only, no_seed,
        sweep_before_seed_para, seed_source, seed_file, techniques, max_sequences, order, order_seed, stack_only, stack_random,
        stack_seed, no_early_stop, max_attempts,
        no_judge, no_context, no_legitimize, target_thinking, target_max_tokens, ipi_behaviors_path,
        ipi_ignore_concealment, taxonomy_path, limit, shard, num_shards, logdir, resume,
        worldsim, models_config, strategy, target_tokenizer, refiner_tokenizer, trace_dir, dry_run,
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
    from autopi.harness.runner import CorruptScenarioLogError, TargetUnavailableError
    from autopi.models.endpoints import load_model_catalog, override_endpoint, resolve_endpoint

    if num_shards < 1 or not 0 <= shard < num_shards:
        raise click.UsageError("--num-shards must be positive and 0 <= --shard < --num-shards")
    if resume and not logdir:
        raise click.UsageError("--resume reads each scenario's log, so it needs --logdir")
    if benchmark == "ipiarena" and (defense or benchmark_version or (injection_tasks and set(injection_tasks) != {"ipi"})):
        raise click.UsageError("IPI Arena uses scenario ids via --user-task and its own scoring; AgentDojo defense/version/task options do not apply")
    if benchmark == "ipiarena":
        # IPI Arena's own client always calls tools natively and places each scenario's injection
        # itself; the AgentDojo options are refused on the command line and ignored from the env.
        context = click.get_current_context()
        given = [f"--{name.replace('_', '-')}" for name in ("tool_calling", "injection_placement")
                 if context.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE]
        if given:
            raise click.UsageError(f"{' and '.join(given)} {'apply' if len(given) > 1 else 'applies'} to "
                                   "AgentDojo; IPI Arena calls tools natively and places the injection where "
                                   "each scenario defines it")
        tool_calling = injection_placement = None
    else:
        # The IPI Arena options have no meaning on AgentDojo (its scoring is programmatic and it
        # simulates no tools), so they are refused rather than silently ignored.
        context = click.get_current_context()
        given = [f"--{name.replace('_', '-')}" for name in ("judge", "worldsim", "ipi_behaviors_path",
                                                             "ipi_ignore_concealment", "ipi_max_steps",
                                                             "ipi_harness_retries")
                 if context.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE]
        if given:
            raise click.UsageError(f"{', '.join(given)} {'apply' if len(given) > 1 else 'applies'} to IPI Arena "
                                   "only; AgentDojo scores attempts with its own checks and simulates no tools")
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
            schedule_flags = {"paraphrase_reps", "stack_depth", "stack_pool", "seed_para_reps", "seed_only",
                              "no_seed", "sweep_before_seed_para", "techniques", "max_sequences", "order", "order_seed",
                              "stack_only", "stack_random", "stack_seed"}
            context = click.get_current_context()
            conflicts = [name for name in schedule_flags
                         if context.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE]
            if conflicts:
                raise ValueError("put schedule choices inside --strategy; conflicting flags: " + ", ".join(sorted(conflicts)))
            no_seed = not any(block["kind"] == "seed" for block in strategy_data["blocks"])
        elif no_seed is None:
            no_seed = os.environ.get("TAX_NO_SEED", "0") != "0"
        if no_seed and (seed_only or seed_para_reps > 0):
            raise ValueError("--no-seed leaves no seed to rewrite: drop --seed-only / --seed-para-reps")
        refiner = refiner or target
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
        refiner_endpoint = override_endpoint(resolve_endpoint(refiner, catalog, role="refiner"),
                                             tokenizer_path=refiner_tokenizer)
        if refiner_endpoint.provider == "ollama" and refiner_endpoint.reasoning_effort:
            # The alias may serve as the target too, where Ollama's OpenAI-compatible API reads it.
            click.echo("[autopi] note: the Ollama generator calls Ollama's native chat API, which takes no "
                       "reasoning_effort; it sends think from the alias's thinking setting. Put the native "
                       "API's own field in extra_body to set an effort there.", err=True)
        if refiner_endpoint.tokenizer_path:
            from autopi.models.recovery import get_decoder

            get_decoder(refiner_endpoint.tokenizer_path)
        if refiner_endpoint.max_tokens is None:
            refiner_endpoint = override_endpoint(refiner_endpoint,
                                                 max_tokens=int(os.environ.get("VLLM_REFINER_MAX_TOKENS", "1024")))
        judge_endpoint = resolve_endpoint(judge or target, catalog)
        worldsim_endpoint = resolve_endpoint(worldsim or judge or target, catalog)
        secret_envs = {ep.api_key_env for ep in (target_endpoint, refiner_endpoint, judge_endpoint, worldsim_endpoint)
                       if ep.api_key_env}
        journal = (Journal(trace_dir or Path(logdir) / "traces", extra_secret_envs=secret_envs)
                   if (trace_dir or logdir) and not dry_run else None)
        adapter = _build_adapter(benchmark, target, judge, list(suites), list(user_tasks),
                                 list(injection_tasks), benchmark_version, defense,
                                 ipi_ignore_concealment=ipi_ignore_concealment,
                                 target_thinking=target_thinking, target_max_tokens=target_max_tokens,
                                 ipi_behaviors_path=ipi_behaviors_path, target_endpoint=target_endpoint,
                                 judge_endpoint=judge_endpoint, worldsim=worldsim, worldsim_endpoint=worldsim_endpoint,
                                 journal=journal, ipi_max_steps=ipi_max_steps, ipi_harness_retries=ipi_harness_retries,
                                 injection_placement=injection_placement or "agentdojo")
        seed_template, seed_name = resolve_seed(
            no_seed=bool(no_seed), seed_source=seed_source or os.environ.get("TAX_SEED_SOURCE"),
            seed_file=seed_file or os.environ.get("TAX_SEED_FILE"),
            benchmark_default_template=getattr(adapter, "default_seed_template", None),
            benchmark_default_name=getattr(adapter, "default_seed_name", None), benchmark=benchmark,
            no_seed_hint="use a strategy without a seed block" if strategy_data else None)
        cfg = RefinerConfig(endpoint=refiner_endpoint, journal=journal)
        common = dict(refiner_model=refiner_endpoint.model, refiner_config=cfg, awareness_mode=mode,
                      taxonomy_path=taxonomy_path, use_judge=False if no_judge else None,
                      use_context=False if no_context else None, use_legitimize=False if no_legitimize else None,
                      seed_template=seed_template, seed_name=seed_name)
        if strategy_data:
            engine = StrategyEngine(strategy_data, **common)
        else:
            engine = TaxonomyEngine(
                **common, paraphrase_reps=paraphrase_reps, stack_depth=stack_depth, stack_pool=stack_pool,
                seed_paraphrase_reps=seed_para_reps, seed_only=True if seed_only else None,
                use_seed=False if no_seed else None, sweep_before_seed_para=True if sweep_before_seed_para else None,
                techniques=techniques, max_sequences=max_sequences, order=order, order_seed=order_seed,
                stack_only=stack_only, stack_random=True if stack_random else None, stack_seed=stack_seed)
        # AgentDojo embeds the injection in a double-quoted YAML scalar (quotes escaped at
        # placement); IPI Arena places it in JSON, so quotes/backslashes are kept as written.
        engine.yaml_safe_output = benchmark == "agentdojo"
        stop = not no_early_stop and (not strategy_data or strategy_data["stop"] == "first_success")
        points = list(adapter.iter_points())[shard::num_shards]
        if limit is not None:
            points = points[:limit]
        if not points:
            raise ValueError("no scenarios match the selection/shard; check the suite and task ids")
        click.echo(f"[autopi] benchmark={benchmark} mode={mode} target={target} refiner={refiner} "
                   f"combos={engine.num_combos} budget/pt={min(max_attempts or engine.attempt_budget, engine.attempt_budget)} "
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
                "models": {"target": target_endpoint.public_config(), "refiner": refiner_endpoint.public_config(),
                           "judge": judge_endpoint.public_config(), "worldsim": worldsim_endpoint.public_config()},
                "strategy": strategy_data or {"seed_only": engine.seed_only, "use_seed": engine.use_seed,
                    "k": paraphrase_reps, "seed_para_reps": seed_para_reps, "stack_depth": engine.stack_depth,
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
        # Skipped when the run makes no generator request (only the seed is tried).
        budget = min(max_attempts or engine.attempt_budget, engine.attempt_budget)
        if not (engine.use_seed and budget <= 1):
            _check_generator(engine)
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
