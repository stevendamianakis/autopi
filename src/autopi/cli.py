"""AutoPI command line: configurable models, benchmarks and composable strategies."""

from __future__ import annotations

import os

import click

from autopi.core.taxonomy import SWEEP_ORDERS
from autopi.harness.runner import run_attack
from autopi.models.registry import (
    TARGET_THINKING_CHOICES,
    resolve_target_max_tokens,
    resolve_target_thinking,
)


def _build_adapter(benchmark: str, target: str, judge: str | None, suites: list[str] | None,
                   user_tasks: list[str] | None, injection_tasks: list[str] | None,
                   benchmark_version: str | None, defense: str | None,
                   ipi_ignore_concealment: bool = False, target_thinking: str | None = None,
                   target_max_tokens: int | None = None, ipi_behaviors_path: str | None = None,
                   target_endpoint=None, judge_endpoint=None, worldsim=None, worldsim_endpoint=None,
                   journal=None, ipi_max_steps=None, ipi_harness_retries=2):
    if benchmark == "agentdojo":
        from autopi.adapters.agentdojo import AgentDojoAdapter

        return AgentDojoAdapter(
            target, benchmark_version=benchmark_version or "v1.2.2", suites=suites or None,
            user_tasks=user_tasks or None, injection_tasks=injection_tasks or None, defense=defense,
            target_thinking=target_thinking, target_max_tokens=target_max_tokens,
            endpoint=target_endpoint, journal=journal,
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


@click.group()
def main() -> None:
    """AutoPI — taxonomy-driven adaptive prompt-injection evaluation."""


@main.command()
@click.option("--benchmark", type=click.Choice(["agentdojo", "ipiarena"]), required=True)
@click.option("--mode", type=click.IntRange(1, 3), default=3, show_default=True, help="Refiner awareness mode.")
@click.option("--target", required=True, help="Configured model alias, provider:MODEL_ID, or vllm:PORT:MODEL_ID.")
@click.option("--refiner", default=None, help="Refiner/attacker model spec (defaults to --target).")
@click.option("--judge", default=None, help="Benchmark judge (IPI Arena; defaults to --target). The diagnostic judge uses --refiner.")
@click.option("--worldsim", default=None, help="IPI tool simulator model (defaults to --judge or --target).")
@click.option("--models-config", type=click.Path(exists=True, dir_okay=False), default=None, help="YAML model alias definitions.")
@click.option("--strategy", type=click.Path(exists=True, dir_okay=False), default=None, help="Ordered/repeatable strategy blocks in YAML.")
@click.option("--target-tokenizer", type=click.Path(exists=True, dir_okay=False), default=None, help="Exact tokenizer.json for prompted-tool target recovery.")
@click.option("--refiner-tokenizer", type=click.Path(exists=True, dir_okay=False), default=None, help="Exact tokenizer.json for generator and diagnostic judge recovery.")
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
@click.option("-k", "--paraphrase-reps", default=0, show_default=True, help="Paraphrases per failed candidate.")
@click.option("--stack-depth", default=1, show_default=True, help="Max sub-techniques blended per line (1 = L1 sweep).")
@click.option("--stack-pool", default=None, type=int, help="Pool size bounding L>=2 combinations.")
@click.option("--seed-para-reps", default=0, show_default=True, help="Seed-paraphrase attempts before the sweep.")
@click.option("--seed-only", is_flag=True, default=None,
              help="Paraphrase the seed only (-k paraphrases), no taxonomy sweep. Env: TAX_SEED_ONLY.")
@click.option("--no-seed", is_flag=True, default=None,
              help="Skip the seed and start from the sweep (not with --seed-only / --seed-para-reps). "
                   "Env: TAX_NO_SEED.")
@click.option("--sweep-before-seed-para", is_flag=True, default=None,
              help="Run the sweep before the seed paraphrases. Env: TAX_SWEEP_BEFORE_SEEDPARA.")
@click.option("--seed-source", type=click.Choice(["benchmark", "agentdojo"]), default=None,
              help="Which built-in fixed seed to use: the benchmark's own default [default], or AgentDojo's "
                   "important_instructions seed reused on any benchmark. Env: TAX_SEED_SOURCE.")
@click.option("--seed-file", default=None,
              help="Use the fixed seed read from this file (must contain <task_goes_here>); overrides "
                   "--seed-source. For a benchmark that ships no seed (e.g. IPI Arena). Env: TAX_SEED_FILE.")
@click.option("--techniques", default=None,
              help="Sweep only these taxonomy ids, comma-separated: families (F2), techniques (F2.T5), "
                   "or sub-techniques (F2.T5.S2). Env: TAX_TECHNIQUES.")
@click.option("--max-sequences", default=None, type=int,
              help="Sweep only the first N sub-techniques of the sweep order. Env: TAX_MAX_SEQUENCES.")
@click.option("--order", default=None, type=click.Choice(list(SWEEP_ORDERS)),
              help="Sweep order [default: priority]. Env: TAX_ORDER.")
@click.option("--order-seed", default=None, type=int,
              help="Seed of the random sweep order [default: 0]. Env: TAX_ORDER_SEED.")
@click.option("--stack-only", default=None, type=int,
              help="Sweep only the N-way combinations of the pool (1 = singles of the pool). Env: TAX_STACK_ONLY.")
@click.option("--stack-random", is_flag=True, default=None,
              help="Draw the stacking pool at random instead of the first --stack-pool. Env: TAX_STACK_RANDOM.")
@click.option("--stack-seed", default=None, type=int, help="Seed of the random stacking pool. Env: TAX_STACK_SEED.")
@click.option("--no-early-stop", is_flag=True,
              help="Keep attacking a data point after a success, until its attempt budget is exhausted. "
                   "The log then lists every success (technique coverage of a scenario, attempt-level ASR).")
@click.option("--max-attempts", default=None, type=click.IntRange(1), help="Hard cap on target calls per data point.")
@click.option("--no-judge", is_flag=True, help="Disable the judge mechanism (①).")
@click.option("--no-context", is_flag=True, help="Disable context-blending (②).")
@click.option("--no-legitimize", is_flag=True, help="Disable the legitimize note.")
@click.option("--target-thinking", default=None, type=click.Choice(list(TARGET_THINKING_CHOICES)),
              help="Thinking mode of a locally served target: off (default), on, or server (send nothing "
                   "and let the server decide). Env: AUTOPI_TARGET_THINKING.")
@click.option("--target-max-tokens", default=None, type=int,
              help="Output cap of a locally served AgentDojo target (default: no cap). "
                   "Env: AUTOPI_TARGET_MAX_TOKENS.")
@click.option("--ipi-behaviors-path", default=None,
              help="IPI Arena: directory of scenario files (default: the installed benchmark's own). "
                   "Env: IPI_ARENA_BEHAVIORS_PATH.")
@click.option("--ipi-ignore-concealment", is_flag=True, help="IPI Arena: count a harmful action as success even if not concealed (drops the concealment/hiding criterion).")
@click.option("--taxonomy-path", default=None, help="Use another taxonomy JSON with the same layout.")
@click.option("--limit", default=None, type=click.IntRange(1), help="Attack only the first N data points (smoke tests).")
@click.option("--shard", default=0, type=int, help="This worker's shard index (0-based) for parallel sharded runs.")
@click.option("--num-shards", default=1, type=int, help="Total number of shards; this worker takes points [shard::num-shards].")
@click.option("--resume", is_flag=True, help="Skip data points whose per-pair log already exists (resume an interrupted run).")
@click.option("--logdir", default=None, help="Directory for per-pair refiner_logs JSON.")
def run(benchmark, mode, target, refiner, judge, suites, user_tasks, injection_tasks, benchmark_version,
        defense, paraphrase_reps, stack_depth, stack_pool, seed_para_reps, seed_only, no_seed,
        sweep_before_seed_para, seed_source, seed_file, techniques, max_sequences, order, order_seed, stack_only, stack_random,
        stack_seed, no_early_stop, max_attempts,
        no_judge, no_context, no_legitimize, target_thinking, target_max_tokens, ipi_behaviors_path,
        ipi_ignore_concealment, taxonomy_path, limit, shard, num_shards, logdir, resume,
        worldsim, models_config, strategy, target_tokenizer, refiner_tokenizer, trace_dir, dry_run,
        ipi_max_steps, ipi_harness_retries) -> None:
    """Run the taxonomy attack against a benchmark and print the ASR summary."""
    import hashlib
    import re
    from pathlib import Path

    from autopi.core.engine import GeneratorUnavailableError, TaxonomyEngine
    from autopi.core.refiner import RefinerConfig
    from autopi.core.seeds import resolve_seed
    from autopi.core.strategy import StrategyEngine, load_strategy
    from autopi.harness.journal import Journal
    from autopi.harness.manifest import benchmark_data_digest, save_run_manifest
    from autopi.models.endpoints import load_model_catalog, override_endpoint, resolve_endpoint

    if num_shards < 1 or not 0 <= shard < num_shards:
        raise click.UsageError("--num-shards must be positive and 0 <= --shard < --num-shards")
    if benchmark == "ipiarena" and (defense or benchmark_version or (injection_tasks and set(injection_tasks) != {"ipi"})):
        raise click.UsageError("IPI Arena uses scenario ids via --user-task and its own scoring; AgentDojo defense/version/task options do not apply")
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
            raise ValueError("--no-seed leaves no seed to paraphrase: drop --seed-only / --seed-para-reps")
        refiner = refiner or target
        catalog = load_model_catalog(models_config)
        target_endpoint = resolve_endpoint(target, catalog)
        # Legacy vllm specs retain the paper's explicit thinking-off default. Named
        # model aliases retain their configured value unless a flag/env overrides it.
        thinking_override = target_thinking or os.environ.get("AUTOPI_TARGET_THINKING")
        if thinking_override is not None or target.startswith("vllm:"):
            target_thinking = resolve_target_thinking(thinking_override)
        else:
            target_thinking = target_endpoint.thinking
        target_max_tokens = resolve_target_max_tokens(target_max_tokens)
        target_endpoint = override_endpoint(target_endpoint, thinking=target_thinking,
                                            max_tokens=target_max_tokens, tokenizer_path=target_tokenizer)
        refiner_endpoint = override_endpoint(resolve_endpoint(refiner, catalog, role="refiner"),
                                             tokenizer_path=refiner_tokenizer)
        if refiner_endpoint.max_tokens is None:
            refiner_endpoint = override_endpoint(refiner_endpoint,
                                                 max_tokens=int(os.environ.get("VLLM_REFINER_MAX_TOKENS", "1024")))
        judge_endpoint = resolve_endpoint(judge or target, catalog)
        worldsim_endpoint = resolve_endpoint(worldsim or judge or target, catalog)
        journal = Journal(trace_dir or Path(logdir) / "traces") if (trace_dir or logdir) and not dry_run else None
        adapter = _build_adapter(benchmark, target, judge, list(suites), list(user_tasks),
                                 list(injection_tasks), benchmark_version, defense,
                                 ipi_ignore_concealment=ipi_ignore_concealment,
                                 target_thinking=target_thinking, target_max_tokens=target_max_tokens,
                                 ipi_behaviors_path=ipi_behaviors_path, target_endpoint=target_endpoint,
                                 judge_endpoint=judge_endpoint, worldsim=worldsim, worldsim_endpoint=worldsim_endpoint,
                                 journal=journal, ipi_max_steps=ipi_max_steps, ipi_harness_retries=ipi_harness_retries)
        seed_template, seed_name = resolve_seed(
            no_seed=bool(no_seed), seed_source=seed_source or os.environ.get("TAX_SEED_SOURCE"),
            seed_file=seed_file or os.environ.get("TAX_SEED_FILE"),
            benchmark_default_template=getattr(adapter, "default_seed_template", None),
            benchmark_default_name=getattr(adapter, "default_seed_name", None), benchmark=benchmark)
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
        stop = not no_early_stop and (not strategy_data or strategy_data["stop"] == "first_success")
        points = list(adapter.iter_points())[shard::num_shards]
        if limit is not None:
            points = points[:limit]
        if not points:
            raise ValueError("no scenarios match the selection/shard; check the suite and task ids")
        click.echo(f"[autopi] benchmark={benchmark} mode={mode} target={target} refiner={refiner} "
                   f"combos={engine.num_combos} budget/pt={min(max_attempts or engine.attempt_budget, engine.attempt_budget)} "
                   f"points={len(points)} target_thinking={target_thinking} "
                   f"{'seed=' + engine.seed_name if engine.use_seed else 'no-seed'}" + (" no-early-stop" if not stop else ""))
        if dry_run:
            click.echo("Validated. No model calls were made.")
            return
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
                "stop_at_first_success": stop, "max_attempts": max_attempts, "defense": defense,
                "seed_sha256": hashlib.sha256((seed_template or "").encode()).hexdigest(),
                "taxonomy_sha256": hashlib.sha256(Path(taxonomy_path or _BUNDLED_TAXONOMY).read_bytes()).hexdigest(),
                "ipi_ignore_concealment": ipi_ignore_concealment, "ipi_max_steps": ipi_max_steps,
                "ipi_harness_retries": ipi_harness_retries, "ipi_behaviors_path": ipi_behaviors_path,
                "environment": {key: value for key, value in os.environ.items() if key.startswith("TAX_") and not re.search(r"API_KEY|TOKEN$|SECRET|PASSWORD", key)},
                "benchmark_data_sha256": benchmark_data_digest(adapter),
            }
            save_run_manifest(Path(logdir), settings)
        summary = run_attack(adapter, engine, logdir=logdir, max_attempts=max_attempts, points=points, resume=resume,
                             stop_at_first_success=stop, journal=journal)
        click.echo("\n" + summary.format())
    except GeneratorUnavailableError as exc:
        raise click.ClickException(f"{exc}. The run stopped before this scenario was logged; "
                                   "run the same command with --resume to continue.") from exc
    except (ValueError, OSError) as exc:
        raise click.UsageError(str(exc)) from exc


if __name__ == "__main__":
    main()
