# Glossary: code names and the paper's terms

The command-line options and the documentation use the paper's terms. Options renamed in
version 0.4.0 still accept their earlier names, so older commands and scripts run unchanged
(first table). Other names in the code (environment variables, the fields of saved files,
Python names) predate the paper's terms and are kept, so saved logs, analysis scripts and
`--resume` keep working (second table).

## Option names before version 0.4.0

| Earlier name (still accepted) | Name since 0.4.0 | In the paper |
|---|---|---|
| `--refiner` | `--generator` | the **generator**; its endpoint also serves the judge |
| `--refiner-tokenizer` | `--generator-tokenizer` | |
| `-k`, `--paraphrase-reps` | `--sweep-exploits` | **sweep exploit** ×r, variants of each failed sweep line. With `--seed-only`, `-k` sets the seed exploits, as before |
| `--seed-para-reps` | `--seed-exploits` | **seed exploit** ×r₀, rewrites of the seed |
| `--sweep-before-seed-para` | `--sweep-before-seed-exploits` | |
| `--stack-depth` | `--max-sweep-level` | **Ln sweeps** for n = 1 up to this level |
| `--stack-only` | `--only-sweep-level` | a single **Ln sweep** |
| `--stack-pool` | `--sweep-pool-size` | the sub-techniques an Ln sweep combines; YAML strategies use `pool_size` |
| `--stack-random` | `--random-sweep-pool` | |
| `--stack-seed` | `--sweep-pool-seed` | |
| `--max-sequences` | `--max-subtechniques` | **sub-techniques** |
| `--judge` | `--ipi-judge` | IPI Arena's **LLM judge**, part of its own scoring, not the paper's judge. These docs call it the scoring judge |
| `--worldsim` | `--ipi-worldsim` | IPI Arena's **tool simulator** (WorldSim), which simulates the tools the agent calls |

An earlier and a current name of the same option can be given together only with the same value.

## Other names in the code

| In the code | In the paper | Notes |
|---|---|---|
| refiner: `RefinerConfig`, `refiner_model`, `VLLM_REFINER_*`, `TAX_*REFINER*`, the `refiner_logs/` folder, `models.refiner` in the run manifest, the `refiner` role in traces | **generator** | The LLM that writes every injection line except the seed |
| diagnostic judge: the `diagnostic_judge` role in traces, `--no-judge` | **judge** | Diagnoses why an attempt failed; it shares the generator's endpoint |
| point, `AttackPoint` | **data point (scenario)** | One user task paired with one injection goal |
| `injection_goal` | **injection goal** | The injection task as text |
| `security=True` | the injection goal was **achieved** | An attack success, not a secure agent |
| family: `families[]`, `family_id`, `F1`–`F6` | **class** | The taxonomy's top level; techniques and sub-techniques keep their names |
| sequence: `sequence_id`, `num_sequences`, `max_sequences`, `TAX_MAX_SEQUENCES` | **sub-technique** | One unit of an L1 sweep |
| combination (combo), stacking: `stack_*` fields and arguments, `TAX_STACK_*` | **Ln sweep** | One line for each set of n sub-techniques; YAML strategies use `level` and `pool_size` |
| template: `template`, `failed_templates`, `successful_templates`, `final_template` | **line** | A generated injection line. The seed, and the output of modes 1 and 2, are templates with the `<task_goes_here>` slot for the goal |
| seed paraphrase: `seed_paraphrase_reps`, `TAX_SEED_PARA_REPS`, `seed_para_reps` in the run manifest, phase `seed_paraphrase`, sequence id `PARA:SEED` | **seed exploit** (rewrites of the seed) | YAML block `seed_exploit`. A `--seed-only` run passes their number as `paraphrase_reps` (`k` in the run manifest) and logs them with phase `paraphrase`, so the `PARA:SEED` id is what marks them |
| paraphrase: `paraphrase_reps`, `k` in the run manifest, phase `paraphrase`, sequence ids `PARA:<id>` other than `PARA:SEED` | **sweep exploit** (variants of a failed sweep line) | YAML block `sweep_exploit` |
| phase `seed` / `taxonomy` in scenario logs | the **seed** / the **sweep** block | A log's `phase` field names the block; the paper's phases are Initialize (seed), Explore (sweep) and Exploit (seed exploit, sweep exploit) |
| `TaxonomyEngine`, `StrategyEngine` | the **engine** | `StrategyEngine` runs a YAML search strategy |
| strategy YAML: `blocks`, `budget`, `stop` | **search strategy** σ: blocks, attempt budget B, stop rule ρ | See [the strategy guide](strategies.md) |
| `attempt_budget` in scenario logs | **attempt budget** | One attempt is one target call |
| `max_attempts` in scenario logs | (no paper term) | The schedule that follows the seed: the budget minus one when the run has a seed (132 for an L1 run, whose budget is 133) |
| `--max-attempts` (`max_attempts_cap` in scenario logs, `max_attempts` in the run manifest) | (no paper term) | An optional extra cap on attempts; a run without it uses the whole budget |
| `--mode 3`, `awareness_mode=3` | the **scenario-aware** setting | The paper's setting and the default; modes 1 and 2 are not used in the paper |
| `--no-context`, `--no-legitimize` | **context blending**, **legitimize** | With the judge, the three adaptive mechanisms |
| ASR, ASR@k in the summary | **scenario-level ASR**, ASR@k | The summary of a run without early stopping also prints an attempt-level rate, which the paper does not use |
| `covered_subtechniques`, `solved_sources` in scenario logs | provenance, not **coverage** | The sub-techniques a line was generated for. The paper's coverage labels each successful injection after the run |
