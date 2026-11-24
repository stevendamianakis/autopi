# Glossary: code names and the paper's terms

The documentation uses the paper's terms. Some names in the code (command-line
flags, environment variables, log fields, Python names) predate them and are
kept, so scripts, saved logs and `--resume` keep working. This table maps them.

| In the code | In the paper | Notes |
|---|---|---|
| refiner: `--refiner`, `--refiner-tokenizer`, `RefinerConfig`, `refiner_model`, `VLLM_REFINER_*`, `TAX_*REFINER*`, the `refiner_logs/` folder, the `[TAXREFINER]` console prefix | **generator** | The LLM that writes every injection line except the seed |
| diagnostic judge: the `diagnostic_judge` role in traces, `--no-judge` | **judge** | Diagnoses why an attempt failed; it shares the generator's endpoint |
| `--judge` | IPI Arena's **LLM judge** | Part of IPI Arena's own scoring, not the paper's judge. These docs call it the scoring judge |
| `--worldsim` | IPI Arena's **tool simulator** (WorldSim) | Simulates the tools the IPI Arena agent calls |
| point, `AttackPoint` | **data point (scenario)** | One user task paired with one injection goal |
| `injection_goal` | **injection goal** | The injection task as text |
| `security=True` | the injection goal was **achieved** | An attack success, not a secure agent |
| family: `families[]`, `family_id`, `F1`–`F6` | **class** | The taxonomy's top level; techniques and sub-techniques keep their names |
| sequence: `sequence_id`, `num_sequences`, `--max-sequences` | **sub-technique** | One unit of an L1 sweep |
| combination (combo), stacking: `--stack-depth`, `--stack-only`, `--stack-pool`, `stack_*` fields | **Ln sweep** | One line for each set of n sub-techniques; YAML strategies use `level` and `pool_size` |
| template: `template`, `failed_templates`, `successful_templates`, `final_template` | **line** | A generated injection line. The seed, and the output of modes 1 and 2, are templates with the `<task_goes_here>` slot for the goal |
| seed paraphrase: `--seed-para-reps`, `seed_paraphrase_reps`, `--seed-only`, phase `seed_paraphrase`, sequence id `PARA:SEED` | **seed exploit** (rewrites of the seed) | YAML block `seed_exploit`. A `--seed-only` run takes their number from `-k` and logs them with phase `paraphrase`, so the `PARA:SEED` id is what marks them |
| paraphrase: `-k`/`--paraphrase-reps`, `paraphrase_reps`, phase `paraphrase`, sequence ids `PARA:<id>` other than `PARA:SEED` | **sweep exploit** (variants of a failed sweep line) | YAML block `sweep_exploit`. With `--seed-only`, `-k` counts seed exploits instead |
| phase `seed` / `taxonomy` in scenario logs | the **seed** / the **sweep** block | A log's `phase` field names the block; the paper's phases are Initialize (seed), Explore (sweep) and Exploit (seed exploit, sweep exploit) |
| `TaxonomyEngine`, `StrategyEngine` | the **engine** | `StrategyEngine` runs a YAML search strategy |
| strategy YAML: `blocks`, `budget`, `stop` | **search strategy** σ: blocks, attempt budget B, stop rule ρ | See [the strategy guide](strategies.md) |
| `max_attempts`, `attempt_budget`, `--max-attempts` | **attempt budget** | One attempt is one target call |
| `--mode 3`, `awareness_mode=3` | the **scenario-aware** setting | The paper's setting and the default; modes 1 and 2 are not used in the paper |
| `--no-context`, `--no-legitimize` | **context blending**, **legitimize** | With the judge, the three adaptive mechanisms |
| ASR, ASR@k in the summary | **scenario-level ASR**, ASR@k | The summary of a run without early stopping also prints an attempt-level rate, which the paper does not use |
| `covered_subtechniques`, `solved_sources` in scenario logs | provenance, not **coverage** | The sub-techniques a line was generated for. The paper's coverage labels each successful injection after the run |
