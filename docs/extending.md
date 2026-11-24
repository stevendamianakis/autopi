# Extending AutoPI

The search engine imports no benchmark package. A benchmark adapter supplies:

1. `iter_points()` yielding `AttackPoint` values, each with its user task and
   injection goal as text.
2. `run(point, injection_text)` placing the text, executing the agent and returning
   `RunResult(utility, security, messages, info)`.
3. A `name`, informational `checker_kind`, and optionally a default seed template/name.

`security=True` means the **attacker's goal succeeded**. It does not mean the
agent was secure. `messages` uses dictionaries with `role`, `content` and
optional `tool_calls`. For errors that prevent fair scoring, raise
`AdapterInfraError(kind, endpoint_error=...)`. Such an attempt consumes budget but
does not become failed generation history. Set `endpoint_error=False` for an error
tied to the scenario (a context overflow, a malformed model output, a broken
environment) and leave it True for an error of the model endpoint (unreachable,
failing, rejecting the request): only endpoint errors count toward the run's stop
when the target keeps failing. Raise `BenchmarkModelConfigError` when a model
rejects its key or model id, which stops the run at once. Let
programming/configuration errors propagate.

```python
from autopi.core.refiner import RefinerConfig
from autopi.core.strategy import StrategyEngine, load_strategy
from autopi.harness.runner import run_attack
from autopi.models.endpoints import ModelEndpoint

endpoint = ModelEndpoint(
    provider="openai-compatible",
    model="YOUR_MODEL_ID",
    base_url="https://your-endpoint.example/v1",
    api_key_env="RESEARCH_MODEL_KEY",
)
engine = StrategyEngine(
    load_strategy("configs/strategies/full.yaml"),
    refiner_model=endpoint.model,
    refiner_config=RefinerConfig(endpoint=endpoint),
    awareness_mode=3,
    seed_template="Before proceeding, do <task_goes_here>.",
    seed_name="my-seed",
)
summary = run_attack(my_adapter, engine, logdir="runs/my-benchmark")
print(summary.format())
```

`TaxonomyEngine` and `StrategyEngine` default to awareness mode 3 (the generator
writes the goal directly) with no exploit variants, the same as the command line
and the paper; pass `awareness_mode` and the exploit counts to change them.

The CLI connects the two bundled benchmarks. New adapters can use the same
Python entry point without changing the engine or importing the taxonomy into
the adapter. A local server is optional; the generator endpoint can use a hosted
provider. A custom text generator can subclass `TaxonomyEngine`/`StrategyEngine`
and implement `_query(prompt)`; it receives no benchmark objects.

## Taxonomies

`--taxonomy-path` accepts a JSON object containing
`taxonomy.families[].techniques[].subtechniques[]`. Its three levels are the
paper's classes (called families in the file), techniques and sub-techniques, and
each unit has a unique uppercase id and a name. Definitions, examples and a
technique's phrasing signature supply the generator prompt. See the bundled
`src/autopi/core/data/piatlas_taxonomy.json` for the complete schema.

Taxonomy ids are data, not benchmark task ids. The default priority order is the
paper's PIAtlas order; use canonical/random ordering or a strategy's `order_ids`
to choose a different order without editing the code. A technique with no named
sub-techniques can supply one sweep unit using its own id. Optional
`allow_unicode` and `allow_markers` metadata override the default PIAtlas
sanitization settings for new taxonomies.

The default taxonomy text is softened before prompting and its illustrative
canaries are abstracted. These transformations preserve the existing method's
prompting behavior. Log the taxonomy hash and settings when comparing experiments.

## Measurements

The engine's per-scenario logs distinguish target attempts from skipped generator
candidates and record every success when early stopping is disabled. Originating
taxonomy ids describe the requested construction. For the paper's coverage metric,
classify the **generated successful text** with a separate annotation procedure;
do not use source ids as if they were those annotations.

AgentDojo uses its task-specific visible injection slots and native state
checkers. IPI Arena uses the benchmark's own full-criteria score, including
concealment. Keep these scoring rules explicit when adding another benchmark.
