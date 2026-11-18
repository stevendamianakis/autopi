# Release notes — version 0.2.0

This release packages the AutoPI engine together with the two benchmarks it runs
on and a general interface for search strategies.

| Area | Behavior in this release |
|---|---|
| Packaging | One AutoPI package at the repository root, with AgentDojo and IPI Arena source snapshots under `benchmarks/` |
| Seeds | The benchmark's default seed, a seed from a file, AgentDojo's seed reused on another benchmark, or no seed |
| Search strategies | Any finite ordering and repetition of the four blocks `seed`, `sweep`, `seed_exploit` and `sweep_exploit` in a YAML file. The earlier command-line flags still work |
| AgentDojo injection placement | The slots AgentDojo marks as visible to the user task (`get_injection_candidates(user_task)`) |
| AgentDojo execution and scoring | The benchmark's own `TaskSuite.run_task_with_pipeline`, with the transcript recorded for the judge and for context blending |
| IPI Arena conversation conversion | An empty tool result is inserted for an unanswered prefilled tool call, and the real image MIME type is detected |
| Model routing | Named endpoints per role, any OpenAI-compatible URL, explicit provider and model ids |
| Text removed by a server-side tool parser | Optional recovery of the generated text from token ids, per endpoint (see [model configuration](models.md)) |
| Failures | A provider or transport error is recorded as an infrastructure failure, never as an empty answer of the target. Malformed tool arguments written by a model go through the benchmark's own parser unchanged |
| Provenance | Every run writes a manifest with its resolved settings, package versions and hashes, and refuses to mix runs with different settings in one output directory |

Changes to the bundled benchmark sources are listed in
[THIRD_PARTY.md](../THIRD_PARTY.md).

The experiments in the paper were run with earlier versions of this code and the
settings the paper lists. Results obtained with this release should state their
models, server and parser versions, strategy and settings. The release
validation ([validation report](validation.md)) checks software behavior and
packaging, and is not a new ASR measurement.
