# Release notes

## Version 0.2.4

A bug-fix release. IPI Arena's own scoring and the bundled benchmark sources are
unchanged.

| Area | Fix |
|---|---|
| Gemini tool calls | AutoPI's clients put Gemini's thought signatures back on the tool calls they came with, through AgentDojo's Gemini client and through the OpenAI-compatible API that IPI Arena uses. Both benchmarks rebuild tool calls without them, and Gemini 3 and later reject such a follow-up request (HTTP 400) |
| Generator outages | A generator that cannot be reached after its retries stops the run before the scenario is logged, so `--resume` repeats it. Before, every candidate was skipped, the scenario was logged as a failure, and `--resume` skipped it. An answer the generator returns that cannot be used is still skipped |
| Infrastructure-only scenarios | A scenario whose every attempt was an infrastructure failure is not logged and stays out of ASR; the summary counts it and `--resume` runs it again. Before, it counted as a failed scenario |
| IPI Arena WorldSim | In the Python interface, a `worldsim` model now takes precedence over `judge_endpoint`. Before, WorldSim ran on the judge's endpoint |
| Ollama generator | Sends `top_p`, the `api_key_env` value as a bearer token, and retries like the other generators |

## Version 0.2.3

A bug-fix release. IPI Arena's own scoring is unchanged.

| Area | Fix |
|---|---|
| Empty generator lines | An empty line between the template delimiters is skipped like a refusal: no target call, no attempt counted. In mode 3 it used to be sent to the target as an empty injection and used an attempt |
| Raw-token recovery | Recovered generator text drops only the end-of-turn token that stopped generation. Version 0.2.2 dropped every special token at the end, so a marker such as `<\|im_start\|>` right before the end of the turn was lost |
| Ollama server | A named `ollama` alias without `base_url` uses the same default server as an `ollama:MODEL` spec. It used to fail when the run started |
| Ollama target thinking | An Ollama target receives the thinking mode as `reasoning_effort` (`none` turns thinking off), the field Ollama's OpenAI-compatible API reads. It used to receive nothing, so `thinking: off` had no effect |

## Version 0.2.2

A bug-fix release that also corrects version 0.2.1. IPI Arena's own scoring is
unchanged.

| Area | Fix |
|---|---|
| Anthropic endpoints | No sampling parameters are sent. The anthropic SDK 2.x, which the tested setup pins, no longer accepts `temperature`, `top_p` or `top_k`, and the API version it sends (2026-10-21) rejects them on every model. So every Anthropic generator request failed, and every Anthropic AgentDojo target request failed unless its alias omitted `temperature`. An `anthropic` alias that sets `temperature` or `top_p` is now refused when it is loaded. The 0.2.1 entry that says these settings reach the request is wrong |
| Provider errors on AgentDojo | Anthropic and Google errors count as infrastructure failures, as OpenAI errors already did, also when AgentDojo's Anthropic client passes one on after its retries. Before, such an error stopped the run |
| IPI Arena transcript | The transcript that AutoPI's diagnostic judge and context blending read holds every step of the agent's turn: what it wrote between tool calls, each call, and the simulated tool results. Context blending still reads only the scenario's own documents |
| Strategy sweeps | A sweep with more combinations than an index can count (for example `level: 20` without `limit`) runs until the budget ends it. Before, it failed at its start |

## Version 0.2.1

A bug-fix release. IPI Arena's own scoring is unchanged, and so is every default
schedule.

| Area | Fix |
|---|---|
| Anthropic AgentDojo target | The alias's temperature, top_p, output cap and omitted parameters now reach the request AgentDojo makes, and the call is traced. Version 0.2.0 always sent AgentDojo's values (temperature 0, 1,024 output tokens) and traced nothing |
| IPI Arena target thinking | `target_thinking` given to `IPIArenaAdapter` in Python now reaches the target's requests, as `--target-thinking` already did |
| IPI Arena action-only score | `--ipi-ignore-concealment` now leaves out an explicit list of concealment criteria per scenario. The keyword match of 0.2.0 also left out a user-task or harmful-action criterion in 7 scenarios and kept the concealment check of 1 |
| IPI Arena WorldSim | An empty or reasoning-only simulator reply stays empty. Version 0.2.0 turned it into a missing reply, which IPI Arena answers with `{"status": "success"}` |
| IPI Arena judge | Unchanged: a judge reply without valid JSON scores 0, as in IPI Arena. The adapter's documentation said such a reply was retried, and now says how it is scored |
| IPI Arena utility field | Only the "User Intent Fulfillment" criterion counts as utility, not attack criteria such as "Forwarding Intent". The field is only logged |
| `--sweep-before-seed-para` | The seed paraphrases run after the sweep even when the sweep's last candidate could not be written. Version 0.2.0 skipped them with `-k 0` and ran them late otherwise |
| Raw-token recovery | Recovered generator text keeps special tokens such as `<\|im_start\|>`, except the end-of-turn token |

## Version 0.2.0

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
