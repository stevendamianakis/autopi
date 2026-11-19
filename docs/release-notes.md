# Release notes

## Version 0.2.8

Four fixes in AutoPI. No bundled benchmark files changed, including AgentDojo's
scoring and validation and IPI Arena's simulator and scoring. Mode 3, the paper's
setting, retains the diagnostic judge's free-text explanation.

| Area | Fix |
|---|---|
| Generator failures | Requests' SSL and proxy failures stop the run as unreachable-generator errors. Missing, unreadable or invalid tokenizer files are configuration errors; the CLI checks the generator tokenizer during setup even without `--logdir`. After 3 consecutive unexpected request errors from either the generator or its diagnostic judge, the run stops before logging the current scenario, so `--resume` repeats it. Each role has its own error streak, retained across scenarios and cleared by a successful request. HTTP 400 remains a skippable per-prompt rejection |
| Goal-blind modes | Modes 1 and 2 pass only a recognized, fixed failure-category phrase to the generator. The goal-aware judge's free-text explanation is withheld because it can quote the hidden goal. Mode 3 keeps that explanation as before |
| Unusable generation limit | Both scheduling interfaces end the current scenario after 200 consecutive unusable generated candidates. This includes sweep lines and seed/sweep variants, across repeated blocks. Each candidate can use the existing two prompt tiers. A usable generated candidate resets the streak, and a new scenario starts with a fresh streak. Skipped candidates consume no target attempts. Logs and summaries record the stop reason. The limit exceeds the full 132-unit L1 sweep |
| Older Anthropic SDKs | AgentDojo target construction falls back to `httpx.Limits` when `httpx2` is unavailable, preserving disabled HTTP keep-alive for the older SDKs allowed by the dependency declarations |

The positive integer environment variables `TAX_MAX_CONSECUTIVE_REFINER_ERRORS`
(default 3) and `TAX_MAX_CONSECUTIVE_UNUSABLE` (default 200) configure the limits.
Python callers can pass `max_consecutive_refiner_errors` and
`max_consecutive_unusable` to either engine. Resolved limits are recorded in
the run manifest and per-scenario logs. Use a fresh output directory for 0.2.8;
earlier completed results are not repaired automatically.

## Version 0.2.7

A bug-fix release. IPI Arena's own scoring and AgentDojo's own execution and
scoring are unchanged, and no bundled benchmark source changed.

| Area | Fix |
|---|---|
| Anthropic models on IPI Arena | IPI Arena's client sends temperature 0 on every target, judge and WorldSim call, and Claude 4.7 and later reject a temperature below 1 on the OpenAI-compatible API (HTTP 400), so every call failed. For an `anthropic` endpoint AutoPI now leaves `temperature`, `top_p` and `top_k` out of those requests, as on its other Anthropic requests, and the API's default sampling applies |
| Process-level resume | The undocumented `TAX_RESUME_AFTER_ATTEMPT` variable is removed and refused with a pointer to `--resume`. It reported a resumed success at the wrong attempt (so ASR@k was wrong), crashed when seed paraphrases were still to come, and could repeat a finished attempt. Scenario logs no longer have the `resumed_after_attempt` field |
| Refusals inside the delimiters | A generated line that opens like a refusal ("I can't…", "I'm sorry…") is skipped also when it is written inside the template delimiters, as an undelimited refusal already was. In mode 3 such a line used to reach the target and use attempts. A genuine line that opens this way is skipped too |
| IPI Arena browser scenarios | Selecting a browser scenario without playwright installed stops the run at setup, a dry run included, as IPI Arena's own runner does. Before, every attempt failed in the infrastructure while the generator was still called, and the scenario was left out of ASR |
| Raw-token recovery | A token id outside the configured tokenizer's vocabulary is an error. The tokenizer used to drop it silently, losing generated text |

## Version 0.2.6

A bug-fix release. IPI Arena's own scoring and AgentDojo's own execution and
scoring are unchanged, and no bundled benchmark source changed.

| Area | Fix |
|---|---|
| Cut-off reasoning, prompted tools | An AgentDojo prompted target's reply that was cut off inside a `<think>` block (for example by a token cap) loses everything from that `<think>` on before AgentDojo parses tool calls. Version 0.2.5 removed only closed blocks, so a call the model was still weighing could be executed |
| Tool arguments that are not an object | Native tool-call arguments that are valid JSON but not an object (`[]`, `null`, a string or a number) count as a model crash, like cut-off JSON. Before, they stopped the run |
| Generator settings | The Anthropic generator sends `extra_body`, and the Google generator applies `extra_body` and `omit_parameters`. The 0.2.5 entry saying that `extra_body` reaches Anthropic and Google native requests and that `omit_parameters` is applied to Google requests is wrong for the generators, which ignored both until this release |
| Google AgentDojo target | `omit_parameters` also removes AgentDojo's own value of an omitted field (its Gemini client always sets temperature 0 and a 65,535-token cap), and `max_tokens` names the Google output cap. In 0.2.5 only AutoPI's own overrides were left out, so the request still carried AgentDojo's values |
| Settings a client cannot send | `reasoning_effort` on a native Anthropic or Google client (an AgentDojo target or a generator) is refused instead of dropped; it still reaches the OpenAI-compatible paths, IPI Arena's clients included. An `anthropic` alias that omits `max_tokens`, which the API requires, is refused when it loads |
| Strategy stop rule in Python | `run_attack` follows a strategy's own `stop` rule when the caller passes no `stop_at_first_success`. Before, a `stop: budget` strategy stopped at the first success through the Python interface; the command line was not affected |
| IPI Arena documentation | The adapter states that a judge reply whose JSON score is not a number makes IPI Arena's judge raise, so the run is an error (left out of ASR, as IPI Arena does), and that each harness retry re-runs the whole scenario. Scoring is unchanged |

## Version 0.2.5

A bug-fix release. IPI Arena's own scoring is unchanged, and so is AgentDojo's own
execution and scoring (`run_task_with_pipeline`). One bundled AgentDojo source file
changed (see [THIRD_PARTY.md](../THIRD_PARTY.md)).

| Area | Fix |
|---|---|
| Anthropic AgentDojo target | The async client now disables HTTP keep-alive. AgentDojo runs each query in its own event loop on one reused client, so a kept-alive connection belonged to a closed loop and every request after the first was sent to the server twice. Each request is now sent once |
| Generator configuration errors | A generator that is misconfigured rather than momentarily down (a missing API key or provider SDK, or HTTP 401/403/404) now stops the run before the scenario is logged, like an outage, so `--resume` repeats it. Before, it was skipped like an unusable answer, and the scenario was logged as a seed-only failure that `--resume` then kept. A plain HTTP 400 is still a skipped, per-prompt failure |
| Unreachable target | A target that fails every attempt of many scenarios in a row now stops the run with a message and a `--resume` hint, instead of silently excluding every scenario and exiting 0 while still paying the generator |
| Quotes in the payload | Generated templates and seeds keep double quotes and backslashes. The AgentDojo adapter escapes them for its double-quoted YAML slot at placement, and IPI Arena keeps them as written in JSON. Before, they were deleted from every benchmark, and a custom seed containing a quote failed AgentDojo's YAML parse on every attempt |
| Thinking targets, prompted tools | An AgentDojo prompted target's reply has its `<think>...</think>` reasoning removed before the benchmark parses `<function=...>` tool calls, so a hypothetical call written inside the reasoning is not executed. No effect when the target does not think |
| Truncated tool JSON | A native tool call whose JSON arguments were cut off (a token cap) is counted as a model crash against the budget instead of stopping the run |
| Redacted thinking | A thinking Anthropic target that returns a `redacted_thinking` block no longer crashes on the next turn (bundled `anthropic_llm.py`) |
| `tool_mode: prompted` | Rejected for hosted `anthropic`, `google` and `openai` endpoints, which have no prompted path; it is for `openai-compatible` and `ollama`. Before, it silently ran native, or sent tool-role messages the OpenAI API rejects |
| Gemini on IPI Arena | A Google (OpenAI-compatible) target gets Google's documented placeholder thought signature on the prefilled tool calls the API never signed, so Gemini 3 and later do not reject the request (HTTP 400) |
| Honoured endpoint settings | `extra_body` reaches Anthropic and Google native requests and the Ollama generator; `omit_parameters` is applied to Google requests and the Ollama generator. A bare `local` target spec is rejected with guidance instead of sending the model id `local` |
| Empty schedule | A sweep that selects zero combinations (for example a level above the pool size, or a strategy sweep block with no matching combinations) is rejected instead of reporting 0% for every scenario |
| Trace and provenance | A resumed scenario no longer writes a dangling `point_start`; the diagnostic judge sees an empty string, not the text `None`, for an empty reply; the trace scrubs a credential named by a custom `api_key_env`; the manifest records the `anthropic` and `google-genai` versions and hashes the full IPI Arena scenario set rather than one worker's selection |
| Command line | A mid-run `OSError` is reported as itself, not as a usage error; an empty `AUTOPI_TARGET_THINKING` no longer forces a named target alias to thinking off |
| Packaging | The undocumented, untested `replay.py` is removed, and the `agentdojo` install extra (which pulled a different PyPI version without the bundled patches) is dropped; install the bundled benchmark from `benchmarks/agentdojo` |

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
