# Comparing results

How to compare a run with published results or with another method: the settings to match,
what to report, and why two runs of the same setup can differ. The [README](../README.md)
covers installation and running.

## Recommended settings

To compare a run with published results or with another method, run the target the way the
benchmark runs it, and report every setting you change. AutoPI's defaults match these settings
except for thinking on a `vllm:PORT:MODEL_ID` target and IPI Arena's scoring judge. To reproduce
the paper, use the settings it lists instead.

| Setting | Recommended | AutoPI default | How to set |
|---|---|---|---|
| Target temperature | 0, as both benchmarks request | 0 | `temperature` in the target alias |
| Target reasoning effort | The model's default | None sent | `reasoning_effort` in the target alias |
| Target thinking | The model's default, on or off | None sent (`server`), except that a `vllm:PORT:MODEL_ID` target runs with thinking off | `--target-thinking server`, or `thinking` in the alias |
| Target output cap | The benchmark's | AgentDojo: no cap, but 1,024 tokens for Anthropic and 65,535 for Google targets. IPI Arena: 4,096 | `--target-max-tokens` (either benchmark), or `max_tokens` in the target alias |
| Tool calling (AgentDojo) | Native | Native | `--tool-calling` |
| Injection placement (AgentDojo) | `agentdojo`, the vectors the user task reads | `agentdojo` | `--injection-placement` |
| Version and defense (AgentDojo) | v1.2.2, no defense | v1.2.2, no defense | `--benchmark-version`, `--defense` |
| Scoring judge (IPI Arena) | IPI Arena's default model, `google/gemini-2.5-flash` | The target | `--ipi-judge` |
| Tool simulator (IPI Arena) | The scoring judge, as in IPI Arena | The scoring judge | `--ipi-worldsim` |
| Generator and AutoPI's judge | AutoPI's defaults, since their sampling is part of the method; the paper runs the judge on the generator | Temperature 0.7, 1,024 output tokens; the judge runs on the generator | `temperature` and `max_tokens` in the `--generator` alias (and in the `--autopi-judge` alias, if given) |

- A model alias's `temperature`, `top_p` and `max_tokens` shape the target's and the generator's
  requests. On IPI Arena the scoring judge and the tool simulator keep the benchmark's own per-call
  values, which are part of its scoring; pick their models with `--ipi-judge` and `--ipi-worldsim`.
- Anthropic targets accept no sampling parameters, so they run with the API's default sampling.
- OpenAI's o-series reasoning models reject a non-default temperature and take the output cap only as
  `max_completion_tokens`. Use an alias with `max_tokens_field: max_completion_tokens` and
  `omit_parameters: [temperature]` (see `configs/models.example.yaml`), for the target, the generator and
  IPI Arena's clients. A model that supports tool calling only through OpenAI's Responses API cannot be a
  target, since every client here uses Chat Completions.
- With thinking on, the reasoning counts toward the output cap, so a cap can cut an answer short; the
  generator's default 1,024-token cap can be too small for a hosted reasoning model.
- A self-hosted **native** target that thinks should be served with a reasoning parser (for example vLLM
  `--reasoning-parser`), so its reasoning is not scored and does not reach AutoPI's judge.
- Model cards often recommend sampling instead of greedy decoding. Qwen3.6's suggests temperature
  1.0 with thinking on, and Qwen3's advises against greedy decoding there. Google advises keeping
  Gemini 3 at its default sampling, and its newer models ignore a custom temperature, so AgentDojo's
  temperature 0 does not apply to them. Keep 0 to match the benchmarks where it applies, and report it
  if you follow the card instead.

## Reproducibility

The paper's experiments ran with earlier versions of this code and the settings the paper lists.
Results obtained with this release should state their models, server and parser versions, strategy
and settings, including the tool calling and the injection placement. Version 0.3.0 changed the
default tool calling of the short `vllm:` and `ollama:` target specs from prompted to native; pass
`--tool-calling prompted` for the earlier behavior. AgentDojo's own runner counts an attempt that
ends in a context overflow or a provider's server error as a successful attack; AutoPI leaves such an
attempt unmeasured (see the [strategy guide](strategies.md)), so compare with results from that
runner with this in mind. AutoPI keeps the bundled benchmarks' own scoring unchanged, including the known
issues listed in [THIRD_PARTY.md](../THIRD_PARTY.md#known-issues-in-the-bundled-benchmarks), so results stay
comparable with other work on them. The repository holds no model weights, credentials, server
addresses or experiment outputs.

## Run-to-run variation

Two runs with the same models, versions and settings can give different results. Judge a difference
between runs against this variation.

- The generator and AutoPI's judge, which shares its endpoint unless `--autopi-judge` names another model,
  sample at temperature 0.7 unless their alias sets another value, so each run can write different injections.
  An Anthropic generator or AutoPI judge uses the API's default sampling.
- Temperature 0 does not make a self-hosted target deterministic. Server load changes the batch a
  request is computed in, which changes the order in which the GPU adds numbers, so the same
  request can get a different answer, and a multi-step agent run carries the change forward. vLLM
  does not guarantee identical outputs by default. Its opt-in batch-invariant mode
  (`VLLM_BATCH_INVARIANT=1`, beta) targets this on supported GPUs and models. The GPU type and the
  server and library versions change the numbers too, so report them.
- Hosted APIs do not promise identical answers either, even at temperature 0. AutoPI's Anthropic
  requests carry no sampling parameters and use the API's default sampling (see
  [Model configuration](models.md)). A provider can also update its serving or point an alias
  at a newer snapshot.
- The benchmarks add variation of their own. AgentDojo orders a new calendar event's participants
  by Python's hash seed (set the `PYTHONHASHSEED` environment variable to fix the order), and it
  reruns an agent that ends without a final answer, up to three attempts, keeping the changes the
  earlier attempts made. IPI Arena's scoring judge and its tool simulator, WorldSim, are LLMs.
- AutoPI's own random choices, a random sweep order or a random pool of sub-techniques for an Ln
  sweep, use fixed seeds (`order_seed` and `stack_seed`, 0 by default), so they repeat on the same
  Python version.

Report how many runs you made and their confidence intervals (for example 95% Wilson intervals for
ASR), and compare methods on the same scenarios with a paired test such as McNemar's.
