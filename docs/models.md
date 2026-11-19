# Model endpoints and response parsing

Each role selects a named endpoint from `--models-config`, or a short spec such
as `vllm:8000:MODEL_ID`, `openai:MODEL_ID`, `anthropic:MODEL_ID`,
`google:MODEL_ID` or `ollama:MODEL_ID`. An endpoint's URL can be on any host.
There is no dependency on the experiment servers or a particular model family.

The YAML `models` mapping supports these fields:

| Field | Meaning |
|---|---|
| `provider` | `openai-compatible`, `openai`, `anthropic`, `google`, `ollama` |
| `model` | Exact served/API model id; required |
| `base_url` | Optional custom API URL; required for generic compatible endpoints; `ollama` defaults to `OLLAMA_BASE_URL` or `http://localhost:11434` |
| `api_key_env` | Name of an environment variable containing the credential |
| `tool_mode` | AgentDojo: `prompted` text tools or `native` API tools |
| `thinking` | Compatible endpoint template mode: `off`, `on`, `server`; an `ollama` target receives it as `reasoning_effort` (`none` for off, else the alias's effort or `medium`) |
| `tokenizer_path` | Optional exact tokenizer JSON for raw text recovery |
| `temperature`, `top_p`, `max_tokens` | Optional decoding overrides (`anthropic`: `max_tokens` only) |
| `timeout`, `retries` | Request timeout in seconds and provider retry count |
| `reasoning_effort` | Optional provider reasoning-effort setting |
| `extra_body` | Additional provider-specific request fields for compatible APIs |
| `omit_parameters` | Parameter names to leave out of compatible requests, such as `temperature` or `max_tokens` |

The generator and its diagnostic judge use `--refiner` and share one configured
endpoint, as in the paper. `--judge` is IPI Arena's **benchmark scoring judge**;
it defaults to the target. `--worldsim` is its tool simulator and defaults to
the benchmark judge. These have independent model aliases and URLs.

Hosted AgentDojo targets use the benchmark's native provider clients. Anthropic
requests carry no sampling parameters: the Anthropic API version that the
anthropic SDK 2.x sends (2026-10-21) rejects `temperature`, `top_p` and `top_k`
on every model, so an `anthropic` alias cannot set them and requests use the
API's default sampling. An Anthropic AgentDojo target allows 1,024 output tokens
unless its alias sets `max_tokens`; for a model that thinks by default, the
thinking counts toward that cap. IPI Arena
uses native tools via the provider's OpenAI-compatible API. Provider compatibility
layers may differ from native APIs; record the interface in your results.
See the official [Claude compatibility documentation](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk)
and [Gemini compatibility documentation](https://aistudio.google.com/docs/openai).
Any other service can be configured as `openai-compatible` with its own URL and
API-key environment variable. Custom model implementations can also be supplied
through the Python adapter interface.

`--target-thinking` and `--target-max-tokens` override the target alias. Named
aliases otherwise retain their own settings. Legacy `vllm:PORT:MODEL_ID` specs
default to thinking off. Use `thinking: server` for endpoints that do not support
the `chat_template_kwargs.enable_thinking` field. Hosted providers receive no
template field by default. Token caps and thinking modes can change results.

## Recovering text removed by a server-side tool parser

A vLLM server with a native tool parser can remove `<function=...>` text from
`message.content`, even when AgentDojo expects to parse that text itself. With
recovery enabled, AutoPI requests `return_token_ids: true` and decodes the
generated token ids with the exact served tokenizer before normal template,
judge or prompted-tool parsing. Generator and diagnostic-judge text keeps special
tokens such as `<|im_start|>`, since a structural injection can be written with
them, and drops only the end-of-turn token that stopped generation. Target text
skips special tokens, as the server's own text does.

To enable it, set `tokenizer_path` for the affected model alias,
or pass `--target-tokenizer FILE` / `--refiner-tokenizer FILE`. Install the
`recovery` extra. Recovery requires thinking off and non-streaming text requests.
For an AgentDojo target it also requires `tool_mode: prompted`. The server must
support returning token ids in each choice. Missing/invalid ids cause an error;
AutoPI never falls back to the stripped text or invents tool arguments. Original
responses and recovered text are saved when tracing is enabled.

Recovery is opt-in because it is not a standard OpenAI API feature and requires
the correct tokenizer. Ordinary endpoints use their visible text. **Do not apply
raw recovery to IPI Arena target/judge/WorldSim clients**: those use native tools
and must retain the server's parsed tool calls. Its generator can use recovery.
Reasoning delimiters are removed from IPI Arena replies.

## Output provenance

`--logdir` writes a run manifest with resolved model settings, package versions,
source hashes, taxonomy and tokenizer hashes, seed hash and strategy. It refuses
to mix a different configuration/code/tokenizer into the same output folder.
`--resume` skips completed scenarios; it does not resume a half-finished scenario.
`--num-shards N --shard I` distributes scenarios among workers. Workers may share
one output directory when their experiment settings match.

Traces default to `LOGDIR/traces`; use `--trace-dir` for another location. They
include prompts, original responses, recovered responses when enabled, tool
messages and outcomes, flushed incrementally. Headers and API-key values are
excluded. Treat prompts and benchmark conversations as research data when
sharing your own traces.
