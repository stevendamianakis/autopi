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
| `tool_mode` | AgentDojo: `native` API tools (default) or `prompted` text tools; `--tool-calling` overrides it |
| `thinking` | Compatible endpoint template mode: `off`, `on`, `server`; an `ollama` target receives it as `reasoning_effort` (`none` for off, else the alias's effort or `medium`) |
| `tokenizer_path` | Optional exact tokenizer JSON for raw text recovery |
| `temperature`, `top_p`, `max_tokens` | Optional decoding overrides (`anthropic`: `max_tokens` only) |
| `timeout`, `retries` | Request timeout in seconds and provider retry count |
| `reasoning_effort` | Optional reasoning-effort field, sent on OpenAI-compatible request paths (`openai-compatible`, `openai`, `ollama`, and every IPI Arena client). The native `anthropic` and `google` clients (AgentDojo targets, generators) have no such field and refuse it; use `extra_body` there |
| `extra_body` | Additional provider-specific request fields: added to the request body on OpenAI-compatible and Anthropic requests, and to the `GenerateContentConfig` on native Google requests (for example `thinking_config`) |
| `omit_parameters` | Parameter names to leave out of the request, such as `temperature` or `max_tokens` (`max_output_tokens` on Google). On a native Google AgentDojo target this also removes AgentDojo's own value (its temperature 0 and 65,535-token cap). An `anthropic` alias cannot omit `max_tokens`, which the API requires |

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
thinking counts toward that cap. Its HTTP client also supports older Anthropic
SDKs that use `httpx` rather than `httpx2`, retaining disabled HTTP keep-alive.
IPI Arena
uses native tools via the provider's OpenAI-compatible API. Its client sends
temperature 0 on every target, judge and WorldSim call; Claude 4.7 and later reject
a temperature below 1 there, so for an `anthropic` endpoint AutoPI leaves sampling
parameters out of those requests too and they use the API's default sampling.
Provider compatibility layers may differ from native APIs; record the interface in
your results.
Gemini 3 and later reject a request whose tool calls lack the thought signatures
they came with, and both benchmarks rebuild tool calls without them, so AutoPI's
clients put each signature back on its call: through AgentDojo's Gemini client and
through the OpenAI-compatible API (`extra_content`). An Ollama generator sends
`top_p`, the `api_key_env` value as a bearer token, and retries like the other
generators.
See the official [Claude compatibility documentation](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk)
and [Gemini compatibility documentation](https://aistudio.google.com/docs/openai).
Any other service can be configured as `openai-compatible` with its own URL and
API-key environment variable. Custom model implementations can also be supplied
through the Python adapter interface.

`--target-thinking`, `--target-max-tokens` and `--tool-calling` override the
target alias. Named aliases otherwise retain their own settings. Legacy
`vllm:PORT:MODEL_ID` specs default to thinking off. Use `thinking: server` for
endpoints that do not support the `chat_template_kwargs.enable_thinking` field.
Hosted providers receive no template field by default. Token caps, thinking modes
and the tool calling can change results.

## Tool calling (AgentDojo)

An AgentDojo target calls tools in one of two ways, set by `--tool-calling` (env
`AUTOPI_TOOL_CALLING`) or the alias's `tool_mode`:

- **`native`, the default.** AgentDojo's own provider clients send the tool
  definitions through the provider's or server's tool-calling API and read the
  calls it parses, as AgentDojo runs every model with tool support (its
  `vllm_parsed` pipeline for a local server). Tool results go back as tool
  messages with their text unchanged. For `openai-compatible` and `ollama`
  endpoints the system prompt is sent with the `system` role, which a
  self-hosted model's chat template knows, rather than the OpenAI `developer`
  role AgentDojo's client uses.
- **`prompted`.** AgentDojo's `LocalLLM`: the tool definitions are written into
  the system prompt, the model writes `<function=name>{...}</function>` text that
  AgentDojo parses, and each tool result goes back as a JSON string
  (`{"result": ...}`), so its newlines and quotes are escaped. For
  `openai-compatible` and `ollama` endpoints only. Hosted `anthropic`, `google`
  and `openai` targets use their native tool API and reject `prompted`.

In both modes AgentDojo's client asks for temperature 0, unless the alias sets its
own (Anthropic targets send none, see above). The short `vllm:PORT:MODEL_ID` and
`ollama:MODEL_ID` specs call tools natively since 0.3.0 (prompted before); named
aliases keep their `tool_mode`, `native` unless set. IPI Arena always calls tools
natively through its own client, so `--tool-calling` applies to AgentDojo only.

`--injection-placement` (env `AUTOPI_INJECTION_PLACEMENT`) sets where an AgentDojo
injection goes: `agentdojo` (default) fills the injection vectors AgentDojo's own
attacks fill, the ones the user task's ground-truth run reads; `all` fills every
injection vector of the suite, including ones the user task never reads; `first`
fills only the first of AgentDojo's vectors, in the suite's vector order. IPI Arena
places each scenario's injection where the scenario defines it.

## Serving a vLLM target

- **Native tool calls** (AgentDojo's default and IPI Arena) send the tools with
  `tool_choice: auto`, so the server must parse tool calls: launch vLLM with
  `--enable-auto-tool-choice` and the `--tool-call-parser` that matches the
  model's chat template, for example `hermes` for Qwen2.5 and Qwen3, and
  `qwen3_coder` for Qwen3.5, Qwen3.6 and Qwen3-Coder (the model card or the vLLM
  recipe names it). With a wrong parser the model's calls stay unparsed text, the
  agent never calls a tool and every scenario fails, so check that a smoke run
  makes tool calls. One such server serves both benchmarks.
- **AgentDojo prompted target** (`--tool-calling prompted`) expects the model's
  `<function=...>` text in `message.content` and parses it itself. A server-side
  tool-call parser can remove that text, so an AgentDojo prompted target must run
  **without** `--enable-auto-tool-choice`, or with raw-token recovery (below) to
  read the generated tokens before the parser. A prompted target whose reply comes
  back with the server's own parsed tool calls and no `<function=...>` text
  produces no tool calls and silently scores zero; serve it without the tool-call
  parser, and do not point it at the server of a native target.

The generator's requests carry no tools. If a server's tool-call parser still
rewrites tool-like text in such replies, the generated injections lose that text;
check them in the traces, and serve the generator without a parser or enable
recovery (below).

An AgentDojo prompted target also has its `<think>...</think>` reasoning removed
from the reply before tool parsing, so a thinking model does not have a
hypothetical call inside its reasoning executed. A reply cut off inside a
`<think>` block (for example by a token cap) loses everything from that `<think>`
on. A reply whose `<think>` opening is in the chat template rather than the
reply cannot be told apart from an answer, so for a thinking target serve vLLM
with a `--reasoning-parser`, which keeps the reasoning out of the reply.

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
For an AgentDojo target it also requires prompted tool calling (`--tool-calling prompted`
or `tool_mode: prompted`). The server must
support returning token ids in each choice. Missing/invalid ids, and ids outside
the tokenizer's vocabulary, cause an error;
AutoPI never falls back to the stripped text or invents tool arguments. Original
responses and recovered text are saved when tracing is enabled.

Missing, unreadable or invalid tokenizer files are configuration errors. The CLI
checks the generator tokenizer at setup, with or without logging. Through Python,
loading a bad generator tokenizer stops the run before the scenario is logged.

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
