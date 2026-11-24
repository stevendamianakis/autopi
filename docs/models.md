# Model endpoints and response parsing

Each role selects a named endpoint from `--models-config`, or a short spec such
as `vllm:8000:MODEL_ID`, `openai:MODEL_ID`, `anthropic:MODEL_ID`,
`google:MODEL_ID` or `ollama:MODEL_ID`. An endpoint's URL can be on any host.
There is no dependency on the experiment servers. Two settings follow particular
model families and are described below: the thinking switch (`thinking`) and the
removal of reasoning a self-hosted model leaves in its reply.

The YAML `models` mapping supports these fields:

| Field | Meaning |
|---|---|
| `provider` | `openai-compatible`, `openai`, `anthropic`, `google`, `ollama` |
| `model` | Exact served/API model id; required |
| `base_url` | Optional custom API URL; required for generic compatible endpoints; `ollama` defaults to `OLLAMA_BASE_URL` or `http://localhost:11434` |
| `api_key_env` | Name of an environment variable containing the credential |
| `tool_mode` | AgentDojo: `native` API tools (default) or `prompted` text tools; `--tool-calling` overrides it |
| `thinking` | Compatible endpoint template mode: `off`, `on`, `server`; an `ollama` target receives it as `reasoning_effort` (`none` for off, else the alias's effort or `medium`), and an `ollama` generator, which calls Ollama's native chat API, as `think` (`true` or `false`). It is sent as `chat_template_kwargs.enable_thinking`, the switch Qwen-style templates read; a family with another switch (GPT-OSS, DeepSeek) ignores it, so use `server` and set the family's own field in `extra_body` |
| `tokenizer_path` | Optional exact tokenizer JSON for raw text recovery |
| `temperature`, `top_p`, `max_tokens` | Optional decoding overrides for the target and the generator (`anthropic`: `max_tokens` only). An IPI Arena scoring judge or WorldSim alias does not send them: the benchmark's own per-call values, part of its scoring, are kept |
| `max_tokens_field` | `max_tokens` (default) or `max_completion_tokens`, for `openai`/`openai-compatible` endpoints. OpenAI's o-series reasoning models take the output cap only as `max_completion_tokens`; set this and `omit_parameters: [temperature]` for them. Models that support tool calling only through the Responses API cannot be targets (see below) |
| `timeout`, `retries` | Request timeout in seconds and the retry count for connection, timeout, rate-limit and server errors. On the generator's HTTP path (`openai-compatible`, `openai`, `ollama`) a rate limit (HTTP 429) is retried at least five times and waits as long as the server's `Retry-After` asks, up to 120 seconds a wait; the provider SDKs (targets, IPI Arena clients, Anthropic and Google generators) apply their own retry rules with this count |
| `reasoning_effort` | Optional reasoning-effort field, sent on OpenAI-compatible request paths (`openai-compatible` and `openai` endpoints, `ollama` targets, and every IPI Arena client). The native `anthropic` and `google` clients (AgentDojo targets, generators) have no such field and refuse it; use `extra_body` there. An `ollama` generator calls Ollama's native chat API, which has no such field either: the CLI prints a note, and the native API's own field goes in `extra_body` |
| `extra_body` | Additional provider-specific request fields: added to the request body on OpenAI-compatible and Anthropic requests, and to the `GenerateContentConfig` on native Google requests (for example `thinking_config`, or the API's own spelling `thinkingConfig`). Google fields are checked against the SDK's `GenerateContentConfig`, so a misspelt field is refused at setup |
| `omit_parameters` | Parameter names to leave out of the request, such as `temperature` or `max_tokens` (`max_output_tokens` on Google; with `max_tokens_field: max_completion_tokens`, also that field). On a native Google AgentDojo target this also removes AgentDojo's own value (its temperature 0 and 65,535-token cap). An `anthropic` alias cannot omit `max_tokens`, which the API requires. On an Ollama generator, `max_tokens` here drops its `num_predict` option |

The generator and the judge (the LLM that diagnoses failed attempts) use
`--refiner` and share one configured endpoint, as in the paper. `--judge` is IPI Arena's **benchmark scoring judge**;
it defaults to the target. `--worldsim` is its tool simulator and defaults to
the benchmark judge. These have independent model aliases and URLs. A judge or
WorldSim alias's `temperature`, `top_p` and `max_tokens` are **not** sent: IPI
Arena sets its own on each judge and simulator call (temperature 0, a 4,096-token
judge cap, a 2,048-token WorldSim cap), and those are part of its scoring. That
cap includes any reasoning, so a reasoning model there may need a low reasoning
effort or thinking off. (`omit_parameters`, `extra_body`, `reasoning_effort` and the
removals a provider requires still apply to every role, since they describe what a
model accepts, not how it is scored.)

An Ollama generator alias merges an `extra_body.options` mapping (for example
`num_ctx`) into AutoPI's sampling options rather than replacing them, and sends
`top_p`, the `api_key_env` value as a bearer token, and retries like the other
generators. Such `options` reach only the generator, which uses Ollama's native
API: an Ollama target uses the OpenAI-compatible API, which cannot set the context
window, so give the model enough context on the server (for example
`OLLAMA_CONTEXT_LENGTH` or a Modelfile); Ollama's default context is small enough to
cut long agent conversations.

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
through the OpenAI-compatible API (`extra_content`).
See the official [Claude compatibility documentation](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk)
and [Gemini compatibility documentation](https://aistudio.google.com/docs/openai).
Any other service can be configured as `openai-compatible` with its own URL and
API-key environment variable. Custom model implementations can also be supplied
through the Python adapter interface.

**Provider coverage.** Open-weight models (vLLM, Ollama and other OpenAI-compatible
servers), Anthropic, Google Gemini and OpenAI chat models are configured directly.
All of AutoPI's and the benchmarks' clients use the Chat Completions API (or the
provider's native API), not OpenAI's Responses API:
- **OpenAI o-series reasoning models** need `max_tokens_field: max_completion_tokens`
  and `omit_parameters: [temperature]`; see the `openai-reasoning-example` alias in
  `configs/models.example.yaml`.
- A model that supports **tool calling only through the Responses API** (OpenAI's
  model pages list which, for example GPT-6 Astra and GPT-6.1 Sol) cannot be a target
  on either benchmark. It can still be the generator, which sends no tools.
- **Azure OpenAI** works as an `openai-compatible` endpoint through its
  OpenAI-compatible (v1) URL with key authentication.
- **A custom `base_url` on an `anthropic` or `google` alias** is the root the native
  SDK adds its own path to (`/v1/messages`, `/v1/models/...`), while IPI Arena's
  clients use the provider's OpenAI-compatible API and add `/chat/completions` to
  the same `base_url`. One custom URL therefore cannot serve both; leave
  `base_url` unset (each client then uses the provider's public URL) or define
  separate aliases for the two uses.
- **AWS Bedrock** (SigV4) and **Vertex AI** sign-in with service accounts or
  application-default credentials are not supported. A Gemini model is reached
  through the Gemini API with an API key (`GOOGLE_API_KEY`, or `GEMINI_API_KEY` as
  the google-genai SDK reads it).

The provider SDKs also read environment variables that reroute requests
(`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_GEMINI_BASE_URL`,
`GOOGLE_VERTEX_BASE_URL`, `GOOGLE_GENAI_USE_VERTEXAI`). AgentDojo targets and the
Anthropic and Google generators go through those SDKs, and the OpenAI generator
follows `OPENAI_BASE_URL` too. IPI Arena's clients use each endpoint's `base_url` or
the provider's public URL. The run manifest records which of these variables were
set (a URL without any `user:password` part).

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

### Reasoning in replies

The threat model gives the attacker the agent's visible replies and tool calls, not
its reasoning. How AutoPI keeps reasoning out depends on where it appears:

- **Anthropic** returns thinking as typed blocks. AgentDojo keeps them in its own
  messages (and its scoring reads them), but AutoPI drops them from the transcript its
  judge and context blending read.
- **Gemini** returns no thoughts unless `include_thoughts` is requested, and AgentDojo
  would turn returned thoughts into ordinary text. A target alias that requests them,
  in either spelling (`include_thoughts` or the API's `includeThoughts`), is refused
  at setup.
- **A self-hosted model that thinks** can leave reasoning in the reply text when the
  server has no reasoning parser. Serve such a target with a vLLM
  `--reasoning-parser` (or the server's equivalent), which keeps reasoning out of the
  scored reply, the tool-call text and the attacker's view. A **native** AgentDojo
  target needs it, since AgentDojo scores its reply as returned.
- **A server with a reasoning parser** returns the reasoning in a field of its own
  (`reasoning_content` or `reasoning`). AutoPI then keeps the reply as it is.
- **Without a parser**, AutoPI removes leaked reasoning from an `openai-compatible`
  or `ollama` endpoint whose thinking is not `off`: from a prompted AgentDojo target
  before tool parsing (so a call drafted inside the reasoning is not executed) and
  from IPI Arena replies. Only reasoning that opens the reply is removed: one or more
  leading `<think>`, `[THINK]`, `◁think▷` or `<seed:think>` blocks, a leading block
  cut off before it closes (all of the reply), and, with `thinking: on`, the text
  before a first `</think>` that no `<think>` precedes (a template that puts the
  opener in the prompt). A tag later in a reply is content and is kept, for example
  an injected token the agent quotes in a tool argument, or IPI Arena's judge
  quoting the injection.
- **A thinking-off endpoint** does not reason, so nothing is removed. A hosted service
  configured as `openai-compatible` (Azure, for example) returns no reasoning in its
  reply; leaving its thinking at the default `server` is safe too, since only a reply
  that opens with reasoning is changed. A family that reasons whatever `thinking`
  says (GPT-OSS, DeepSeek-R1) should be configured with `thinking: server` and served
  with a reasoning parser; without one, and with the opener in the prompt, use
  `thinking: on` so that the text before `</think>` is removed.
- **The generator and the judge**, when self-hosted and not thinking-off,
  have reasoning that opens their reply removed before the line or the
  `FAILURE_MODE` is parsed, including the text before a closing tag when AutoPI's
  answer follows it. A closing tag inside the generated line itself is kept.

## Recovering text removed by a server-side tool parser

A vLLM server with a native tool parser can remove `<function=...>` text from
`message.content`, even when AgentDojo expects to parse that text itself. With
recovery enabled, AutoPI requests `return_token_ids: true` and decodes the
generated token ids with the exact served tokenizer before normal template,
judge or prompted-tool parsing. Generator and judge text keeps special
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
Recovery requires thinking off, so recovered text is used as decoded, with no
reasoning removed.

## Output provenance

`--logdir` writes a run manifest with resolved model settings, package versions,
source hashes, taxonomy and tokenizer hashes, seed hash and strategy. It refuses
to mix a different configuration/code/tokenizer into the same output folder.
`--resume` skips completed scenarios; it does not resume a half-finished scenario.
`--num-shards N --shard I` distributes scenarios among workers. Workers may share
one output directory when their experiment settings match.

Traces default to `LOGDIR/traces`; use `--trace-dir` for another location. They
include prompts, original responses, recovered responses when enabled, tool
messages and outcomes, flushed incrementally. Authorization headers, API-key
fields, API-key values from the environment, the Google SDK's HTTP response
headers and any `user:password` part of a URL are left out. A field named like a
password or a secret keeps its name, with its value shown as `<redacted>`, so a
tool call such as AgentDojo's `update_password` stays readable. Treat prompts and
benchmark conversations as research data when sharing your own traces.

### Content-policy refusals

A provider's refusal of a prompt on content-policy grounds is recognised in three
forms: Azure OpenAI's `content_filter` (HTTP 400), OpenAI's `invalid_prompt` for a
flagged prompt (HTTP 400), and a moderation or guardrail block answered with HTTP
403 (OpenRouter). On AgentDojo such a refusal of the target is scored by AgentDojo
as a defended attempt; on IPI Arena it is a failed run, left out of the score as
IPI Arena leaves it out, and it does not count toward the stop for a failing
target. A refused generator prompt is skipped like a refusal. Another provider's
refusal in another form counts as a rejected request, that is an endpoint error.
