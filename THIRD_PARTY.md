# Third-party sources and provenance

The repository vendors source snapshots so cloning AutoPI also supplies both
benchmarks. It contains no nested Git repositories or Git submodules.

| Component | Source | Upstream revision | License |
|---|---|---|---|
| AgentDojo | https://github.com/ethz-spylab/agentdojo | `089ed468cf3ed0322acc66b0211f26d9d90dbf60` | [MIT](benchmarks/agentdojo/LICENSE), Edoardo Debenedetti, Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc Fischer, and Florian Tramèr |
| IPI Arena | https://github.com/GraySwanAI/ipi_arena_os | `5e2e285910d581c97767195f1d113ef323f9ab07` | [MIT](benchmarks/ipi_arena/LICENSE), Gray Swan AI |
| PIAtlas taxonomy | PIAtlas, arXiv:2610.24999 | Paper taxonomy used by AutoPI | Distributed as part of the AutoPI research code |

Portions of AutoPI are derived from AgentDojo, so the root [LICENSE](LICENSE) also
carries AgentDojo's copyright notice.

Benchmark `src`, tests, metadata, README, license and required scenario data are
included. Historical run outputs, notebook artifacts, environments and VCS
metadata are excluded. The benchmark READMEs describe the upstream projects;
AutoPI's root README describes this integration.

Changes relative to the bundled upstream revisions:

- AgentDojo `local_llm.py` (the prompted-tool client): remove the catch-all
  `except Exception` that turned any failed request into an empty reply. Every
  request error (a transport failure, a rate limit, an HTTP 400 such as a context
  overflow) now reaches AutoPI, which records the attempt as unmeasured, or, for a
  content-policy refusal, has AgentDojo score it as a defended attempt. Upstream
  scored an errored prompted request as an empty final reply. Tool argument parsing
  and benchmark checkers are retained.
- AgentDojo `anthropic_llm.py`: construct message dictionaries directly instead
  of calling the SDK's `MessageParam` type alias, which is a union in current
  Anthropic SDKs. Its test fixtures use the same representation. A
  `redacted_thinking` content block is also converted back to an Anthropic
  `redacted_thinking` block instead of raising, so a thinking target that returns
  one does not crash on the following turn. Scoring is unchanged.
- IPI Arena `behavior.py`: insert an empty tool result for unanswered
  prefilled tool calls and detect the actual image MIME type. Its data loader also locates scenario data inside installed wheels.
- IPI Arena packaging: include scenario data in wheels and declare the optional
  Hugging Face dataset dependency. A conversion regression test was updated to
  expect the required tool-result message.

The default important-instructions seed is benchmark content. Names and goals
inside benchmark fixtures and taxonomy examples are intentional research data,
not private credentials or environment-specific settings. Researchers can
choose another seed or taxonomy without changing code.
