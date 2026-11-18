# Third-party sources and provenance

The repository vendors source snapshots so cloning AutoPI also supplies both
benchmarks. It contains no nested Git repositories or Git submodules.

| Component | Source | Upstream revision | License |
|---|---|---|---|
| AgentDojo | https://github.com/ethz-spylab/agentdojo | `089ed468cf3ed0322acc66b0211f26d9d90dbf60` | [MIT](benchmarks/agentdojo/LICENSE), ETH Zurich |
| IPI Arena | https://github.com/GraySwanAI/ipi_arena_os | `5e2e285910d581c97767195f1d113ef323f9ab07` | [MIT](benchmarks/ipi_arena/LICENSE), original authors |
| PIAtlas taxonomy | PIAtlas, arXiv:2610.24999 | Paper taxonomy used by AutoPI | Distributed as part of the AutoPI research code |

Benchmark `src`, tests, metadata, README, license and required scenario data are
included. Historical run outputs, notebook artifacts, environments and VCS
metadata are excluded. The benchmark READMEs describe the upstream projects;
AutoPI's root README describes this integration.

Changes relative to the bundled upstream revisions:

- AgentDojo `local_llm.py`: propagate provider transport failures rather than
  returning empty text, so AutoPI can identify infrastructure failures. Tool
  argument parsing and benchmark checkers are retained. Thinking mode is not
  forced inside the benchmark; AutoPI applies the configured per-role setting
  in its own client.
- AgentDojo `anthropic_llm.py`: construct message dictionaries directly instead
  of calling the SDK's `MessageParam` type alias, which is a union in current
  Anthropic SDKs. Its test fixtures use the same representation.
- IPI Arena `behavior.py`: insert an empty tool result for unanswered
  prefilled tool calls and detect the actual image MIME type. Its data loader also locates scenario data inside installed wheels.
- IPI Arena packaging: include scenario data in wheels and declare the optional
  Hugging Face dataset dependency. A conversion regression test was updated to
  expect the required tool-result message.

The default important-instructions seed is benchmark content. Names and goals
inside benchmark fixtures and taxonomy examples are intentional research data,
not private credentials or environment-specific settings. Researchers can
choose another seed or taxonomy without changing code.
