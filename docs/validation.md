# Release validation — version 0.2.6, November 19, 2026

Validated locally on Python 3.11.18, macOS arm64.

- **222 tests passed**, covering AutoPI and both bundled benchmark test suites.
  The AutoPI tests also passed on Python 3.13.16 against the built wheels.
- The lint check passed on the source and the tests.
- Offline AgentDojo checks compared the adapter's selected slots and native
  execution against direct benchmark execution for **all 97 user tasks**.
- Setup validation enumerated **949 AgentDojo scenarios**, **41 IPI Arena
  scenarios** (18 tool, 15 coding, 8 browser), and **132 taxonomy sweep units**.
- The L1, Full and Seed-Exploit strategies were exercised with offline responses
  at their complete 133, 529 and 601 attempt budgets. Repeated/reordered phases,
  combination limits, histories, failures and stopping rules were checked.
- A local HTTP test exercised real OpenAI-compatible clients and the benchmark's
  prompted-tool parser with recovered token text. Missing token ids are rejected;
  generator output preserves structural tokens, including special tokens
  recovered from token ids.
- Source distributions and wheels built successfully. The AutoPI source archive
  includes both benchmark trees; its wheel includes the taxonomy. The IPI wheel
  includes all 41 scenario files.
- A separate environment installed the built wheels without editable source
  imports and successfully loaded all benchmark/taxonomy data. Both installed
  CLIs passed model-free setup validation.
- OpenAI, Anthropic and Google AgentDojo pipelines were constructed with arbitrary
  model ids and dummy credentials without contacting model endpoints. The
  Anthropic target and generator also ran through the installed anthropic SDK's
  own request code over an offline HTTP transport: no sampling parameter is
  sent, the output cap applies, and the target call is traced. Provider errors
  of the OpenAI, Anthropic and Google clients count as infrastructure failures.
- IPI Arena attempts were checked against the benchmark's own scoring rules: an
  unusable judge reply scores 0, and only a missing WorldSim reply becomes
  `{"status": "success"}`. The action-only score was checked against an explicit
  list of the concealment criteria of all 41 scenarios. The transcript AutoPI's
  diagnostic judge reads holds every step of the agent's turn.
- The publication folder was scanned for the workspace's private paths, server
  addresses, private keys and credential patterns. No such values were found.

Reproduce the test suite after installing the README's development dependencies:

```bash
python -m pytest -q tests benchmarks/agentdojo/tests benchmarks/ipi_arena/tests --import-mode=importlib
python -m ruff check src
python -m build
```

These checks establish software behavior and packaging. No new ASR experiment or
paid model request was performed. Browser rendering and real provider/model
behavior require the researcher's configured endpoints and browser installation.
The CI workflow also targets Python 3.10 and 3.12; those jobs have not been run
locally for this release. Exact locally tested dependency versions are recorded
in `requirements-tested.txt`.
