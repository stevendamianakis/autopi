# Release validation — version 0.2.8, November 19, 2026

- **272 tests passed** on Python 3.11.18 with `requirements-tested.txt`, and on
  Python 3.13.16 with the review environment's dependencies, on macOS arm64.
- The suite includes **45 new regression cases** for the four AutoPI fixes:
  SSL/proxy failures and resume, tokenizer setup with and without logging,
  consecutive unexpected errors and independent generator/judge streaks,
  category-only feedback in modes 1/2, generation limits in both scheduling
  interfaces and every refinement phase, resets, logs, traces and resume, and
  Anthropic HTTP transport compatibility.
- The complete 132-unit L1 sweep still runs when every generation is unusable,
  including both prompt tiers. Large level-20 sweeps stop at the configured
  unusable-candidate limit without adding target attempts. Existing tests also
  exercise the paper strategies' complete 133, 529 and 601 target-attempt budgets
  with offline responses.
- With **Anthropic SDK 0.47.0**, three targeted tests passed: construction using
  either HTTP transport, including `httpx2` being unavailable at construction,
  and four real local HTTP requests across AgentDojo's separate event loops,
  received exactly once each. The full suite covers the current SDK 2.10.0 too.
- Lint passed on AutoPI's source and tests; AutoPI source also parses with Python
  3.10 syntax rules. This is not an execution test on Python 3.10.
- All **193 files under `benchmarks/`** matched the pre-0.2.8 snapshot byte for
  byte. No benchmark source, scenario, scoring, simulator or validation file
  changed.
- AutoPI's source distribution and wheel, and both bundled benchmark wheels,
  built successfully. Package checks verified the 0.2.8 version, bundled
  taxonomy, both benchmark trees in the source archive, and all 41 IPI scenarios.
- A separate environment installed the built wheels without source imports.
  Setup validation enumerated **949 AgentDojo scenarios**, **33 IPI Arena
  tool/coding scenarios**, and **132 taxonomy units**. All 41 IPI scenarios load
  from the installed package; browser rendering was not exercised.

Reproduce the test suite after installing the README's development dependencies:

```bash
python -m pytest -q tests benchmarks/agentdojo/tests benchmarks/ipi_arena/tests --import-mode=importlib
python -m ruff check src tests
python -m build
```

These checks establish AutoPI behavior and packaging, with the bundled
benchmarks' existing semantics. The reviewed AgentDojo file-deletion checks,
thinking-text scoring and native injectability validator, and IPI Arena's
omission of conversation history from WorldSim requests, remain unchanged.
No new ASR experiment or paid model request was performed. Real provider/model
behavior and browser rendering require configured endpoints and a browser
installation. Python 3.10 and 3.12 execution remain covered by the CI workflow,
not by local runs for this release.
