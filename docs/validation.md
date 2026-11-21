# Release validation — version 0.3.0, November 20, 2026

- **296 tests passed** on Python 3.11.18 on macOS arm64: 222 AutoPI tests and the
  74 tests of the bundled benchmarks.
- The suite includes **24 new regression cases** for the two AgentDojo options:
  native and prompted target clients and what each sends (tools and
  `tool_choice`, the system prompt's role, temperature 0, thinking off, tool
  definitions in the prompt), the short specs' native default, `--tool-calling`
  and `AUTOPI_TOOL_CALLING` with their precedence over a named alias, the refusal
  of prompted tools for hosted targets and of both options on IPI Arena,
  raw-token recovery's error, all three placements checked against AgentDojo's
  own slot choice for **all 97 user tasks** and run through AgentDojo's own
  execution and scoring, and the run manifest. Removing the system-role mapping,
  the native default or the `first` placement makes a test fail.
- An end-to-end run against a local stub OpenAI-compatible server exercised the
  CLI, AgentDojo's clients and real HTTP requests in four configurations (native
  and prompted tools, and the `all` and `first` placements) on one banking
  scenario. The native target received the suite's tools, the `system` role,
  temperature 0 and thinking off, and tool results with the injected text
  unchanged; the prompted target received the tools in its system prompt and
  JSON tool results. The generator's requests were unchanged.
- Lint passed on AutoPI's source and tests.
- All **193 files under `benchmarks/`** are unchanged from 0.2.8. No benchmark
  source, scenario, injection, scoring, simulator or validation file changed.
- AutoPI's source distribution and wheel built successfully; the wheel carries
  the 0.3.0 version and the bundled taxonomy.
- Setup validation enumerated **949 AgentDojo scenarios** with each tool-calling
  mode and placement, and **33 IPI Arena tool/coding scenarios**.

Reproduce the test suite after installing the README's development dependencies:

```bash
python -m pytest -q tests benchmarks/agentdojo/tests benchmarks/ipi_arena/tests --import-mode=importlib
python -m ruff check src tests
python -m build
```

These checks establish AutoPI behavior and packaging, with the bundled
benchmarks' existing semantics. No model was run for this release: a real
server's tool-call parser, real provider behavior and browser rendering require
configured endpoints and a browser installation. No new ASR experiment or paid
model request was performed. Python 3.10 and 3.12 execution remain covered by the
CI workflow, not by local runs for this release.
