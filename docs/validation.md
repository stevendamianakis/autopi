# Release validation — version 0.4.1, November 24, 2026

- **Bundled benchmarks are the latest upstream versions**, checked against upstream on
  November 24, 2026:
  - AgentDojo is upstream `main` at `089ed468`, the newest code; its latest release,
    v0.1.35, is five commits older. Its newest task-suite version is v1.2.2.
  - IPI Arena is release v0.1.0 (`41d40b29`, also upstream `main`). Of the 65 bundled
    files, 62 are identical to v0.1.0 and the other three hold only the changes listed in
    THIRD_PARTY.md (`behavior.py`, `pyproject.toml`, one test). Upstream's example attacks
    and sample output are not bundled, as before.
  - Open upstream pull requests that would change scoring are not merged, so they are not
    part of these versions.
- **IPI Arena update:** between the previous revision (`5e2e285`) and v0.1.0, upstream
  changed four files and no scenario, judge, simulator or scoring file. The 41 scenarios
  and the scenario-data digest the run manifest records are unchanged, and the scenario
  data is found both in the source tree and inside an installed wheel.
- **393 tests passed** on Python 3.11.18 on macOS arm64: 319 AutoPI tests and the
  74 tests of the bundled benchmarks, including the guard that the known benchmark
  issues stay as upstream has them.
- **Behaviour fingerprint:** every generator and judge prompt in all three modes, the
  notes added to prompts, context blending, scrubbing, masking and softening, the
  priority order, and the full scenario logs of six deterministic runs are identical
  to 0.4.0's. AutoPI's own code did not change in this release.
- Lint passed on AutoPI's source and tests.
- Setup validation enumerated **949 AgentDojo scenarios**. IPI Arena loads all **41
  scenarios**; without playwright, a dry run of its 33 tool and coding scenarios
  validates, while a selection holding a browser scenario stops at setup.
- The source distribution and wheel are built by the CI workflow on every push; the
  offline test environment used for this release has no build backend installed.

Reproduce the test suite after installing the README's development dependencies
and the `build` package:

```bash
python -m pip install build
python -m pytest -q tests benchmarks/agentdojo/tests benchmarks/ipi_arena/tests --import-mode=importlib
python -m ruff check src tests
python -m build
```

These checks establish AutoPI behavior and packaging, with the bundled
benchmarks' existing semantics. No model was run for this release: a real
server's tool-call parser, real provider behavior and browser rendering require
configured endpoints and a browser installation. No new ASR experiment or paid
model request was performed. Python 3.10 and 3.12 execution are covered by the
CI workflow, not by local runs for this release.
