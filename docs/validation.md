# Release validation — version 0.5.0, November 24, 2026

- **Bundled benchmarks unchanged since 0.4.1**, which updated them to the latest upstream
  versions, checked against upstream on November 24, 2026:
  - AgentDojo is upstream `main` at `089ed468`, the newest code; its latest release,
    v0.1.35, is five commits older. Its newest task-suite version is v1.2.2.
  - IPI Arena is release v0.1.0 (`41d40b29`, also upstream `main`). Of the 65 bundled
    files, 62 are identical to v0.1.0 and the other three hold only the changes listed in
    THIRD_PARTY.md (`behavior.py`, `pyproject.toml`, one test). Upstream's example attacks
    and sample output are not bundled, as before.
  - Open upstream pull requests that would change scoring are not merged, so they are not
    part of these versions.
- **AutoPI's judge on a model of its own, end to end:** one AgentDojo scenario
  (`workspace`, `user_task_0` with `injection_task_0`) and one IPI Arena scenario
  (`book-hotel`) ran through the command line against local OpenAI-compatible stub
  servers, one port per model, with `--sweep-exploits 1` on six sub-techniques. Every
  generator request reached the `--generator` port (13), every request of AutoPI's judge
  the `--autopi-judge` port (one test request and six diagnoses), every target call the
  `--target` port (13), and on IPI Arena every scoring request the `--ipi-judge` port
  (13). The traces record these as the roles `refiner`, `diagnostic_judge`, `target` and
  `benchmark_judge`. Without `--autopi-judge`, the six diagnoses reached the generator's
  port, as in 0.4.1.
- **407 tests passed** on Python 3.11.18 on macOS arm64: 333 AutoPI tests and the
  74 tests of the bundled benchmarks, including the guard that the known benchmark
  issues stay as upstream has them. Each of the 14 new tests fails when the change it
  covers is reverted (22 reverted changes, each caught).
- **Behaviour fingerprint:** every generator and judge prompt in all three modes, the
  notes added to prompts, context blending, scrubbing, masking and softening, the
  priority order, and the full scenario logs of six deterministic runs are identical
  to 0.4.1's.
- Lint passed on AutoPI's source and tests.
- Setup validation enumerated **949 AgentDojo scenarios**. IPI Arena loads all **41
  scenarios**; without playwright, a dry run of its 33 tool and coding scenarios
  validates, while a selection holding a browser scenario stops at setup.
- The source distribution and wheel built offline (`uv build`); the wheel holds the
  taxonomy. The CI workflow builds both on every push.

Reproduce the test suite after installing the README's development dependencies
and the `build` package:

```bash
python -m pip install build
python -m pytest -q tests benchmarks/agentdojo/tests benchmarks/ipi_arena/tests --import-mode=importlib
python -m ruff check src tests
python -m build
```

These checks establish AutoPI behavior and packaging, with the bundled
benchmarks' existing semantics. No model was run for this release (the end-to-end
check used stub servers): a real server's tool-call parser, real provider behavior,
whether a given model refuses the judge's task, and browser rendering require
configured endpoints and a browser installation. No new ASR experiment or paid
model request was performed. Python 3.10 and 3.12 execution are covered by the
CI workflow, not by local runs for this release.
