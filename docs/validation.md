# Release validation — version 0.3.3, November 23, 2026

- **375 tests passed** on Python 3.11.18 on macOS arm64: 301 AutoPI tests and the
  74 tests of the bundled benchmarks. The release tests also pass when pytest is
  started from another directory.
- The suite includes **16 new regression cases** (`tests/test_release_033.py`):
  - `--resume` refusing seven kinds of incomplete or malformed scenario logs, and
    reusing the logs AutoPI writes;
  - the adapter check on the scenarios a run attacks: in `run_attack` (finished
    scenarios are not checked again on `--resume`), in the CLI after `--limit` and
    `--shard`, and for IPI Arena with and without browser scenarios;
  - a refused run leaving the other run's folder unchanged, the trace folder created
    with the first event, an unwritable trace location reported at setup, and a trace
    write error reported as such;
  - a guard that the bundled benchmarks keep their known upstream behaviour
    (AgentDojo's banking checks, IPI Arena's `tool_used` check).
- **Mutation check:** each of the eight changes was reverted, and each of the two
  guarded benchmark issues "fixed", one at a time in a copy of the code; in all ten
  cases the tests meant to catch it failed. Without any change they pass.
- **Behaviour fingerprint:** every generator and judge prompt in all three modes, the
  notes added to prompts, context blending, scrubbing, masking and softening, the
  priority order, and the full scenario logs of six deterministic runs are identical
  to 0.3.2's.
- The known benchmark issues listed in THIRD_PARTY.md were reproduced on the bundled
  code, which is identical to the pinned upstream revisions, and each is still present
  in the upstream repositories.
- Lint passed on AutoPI's source and tests.
- All **193 files under `benchmarks/`** are unchanged from 0.2.8. No benchmark
  source, scenario, injection, scoring, simulator or validation file changed.
- Setup validation enumerated **949 AgentDojo scenarios**. On IPI Arena without
  playwright, `--limit 1` (a tool scenario) validates, while a selection holding a
  browser scenario stops at setup.
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
