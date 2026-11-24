# Release validation — version 0.4.0, November 24, 2026

- **393 tests passed** on Python 3.11.18 on macOS arm64: 319 AutoPI tests and the
  74 tests of the bundled benchmarks. The release tests also pass when pytest is
  started from another directory.
- The suite includes **18 new regression cases** (`tests/test_release_040.py`):
  - `--help` listing the 13 new option names and none of the earlier ones;
  - each earlier name writing the same run manifest, scenario log and first output line as
    its new name (nine cases on a mock benchmark with a stub generator, plus the tokenizer
    and IPI Arena options), and `--resume` accepting a run started with the earlier names;
  - an earlier and a new name with different values refused;
  - the seed exploits of a seed-only run taken from `--seed-exploits`, `--seed-para-reps`
    or `-k`, also with `TAX_SEED_ONLY`, while `--sweep-exploits` and two different counts
    are refused, and the Python engine refusing a positive `seed_paraphrase_reps` with
    `seed_only`;
  - messages naming the options as the command line gave them, and the run's console words.
- The older tests keep the earlier option names, so they also check that commands written
  for 0.3.3 still run. One test that looked for the earlier names in `--help` now looks
  for the new ones.
- **Mutation check:** each of 18 parts of the change was reverted, one at a time in a copy
  of the code; in all 18 cases the tests meant to catch it failed. Without any change they
  pass.
- **Behaviour fingerprint:** every generator and judge prompt in all three modes, the
  notes added to prompts, context blending, scrubbing, masking and softening, the
  priority order, and the full scenario logs of six deterministic runs are identical
  to 0.3.3's.
- Lint passed on AutoPI's source and tests.
- All **193 files under `benchmarks/`** are unchanged from 0.2.8. No benchmark
  source, scenario, injection, scoring, simulator or validation file changed.
- Setup validation enumerated **949 AgentDojo scenarios**, with the same budget per
  scenario under the new and the earlier option names (529 for the Full strategy and for
  `--sweep-exploits 3`, as for `-k 3`). On IPI Arena without playwright, `--limit 1` (a tool
  scenario) validates, while a selection holding a browser scenario stops at setup.
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
