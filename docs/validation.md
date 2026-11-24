# Release validation — version 0.3.2, November 23, 2026

- **359 tests passed** on Python 3.11.18 on macOS arm64: 285 AutoPI tests and the
  74 tests of the bundled benchmarks. The release tests also pass when pytest is
  started from another directory.
- The suite includes **33 new regression cases** (`tests/test_release_032.py`). They
  cover every 0.3.2 change and the 0.3.1 fixes that had no test of their own:
  - reasoning removed only where it opens a reply, and not at all when the server
    returned it apart; IPI Arena's judge verdict quoting a tag kept intact; a reply cut
    off while reasoning turned into an empty reply;
  - the shared context-overflow wordings on both benchmarks; a scenario that fails on
    its own ending after ten attempts while the run goes on; `--resume` running such
    scenarios last; a dead endpoint still stopping within 25 attempts; the ten-scenario
    streak ended by a scenario lost to its own errors; `n/a` and an error exit when
    nothing was scored;
  - the IPI Arena options refused on AgentDojo; a folder of another run refused
    before anything is written to it; `--version`; the missing-seed hint with a
    strategy; `--resume` without `--logdir`; a corrupt scenario log; the manifest's
    provider variables;
  - the Ollama effort, the Gemini settings check and `includeThoughts`, the generator
    error classes, the leading reasoning of a generator reply, the Python API's
    default mode, IPI Arena's scoring judge and WorldSim keeping their own decoding,
    and the target reasoning guard on both benchmarks;
  - traces that keep password fields readable and leave credentials, URL credentials
    and Google response headers out;
  - the Chromium check for IPI Arena's browser scenarios, IPI Arena's credential
    stop, a non-injectable AgentDojo task, every AgentDojo restart in the transcript,
    and the example L2 sweep.
- **Mutation check:** each of 44 fixes (the 0.3.2 changes, the 0.3.1 fixes that gained a
  test, and two strengthened older tests) was reverted on its own in a copy of the
  code, and in every case the tests meant to guard it failed; without any change they
  pass.
- **Behaviour fingerprint:** every generator and judge prompt in all three modes, the
  notes added to prompts, context blending, scrubbing, masking and softening, the
  priority order, and the full scenario logs of six deterministic runs are identical
  to 0.3.1's, except the run of `repeated-phases.yaml`, whose example L2 sweep changed
  on purpose. With 0.3.1's version of that file the fingerprint is byte-identical.
- **End-to-end runs against a local stub OpenAI-compatible server** went through the
  CLI, AgentDojo's client and real HTTP requests:
  - a scenario whose target request failed with llama.cpp's context-overflow wording
    ended after ten attempts as an error of its own, the other scenario was scored,
    and the run exited normally;
  - `--resume` ran that scenario after the others and ended it early again;
  - with every request failing that way, the summary printed `ASR=n/a` and the command
    exited with an error.
- Lint passed on AutoPI's source and tests.
- All **193 files under `benchmarks/`** are unchanged from 0.2.8. No benchmark
  source, scenario, injection, scoring, simulator or validation file changed.
- Setup validation enumerated **949 AgentDojo scenarios** and **33 IPI Arena
  tool/coding scenarios**; selecting the browser scenarios without playwright stops at
  setup.
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
