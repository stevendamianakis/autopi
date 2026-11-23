# Release validation — version 0.3.1, November 23, 2026

- **326 tests passed** on Python 3.11.18 on macOS arm64: 252 AutoPI tests and the
  74 tests of the bundled benchmarks.
- The suite includes **31 new regression cases** (`tests/test_release_031.py`) for
  the review fixes. They cover:
  - a target's reasoning kept out of the diagnostic judge's transcript, and a
    Gemini target refused when it asks for its thoughts;
  - `seed_exploit` running after an infrastructure-failed seed, with the legacy and
    strategy engines agreeing;
  - the stop when the target keeps failing: a target rejecting every request
    stops within 25 attempts, scenario-specific errors such as a context overflow
    never stop the run, and a rejected key, payment or model id stops it at once;
  - content-policy refusals on AgentDojo, including a moderation HTTP 403, scored as
    defended attempts through AgentDojo's own scoring, with an empty final reply;
  - the generator: the test request before the first scenario, the HTTP 400
    streak, content-policy refusals kept out of that streak, and rate-limit backoff
    honouring `Retry-After`;
  - reasoning tags kept as content for a thinking-off target (a tool call quoting
    them stays intact), and a template that puts `<think>` in the prompt handled for
    the generator and the judge;
  - the OpenAI reasoning-model `max_completion_tokens` field and its `omit_parameters`
    mapping, the Ollama generator's `options` merge, `OPENAI_BASE_URL` for the OpenAI
    generator, the manifest's provider variables without credentials, and
    `GEMINI_API_KEY`;
  - strategy validation of repeated `order_ids`.
- The test file needs 0.3.1's interfaces, so it does not import on 0.3.0. The bugs
  it covers were reproduced separately with local scripts:
  - the reasoning leak to the judge and the skipped seed exploit, on 0.3.0;
  - five more on the first 0.3.1 draft, which they came from: a target rejecting
    every request never stopping the run, a moderation 403 stopping it, a no-op
    `omit_parameters: [max_tokens]`, a quoted closing tag destroying a tool call, and
    a lone `</think>` left in a generator reply.
- **End-to-end runs against local stub OpenAI-compatible servers** went through the
  CLI, AgentDojo's clients and real HTTP requests:
  - a target answering every request with HTTP 400 (a server without the
    tool-call parser) stopped after 25 target and 25 generator requests;
  - a working stub ran a capped banking scenario, and its log recorded the cap and
    the effective budget;
  - `--resume` without `--logdir` was rejected as a usage error.
- Lint passed on AutoPI's source and tests.
- All **193 files under `benchmarks/`** are unchanged from 0.2.8. No benchmark
  source, scenario, injection, scoring, simulator or validation file changed.
- AutoPI's source distribution and wheel built successfully. The wheel carries the
  0.3.1 version and the bundled taxonomy. The AutoPI tests pass on the **3.13 wheel**
  with the provider SDKs installed; the AgentDojo- and IPI Arena-gated cases are
  skipped there without the bundled benchmarks.
- Setup validation enumerated **949 AgentDojo scenarios** and **33 IPI Arena
  tool/coding scenarios**.

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
model request was performed. Python 3.10 and 3.12 execution remain covered by the
CI workflow, not by local runs for this release.
