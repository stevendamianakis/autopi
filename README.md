# AutoPI

AutoPI searches for indirect prompt injections against tool-using LLM agents. Its
benchmark-independent engine combines optional seeds, taxonomy exploration and
local exploitation with diagnostic feedback, context blending and style guidance.

This repository includes the runnable **AgentDojo** and **IPI Arena** benchmark
sources and data under `benchmarks/`, with their original licenses. No model
weights, credentials, private paths, server addresses or experiment outputs are included.

## Install

Python 3.11 is recommended. Exact locally validated dependency versions are in
`requirements-tested.txt` (usable with pip `-c requirements-tested.txt`). From this repository's root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,recovery]' -e benchmarks/agentdojo -e 'benchmarks/ipi_arena[dev]'
# Required for IPI Arena browser scenarios:
python -m pip install -e 'benchmarks/ipi_arena[browser]'
python -m playwright install chromium
```

Configure model aliases in a copy of `configs/models.example.yaml`. Model names,
URLs, credentials (by environment variable), decoding settings and tokenizers
are configuration choices. Use an exact served model id; no model is silently substituted.

```bash
cp configs/models.example.yaml configs/models.local.yaml
# Edit the model ids and endpoint URLs in configs/models.local.yaml.
autopi run --benchmark agentdojo --models-config configs/models.local.yaml \
  --target target --refiner generator --strategy configs/strategies/full.yaml \
  --logdir runs/agentdojo-full

autopi run --benchmark ipiarena --models-config configs/models.local.yaml \
  --target arena-target --refiner generator --judge arena-judge \
  --worldsim arena-judge --seed-source agentdojo \
  --strategy configs/strategies/full.yaml --logdir runs/ipiarena-full
```

IPI Arena defaults to its full `is_break` rule: **every criterion must pass**.
Its target uses native tool calls; a local model server must support that model's
tool-call format. Browser scenarios also require an image-capable target. `--ipi-ignore-concealment` selects a different, action-only metric.

## Search strategies

The three reported configurations are examples, not fixed restrictions:

| Strategy file | Blocks | Attempt budget per scenario |
|---|---|---:|
| `configs/strategies/l1.yaml` | seed, L1 sweep | 133 |
| `configs/strategies/full.yaml` | seed, L1 sweep, 3 variants per failed sweep line | 529 |
| `configs/strategies/seed-exploit.yaml` | seed, 600 seed variants | 601 |

The YAML strategy format supports any finite ordering and repetition of `seed`,
`sweep`, `seed_exploit` and `sweep_exploit` blocks. Each sweep can select a different
taxonomy subset, order, combination level and candidate limit. Higher-level
combinations are generated lazily. See [the strategy guide](docs/strategies.md).

The familiar flags (`-k`, `--seed-only`, `--no-seed`, `--stack-depth`, `--techniques`,
`--order`, and others) remain available when `--strategy` is omitted. Run
`autopi run --help` for all options. The default is a seed and L1 sweep.

Modes 1 and 2 generate goal-blind templates; mode 2 also receives the user task.
Mode 3 receives both the user task and injection goal. Custom taxonomies use
`--taxonomy-path`; custom seeds use `--seed-file`; `--no-seed` opens with exploration.
All three adaptive mechanisms can be disabled independently. `--no-early-stop`
continues through the schedule after successes, subject to its budget.

## Parsing and reproducibility

AutoPI places each injection in the slots AgentDojo marks as visible to the user
task and runs every attempt through the benchmark's own task runner. The bundled
IPI Arena includes two conversation-conversion fixes (see
[THIRD_PARTY.md](THIRD_PARTY.md)). Optional raw-token text recovery handles vLLM
endpoints whose tool parser removes prompted tool calls. Recovery needs the exact
served tokenizer and thinking disabled; it never applies to native IPI tool
calls. See [model configuration](docs/models.md) and the
[release notes](docs/release-notes.md). Validation details are in
[the release validation report](docs/validation.md).

Each run saves its resolved settings and per-scenario attempts. Request and response
traces are written incrementally when a log directory is provided. Infrastructure
errors use the attempt budget but do not enter the generator's failed-line history.
Unparseable generated candidates are skipped without a target attempt.

The bundled default taxonomy has 6 classes, 34 techniques and 131 named
sub-techniques plus one technique without sub-techniques: **132 sweep units**.
Source-technique provenance in logs is not post-hoc technique classification and
must not be reported as the paper's annotated coverage metric.

## Verify and extend

```bash
python -m pytest -q tests
python -m pytest -q benchmarks/agentdojo/tests benchmarks/ipi_arena/tests --import-mode=importlib
python -m build
```

See [extending AutoPI](docs/extending.md) for the benchmark adapter protocol and
the Python API. Upstream benchmark provenance and local changes are recorded in
[THIRD_PARTY.md](THIRD_PARTY.md). This release enables new experiments; it does
not claim that changed models, server versions or settings reproduce published ASR.

## License

AutoPI is MIT licensed. The bundled benchmarks retain their own MIT licenses and
copyright notices. See [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md).
