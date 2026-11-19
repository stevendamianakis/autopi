# Composing a search strategy

Section 4 and Appendix G define a strategy as an ordered list of blocks, an
attempt budget per scenario, and a stop rule. `--strategy FILE.yaml` implements
that interface. Blocks can be omitted, reordered or repeated; each occurrence
has independent sweep parameters. A block's `id` must be unique and defaults to
`block_1`, `block_2`, and so on.

```yaml
version: 1
budget: 40
stop: first_success    # or budget, to continue after success
blocks:
  - kind: seed
  - kind: sweep
    id: first
    level: 1
    techniques: [F2.T5, F1.T2]
    order: random
    order_seed: 17
    limit: 6
  - kind: sweep_exploit
    source: first
    reps: 2
  - kind: seed_exploit
    reps: 3
  - kind: sweep
    id: second
    level: 2
    techniques: [F1, F2]
    pool_size: 5
    limit: 8
  - kind: sweep_exploit
    source: second
    reps: 2
```

`seed` yields the selected fixed seed without a generator request. The seed file
must contain `<task_goes_here>`; its goal is inserted only when placing it.
On AgentDojo the default is the paper's important-instructions seed. IPI Arena
has no default seed template: choose `--seed-source agentdojo`, `--seed-file`,
or a strategy with no seed block. The seed is benchmark content, not a model
or infrastructure configuration. Its names/text can be changed with a seed file.

`sweep` generates one candidate per set of `level` taxonomy units. `techniques`
accepts family, technique or sub-technique ids. Omit it for the full taxonomy.
A sweep block that selects zero combinations (for example a `level` above the
number of selected units, or an empty selection) is rejected when the strategy
loads, rather than running an empty schedule.
`order` is `priority`, `canonical` (file order), or `random` with `order_seed`.
Optional `order_ids` is an ordered list of sub-technique ids to put first;
unlisted selected units follow in the chosen order. `pool_size` restricts the
combination pool after ordering. `limit` caps the number of combinations tried.
Combinations are unordered sets, sampled without repetition within a sweep.
Higher levels are generated lazily; setting a small target budget does not
allocate billions of combinations.

`seed_exploit` makes `reps` variants of the latest earlier failed seed. A seed
block must precede it. `sweep_exploit` makes `reps` variants of each failed sweep
line from the named earlier `source`. If source is omitted, it uses all earlier
sweeps. Repeating an exploit block continues that source's failed-variant history
and accumulated diagnostic feedback. Repeating a sweep creates fresh source
lines, tracked by their originating block and target attempt.

Only actual target runs consume the attempt budget. Invalid, empty or refused
generator outputs are logged and skipped. A line that opens like a refusal ("I
can't…", "I'm sorry…") counts as one, also inside the template delimiters, so a
written line that opens this way is skipped too. A generator that cannot be reached (a
connection failure, timeout, rate limit or server error after the endpoint's
retries) stops the run before the scenario is logged, so `--resume` repeats it.
This includes TLS/SSL and proxy failures. A missing, unreadable or invalid
tokenizer is a setup error. Three consecutive unexpected request errors from
the generator or diagnostic judge also stop the run before logging the current
scenario. Each role keeps its own streak across scenarios; a successful request
clears that role's streak. HTTP 400 remains a skippable per-prompt rejection.
Infrastructure failures consume budget but never become failed source lines or
failed-variant examples. A scenario whose every attempt was an infrastructure
failure is not logged and stays out of ASR; the summary counts it, and `--resume`
runs it again. Successful source lines are not selected for exploitation. If
continuing after a successful variant, that success is not sent to the diagnostic
judge as a failure.

The finite schedule can end before the budget (for example after skipped
generator outputs or when no failed sources exist). `--max-attempts` can impose
a smaller cap. To bound huge sweeps whose generator keeps refusing or returning
unusable output, the scenario also ends after **200 consecutive unusable generated
candidates**. A candidate can use the existing two prompt tiers before it is
skipped; neither tier adds a target attempt. The streak spans sweep and exploit
blocks, ignores fixed-seed blocks and diagnostic replies, resets on a usable
generated candidate, and starts fresh for each scenario. The 132-unit L1 sweep
fits below the default limit even if every candidate is unusable. The scenario
is logged with `generation_stop_reason: consecutive_unusable_generations`, its
actual target-attempt count and any successes already found. Resume retains
this completed result, and the summary reports how many scenarios hit the limit.

Configure positive integer limits with `TAX_MAX_CONSECUTIVE_UNUSABLE` (default
200) and `TAX_MAX_CONSECUTIVE_REFINER_ERRORS` (default 3). Python callers can use
the corresponding `max_consecutive_unusable` and `max_consecutive_refiner_errors`
constructor arguments. Resolved limits appear in the manifest and scenario logs.

Modes 1 and 2 receive only the diagnostic judge's fixed failure-category phrase;
its free-text explanation is withheld to keep the injection goal hidden. Mode 3
retains that explanation.

`--no-early-stop` overrides `stop: first_success` for a coverage
run. Use a fresh output folder when changing settings or code.

Without `--strategy`, the existing schedule flags remain available. Do not mix
those scheduling flags with `--strategy`; define the order and counts in one place.
Seed selection, model settings, awareness mode, mechanism switches, taxonomy
path, task selection, sharding and output options apply to either interface.

The included L1, Full and Seed-Exploit files match the paper's intended budgets;
`repeated-phases.yaml` is a composition example (repeated and reordered blocks).
Use `--max-attempts 350` with Seed-Exploit for its IPI equal-cost budget. A changed
taxonomy may contain a different number of units; adjust the budget accordingly.
