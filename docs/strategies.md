# Composing a search strategy

Section 4 and Appendix G define a strategy as an ordered list of blocks, an
attempt budget per scenario, and a stop rule. `--strategy FILE.yaml` implements
that interface. Blocks can be omitted, reordered or repeated, as long as at least
one remains and each exploit block follows the block it refines; each occurrence
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
    techniques: [F1.T2, F2.T5]
    order_ids: [F1.T2.S1, F2.T5.S1, F1.T2.S2, F2.T5.S2]
    pool_size: 4
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

`sweep` generates one line per set of `level` sub-techniques. `techniques`
accepts class, technique or sub-technique ids. Omit it for the full taxonomy.
A sweep block that selects zero combinations (for example a `level` above the
number of selected sub-techniques, or an empty selection) is rejected when the strategy
loads, rather than running an empty schedule.
`order` is `priority`, `canonical` (file order), or `random` with `order_seed`.
Optional `order_ids` is an ordered list of sub-technique ids to put first;
unlisted selected sub-techniques follow in the chosen order. `pool_size` restricts the
combination pool after ordering. `limit` caps the number of combinations tried.
Combinations are unordered sets without repetition, listed in a fixed order (by
their members' positions in the pool), so `pool_size` and `limit` keep the first
ones; use `order` or `order_ids` to choose which sub-techniques come first. In the
example above, `order_ids` puts two sub-techniques of each technique first, so the
four-unit pool gives six pairs, four of them across the two techniques.
Higher levels are generated lazily; setting a small target budget does not
allocate billions of combinations.

`seed_exploit` makes `reps` rewrites of the latest earlier seed. A seed
block must precede it, and the seed must not have succeeded; a seed attempt that
failed in the infrastructure is refined all the same (its first rewrite simply
has no judge feedback to work from). `sweep_exploit` makes `reps` variants of
each failed sweep line from the named earlier `source`. If source is omitted, it
uses all earlier sweeps. Repeating an exploit block continues that source's
history of failed rewrites or variants and accumulated judge feedback. Repeating a sweep
creates fresh source lines, tracked by their originating block and target attempt.

A candidate the generator could not write makes no target run and does **not**
consume the attempt budget: an invalid, empty or refused generator output is
logged and skipped. A line that opens like a refusal ("I can't…", "I'm sorry…")
counts as one, also inside the `[TEMPLATE_START]` and `[TEMPLATE_END]` delimiters, so a written line that opens
this way is skipped too, and so is a prompt the provider refuses on content-policy
grounds. Before the first scenario, a command-line run sends the generator **one
short test request** with the run's settings and stops if it fails, so a generator
that rejects those settings (an unsupported `temperature` or `max_tokens`, a wrong
model id or key) is caught before any target call; a run that tries only the seed
skips the test. During the run, a generator that cannot be reached (a connection
failure, timeout, rate limit or server error after the endpoint's retries) stops the
run before the scenario is logged, so `--resume` repeats it. This includes TLS/SSL
and proxy failures. A missing, unreadable or invalid tokenizer is a setup error, and
so is a rejected key, account or model id (HTTP 401, 402, 403 other than a
moderation block, 404, or an invalid API key). Three consecutive unexpected request
errors from the generator or the judge also stop the run before logging the
current scenario. Any other HTTP 400 is a per-prompt rejection and stays a skippable
candidate, but **25 consecutive HTTP 400s** from a role stop the run, since a
generator that rejects every request would otherwise turn it into seed-only results.
Each role keeps its own error and 400 streaks across scenarios; a successful request
or a content-policy refusal clears them.

A target run that fails in the infrastructure (a transport, provider or parse
error, or a request the provider rejected) does consume the budget, but never
becomes a failed sweep line to refine or a failed rewrite or variant listed in an exploit prompt. A scenario whose every
attempt was an infrastructure failure is not logged and stays out of ASR; the
summary counts it, and `--resume` runs it again. A scenario whose **first ten
attempts** all fail in the infrastructure, whatever the error, ends there instead
of spending the rest of its budget. Such scenarios are listed in
`LOGDIR/infrastructure_only.jsonl`, and `--resume` runs them after the others, so a
group of scenarios that fail on their own cannot stop every resumed run before the
rest. The run stops when the target keeps failing with **endpoint errors**: it
cannot be reached, times out, is rate limited, fails on the server, or rejects the
request (for example a parameter the model does not support, or a server started
without the tool-call parser native tools need). It stops after **25 target
attempts in a row** that failed that way, or after ten scenarios in a row whose
every attempt did, before the current scenario is logged. Errors tied to one
scenario (a context overflow, a malformed tool call, a broken environment) do not
count, and since they show that the endpoint answers, they end both streaks. A
context overflow is recognised by its wording (OpenAI, vLLM, Anthropic, Gemini,
llama.cpp and Hugging Face TGI wordings); one worded otherwise counts as an
endpoint error. A target that rejects its key, account or model id (HTTP 401, 402,
403 other than a moderation block, 404, or an invalid API key) stops the run at
once. When no scenario of a run was scored, the summary prints the ASR as `n/a`
and the command exits with an error. AgentDojo's own runner
(`benchmark_suite_with_injections`) scores a context overflow or a provider's
server error as a successful attack (utility false, security true); AutoPI leaves
such an attempt unmeasured instead, which matters when comparing with results
produced by that runner.
On AgentDojo, a request the target's provider refuses on content-policy grounds
(Azure `content_filter`, OpenAI `invalid_prompt`, a moderation 403) ends the agent's
run there and is scored by AgentDojo on the state it reached, as a defended attempt
when the goal was not met, so a fully blocked scenario stays a miss in the
denominator. On IPI Arena such a refusal is a failed run that IPI Arena leaves out of
its ASR, and AutoPI does the same. Infrastructure failures inside a scored scenario
are counted in the summary and the scenario log. Successful source lines are not
selected for exploitation. If continuing after a successful variant, that success
is not sent to the judge as a failure.

The finite schedule can end before the budget (for example after skipped
generator outputs or when no failed sources exist). `--max-attempts` can impose
a smaller cap. To bound huge sweeps whose generator keeps refusing or returning
unusable output, the scenario also ends after **200 consecutive unusable generated
candidates**. A candidate can use the existing two prompt tiers before it is
skipped; neither tier adds a target attempt. The streak spans sweep and exploit
blocks, ignores seed blocks and judge replies, resets on a usable
generated candidate, and starts fresh for each scenario. The 132-unit L1 sweep
fits below the default limit even if every candidate is unusable. The scenario
is logged with `generation_stop_reason: consecutive_unusable_generations`, its
actual target-attempt count and any successes already found. Resume retains
this completed result, and the summary reports how many scenarios hit the limit.

Configure positive integer limits with `TAX_MAX_CONSECUTIVE_UNUSABLE` (default
200), `TAX_MAX_CONSECUTIVE_REFINER_ERRORS` (default 3) and
`TAX_MAX_CONSECUTIVE_REFINER_400` (default 25). Python callers can use the
corresponding `max_consecutive_unusable`, `max_consecutive_refiner_errors` and
`max_consecutive_refiner_400` constructor arguments. Resolved limits appear in the
manifest and scenario logs.

An exploit prompt lists the line's earlier failed variants (or the seed's earlier
failed rewrites) and the judge's accumulated feedback, keeping only the most
recent ones: at most 10 of each by default, set with `TAX_PARA_HISTORY_WINDOW` and
`TAX_SIGNAL_HISTORY_WINDOW`.

A scenario log lists every attempt that did not succeed under `failed_templates`,
with its `status`: `failed` for an attempt the target resisted, and an `infra_`
status for one that failed in the infrastructure (also listed under
`infra_failures`).

Modes 1 and 2 receive only the judge's fixed failure-category phrase;
its free-text explanation is withheld to keep the injection goal hidden. Mode 3
retains that explanation.

`--no-early-stop` overrides `stop: first_success` for a coverage
run. Use a fresh output folder when changing settings or code.

Without `--strategy`, the schedule flags remain available (`--seed-exploits`,
`--sweep-exploits`, `--max-sweep-level` and the other sweep options in `autopi run --help`).
Do not mix those scheduling flags with `--strategy`; define the order and counts in one place.
Seed selection, model settings, awareness mode, mechanism switches, taxonomy
path, task selection, sharding and output options apply to either interface.

The included L1, Full and Seed-Exploit files match the paper's intended budgets;
`repeated-phases.yaml` is an example of repeated and reordered blocks.
Use `--max-attempts 350` with Seed-Exploit for its IPI equal-cost budget. A changed
taxonomy may contain a different number of units; adjust the budget accordingly.
