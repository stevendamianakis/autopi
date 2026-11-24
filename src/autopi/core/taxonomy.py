"""PIAtlas construction taxonomy: loading, softening, prioritization, and the
Ln-sweep (sub-technique combination) schedule. Its top level is a class (the paper's term;
the file and this code call it a family).

This is the data core of AutoPI and is completely benchmark- and model-agnostic.

The taxonomy JSON ships in ``core/data/`` and is swappable via ``taxonomy_path`` (the
``--taxonomy-path`` CLI flag), so another taxonomy with the same file layout can be used.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable

# Default = the PIAtlas taxonomy (arXiv:2610.24999): 6 classes / 34 techniques / 131 sub-techniques.
# Technique F3.T6 has no sub-technique and is swept as one unit, so the sweep has 132 units.
_BUNDLED_TAXONOMY = Path(__file__).parent / "data" / "piatlas_taxonomy.json"

# Taxonomy examples use benchmark canaries to illustrate technique SHAPE; strip them to a
# neutral marker so the generator learns the structure, not a stray payload.
_CANARY_PATTERNS = [
    re.compile(r"I have been PWNED", re.IGNORECASE),
    re.compile(r"ACCESS[ _]GRANTED", re.IGNORECASE),
    re.compile(r"(print|reveal|output|show)[^.\n]{0,20}system prompt", re.IGNORECASE),
    re.compile(r"PWNED", re.IGNORECASE),
]
_CANARY_REPLACEMENT = "[MY TASK]"

# Light semantic softening of taxonomy free-text before it is shown to the generator: the
# MEANING is preserved; only vocabulary likely to trip a safety-tuned generator is swapped
# for neutral equivalents.
_SOFTEN = [
    # Order matters: multi-word / more-specific stems first.
    (re.compile(r"\bprompt[- ]inject\w*\b", re.IGNORECASE), "embedded-instruction"),
    (re.compile(r"\battacker\w*\b", re.IGNORECASE), "sender"),
    (re.compile(r"\battack\w*\b", re.IGNORECASE), "pattern"),
    (re.compile(r"\bexploit\w*\b", re.IGNORECASE), "leverage"),
    (re.compile(r"\bmalicious\w*\b", re.IGNORECASE), "target"),
    (re.compile(r"\bpayload\w*\b", re.IGNORECASE), "content"),
    (re.compile(r"\bvictim\w*\b", re.IGNORECASE), "assistant"),
    (re.compile(r"\badversar\w*\b", re.IGNORECASE), "alternative"),
    (re.compile(r"\bjailbr\w*\b", re.IGNORECASE), "override"),  # jailbreak / jailbroken
    (re.compile(r"\binject\w*\b", re.IGNORECASE), "embed"),
]

# Techniques whose sub-techniques legitimately require non-ASCII characters (homoglyph /
# zero-width; other-language content): sanitisation does NOT strip Unicode for these.
_UNICODE_TECHNIQUES = {"F3.T1", "F3.T3"}
# Classes (families in the file) that legitimately use structural markers / angle-bracket tokens.
_STRUCTURAL_FAMILIES = {"F2"}

# Sweep order: front-load the techniques most likely to crack an instruction-tuned agent
# so solvable pairs succeed in the fewest tries. Nothing is dropped — only reordered. Any
# technique id not listed here is appended afterwards in canonical order.
_PRIORITY_TECHNIQUES = [
    "F2.T2", "F2.T1", "F2.T5",            # structural spoofing: tokens, role-tags, fake tool output
    "F1.T2", "F1.T3",                       # context override / negation, task hijacking
    "F6.T1", "F6.T3", "F6.T2",            # output coercion / refusal suppression
    "F2.T3", "F2.T4",                       # syntax/format injection, synthetic dialogue
    "F1.T1", "F1.T4", "F1.T5",            # naked imperative, salience, indirect/declarative
    "F5.T1", "F5.T2", "F5.T3", "F5.T4",            # scenario framing / personas
    "F4.T1", "F4.T4", "F4.T7", "F4.T3", "F4.T6", "F4.T5", "F4.T2",   # social engineering
    "F3.T1", "F3.T4", "F3.T5", "F3.T3", "F3.T2", "F3.T6",   # obfuscation (last)
]


def soften(text: str) -> str:
    for pat, repl in _SOFTEN:
        text = pat.sub(repl, text)
    return text


def _abstract_example(text: str) -> str:
    """Replace benchmark canaries in an illustrative example with a neutral marker."""
    for pat in _CANARY_PATTERNS:
        text = pat.sub(_CANARY_REPLACEMENT, text)
    return " ".join(text.split()).strip()


def load_sequences(taxonomy_path: str | Path | None = None) -> list[dict]:
    """Flatten the PIAtlas taxonomy into an ordered list of sub-technique sequences.

    Order follows the JSON (F1.T1.S1 ... F6.T3.S2). Each entry carries its own definition
    plus the parent technique and class context needed to prompt the generator.
    """
    path = Path(taxonomy_path) if taxonomy_path else _BUNDLED_TAXONOMY
    try:
        data = json.loads(path.read_text(encoding="utf-8"))["taxonomy"]
        families = data["families"]
    except (KeyError, TypeError) as exc:
        raise ValueError("taxonomy JSON must contain taxonomy.families") from exc
    sequences: list[dict] = []
    for family in families:
        for technique in family["techniques"]:
            for sub in technique.get("subtechniques") or [technique]:
                examples = sub.get("examples") or []
                sequences.append(
                    {
                        "id": sub["id"],
                        "name": soften(sub["name"]),
                        "definition": soften(sub.get("definition", "")),
                        "family_id": family["id"],
                        "family_name": soften(family["name"]),
                        "technique_id": technique["id"],
                        "technique_name": soften(technique["name"]),
                        "phrasing_signature": soften(technique.get("phrasing_signature", "")),
                        "abstracted_example": soften(_abstract_example(examples[0])) if examples else "",
                        "raw_name": sub["name"],  # untouched canonical name for logging/stats
                        "allow_unicode": sub.get("allow_unicode", technique.get("allow_unicode", technique["id"] in _UNICODE_TECHNIQUES)),
                        "allow_markers": sub.get("allow_markers", technique.get("allow_markers", family["id"] in _STRUCTURAL_FAMILIES)),
                    }
                )
    if not sequences or len({sequence["id"] for sequence in sequences}) != len(sequences):
        raise ValueError("taxonomy must contain nonempty, uniquely identified sweep units")
    return sequences


def prioritize(sequences: list[dict]) -> list[dict]:
    """Reorder by technique priority (stable within a technique, so S1,S2,... order is
    preserved). Unlisted techniques keep canonical order and go after all listed ones.
    Never adds or drops sequences."""
    rank = {tech: i for i, tech in enumerate(_PRIORITY_TECHNIQUES)}
    default = len(_PRIORITY_TECHNIQUES)
    return [
        s
        for _, s in sorted(
            enumerate(sequences),
            key=lambda pair: (rank.get(pair[1]["technique_id"], default), pair[0]),
        )
    ]


SWEEP_ORDERS = ("priority", "canonical", "random")


def select_sequences(sequences: list[dict], selectors: Iterable[str] | None) -> list[dict]:
    """Keep only the sequences named by ``selectors``, in their current order.

    A selector is a class id (``F2``), a technique id (``F2.T5``), or a sub-technique id
    (``F2.T5.S2``). ``None`` or an empty selection keeps everything. A selector that matches
    nothing is an error, so a typo cannot silently shrink the sweep."""
    wanted = [s.strip().upper() for s in (selectors or []) if s and s.strip()]
    if not wanted:
        return list(sequences)
    known = {v for s in sequences for v in (s["family_id"], s["technique_id"], s["id"])}
    unknown = [w for w in wanted if w not in known]
    if unknown:
        raise ValueError(f"unknown taxonomy id(s) in the technique selection: {', '.join(unknown)}")
    keep = set(wanted)
    return [s for s in sequences if keep & {s["family_id"], s["technique_id"], s["id"]}]


def order_sequences(sequences: list[dict], order: str = "priority", seed: int = 0) -> list[dict]:
    """Put the sweep in ``order``: ``priority`` (the default, :func:`prioritize`),
    ``canonical`` (taxonomy-file order), or ``random`` (one fixed shuffle drawn from
    ``seed``, so every worker and every data point of a run sees the same order).
    Never adds or drops sequences."""
    if order == "priority":
        return prioritize(sequences)
    if order == "canonical":
        return list(sequences)
    if order == "random":
        shuffled = list(sequences)
        random.Random(seed).shuffle(shuffled)
        return shuffled
    raise ValueError(f"sweep order must be one of {', '.join(SWEEP_ORDERS)}; got {order!r}")


def seq_allows_unicode(seq: dict) -> bool:
    return seq.get("allow_unicode", seq["technique_id"] in _UNICODE_TECHNIQUES)


def seq_allows_markers(seq: dict) -> bool:
    return seq.get("allow_markers", seq["family_id"] in _STRUCTURAL_FAMILIES)


def as_seq_list(seqs: dict | list[dict]) -> list[dict]:
    """Normalise a single seq dict or a list of them (an Ln-sweep set) to a list."""
    return [seqs] if isinstance(seqs, dict) else list(seqs)


def combo_allows_unicode(seqs: dict | list[dict]) -> bool:
    """An Ln-sweep set may use Unicode if ANY of its sub-techniques legitimately needs it."""
    return any(seq_allows_unicode(s) for s in as_seq_list(seqs))


def combo_id(combo: list[dict]) -> str:
    """Canonical id for a combination, e.g. 'F1.T1.S1' or 'F1.T2.S1+F2.T2.S1'. itertools
    yields each unordered combination once in pool order, so this is unique."""
    return "+".join(s["id"] for s in combo)


def combo_name(combo: list[dict]) -> str:
    return " + ".join(s["raw_name"] for s in combo)


class ComboSchedule(Sequence):
    """Indexable combinations without allocating C(N, L) candidates in memory."""

    def __init__(self, singles, pool, levels):
        self.singles = singles
        self.pool = pool
        self.levels = list(levels)
        self.counts = [math.comb(len(pool), level) for level in self.levels]

    def __len__(self):
        return self.size

    @property
    def size(self):
        return len(self.singles) + sum(self.counts)

    def __iter__(self):
        yield from ([s] for s in self.singles)
        for level in self.levels:
            yield from (list(combo) for combo in itertools.combinations(self.pool, level))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.size))]
        if index < 0:
            index += self.size
        if index < 0 or index >= self.size:
            raise IndexError(index)
        if index < len(self.singles):
            return [self.singles[index]]
        index -= len(self.singles)
        for level, count in zip(self.levels, self.counts, strict=True):
            if index >= count:
                index -= count
                continue
            selected, start = [], 0
            for remaining in range(level, 0, -1):
                for position in range(start, len(self.pool)):
                    suffixes = math.comb(len(self.pool) - position - 1, remaining - 1)
                    if index < suffixes:
                        selected.append(self.pool[position])
                        start = position + 1
                        break
                    index -= suffixes
            return selected
        raise IndexError(index)


def build_combo_schedule(
    sequences: list[dict],
    *,
    stack_depth: int = 1,
    stack_pool: int | None = None,
    stack_random: bool = False,
    stack_seed: int = 0,
    stack_only: int = 0,
) -> tuple[ComboSchedule, list[str]]:
    """Ordered list of sub-technique COMBINATIONS to sweep, plus the Ln-sweep pool ids.

    Level 1 = every sequence (the single-technique sweep). Levels 2..stack_depth = every
    UNIQUE UNORDERED combination of that length drawn from the top ``stack_pool`` prioritized
    sequences (all if None). Order = all L=1, then all L=2, ..., then all L=stack_depth.
    ``stack_only=L`` sweeps ONLY the L-way combinations of the pool. ``stack_random`` draws
    the pool as a fixed seeded random sample instead of the top-P; the seed is
    constant so every worker samples the same pool.
    """
    if stack_depth < 1 or stack_only < 0 or (stack_pool is not None and stack_pool < 1):
        raise ValueError("stack depth/pool must be positive and stack_only nonnegative")
    depth = stack_depth
    pool: list[dict] = []
    pool_ids: list[str] = []
    if stack_only or depth >= 2:
        if stack_random:
            k = min(stack_pool or len(sequences), len(sequences))
            pool = random.Random(stack_seed).sample(sequences, k)
            order = {s["id"]: i for i, s in enumerate(sequences)}
            pool.sort(key=lambda s: order[s["id"]])  # canonical order -> stable combo ids
        else:
            pool = sequences[:stack_pool] if stack_pool else list(sequences)
        pool_ids = [s["id"] for s in pool]
    if stack_only:
        return ComboSchedule([], pool, [stack_only]), pool_ids
    return ComboSchedule(sequences, pool, range(2, min(depth, len(pool)) + 1)), pool_ids
