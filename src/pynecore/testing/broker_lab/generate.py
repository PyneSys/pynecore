"""Deterministic scenario generation without third-party dependencies."""

import random
from collections.abc import Iterable, Mapping
from itertools import combinations, product
from typing import Any


def pairwise_cases(
    axes: Mapping[str, Iterable[Any]],
    *,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Return a deterministic greedy pairwise covering array.

    Every value pair from every pair of axes appears in at least one returned
    case. The seed only breaks equal-coverage ties, so the same seed is exactly
    reproducible across processes.
    """
    names = list(axes)
    values = {name: tuple(axes[name]) for name in names}
    if any(not value for value in values.values()):
        raise ValueError("pairwise axes must not be empty")
    if not names:
        return [{}]
    all_cases = [dict(zip(names, case)) for case in product(*(values[n] for n in names))]
    if len(names) == 1:
        return all_cases

    uncovered = {(a, repr(av), b, repr(bv)) for a, b in combinations(names, 2) for av in values[a] for bv in values[b]}
    rng = random.Random(seed)
    tie_order = list(range(len(all_cases)))
    rng.shuffle(tie_order)
    selected: list[dict[str, Any]] = []
    while uncovered:
        best_idx = max(
            tie_order,
            key=lambda idx: sum(
                (a, repr(all_cases[idx][a]), b, repr(all_cases[idx][b])) in uncovered for a, b in combinations(names, 2)
            ),
        )
        case = all_cases[best_idx]
        selected.append(case)
        for a, b in combinations(names, 2):
            uncovered.discard((a, repr(case[a]), b, repr(case[b])))
        tie_order.remove(best_idx)
    return selected
