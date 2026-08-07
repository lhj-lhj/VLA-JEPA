"""Posterior camera-view sampling helpers."""

from __future__ import annotations

import random
from collections.abc import Sequence


def balanced_random_view_indices(
    *,
    batch_size: int,
    view_indices: Sequence[int],
    rng: random.Random | None = None,
) -> list[int]:
    """Return a shuffled batch containing exactly half of each of two views."""

    if batch_size <= 0 or batch_size % 2:
        raise ValueError("Balanced posterior-view sampling requires an even batch size.")
    if len(view_indices) != 2 or view_indices[0] == view_indices[1]:
        raise ValueError("Balanced posterior-view sampling requires two distinct views.")

    assignments = [
        int(view_indices[0])
    ] * (batch_size // 2) + [
        int(view_indices[1])
    ] * (batch_size // 2)
    (rng or random).shuffle(assignments)
    return assignments
