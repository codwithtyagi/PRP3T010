from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np

from probabilities_in_the_sky.models.markov_chain import MarkovChain


def simulate_markov_chain(
    mc: MarkovChain,
    initial_state: str | int,
    days: int,
    seed: int | None = None,
) -> Tuple[List[str], Dict[str, int]]:
    """Simulate a Markov Chain trajectory for the given number of days.

    Returns the visited state names list (length = days + 1 including day 0) and counts.
    """
    if days < 0:
        raise ValueError("days must be >= 0")

    rng = np.random.default_rng(seed)

    if isinstance(initial_state, int):
        current_idx = int(initial_state)
    else:
        current_idx = mc.state_index(initial_state)

    trajectory_indices: List[int] = [current_idx]
    for _ in range(days):
        current_idx = mc.step(current_idx, rng)
        trajectory_indices.append(current_idx)

    trajectory_states = [mc.states[i] for i in trajectory_indices]
    counts: Dict[str, int] = {s: trajectory_states.count(s) for s in mc.states}
    return trajectory_states, counts


