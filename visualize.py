from __future__ import annotations

from typing import List, Dict, Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(states: Sequence[str], title: str = "Weather Trajectory") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2.2))
    unique = list(dict.fromkeys(states))
    mapping = {s: i for i, s in enumerate(unique)}
    y = [mapping[s] for s in states]
    ax.step(range(len(states)), y, where="post")
    ax.set_yticks(list(mapping.values()), list(mapping.keys()))
    ax.set_xlabel("Day")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_distribution_over_time(
    states: Sequence[str],
    state_space: Sequence[str],
    title: str = "State Counts Over Time",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    xs = list(range(len(states)))
    for s in state_space:
        ys = [states[:i + 1].count(s) / (i + 1) for i in range(len(states))]
        ax.plot(xs, ys, label=s)
    ax.set_xlabel("Day")
    ax.set_ylabel("Empirical Probability")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_transition_heatmap(
    transition_matrix: np.ndarray,
    state_space: Sequence[str],
    title: str = "Transition Probabilities",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.8, 4))
    im = ax.imshow(transition_matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(state_space)), state_space, rotation=45, ha="right")
    ax.set_yticks(range(len(state_space)), state_space)
    ax.set_xlabel("Next State")
    ax.set_ylabel("Current State")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="P(next|current)")
    fig.tight_layout()
    return fig


