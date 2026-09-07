"""Figures for an evaluation run (requires the optional ``plots`` extra).

One muted palette is used everywhere: a deep-blue to pale-rose ramp for the criteria and two
colours for the systems drawn from the same family, so the figures read as one set.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .evaluate import EvaluationRun

# criteria ramp: deep blue -> periwinkle -> muted rose
PALETTE = ["#3f5e8c", "#6b8fb8", "#9a8fbf", "#c08bab", "#e2b6c4"]
# one colour per system, taken from the same family
SYSTEM_COLOURS = ["#5b83b0", "#b57fa8", "#7fa88f", "#a89a7f"]
TARGET_COLOUR = "#6c757d"
TEXT_COLOUR = "#343a40"
GRID_COLOUR = "#dee2e6"


def _matplotlib():
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "figures need matplotlib; install with `uv sync --extra plots` or pass --no-plots"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "text.color": TEXT_COLOUR,
            "axes.labelcolor": TEXT_COLOUR,
            "axes.edgecolor": "#adb5bd",
            "xtick.color": TEXT_COLOUR,
            "ytick.color": TEXT_COLOUR,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def _labels(run: EvaluationRun) -> tuple[list[str], list[str]]:
    metric_labels = [run.inputs.metric_labels[m] for m in run.inputs.metric_ids]
    system_labels = [run.inputs.system_labels[s] for s in run.inputs.system_ids]
    return metric_labels, system_labels


def plot_weights(run: EvaluationRun, path: Path) -> Path:
    """Mean importance rank per criterion, with the derived weight annotated."""
    plt = _matplotlib()
    metric_labels, _ = _labels(run)
    w = run.weights
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(metric_labels))
    ax.barh(y, w.mean, xerr=w.std, color=PALETTE[1], ecolor="#8d99ae", capsize=3, height=0.62)
    for i, (mean, weight) in enumerate(zip(w.mean, w.weight, strict=True)):
        ax.text(mean + w.std[i] + 0.1, i, f"{100 * weight:.1f} %", va="center", fontsize=9.5)
    ax.set_yticks(y, metric_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, w.kendall.n_items + 1.5)
    ax.set_xlabel("Mean importance rank across experts (error bar: standard deviation)")
    ax.set_title(
        f"Criterion weights from {w.kendall.n_raters} expert rankings "
        f"(Kendall's W = {w.kendall.w:.3f})",
        fontsize=11,
    )
    ax.xaxis.grid(True, color=GRID_COLOUR, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_contributions(run: EvaluationRun, path: Path) -> Path:
    """Weighted utility per system, stacked by criterion."""
    plt = _matplotlib()
    metric_labels, system_labels = _labels(run)
    contributions = np.asarray(run.utility.contributions)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    x = np.arange(len(system_labels))
    bottom = np.zeros(len(system_labels))
    for j, label in enumerate(metric_labels):
        ax.bar(
            x,
            contributions[:, j],
            bottom=bottom,
            color=PALETTE[j % len(PALETTE)],
            label=label,
            width=0.5,
            edgecolor="white",
            linewidth=0.8,
        )
        bottom += contributions[:, j]
    for i, total in enumerate(run.utility.totals):
        ax.text(x[i], total + 0.018, f"{total:.3f}", ha="center", fontsize=11, fontweight="600")
    ax.set_xticks(x, system_labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Weighted utility")
    ax.set_title(f"Utility by criterion ({run.rounding.value} arithmetic)", fontsize=11)
    ax.yaxis.grid(True, color=GRID_COLOUR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.95, edgecolor=GRID_COLOUR)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_profile(run: EvaluationRun, path: Path) -> Path:
    """Stakeholder target profile against the measured score profile of each system."""
    plt = _matplotlib()
    metric_labels, system_labels = _labels(run)
    n = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.concatenate([angles, angles[:1]])
    target = np.asarray(run.weights.mean) / run.weights.kendall.n_items
    scores = np.asarray(run.utility.scores)

    fig = plt.figure(figsize=(8.6, 5.4))
    ax = fig.add_subplot(111, polar=True)
    fig.subplots_adjust(left=0.10, right=0.68, top=0.86, bottom=0.10)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(
        closed,
        np.concatenate([target, target[:1]]),
        color=TARGET_COLOUR,
        linestyle="--",
        linewidth=1.6,
        marker="o",
        markersize=3.4,
        label="Stakeholder target profile",
    )
    for i, label in enumerate(system_labels):
        values = np.concatenate([scores[i], scores[i][:1]])
        colour = SYSTEM_COLOURS[i % len(SYSTEM_COLOURS)]
        ax.plot(
            closed,
            values,
            color=colour,
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=3 + i,
        )
        ax.fill(closed, values, color=colour, alpha=0.18, label=label, zorder=2 + i)

    ax.set_xticks(angles, metric_labels, fontsize=10)
    ax.tick_params(axis="x", pad=10)
    ax.set_ylim(0, 1)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8.5, color="#868e96")
    ax.grid(color=GRID_COLOUR, linewidth=0.8)
    ax.spines["polar"].set_edgecolor(GRID_COLOUR)
    fig.suptitle(
        "Stakeholder target profile versus measured scores", fontsize=11, color=TEXT_COLOUR, y=0.97
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.10, 1.02),
        fontsize=9,
        frameon=True,
        framealpha=0.95,
        edgecolor=GRID_COLOUR,
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_sensitivity(run: EvaluationRun, path: Path) -> Path:
    """Utility of each system when one criterion is removed and the weights are renormalised."""
    plt = _matplotlib()
    _, system_labels = _labels(run)
    rows = run.sensitivity.leave_one_out
    fig, ax = plt.subplots(figsize=(7.6, 4.0))

    x = np.arange(len(rows))
    width = 0.74 / len(system_labels)
    for i, label in enumerate(system_labels):
        totals = [row.totals[i] for row in rows]
        offset = (i - (len(system_labels) - 1) / 2) * width
        ax.bar(
            x + offset,
            totals,
            width,
            label=label,
            color=SYSTEM_COLOURS[i % len(SYSTEM_COLOURS)],
            edgecolor="white",
            linewidth=0.8,
        )
    for i, total in enumerate(run.utility.totals):
        ax.axhline(
            total,
            color=SYSTEM_COLOURS[i % len(SYSTEM_COLOURS)],
            linestyle=":",
            linewidth=1.4,
            alpha=0.9,
        )

    ax.set_xticks(x, [run.inputs.metric_labels[row.dropped_metric] for row in rows], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Utility with the criterion removed")
    ax.set_xlabel("Criterion left out, remaining weights renormalised")
    ax.set_title(
        "How the ranking holds up when one criterion is dropped (dotted: all criteria included)",
        fontsize=11,
    )
    ax.yaxis.grid(True, color=GRID_COLOUR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=True, framealpha=0.95, edgecolor=GRID_COLOUR)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_figures(run: EvaluationRun, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    return [
        plot_weights(run, directory / "weights.png"),
        plot_contributions(run, directory / "utility_contributions.png"),
        plot_profile(run, directory / "profile_radar.png"),
        plot_sensitivity(run, directory / "sensitivity.png"),
    ]
