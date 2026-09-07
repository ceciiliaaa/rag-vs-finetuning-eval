"""Figures for an evaluation run (requires the optional ``plots`` extra)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .evaluate import EvaluationRun

PALETTE = ["#1f4e79", "#2e86c1", "#5dade2", "#a9cce3", "#d4e6f1", "#7f8c8d"]
SYSTEM_COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def _matplotlib():
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "figures need matplotlib; install with `uv sync --extra plots` or pass --no-plots"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _labels(run: EvaluationRun) -> tuple[list[str], list[str]]:
    metric_labels = [run.inputs.metric_labels[m] for m in run.inputs.metric_ids]
    system_labels = [run.inputs.system_labels[s] for s in run.inputs.system_ids]
    return metric_labels, system_labels


def plot_weights(run: EvaluationRun, path: Path) -> Path:
    plt = _matplotlib()
    metric_labels, _ = _labels(run)
    w = run.weights
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(metric_labels))
    ax.barh(y, w.mean, xerr=w.std, color=PALETTE[1], capsize=3)
    for i, (mean, weight) in enumerate(zip(w.mean, w.weight, strict=True)):
        ax.text(mean + w.std[i] + 0.08, i, f"weight {100 * weight:.1f} %", va="center", fontsize=9)
    ax.set_yticks(y, metric_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, w.kendall.n_items + 1.6)
    ax.set_xlabel("Mean importance rank across experts (error bar: SD)")
    ax.set_title(
        f"Criterion weights from {w.kendall.n_raters} expert rankings "
        f"(Kendall's W = {w.kendall.w:.3f})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_contributions(run: EvaluationRun, path: Path) -> Path:
    plt = _matplotlib()
    metric_labels, system_labels = _labels(run)
    contributions = np.asarray(run.utility.contributions)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    x = np.arange(len(system_labels))
    bottom = np.zeros(len(system_labels))
    for j, label in enumerate(metric_labels):
        ax.bar(
            x,
            contributions[:, j],
            bottom=bottom,
            color=PALETTE[j % len(PALETTE)],
            label=label,
            width=0.55,
        )
        bottom += contributions[:, j]
    for i, total in enumerate(run.utility.totals):
        ax.text(x[i], total + 0.015, f"{total:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x, system_labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Weighted utility")
    ax.set_title(f"Utility by criterion ({run.rounding.value} rounding)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_profile(run: EvaluationRun, path: Path) -> Path:
    plt = _matplotlib()
    metric_labels, system_labels = _labels(run)
    n = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.concatenate([angles, angles[:1]])
    target = np.asarray(run.weights.mean) / run.weights.kendall.n_items
    fig, ax = plt.subplots(figsize=(5.6, 5.2), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(
        closed,
        np.concatenate([target, target[:1]]),
        color="#555555",
        linestyle="--",
        label="Stakeholder target profile",
    )
    scores = np.asarray(run.utility.scores)
    for i, label in enumerate(system_labels):
        values = np.concatenate([scores[i], scores[i][:1]])
        colour = SYSTEM_COLOURS[i % len(SYSTEM_COLOURS)]
        ax.plot(closed, values, color=colour, label=label)
        ax.fill(closed, values, color=colour, alpha=0.15)
    ax.set_xticks(angles, metric_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(
        "Target profile (mean rank / max rank) versus measured scores", pad=18, fontsize=10
    )
    ax.legend(loc="upper left", bbox_to_anchor=(-0.22, 1.12), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sensitivity(run: EvaluationRun, path: Path) -> Path:
    plt = _matplotlib()
    metric_labels, system_labels = _labels(run)
    sens = run.sensitivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    x = np.arange(len(sens.leave_one_out))
    width = 0.8 / len(system_labels)
    for i, label in enumerate(system_labels):
        totals = [row.totals[i] for row in sens.leave_one_out]
        ax1.bar(
            x + (i - (len(system_labels) - 1) / 2) * width,
            totals,
            width,
            label=label,
            color=SYSTEM_COLOURS[i % len(SYSTEM_COLOURS)],
        )
    for i, total in enumerate(run.utility.totals):
        ax1.axhline(
            total, color=SYSTEM_COLOURS[i % len(SYSTEM_COLOURS)], linestyle=":", linewidth=1
        )
    ax1.set_xticks(
        x,
        [run.inputs.metric_labels[row.dropped_metric] for row in sens.leave_one_out],
        rotation=20,
        ha="right",
        fontsize=8,
    )
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Utility with criterion removed")
    ax1.set_title("Leave one criterion out (dotted: baseline)")
    ax1.legend(fontsize=8)

    current = [row.current_weight for row in sens.rank_reversal]
    thresholds = [
        row.threshold_weight if row.threshold_weight is not None else np.nan
        for row in sens.rank_reversal
    ]
    y = np.arange(len(current))
    ax2.scatter(current, y, color="#333333", label="current weight", zorder=3)
    ax2.scatter(thresholds, y, color="#d62728", marker="|", s=200, label="rank reversal", zorder=3)
    for i in range(len(current)):
        if not np.isnan(thresholds[i]):
            ax2.annotate(
                "",
                xy=(thresholds[i], i),
                xytext=(current[i], i),
                arrowprops={"arrowstyle": "->", "color": "#999999"},
            )
        else:
            ax2.text(
                current[i] + 0.03,
                i,
                "no reversal possible",
                va="center",
                fontsize=8,
                color="#666666",
            )
    ax2.set_yticks(y, metric_labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Weight of the criterion")
    ax2.set_title(
        f"Weight at which {run.inputs.system_labels[sens.baseline_runner_up]} would overtake"
    )
    ax2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
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
