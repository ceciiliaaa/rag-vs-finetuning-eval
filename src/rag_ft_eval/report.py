"""Deterministic Markdown report of one evaluation run.

The report contains no timestamps or machine-specific values so that regenerating it from the
same inputs yields a byte-identical file; the golden test and the CI workflow rely on that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import __version__
from .schema import MetricKind, Rounding

if TYPE_CHECKING:
    from .evaluate import EvaluationRun


def _f(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f} %"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _score_basis(run: EvaluationRun, metric_id: str, system: str) -> str:
    detail = run.scores.details.get(metric_id, {})
    kind = detail.get("kind")
    if kind == MetricKind.MEASURED_BINARY.value:
        return f"{detail['positive'][system]}/{detail['n'][system]}"
    if kind == MetricKind.MEASURED_LATENCY.value:
        stats = detail["stats"][system]
        return f"mean {stats['mean']:.2f} s, sd {stats['std']:.2f} s, n = {stats['n']}"
    if kind == MetricKind.LITERATURE.value:
        hits = sum(item["scores"][system] for item in detail["criteria"])
        return f"{hits}/{len(detail['criteria'])} sub-criteria"
    return ""


def render_report(run: EvaluationRun) -> str:
    inputs = run.inputs
    labels = inputs.metric_labels
    system_labels = inputs.system_labels
    systems = inputs.system_ids
    weights = run.weights
    util = run.utility
    sens = run.sensitivity
    case = inputs.case_study

    out: list[str] = []
    out.append(f"# Evaluation report: {case.get('title', case.get('name', 'case study'))}")
    out.append("")
    if case.get("organisation"):
        out.append(f"Context: {case['organisation']}, {case.get('year', '')}".rstrip(", "))
    mode_note = (
        "exact arithmetic"
        if run.rounding is Rounding.EXACT
        else "rounding convention of the original write-up (intermediate values rounded)"
    )
    out.append(f"Rounding mode: `{run.rounding.value}` ({mode_note}).")
    out.append("")

    # 1. Weights
    out.append("## 1. Criterion weights from expert rankings")
    out.append("")
    out.append(
        f"{weights.kendall.n_raters} experts ranked {weights.kendall.n_items} criteria "
        "(forced ranking, higher rank = more important). Weights are mean ranks normalised to "
        "sum to one; the standard deviation is the sample standard deviation across experts."
    )
    out.append("")
    rows = [
        [
            labels[m],
            _f(weights.mean[i], 2),
            _f(weights.std[i], 2),
            _f(weights.weight[i]),
            _pct(weights.weight[i], 2),
        ]
        for i, m in enumerate(weights.metrics)
    ]
    out.append(_table(["Criterion", "Mean rank", "SD", "Weight", "Weight (%)"], rows))
    out.append("")
    k = weights.kendall
    perm = (
        f", permutation test p = {_f(k.p_perm, 4)} ({k.n_perm} samples)"
        if k.p_perm is not None
        else ""
    )
    out.append(
        f"Agreement: Kendall's W = {_f(k.w, 3)}; chi-square = {_f(k.chi2, 2)} with "
        f"{k.df} degrees of freedom, p = {_f(k.p_chi2, 4)}{perm}."
    )
    out.append("")

    # 2. Scores
    out.append("## 2. Normalised criterion scores")
    out.append("")
    header = ["Criterion", "Kind"]
    for s in systems:
        header += [f"{system_labels[s]} score", f"{system_labels[s]} basis"]
    rows = []
    for j, m in enumerate(inputs.metrics):
        row = [labels[m.id], m.kind.value]
        for i, s in enumerate(systems):
            row += [_f(util.scores[i][j], 3), _score_basis(run, m.id, s)]
        rows.append(row)
    out.append(_table(header, rows))
    out.append("")
    out.append(
        "Binary criteria are the share of positive judgements. Response time uses ratio-to-best "
        "normalisation (fastest mean divided by the system mean). Literature-based criteria are "
        "the share of sub-criteria in favour of the system (section 5)."
    )
    out.append("")

    # 3. Utility
    out.append("## 3. Weighted utility")
    out.append("")
    header = ["Criterion", "Weight"] + [f"{system_labels[s]}" for s in systems]
    rows = []
    for j, m in enumerate(util.metrics):
        rows.append(
            [labels[m], _f(util.weights[j])]
            + [_f(util.contributions[i][j]) for i in range(len(systems))]
        )
    rows.append(
        ["**Total utility**", _f(sum(util.weights))] + [f"**{_f(t)}**" for t in util.totals]
    )
    out.append(_table(header, rows))
    out.append("")
    out.append(
        f"{system_labels[util.winner]} reaches the higher utility "
        f"({_f(util.total_of(util.winner))} versus {_f(util.total_of(util.runner_up))} for "
        f"{system_labels[util.runner_up]}); the gap is {_f(util.gap)}."
    )
    out.append("")
    out.append("### Decomposition of the gap")
    out.append("")
    rows = [
        [labels[m], _f(util.gap_by_metric[j]), _pct(util.gap_share[j])]
        for j, m in enumerate(util.metrics)
    ]
    out.append(_table(["Criterion", "Contribution to gap", "Share of gap"], rows))
    out.append("")
    out.append("Negative shares mark criteria on which the runner-up is ahead.")
    out.append("")

    # 4. Sensitivity
    out.append("## 4. Sensitivity to the weights")
    out.append("")
    out.append("### Leave one criterion out (remaining weights renormalised)")
    out.append("")
    header = ["Dropped criterion"] + [system_labels[s] for s in systems] + ["Delta", "Winner"]
    rows = [
        [labels[row.dropped_metric]]
        + [_f(t) for t in row.totals]
        + [_f(row.delta), system_labels[row.winner]]
        for row in sens.leave_one_out
    ]
    out.append(_table(header, rows))
    out.append("")
    p = sens.perturbation
    out.append("### Random weight perturbation")
    out.append("")
    out.append(
        f"{p.n_samples} weight vectors drawn from Dirichlet(concentration {_f(p.concentration, 1)} "
        f"x expert weights), seed {p.seed}, exact arithmetic. Win share: "
        + ", ".join(f"{system_labels[s]} {_pct(p.win_fraction[s])}" for s in systems)
        + f". Utility advantage of {system_labels[sens.baseline_winner]}: mean {_f(p.delta_mean)}, "
        f"5th to 95th percentile {_f(p.delta_p05)} to {_f(p.delta_p95)}, minimum {_f(p.delta_min)}."
    )
    out.append("")
    out.append("### Rank-reversal thresholds")
    out.append("")
    out.append(
        "Weight a single criterion would need (others rescaled proportionally) for "
        f"{system_labels[sens.baseline_runner_up]} to overtake "
        f"{system_labels[sens.baseline_winner]}."
    )
    out.append("")
    rows = []
    for row in sens.rank_reversal:
        if row.threshold_weight is None:
            rows.append(
                [
                    labels[row.metric],
                    _f(row.current_weight),
                    "none in [0, 1]",
                    "no reversal possible",
                ]
            )
        else:
            rows.append(
                [
                    labels[row.metric],
                    _f(row.current_weight),
                    _f(row.threshold_weight),
                    row.direction or "",
                ]
            )
    out.append(_table(["Criterion", "Current weight", "Reversal at weight", "Direction"], rows))
    out.append("")

    # 5. Literature criteria
    literature = [m for m in inputs.metrics if m.kind is MetricKind.LITERATURE]
    if literature:
        out.append("## 5. Literature-based criteria")
        out.append("")
        cited: list[str] = []
        for m in literature:
            out.append(f"### {labels[m.id]}")
            out.append("")
            header = ["Id", "Sub-criterion"] + [system_labels[s] for s in systems] + ["Sources"]
            rows = []
            for item in run.scores.details[m.id]["criteria"]:
                rows.append(
                    [item["id"], item["statement"]]
                    + [str(item["scores"][s]) for s in systems]
                    + [", ".join(item["sources"])]
                )
                cited.extend(src for src in item["sources"] if src not in cited)
            out.append(_table(header, rows))
            out.append("")
        out.append("### References")
        out.append("")
        for key in cited:
            ref = inputs.references.get(key, {})
            locator = ref.get("doi")
            locator = f"doi:{locator}" if locator else ref.get("url", "")
            out.append(
                f"- `{key}`: {ref.get('authors', '')} ({ref.get('year', '')}). "
                f"{ref.get('title', '')}. {ref.get('venue', '')}. {locator}".rstrip(". ")
            )
        out.append("")

    out.append("---")
    out.append(
        f"Generated by rag-ft-eval {__version__} from `{inputs.config_path.name}` "
        f"({len(inputs.questions)} test questions, {len(inputs.expert_ids)} experts)."
    )
    out.append("")
    return "\n".join(out)
