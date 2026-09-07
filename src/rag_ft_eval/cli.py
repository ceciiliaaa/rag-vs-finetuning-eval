"""Command-line interface: ``rag-ft-eval run | weights | sensitivity | validate``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ._version import __version__
from .evaluate import run_evaluation, write_outputs
from .io import ConfigError, load_config
from .schema import Rounding


def _add_common(parser: argparse.ArgumentParser, with_rounding: bool = True) -> None:
    parser.add_argument("config", type=Path, help="path to the evaluation config (YAML)")
    if with_rounding:
        parser.add_argument(
            "--rounding",
            choices=[mode.value for mode in Rounding],
            default=None,
            help="override the rounding mode declared in the config",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-ft-eval",
        description="Stakeholder-weighted utility analysis for comparing LLM systems.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="compute everything and write results")
    _add_common(run)
    run.add_argument(
        "--output", type=Path, default=None, help="output directory (default: from config)"
    )
    run.add_argument("--no-plots", action="store_true", help="skip figure generation")

    weights = sub.add_parser("weights", help="print criterion weights and Kendall's W")
    _add_common(weights, with_rounding=False)

    sens = sub.add_parser("sensitivity", help="print the sensitivity tables")
    _add_common(sens)

    validate = sub.add_parser("validate", help="check the config and input files only")
    _add_common(validate, with_rounding=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        inputs = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "validate":
        print(
            f"ok: {len(inputs.systems)} systems, {len(inputs.metrics)} criteria, "
            f"{len(inputs.expert_ids)} experts, {len(inputs.questions)} questions, "
            f"{len(inputs.measurements)} measurement rows"
        )
        return 0

    rounding = Rounding(args.rounding) if getattr(args, "rounding", None) else None

    if args.command == "weights":
        run = run_evaluation(inputs)
        frame = run.weights.as_frame()
        frame.index = [inputs.metric_labels[m] for m in frame.index]
        print(frame.to_string(float_format=lambda x: f"{x:.4f}"))
        k = run.weights.kendall
        perm = f", permutation p = {k.p_perm:.4f}" if k.p_perm is not None else ""
        print(
            f"\nKendall's W = {k.w:.3f} "
            f"(chi2 = {k.chi2:.2f}, df = {k.df}, p = {k.p_chi2:.4f}{perm})"
        )
        return 0

    run = run_evaluation(inputs, rounding)

    if args.command == "sensitivity":
        print("Leave one criterion out:")
        print(
            run.sensitivity.leave_one_out_frame(inputs.system_ids).to_string(
                float_format=lambda x: f"{x:.4f}"
            )
        )
        print("\nRank-reversal thresholds:")
        print(run.sensitivity.rank_reversal_frame().to_string(float_format=lambda x: f"{x:.4f}"))
        p = run.sensitivity.perturbation
        print(f"\nWeight perturbation ({p.n_samples} samples): win fraction {p.win_fraction}")
        return 0

    output_dir = args.output if args.output is not None else inputs.output_dir
    try:
        written = write_outputs(run, output_dir, plots=not args.no_plots)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    util = run.utility
    labels = inputs.system_labels
    print(
        f"{labels[util.winner]} {util.total_of(util.winner):.4f} versus "
        f"{labels[util.runner_up]} {util.total_of(util.runner_up):.4f} "
        f"({run.rounding.value} rounding); Kendall's W = {run.weights.kendall.w:.3f}"
    )
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
