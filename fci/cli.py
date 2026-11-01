"""The fci command, one subcommand per step. The README's "Run it" says how the steps chain."""
import argparse
import sys

from fci import StepError, all_steps, fit, page, predict, prep
from fci.paths import Paths

STEPS = {
    "prep": (prep, "raw TradingView exports -> one monthly clean series each"),
    "fit": (fit, "clean series -> a fit record: the lead, the TNO-FCI weights and the line"),
    "predict": (predict, "a fit -> its prediction, one row per predicted month"),
    "page": (page, "the July 2025 fit and the latest fit, with their predictions -> site/index.html"),
    "all": (all_steps, "prep, fit, predict and page in order; with --demo, prep and fit only"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fci", description="Fit the TNO-FCI and predict the ISM PMI.")
    steps = parser.add_subparsers(dest="step", required=True)
    for name, (_, summary) in STEPS.items():
        step = steps.add_parser(name, help=summary)
        step.add_argument("--demo", action="store_true", help="read and write under demo/ (see scripts/make_demo_data.py)")
        if name == "predict":
            step.add_argument("--fit", dest="fit_path", help="a fit record or an archived run folder; default: the newest record")
        if name == "page":
            step.add_argument("--july", help=f"the July 2025 fit; default: {Paths.real().july_2025}")
            step.add_argument("--latest", help="the latest fit; default: the newest record")
            step.add_argument("--out", default="site", help="the folder index.html goes in")
    args = parser.parse_args(argv)

    paths = Paths.demo() if args.demo else Paths.real()
    try:
        options = {k: v for k, v in vars(args).items() if k not in ("step", "demo")}
        lines = STEPS[args.step][0].run(paths, **options)
    except StepError as e:
        print(f"fci {args.step}: {e}", file=sys.stderr)
        return 1
    for line in lines:
        print(line)
    return 0
