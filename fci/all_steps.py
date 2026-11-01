"""`fci all`: prep, fit, predict and page, in order, each reading the previous step's files.

predict runs twice, for the new fit record and for the July 2025 run, since the page shows both.
With --demo it stops after the fit: no page is ever built from demo data (g04).
"""
import os

from fci import StepError, fit, page, predict, prep


def run(paths):
    if not paths.is_demo and not os.path.isdir(paths.july_2025):
        # checked first, so a missing run can't leave a new fit record behind
        raise StepError(f"no July 2025 run at {paths.july_2025}, which the page needs")
    lines = prep.run(paths) + fit.run(paths)
    if paths.is_demo:
        return lines + ["stopped after the fit: no page is built from demo data"]
    lines += predict.run(paths) + predict.run(paths, fit_path=paths.july_2025)
    return lines + page.run(paths)
