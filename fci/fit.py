"""Clean series -> a fit record: the lead, the TNO-FCI weights and the line to target points.

TNO-FCI is a weighted sum of the five input series, each z-scored over the fit's training
months, and scaled to mean 0 and standard deviation 1 over those months. The weights come from
least squares of the target one lead later on those inputs, which gives the weighted sum with
the highest correlation to it. A straight line maps TNO-FCI to target points.

The lead comes from a walk-forward search. For each lead in LEADS it steps through the graded
origins, and at each one refits on the pairs whose target was already known, then predicts the
lead's path of target months after it. A lead's score is the out-of-sample score of those paths
(fci.score), over the same origins for every lead. The fit then refits at the best lead on every
pair. It also records the in-sample R² of the same fit at each lag in LAGS, the target that many
months before the input month, so the page can show that the TNO-FCI leads the target rather
than lags it. Each run writes a new record under the fits folder and leaves the earlier ones as
they are.
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from fci import StepError
from fci.prep import EXPORTS
from fci.score import score

TARGET = "ism_pmi"
INPUTS = [name for name in EXPORTS if name != TARGET]
LEADS = range(9, 19)  # at least GMI's 9 months ahead
LAGS = range(1, 13)  # the lag test: a lag of k months is the fit at lead -k
WARM_UP = 120  # months of pairs a refit needs before its prediction is graded
MIN_ORIGINS = 24  # graded origins the search needs, or it falls back
FALLBACK_LEAD = 9  # GMI's lead, kept as the fallback at the manager's word (g04)


def load_clean(clean_dir):
    """One row per month, from the earliest month of any clean series to the latest."""
    series = {}
    for name in [*INPUTS, TARGET]:
        path = os.path.join(clean_dir, f"{name}.csv")
        if not os.path.exists(path):
            raise StepError(f"no {path}; run fci prep first")
        s = pd.read_csv(path, index_col="month")[name]
        s.index = pd.PeriodIndex(s.index, freq="M")
        series[name] = s
    months = pd.period_range(min(s.index.min() for s in series.values()),
                             max(s.index.max() for s in series.values()), freq="M")
    return pd.DataFrame({name: s.reindex(months) for name, s in series.items()})


def solve(x, y):
    """Least squares of y on the z-scored columns of x: their mean, std, weights and intercept."""
    mean, std = x.mean(axis=0), x.std(axis=0)
    if (std == 0).any():
        raise StepError("an input series doesn't change over the training months")
    design = np.column_stack([np.ones(len(x)), (x - mean) / std])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    return mean, std, coef[1:], coef[0]


def r2(actual, predicted):
    return 1 - ((actual - predicted) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()


def search(table):
    """Out-of-sample score per lead, or None for every lead when too few origins can be graded.
    Also returns the graded origins. At origin o the refit knows the pairs whose target month is
    at or before o, and predicts target months o+1 to o+lead. An origin is graded when the whole
    path is known for the longest lead, so every lead is graded on the same origins."""
    x = table[INPUTS].to_numpy()
    now = table[TARGET].to_numpy()
    complete = ~np.isnan(x).any(axis=1)
    later = {lead: table[TARGET].shift(-lead).to_numpy() for lead in LEADS}
    usable = {lead: complete & ~np.isnan(later[lead]) for lead in LEADS}
    # usable pairs among the input months before row k, for k = 0..len
    counts = {lead: np.concatenate([[0], np.cumsum(usable[lead])]) for lead in LEADS}
    longest = max(LEADS)

    def gradable(o):
        return (o - longest + 1 >= 0 and o + longest < len(table)
                and complete[o - longest + 1:o + 1].all() and not np.isnan(now[o + 1:o + longest + 1]).any()
                and all(counts[lead][o - lead + 1] >= WARM_UP for lead in LEADS))

    origins = [o for o in range(len(table)) if gradable(o)]
    if len(origins) < MIN_ORIGINS:
        return {lead: None for lead in LEADS}, origins
    guess = np.array([np.nanmean(now[:o + 1]) for o in origins])  # the target's average known at o
    scores = {}
    for lead in LEADS:
        predicted = []
        for o in origins:
            rows = np.flatnonzero(usable[lead][:o - lead + 1])  # pairs whose target month is at or before o
            mean, std, weights, intercept = solve(x[rows], later[lead][rows])
            predicted.append(intercept + ((x[o - lead + 1:o + 1] - mean) / std) @ weights)
        actual = np.array([now[o + 1:o + lead + 1] for o in origins])
        scores[lead] = score(np.array(predicted), actual, guess)
    return scores, origins


def fit_at(table, lead):
    """The fit on every usable pair at this lead, as the record's fields."""
    x = table[INPUTS].to_numpy()
    later = table[TARGET].shift(-lead).to_numpy()
    rows = np.flatnonzero(~np.isnan(x).any(axis=1) & ~np.isnan(later))
    if len(rows) <= len(INPUTS) + 1:
        raise StepError(f"{len(rows)} months have the inputs and the target {lead} months later; too few to fit")
    mean, std, weights, intercept = solve(x[rows], later[rows])
    raw = ((x[rows] - mean) / std) @ weights
    scale = raw.std()
    if scale == 0:
        raise StepError("the inputs explain none of the target's movement")
    tnofci = raw / scale  # mean 0, std 1 over the training months
    return {
        "training_months": {"first": str(table.index[rows[0]]), "last": str(table.index[rows[-1]]),
                            "count": int(len(rows))},
        "normalization": {name: {"mean": float(m), "std": float(s)} for name, m, s in zip(INPUTS, mean, std)},
        "weights": {name: float(w / scale) for name, w in zip(INPUTS, weights)},
        "line": {"slope": float(scale), "intercept": float(intercept)},
        "in_sample_r2": float(r2(later[rows], intercept + scale * tnofci)),
    }


def run(paths):
    table = load_clean(paths.clean)
    scores, origins = search(table)
    scored = {lead: s for lead, s in scores.items() if s is not None}
    if scored:
        lead = max(scored, key=lambda k: (scored[k], -k))  # ties go to the shorter lead
        chosen_by, said = "walk-forward", f"out-of-sample score {scored[lead]:.3f}"
    else:
        lead = FALLBACK_LEAD
        chosen_by = "fallback"
        said = (f"the fallback: {len(origins)} months could be graded, the search needs "
                f"{MIN_ORIGINS} after a warm-up of {WARM_UP}")
    fitted = fit_at(table, lead)  # stops on too few months before any record field is read
    complete = table[INPUTS].notna().all(axis=1)
    now = datetime.now()
    record = {
        "fitted_at": now.isoformat(timespec="seconds"),
        "inputs_through": str(table.index[complete][-1]),
        "target_through": str(table[TARGET].dropna().index[-1]),
        "lead": lead,
        "lead_chosen_by": chosen_by,
        "out_of_sample_score": scored.get(lead),
        "lead_scores": {str(k): v for k, v in scores.items()},
        "lag_in_sample_r2": {str(k): fit_at(table, -k)["in_sample_r2"] for k in LAGS},
        "graded_origins": {"first": str(table.index[origins[0]]) if origins else None,
                           "last": str(table.index[origins[-1]]) if origins else None,
                           "count": len(origins)},
        **fitted,
    }
    os.makedirs(paths.fits, exist_ok=True)
    out = os.path.join(paths.fits, now.strftime("%Y%m%d-%H%M%S-%f") + ".json")
    with open(out, "x") as fh:  # a new record every run; never overwrite one
        json.dump(record, fh, indent=2)
        fh.write("\n")
    return [f"lead {lead} months, {said}", out]
