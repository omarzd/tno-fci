"""A fit -> its prediction: the target for the lead's worth of months after the last input month.

The fit is a record that `fci fit` wrote (the newest one by default) or a run folder that the
old numbered scripts archived, such as output/archived_runs/202507261556, the July 2025 fit.
- From a record, TNO-FCI is rebuilt from the clean series with the record's normalization and
  weights, over input months up to its `inputs_through`, then mapped by the record's line.
- From an archived run, TNO-FCI comes from the run's results file, and the line is fitted on it
  against the target one lead later, as the page does.

Writes predictions/<fit name>.csv under the output folder: `month` and `ism_pmi`, one row per
predicted month. The same fit and the same clean series always write the same file; a record's
prediction changes if prep later revises a month the record used.
"""
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fci import StepError
from fci.fit import INPUTS, TARGET, load_clean


def newest_record(fits_dir):
    records = sorted(f for f in os.listdir(fits_dir) if f.endswith(".json")) if os.path.isdir(fits_dir) else []
    if not records:
        raise StepError(f"no fit record in {fits_dir}; run fci fit first")
    return os.path.join(fits_dir, records[-1])


@dataclass(frozen=True)
class LoadedFit:
    """What a fit says, whichever kind it is. `later` is the target one lead later, by input
    month, over the months the fit saw; `model` names the TNO-FCI's terms."""
    name: str
    tnofci: pd.Series
    lead: int
    slope: float
    intercept: float
    later: pd.Series
    r2: float
    generated: pd.Timestamp
    model: str  # "five weights" (fci fit), "92 features" or "five series" (archived runs)
    lag_r2: dict = None  # in-sample R² by lag in months (fci fit's lag test); None if the fit has none


def from_record(path, clean_dir):
    try:
        with open(path) as fh:
            record = json.load(fh)
        through = pd.Period(record["inputs_through"], "M")
        norm, weights, line, lead = record["normalization"], record["weights"], record["line"], record["lead"]
        table = load_clean(clean_dir)
        inputs = table[INPUTS].loc[:through]
        tnofci = sum(weights[s] * (inputs[s] - norm[s]["mean"]) / norm[s]["std"] for s in INPUTS).dropna()
        first, last = (pd.Period(record["training_months"][k], "M") for k in ("first", "last"))
        later = table[TARGET].shift(-lead).loc[first:last].reindex(tnofci.index).dropna()
        r2, generated = record["in_sample_r2"], pd.Timestamp(record["fitted_at"])
        lags = record.get("lag_in_sample_r2")  # records from before the lag test have none
        lag_r2 = {int(k): float(v) for k, v in lags.items()} if lags else None
    except (ValueError, KeyError, TypeError) as e:  # JSONDecodeError is a ValueError
        raise StepError(f"{path} is not a fit record that fci fit writes: {e!r}") from e
    if not len(tnofci) or tnofci.index[-1] != through:
        raise StepError(f"the clean series don't reach {through}, the record's inputs_through; run fci prep")
    return LoadedFit(fit_name(path), tnofci, lead, line["slope"], line["intercept"], later, r2, generated,
                     "five weights", lag_r2)


def from_archived_run(folder):
    names = [n for n in ("fci_results.csv", "tno_fci_results.csv") if os.path.exists(os.path.join(folder, n))]
    if not names:
        raise StepError(f"{folder} has no fci_results.csv or tno_fci_results.csv")
    df = pd.read_csv(os.path.join(folder, names[0]))
    forward = [c for c in df.columns if re.fullmatch(r"ISM_\d+m_Forward", c)]
    if "TNO_FCI" not in df.columns or len(forward) != 1:
        raise StepError(f"{folder}/{names[0]} needs a TNO_FCI column and one ISM_<lead>m_Forward column")
    months = pd.PeriodIndex(pd.to_datetime(df["Date" if "Date" in df.columns else df.columns[0]]), freq="M")
    tnofci = pd.Series(df["TNO_FCI"].to_numpy(), index=months)
    later = pd.Series(df[forward[0]].to_numpy(), index=months).dropna()
    slope, intercept = np.polyfit(tnofci[later.index].to_numpy(), later.to_numpy(), 1)
    reports = ""
    for n in sorted(os.listdir(folder)):
        if n.endswith(".txt"):
            with open(os.path.join(folder, n)) as fh:
                reports += fh.read() + "\n"
    r2 = re.search(r"R-squared: ([\d.]+)", reports)
    generated = re.search(r"Generated on: ([\d\- :]+)", reports)
    return LoadedFit(fit_name(folder), tnofci, int(re.search(r"\d+", forward[0]).group()), slope, intercept, later,
                     float(r2.group(1)) if r2 else None,
                     pd.Timestamp(generated.group(1).strip()) if generated else None,
                     archived_model(reports))


def archived_model(reports):
    """The calculator reports the method it chose: "enhanced" or "linear" weighs its 92 features,
    "baseline" the five series."""
    chosen = re.search(r"(?:Final Selected Method|Model method): (enhanced|linear)\b", reports)
    return "92 features" if chosen else "five series"


def fit_name(path):
    return os.path.splitext(os.path.basename(os.path.normpath(path)))[0]


def load_fit(path, clean_dir):
    if os.path.isdir(path):
        return from_archived_run(path)
    if os.path.isfile(path):
        return from_record(path, clean_dir)
    raise StepError(f"no fit at {path}")


def prediction(fit):
    """The target for the lead's worth of months after the fit's last input month."""
    tnofci = fit.tnofci.dropna()
    last = tnofci.index[-1]
    inputs = tnofci[tnofci.index > last - fit.lead]
    if len(inputs) != fit.lead:
        raise StepError(f"the fit needs TNO-FCI for each of the {fit.lead} input months up to {last}; it has {len(inputs)}")
    return pd.Series(fit.slope * inputs.to_numpy() + fit.intercept, index=inputs.index + fit.lead)


def prediction_path(output_dir, name):
    return os.path.join(output_dir, "predictions", f"{name}.csv")


def run(paths, fit_path=None):
    fit = load_fit(fit_path or newest_record(paths.fits), paths.clean)
    predicted = prediction(fit)
    out = prediction_path(paths.output, fit.name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame({"month": predicted.index.astype(str), TARGET: predicted.to_numpy()}).to_csv(out, index=False)
    return [f"{fit.lead} months, {predicted.index[0]} to {predicted.index[-1]}", out]
