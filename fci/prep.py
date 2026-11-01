"""Raw TradingView exports -> one monthly clean series per input series and one for the target.

Each clean file has two columns, `month` (YYYY-MM) and the series name, and one row for every
month from the series' first to its last. A month with several rows keeps its last close. A
missing month takes the last known value, at most FILL_LIMIT months on; the rest of a longer gap
stays empty.
"""
import os

import pandas as pd

from fci import StepError

# Clean series name -> the start of its export's filename in the raw folder.
EXPORTS = {
    "ism_pmi": "ECONOMICS_USBCOI",
    "fed_rate": "ECONOMICS_USINTR",
    "dxy": "TVC_DXY",
    "gold": "TVC_GOLD",
    "oil": "TVC_USOIL",
    "silver": "TVC_SILVER",
}
FILL_LIMIT = 3  # the old numbered scripts' monthly step filled the same way


def find_export(raw_dir, prefix):
    found = sorted(f for f in os.listdir(raw_dir) if f.startswith(prefix) and f.endswith(".csv"))
    if not found:
        raise StepError(f"no {prefix}*.csv in {raw_dir}")
    if len(found) > 1:
        # TradingView filenames hold ", " themselves, so the list uses "; "
        raise StepError(f"{len(found)} {prefix}*.csv files in {raw_dir}, keep one: {'; '.join(found)}")
    return os.path.join(raw_dir, found[0])


def clean_series(path, name):
    try:
        raw = pd.read_csv(path, usecols=["time", "close"]).dropna()
        raw["time"] = pd.to_datetime(raw["time"])
    except ValueError as e:
        raise StepError(f"{path} needs a time and a close column with dates in time: {e}") from e
    if raw.empty:
        raise StepError(f"{path} has no rows with both a time and a close")
    raw["month"] = raw["time"].dt.to_period("M")
    series = raw.sort_values("time", kind="stable").groupby("month")["close"].last()
    every_month = pd.period_range(series.index.min(), series.index.max(), freq="M")
    series = series.reindex(every_month).ffill(limit=FILL_LIMIT)
    return pd.DataFrame({"month": series.index.astype(str), name: series.to_numpy()})


def run(paths):
    """Write every clean series, or none: each export is read before the first file is written."""
    if not os.path.isdir(paths.raw):
        raise StepError(f"no raw folder at {paths.raw}")
    frames = {name: clean_series(find_export(paths.raw, prefix), name) for name, prefix in EXPORTS.items()}
    os.makedirs(paths.clean, exist_ok=True)
    written = []
    for name, frame in frames.items():
        out = os.path.join(paths.clean, f"{name}.csv")
        frame.to_csv(out, index=False)
        written.append(out)
    return written
