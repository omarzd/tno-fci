#!/usr/bin/env python3
"""Write demo data: six generated series with a planted lead, in the sample files' export format.

The target follows a weighted sum of the input series, `--lead` months later, plus noise. Run the
prep and fit steps with --demo afterwards, and the fit should report the planted lead.
Writes to demo/data/raw_all/, relative to where you run it (the repo root).
"""
import argparse
import os

import numpy as np
import pandas as pd

MONTHS = 360  # 30 years: enough months for every lead from 3 to 18
MONTHS_HELD = np.arange(MONTHS) // 3 * 3  # each month reads the first month of its quarter
DEMO_RAW = os.path.join("demo", "data", "raw_all")

# One file per series, named like a TradingView export so the prep step finds it.
FILES = {
    "target": "ECONOMICS_USBCOI, 1M_demo.csv",
    "fed": "ECONOMICS_USINTR, 1M_demo.csv",
    "dxy": "TVC_DXY, 1M_demo.csv",
    "gold": "TVC_GOLD, 1M_demo.csv",
    "oil": "TVC_USOIL, 1M_demo.csv",
    "silver": "TVC_SILVER, 1M_demo.csv",
}
# How much each input series drives the target.
WEIGHTS = {"fed": -0.5, "dxy": -0.3, "gold": 0.2, "oil": 0.3, "silver": 0.25}


def drift(rng, n, persistence=0.85):
    """A series that wanders but keeps returning to 0, so months far apart are unlike each other."""
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = persistence * x[t - 1] + rng.normal()
    return x


def generate(lead, seed):
    rng = np.random.default_rng(seed)
    wander = {name: drift(rng, MONTHS) for name in WEIGHTS}
    inputs = {
        # set once a quarter, in quarter points, held in between
        "fed": np.clip(np.round((2.5 + 0.5 * wander["fed"][MONTHS_HELD]) * 4) / 4, 0, None),
        "dxy": 95 + 5 * wander["dxy"],
        "gold": 1200 * np.exp(0.15 * wander["gold"]),
        "oil": 60 * np.exp(0.2 * wander["oil"]),
        "silver": 18 * np.exp(0.15 * wander["silver"]),
    }
    signal = sum(w * (inputs[k] - inputs[k].mean()) / inputs[k].std() for k, w in WEIGHTS.items())
    signal /= signal.std()
    target = 50 + rng.normal(0, 0.8, MONTHS)
    target[lead:] += 3 * signal[:-lead]  # the target follows the inputs `lead` months later
    return inputs, np.round(target, 1)


def main():
    parser = argparse.ArgumentParser(description="Write demo data with a planted lead")
    parser.add_argument("--lead", type=int, default=7, help="months the target follows the inputs by (3 to 18)")
    parser.add_argument("--seed", type=int, default=42, help="same seed, same demo data")
    args = parser.parse_args()
    if not 3 <= args.lead <= 18:
        parser.error("--lead must be from 3 to 18, the leads the fit searches")

    inputs, target = generate(args.lead, args.seed)
    months = pd.date_range("1995-01-01", periods=MONTHS, freq="MS").strftime("%Y-%m-%d")
    os.makedirs(DEMO_RAW, exist_ok=True)
    pd.DataFrame({"time": months, "close": target}).to_csv(os.path.join(DEMO_RAW, FILES["target"]), index=False)
    pd.DataFrame({"time": months, "close": inputs["fed"]}).to_csv(os.path.join(DEMO_RAW, FILES["fed"]), index=False)
    for name in ("dxy", "gold", "oil", "silver"):
        close = np.round(inputs[name], 2)
        pd.DataFrame({"time": months, "open": close, "high": close, "low": close, "close": close}).to_csv(
            os.path.join(DEMO_RAW, FILES[name]), index=False)
    print(f"Demo data with a planted lead of {args.lead} months written to {DEMO_RAW}/")
    print("Next: 2_prep_raw_data.py --demo, then 3_tno_fci_calculator.py --demo")


if __name__ == "__main__":
    main()
