# TNO-FCI

A financial conditions index that predicts the ISM Manufacturing PMI about 10 months ahead.
The public results page, https://omarzd.github.io/fci/, is a snapshot from October 2026;
its source is `docs/index.html`.

Inspired by the financial conditions index of Raoul Pal's Global Macro Investor (GMI), which
GMI uses to look about nine months ahead of the ISM PMI
([The GMI Top 5 Weekly Charts That Make You Go Hmmm...](https://raoulpal.substack.com/p/the-gmi-top-5-weekly-charts-that-e8c), August 2023).
TNO is the organization authoring this FOSS copy to distinguish it from the original closed GMI FCI.

## The model

TNO-FCI is one number per month: a weighted sum of five monthly input series, the Fed funds
rate, the US dollar index, gold, oil and silver. A straight line maps it to ISM PMI points
some months later, the lead. Each fit picks its lead by a walk-forward search: for every lead
from 9 to 18 months it refits month by month on the data known at the time and scores the
predictions on months that fit hadn't seen. The docstring of `fci/fit.py` defines the fit and
what its record holds.

The July 2025 prediction came from an earlier model of 92 features built from the same five
series. Its archived run holds the licensed series, so only the real-data copy keeps it, in
`output/archived_runs/202507261556`; the page shows it as published. Fits since use five
weights, because they scored better on months the fit hadn't seen.

## Run it on the demo data

The series are licensed, so this repo can't ship them. The demo data stands in: six generated
series, with the target following a weighted sum of the input series a planted number of months
later. If the fit reports the lead you planted, the lead search works.

Python 3.12 with [uv](https://docs.astral.sh/uv/), from the repo root:

```bash
uv venv --python 3.12
uv pip install -e .
.venv/bin/python scripts/make_demo_data.py --lead 10  # any lead from 9 to 18; --seed changes the data
.venv/bin/fci all --demo                              # prep and fit under demo/; should say "lead 10 months"
```

`--demo` reads and writes only under `demo/`, which git ignores, and stops after the fit: no
page is built from demo data.

## Bring your own data

`data/raw_all/` holds the sample files: the first 5 rows of each series, exactly as exported,
to show the format. They are too short to fit. Replace each with a full monthly export from
TradingView, one file per series. `fci prep` stops if a series has two.

| Series | TradingView symbol | File in `data/raw_all/` | Columns |
| --- | --- | --- | --- |
| ISM Manufacturing PMI (the target) | `ECONOMICS:USBCOI` | `ECONOMICS_USBCOI*.csv` | `time,close` |
| Fed funds rate | `ECONOMICS:USINTR` | `ECONOMICS_USINTR*.csv` | `time,close` |
| US dollar index | `TVC:DXY` | `TVC_DXY*.csv` | `time,open,high,low,close` |
| Gold | `TVC:GOLD` | `TVC_GOLD*.csv` | `time,open,high,low,close` |
| WTI crude oil | `TVC:USOIL` | `TVC_USOIL*.csv` | `time,open,high,low,close` |
| Silver | `TVC:SILVER` | `TVC_SILVER*.csv` | `time,open,high,low,close` |

Use the monthly timeframe (1M). `time` is the date TradingView gives the monthly bar, its first
trading day. `fci prep` ignores extra columns. The docstring of `fci/prep.py` says which close a month
keeps and how a missing month is filled. TradingView no longer exports the two `ECONOMICS:`
series, so their recent rows come by hand: the ISM PMI from ISM's monthly release, and the Fed
funds rate as the upper bound of the Fed's target range at month end, from the FOMC statements.

Each step reads the files the step before it wrote and writes its own, so any one re-runs alone:

```bash
.venv/bin/fci prep      # data/raw_all/ -> output/clean/<series>.csv
.venv/bin/fci fit       # output/clean/ -> a new record in output/fits/<time>.json
.venv/bin/fci predict   # the newest record -> output/predictions/<time>.csv
.venv/bin/fci page      # the July 2025 fit and the newest record, with their predictions -> site/index.html
```

Each step's module docstring in `fci/` says what it reads and writes. `fci page`, and `fci all`,
which runs the four steps in order, also need the archived July 2025 run, so on this copy they
stop and say so. `fci page --july` takes another archived run in its place.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests   # each test file runs in seconds
```

The suite runs on the demo data. Tests that need the archived July 2025 run skip, and say why,
when it is absent.

`scripts/check_export.py` runs before every push of a public copy: it fails if a CSV under
`data/` holds more than 5 rows, among other checks. `scripts/export_public.py` builds the public
copy from the real-data copy.

## License

The code is MIT licensed; see `LICENSE`. The license covers the code only, not the data series,
which belong to their providers.
