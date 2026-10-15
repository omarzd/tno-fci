# TNO-FCI

A financial conditions index that predicts the ISM Manufacturing PMI about 10 months ahead.
The results page is at **https://omarzd.github.io/tno-fci/**.

Inspired by the financial conditions index of Raoul Pal's Global Macro Investor (GMI), which
GMI uses to look about nine months ahead of the ISM PMI
([The GMI Top 5 Weekly Charts That Make You Go Hmmm...](https://raoulpal.substack.com/p/the-gmi-top-5-weekly-charts-that-e8c), August 2023).
TNO is the author's TradingView handle, TechNerdOmar.

## The model

TNO-FCI is one number per month, built from five monthly series: the Fed funds rate, the US
dollar index, gold, oil and silver. `3_tno_fci_calculator.py` builds 92 features from them
(each series' values 3, 6 and 12 months back, momentum, moving averages, volatility, ratios
such as gold to silver, and a few interaction and seasonal terms) and combines them linearly.
It tests every lead from 3 to 18 months and keeps the one where TNO-FCI correlates best with
the PMI that many months later. On data through September 2026 that lead is 10 months, with
an in-sample R² of 0.46. A straight line then maps TNO-FCI to PMI points.

The script also trains several machine-learning models (random forest, gradient boosting,
SVR, regularized linear models) for comparison, and keeps the linear model when they score
worse out of sample, which they do. That comparison is why a run takes several minutes.

## Data

The data is licensed and isn't in this repo. Each file under `data/` holds its first 5 rows
only, to show the exact format the scripts read. These sample files are too short to fit, so
there are two ways to run the model: on demo data, or on your own exports.

### Try it on demo data

`scripts/make_demo_data.py` generates the six series, with the target following a weighted sum
of the inputs a planted number of months later. The fit doesn't know that number. If it reports
the lead you planted, the lead search works.

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
.venv/bin/python scripts/make_demo_data.py --lead 7   # any lead from 3 to 18; --seed changes the data
.venv/bin/python 2_prep_raw_data.py --demo            # demo/data/raw_all/ -> demo/data/clean_all/
.venv/bin/python 3_tno_fci_calculator.py --demo       # fit -> demo/output/; should say "Optimal lag period: 7 months"
```

Everything the demo writes stays in `demo/`, which git ignores. It never touches `data/` or
`output/`. The demo shows the method finds a lead that is really there. It doesn't reproduce the
10-month lead on the real series; for that you need the real data.

### Bring your own data

Replace the sample files with full monthly exports.

| Series | TradingView symbol | File in `data/raw_all/` | Columns |
| --- | --- | --- | --- |
| ISM Manufacturing PMI (the target) | `ECONOMICS:USBCOI` | `ECONOMICS_USBCOI*.csv` | `time,close` |
| Fed funds rate | `ECONOMICS:USINTR` | `ECONOMICS_USINTR*.csv` | `time,close` |
| US dollar index | `TVC:DXY` | `TVC_DXY*.csv` | `time,open,high,low,close` |
| Gold | `TVC:GOLD` | `TVC_GOLD*.csv` | `time,open,high,low,close` |
| WTI crude oil | `TVC:USOIL` | `TVC_USOIL*.csv` | `time,open,high,low,close` |
| Silver | `TVC:SILVER` | `TVC_SILVER*.csv` | `time,open,high,low,close` |

Use the monthly timeframe (1M). `time` is the date TradingView gives the monthly bar, its first trading day. Keep one file per
series: `2_prep_raw_data.py` reads the first file whose name starts with the pattern, so two
exports of one series side by side means either could be used. Extra columns in an export are
ignored.

TradingView no longer exports the two `ECONOMICS:` series. Since July 2025 the page uses ISM's
monthly releases for the PMI and FOMC statements for the Fed funds rate (the upper bound of the
target range, as it stood at month end), entered by hand in the same format.

## Run it

Python 3.12 with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
.venv/bin/python 2_prep_raw_data.py        # raw exports -> data/clean_all/
.venv/bin/python 3_tno_fci_calculator.py   # fit -> output/, archived in output/archived_runs/<time>/
.venv/bin/python 5_build_page.py --july output/archived_runs/<an earlier run> --latest output
```

`1_collect_data.py` reports which raw files are out of date. `pandas` stays below 3 because
the scripts resample with the `'M'` frequency, which pandas 3 removed.

The page step compares an earlier fit's prediction (`--july`) with what followed, and shows
the latest fit (`--latest`). It writes `site/index.html`: static HTML with inline SVG charts,
no script and no data. The published page is that file, copied to `docs/`.

These steps need your own data; the sample files are too short to fit. To run without it, see [Try it on demo data](#try-it-on-demo-data).

## Tests and the pre-push check

```bash
.venv/bin/python -m unittest tests/test_build_page.py   # the page step, on synthetic fits
.venv/bin/python -m unittest tests/test_demo_data.py    # the demo finds a planted lead; about 6 minutes
.venv/bin/python scripts/check_export.py                # run before every push
```

`check_export.py` fails if any data file holds more than the 5 sample rows, if an
unexpected file sits at the top level, if a file names a local home-folder path, or if a
commit carries an email other than the author's GitHub no-reply address.

## License

The code is MIT licensed; see `LICENSE`. The license covers the code only, not the data series, which belong to their providers.
