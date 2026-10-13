#!/usr/bin/env python3
"""Build the public results page: the July 2025 prediction against the target that followed,
and the latest fit's prediction, tracking since 1971 and correlation.

A fit folder is what the fit script leaves: a results CSV (TNO-FCI per month and the target
one lead later) and its text reports. A fit's prediction maps its TNO-FCI to the target with
the straight line of "target, one lead later" on TNO-FCI. The page is static HTML with inline
SVG charts, no script and no series data.
"""
import argparse
import html
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

CODE_URL = "https://github.com/omarzd/tno-fci"
INSPIRATION_URL = "https://raoulpal.substack.com/p/the-gmi-top-5-weekly-charts-that-e8c"
# One palette per theme. Charts are inline SVG and colour by class, so the theme toggle
# recolours them along with the page.
LIGHT = {"bg": "#ffffff", "ink": "#1c2430", "accent": "#d9480f", "soft": "#f0b08c",
         "band": "#fdebdd", "grid": "#e6e8eb", "muted": "#6b7280"}
DARK = {"bg": "#14171c", "ink": "#e6e8eb", "accent": "#ff8a4c", "soft": "#a8582f",
        "band": "#3a2418", "grid": "#2c323b", "muted": "#9aa1ab"}
RECENT_START = pd.Period("2022-07", "M")


@dataclass
class Fit:
    lead: int
    r2: float
    generated: pd.Timestamp
    fci: pd.Series           # TNO-FCI, indexed by input month
    target_later: pd.Series  # the target one lead later, indexed by input month
    slope: float
    intercept: float

    @property
    def last_input(self):
        return self.fci.index[-1]

    @property
    def path(self):
        """TNO-FCI mapped to target points, indexed by the target month it predicts."""
        return pd.Series(self.slope * self.fci.values + self.intercept, index=self.fci.index + self.lead)

    @property
    def prediction(self):
        """The target path for the lead's worth of months after the last input month."""
        return self.path[self.path.index > self.last_input]

    @property
    def fitted(self):
        """The line over months the fit was fitted on, plus the first predicted month so the
        two lines meet."""
        return self.path[self.path.index <= self.last_input + 1]


def load_fit(folder):
    """Read a fit folder: fci_results.csv or tno_fci_results.csv plus text reports. The lead
    comes from the forward column's name; R² and the run time from the reports."""
    results = next(os.path.join(folder, n) for n in ("fci_results.csv", "tno_fci_results.csv")
                   if os.path.exists(os.path.join(folder, n)))
    reports = ""
    for name in sorted(os.listdir(folder)):
        if name.endswith(".txt"):
            with open(os.path.join(folder, name)) as fh:
                reports += fh.read() + "\n"
    r2 = float(re.search(r"R-squared: ([\d.]+)", reports).group(1))
    generated = pd.Timestamp(re.search(r"Generated on: ([\d\- :]+)", reports).group(1).strip())
    df = pd.read_csv(results)
    fwd_col = next(c for c in df.columns if re.fullmatch(r"ISM_\d+m_Forward", c))
    lead = int(re.search(r"\d+", fwd_col).group())
    months = pd.PeriodIndex(pd.to_datetime(df["Date"]), freq="M")
    fci = pd.Series(df["TNO_FCI"].values, index=months)
    later = pd.Series(df[fwd_col].values, index=months).dropna()
    slope, intercept = np.polyfit(fci[later.index].values, later.values, 1)
    return Fit(lead, r2, generated, fci, later, slope, intercept)


def load_target(csv_path):
    df = pd.read_csv(csv_path)
    return pd.Series(df.iloc[:, 1].values, index=pd.PeriodIndex(pd.to_datetime(df.iloc[:, 0]), freq="M"))


def month_name(p):
    return p.strftime("%B %Y")


def joined(months):
    return " and ".join(month_name(p) for p in months)


def svg_open(W, H, title):
    return (f'<svg class="chart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" font-size="15">'
            f"<title>{html.escape(title)}</title>")


def nice_range(lo, hi, step):
    return step * np.floor((lo - step / 5) / step), step * np.ceil((hi + step / 5) / step)


def line_chart(lines, start, end, band=None, marker=None, title=""):
    """An inline SVG line chart over months start..end. lines: (label, series, role, dashed),
    where role is a palette key; classes s-<role> and f-<role> set stroke and fill."""
    W, H, L, R, T, B = 600, 340, 40, 150, 16, 36
    shown = [s[(s.index >= start) & (s.index <= end)] for _, s, _, _ in lines]
    vmin = min(s.min() for s in shown if len(s))
    vmax = max(s.max() for s in shown if len(s))
    step = 5 if vmax - vmin < 30 else 10
    lo, hi = nice_range(vmin, vmax, step)
    span = (end - start).n
    years = span / 12
    every = 1 if years <= 8 else 5 if years <= 30 else 10
    x = lambda p: L + (W - L - R) * (p - start).n / span
    y = lambda v: T + (H - T - B) * (hi - v) / (hi - lo)
    out = [svg_open(W, H, title)]
    if band:
        b0, b1, blabel = band
        out.append(f'<rect class="f-band" x="{x(b0):.1f}" y="{T}" width="{x(b1) - x(b0):.1f}" height="{H - T - B}"/>')
        out.append(f'<text class="f-accent" x="{x(b0) + 6:.1f}" y="{T + 14}">{html.escape(blabel)}</text>')
    for v in np.arange(lo, hi + 0.1, step):
        out.append(f'<line class="s-grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        out.append(f'<text class="f-muted" x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.0f}</text>')
    if lo < 50 < hi:
        out.append(f'<line class="s-muted" x1="{L}" x2="{W - R}" y1="{y(50):.1f}" y2="{y(50):.1f}" stroke-dasharray="2 3"/>')
    for p in pd.period_range(start, end, freq="M"):
        if p.month == 1 and p.year % every == 0:
            out.append(f'<text class="f-muted" x="{x(p):.1f}" y="{H - 12}" text-anchor="middle">{p.year}</text>')
            out.append(f'<line class="s-muted" x1="{x(p):.1f}" x2="{x(p):.1f}" y1="{H - B}" y2="{H - B + 5}"/>')
    if marker:
        mp, mlabel = marker
        out.append(f'<line class="s-muted" x1="{x(mp):.1f}" x2="{x(mp):.1f}" y1="{T}" y2="{H - B}" stroke-dasharray="4 4"/>')
        out.append(f'<text class="f-muted" x="{x(mp) - 6:.1f}" y="{H - B - 8}" text-anchor="end">{html.escape(mlabel)}</text>')
    width = 2.6 if years <= 8 else 1.4
    for (label, _, role, dashed), s in zip(lines, shown):
        if not len(s):
            continue
        pts = " ".join(f"{x(p):.1f},{y(v):.1f}" for p, v in s.items())
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        out.append(f'<polyline class="s-{role}" points="{pts}" fill="none" stroke-width="{width}" stroke-linejoin="round"{dash}/>')
        if label:
            out.append(f'<text class="f-{role}" x="{x(s.index[-1]) + 8:.1f}" y="{y(s.iloc[-1]) + 5:.1f}" font-weight="600">{html.escape(label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def scatter_chart(fit, title=""):
    """TNO-FCI against the target one lead later, with the fitted straight line and the 12
    most recent months whose outcome is known."""
    W, H, L, R, T, B = 600, 380, 44, 24, 16, 52
    xs, ys = fit.fci[fit.target_later.index], fit.target_later
    x_lo, x_hi = nice_range(fit.fci.min(), fit.fci.max(), 1)
    y_lo, y_hi = nice_range(ys.min(), ys.max(), 10)
    x = lambda v: L + (W - L - R) * (v - x_lo) / (x_hi - x_lo)
    y = lambda v: T + (H - T - B) * (y_hi - v) / (y_hi - y_lo)
    out = [svg_open(W, H, title)]
    xstep = 1 if x_hi - x_lo <= 8 else 2.5 if x_hi - x_lo <= 20 else 5
    for v in np.arange(y_lo, y_hi + 0.1, 10):
        out.append(f'<line class="s-grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        out.append(f'<text class="f-muted" x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.0f}</text>')
    for v in np.arange(np.ceil(x_lo / xstep) * xstep, x_hi + 1e-9, xstep):
        out.append(f'<text class="f-muted" x="{x(v):.1f}" y="{H - B + 20}" text-anchor="middle">{v:g}</text>')
    out.append(f'<text class="f-muted" x="{(L + W - R) / 2:.1f}" y="{H - 6}" text-anchor="middle">TNO-FCI</text>')
    recent = ys.index[-12:]
    for p in ys.index:
        if p not in recent:
            out.append(f'<circle class="f-muted" cx="{x(xs[p]):.1f}" cy="{y(ys[p]):.1f}" r="2.6" fill-opacity=".45"/>')
    out.append(f'<line class="s-accent" x1="{x(x_lo):.1f}" y1="{y(fit.slope * x_lo + fit.intercept):.1f}" '
               f'x2="{x(x_hi):.1f}" y2="{y(fit.slope * x_hi + fit.intercept):.1f}" stroke-width="2"/>')
    for p in recent:
        out.append(f'<circle class="f-ink" cx="{x(xs[p]):.1f}" cy="{y(ys[p]):.1f}" r="4.2"/>')
    out.append(f'<text class="f-muted" x="{L + 8}" y="{T + 14}">R-squared {fit.r2:.2f}</text>')
    out.append("</svg>")
    return "\n".join(out)


def table(rows, headers):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def palette_vars(p):
    return " ".join(f"--{k}: {v};" for k, v in p.items())


def stretch_note(fit):
    """Plain words for predicted months that rest on TNO-FCI beyond anything the fit saw."""
    pred, past = fit.prediction, fit.path[fit.path.index <= fit.last_input]
    beyond = pred[pred > past.max()]
    if not len(beyond):
        return ""
    one = len(beyond) == 1
    inputs = beyond.index - fit.lead
    since = (past.index[0] - fit.lead).year
    if len(beyond) <= 2:
        what = f"The predictions for {joined(beyond.index)} rest on {joined(inputs)}"
    else:
        what = f"{len(beyond)} of these predictions rest on input months"
    return (f"<p>{what}, when TNO-FCI rose higher than in any month since {since}. The model has never seen "
            f"readings that high, so {'this number is' if one else 'these numbers are'} less reliable than the rest: "
            f"take {'it' if one else 'them'} as a sign of strong expansion rather than {'a precise value' if one else 'precise values'}.</p>")


def recent_chart(fit, target, end, band_label, label, title, marker=None):
    jp = fit.prediction
    return line_chart(
        [("", fit.fitted, "soft", True), ("", target, "ink", False), (label, jp, "accent", False)],
        RECENT_START, end, band=(jp.index[0], jp.index[-1], band_label) if band_label else None,
        marker=marker, title=title)


def build_page(july_dir, latest_dir, target_csv, out_dir):
    july, latest, target = load_fit(july_dir), load_fit(latest_dir), load_target(target_csv)
    os.makedirs(out_dir, exist_ok=True)

    jp, lp = july.prediction, latest.prediction
    pred_above = jp[jp > 50]
    after = target[target.index > july.last_input]
    actual_above = after[after >= 50]
    crossing = (f"The prediction had the PMI back above 50 from {month_name(pred_above.index[0])}; "
                if len(pred_above) else "The prediction kept the PMI below 50; ")
    crossing += (f"the PMI itself crossed 50 in {month_name(actual_above.index[0])}."
                 if len(actual_above) else "the PMI itself has not crossed 50 since.")
    leads = sorted({july.lead, latest.lead})
    lead_text = " to ".join(str(n) for n in leads)
    made = latest.generated.to_period("M")
    since = latest.path.index[0].year

    chart_july = recent_chart(july, target, max(target.index[-1], jp.index[-1]), "predicted in July 2025",
                              "", "The ISM PMI and the July 2025 prediction")
    chart_latest = recent_chart(latest, target, lp.index[-1], None, "Prediction",
                                f"The ISM PMI and the {month_name(made)} prediction",
                                marker=(latest.last_input, f"inputs end {latest.last_input.strftime('%b %Y')}"))
    chart_history = line_chart(
        [("", target, "ink", False), ("TNO-FCI", latest.path[latest.path.index <= latest.last_input], "accent", False)],
        latest.path.index[0], latest.last_input,
        title=f"The ISM PMI since {since} and TNO-FCI moved {latest.lead} months forward")
    chart_scatter = scatter_chart(latest, title=f"TNO-FCI against the ISM PMI {latest.lead} months later")

    roles = ("ink", "accent", "soft", "band", "grid", "muted")
    chart_css = " ".join(f".s-{r} {{ stroke: var(--{r}); }} .f-{r} {{ fill: var(--{r}); }}" for r in roles)
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TNO-FCI: predicting the ISM Manufacturing PMI {lead_text} months ahead</title>
<style>
  body {{ margin: 0; }}
  .page {{ {palette_vars(LIGHT)} background: var(--bg); color: var(--ink); min-height: 100vh;
    font: 17px/1.6 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
  @media (prefers-color-scheme: dark) {{
    .page {{ {palette_vars(DARK)} }}
    #theme:checked ~ .page {{ {palette_vars(LIGHT)} }}
  }}
  @media not (prefers-color-scheme: dark) {{
    #theme:checked ~ .page {{ {palette_vars(DARK)} }}
  }}
  .wrap {{ max-width: 46rem; margin: 0 auto; padding: 2.5rem 1.2rem; }}
  h1 {{ font-size: 1.7rem; line-height: 1.25; margin: 0 0 .3rem; }}
  h2 {{ font-size: 1.2rem; margin-top: 2.6rem; }}
  .lede, figcaption {{ color: var(--muted); }}
  .lede {{ margin-top: 0; }}
  figure {{ margin: 1.2rem 0; }}
  svg.chart {{ width: 100%; height: auto; display: block; font-family: inherit; }}
  svg.chart text {{ paint-order: stroke; stroke: var(--bg); stroke-width: 4px; stroke-linejoin: round; }}
  {chart_css}
  figcaption {{ font-size: .85rem; }}
  a {{ color: var(--accent); }}
  table {{ border-collapse: collapse; font-size: .9rem; margin: .8rem 0; }}
  th, td {{ padding: .25rem .9rem .25rem 0; text-align: left; border-bottom: 1px solid var(--grid); }}
  td:last-child, th:last-child {{ text-align: right; }}
  #theme {{ position: absolute; opacity: 0; pointer-events: none; }}
  .theme-label {{ position: absolute; top: .9rem; left: 1rem; cursor: pointer; font-size: .85rem; color: var(--muted);
    border: 1px solid var(--grid); border-radius: 999px; padding: .15rem .7rem; user-select: none; }}
  .theme-label:hover {{ color: var(--ink); }}
  #theme:focus-visible ~ .page .theme-label {{ outline: 2px solid var(--accent); }}
  .github-corner svg {{ position: absolute; top: 0; right: 0; border: 0; fill: var(--ink); color: var(--bg); }}
  .github-corner:hover .octo-arm {{ animation: octo-wave 560ms ease-in-out; }}
  @keyframes octo-wave {{ 0%, 100% {{ transform: rotate(0); }} 20%, 60% {{ transform: rotate(-25deg); }} 40%, 80% {{ transform: rotate(10deg); }} }}
</style></head><body>
<input type="checkbox" id="theme" aria-label="Switch between light and dark mode">
<div class="page">
<label for="theme" class="theme-label">&#9680; Light / dark</label>
<a href="{CODE_URL}" class="github-corner" aria-label="View source on GitHub"><svg width="80" height="80" viewBox="0 0 250 250" aria-hidden="true"><path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"/><path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"/><path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"/></svg></a>
<div class="wrap">
<h1>TNO-FCI: predicting the ISM Manufacturing PMI {lead_text} months ahead</h1>
<p class="lede">A financial conditions index built from five monthly market and policy series.</p>
<p>Inspired by the financial conditions index of Raoul Pal's Global Macro Investor (GMI), which GMI uses to look about nine months ahead of the ISM PMI (<a href="{INSPIRATION_URL}">The GMI Top 5 Weekly Charts That Make You Go Hmmm...</a>, August 2023).</p>

<p>TNO-FCI is one number per month, built from the Fed funds rate, the US dollar index, gold, oil and silver. A linear model combines features made from those five series, weighted so that this month's value tracks the ISM Manufacturing PMI {lead_text} months later. The PMI is a monthly survey of US factory activity; above 50 means expansion.</p>

<h2>The July 2025 prediction, and what followed</h2>
<p>On {july.generated.day} July 2025 the model, fitted on data through {month_name(july.last_input)}, predicted the PMI for {month_name(jp.index[0])} to {month_name(jp.index[-1])}. {crossing}</p>
<figure>{chart_july}
<figcaption>Black line (white in dark mode): the ISM PMI. Solid orange: the prediction of {july.generated.strftime('%-d %B %Y')}, TNO-FCI moved {july.lead} months forward and mapped to PMI points. Dashed: the same fit over the months it was fitted on.</figcaption></figure>
{table([(month_name(p), f"{v:.1f}") for p, v in jp.items()], ["Month", "Predicted PMI"])}

<h2>The {month_name(made)} prediction</h2>
<p>Refitted on data through {month_name(latest.last_input)}, the model predicts the PMI for {month_name(lp.index[0])} to {month_name(lp.index[-1])}.</p>
<figure>{chart_latest}
<figcaption>Black line (white in dark mode): the ISM PMI. Solid orange: the prediction from the fit of {latest.generated.strftime('%-d %B %Y')}, lead {latest.lead} months. Dashed: the same fit over the months it was fitted on.</figcaption></figure>
{table([(month_name(p), f"{v:.1f}") for p, v in lp.items()], ["Month", "Predicted PMI"])}
{stretch_note(latest)}

<h2>How it tracks since {since}</h2>
<figure>{chart_history}
<figcaption>Black line (white in dark mode): the ISM PMI. Orange: TNO-FCI from the fit of {latest.generated.strftime('%-d %B %Y')}, moved {latest.lead} months forward and mapped to PMI points.</figcaption></figure>
<figure>{chart_scatter}
<figcaption>Each dot is one month: TNO-FCI against the PMI {latest.lead} months later. Orange line: the straight line that maps TNO-FCI to PMI points. Highlighted dots, black in light mode and white in dark mode: the 12 most recent months whose PMI {latest.lead} months later is already known, showing how far real months land from the line.</figcaption></figure>

<h2>Method</h2>
<p>Alongside the five series, the model uses features built from them: their values 3, 6 and 12 months back, momentum, moving averages, volatility and ratios such as gold to silver. The lead is chosen by testing 3 to 18 months and keeping the one where TNO-FCI correlates best with the PMI that many months later; a straight line then maps TNO-FCI to PMI points. In-sample, the fit explains R² {latest.r2:.2f} of the PMI's variation ({july.r2:.2f} in July 2025), so read the prediction as a direction rather than a precise numerical forecast.</p>

<h2>Data and code</h2>
<p>Monthly series, mostly exported from TradingView: TVC:DXY, TVC:GOLD, TVC:USOIL and TVC:SILVER, plus ECONOMICS:USBCOI (ISM PMI) and ECONOMICS:USINTR (Fed funds rate) through June 2025. From July 2025 the ISM PMI comes from ISM's monthly releases and the Fed funds rate (upper bound of the target range) from FOMC statements. The data isn't redistributed; the <a href="{CODE_URL}">code</a> runs on your own copies.</p>
</div></div>
</body></html>
"""
    path = os.path.join(out_dir, "index.html")
    with open(path, "w") as fh:
        fh.write(page)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--july", default="output/archived_runs/202507261556")
    ap.add_argument("--latest", default="output")
    ap.add_argument("--target", default="data/clean_all/ismpmi_clean.csv")
    ap.add_argument("--out", default="site")
    a = ap.parse_args()
    print(build_page(a.july, a.latest, a.target, a.out))


if __name__ == "__main__":
    main()
