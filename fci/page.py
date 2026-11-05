"""Render the results page: the July 2025 prediction against the target that followed, the
latest fit's prediction, and how that fit's TNO-FCI has tracked the target, as line charts and
a scatter. The page is static HTML with inline SVG charts, no script and no series data.

`fci page` reads the July 2025 fit, the latest fit (the newest record by default), their
prediction files from `fci predict`, and the clean target, and writes site/index.html. When the
latest fit is a record, the Method section also states its lag test from `fci fit`. The footer
states the day the page was written: today, or `--written`.

The look is the studio's round-1 design system (ADR 0005 in fci-private).
"""
import html
import math
import os
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from fci import StepError
from fci import predict as predicting
from fci.fit import TARGET, load_clean

CODE_URL = "https://github.com/omarzd/fci"
INSPIRATION_URL = "https://raoulpal.substack.com/p/the-gmi-top-5-weekly-charts-that-e8c"
# One palette per theme. Charts are inline SVG and colour by class, so the theme toggle
# recolours them along with the page.
LIGHT = {"bg": "#ffffff", "ink": "#1c2430", "accent": "#d9480f", "soft": "#d97a45",
         "band": "#fdebdd", "grid": "#e6e8eb", "muted": "#6b7280"}
DARK = {"bg": "#14171c", "ink": "#e6e8eb", "accent": "#ff8a4c", "soft": "#a8582f",
        "band": "#2e2119", "grid": "#2c323b", "muted": "#9aa1ab"}
SERIF = 'Charter, "Bitstream Charter", "Iowan Old Style", Georgia, serif'
SANS = '-apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
DESCRIPTION = ("A financial conditions index built from five monthly market and policy series, "
               "and what it predicts for the ISM Manufacturing PMI.")
# A small line chart in the dark palette, so the tab icon needs no request.
FAVICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
           "%3Crect width='32' height='32' rx='6' fill='%231c2430'/%3E"
           "%3Cpolyline points='5,22 11,17 16,19 21,10 27,13' fill='none' stroke='%23ff8a4c' "
           "stroke-width='3' stroke-linejoin='round' stroke-linecap='round'/%3E%3C/svg%3E")
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
    model: str  # "five weights" (fci fit), "92 features" or "five series" (archived runs)
    predicted: pd.Series = None  # from the prediction file; None computes it from the line
    lag_r2: dict = None  # in-sample R² by lag in months, from fci fit's lag test

    @property
    def last_input(self):
        return self.fci.index[-1]

    @property
    def shifted(self):
        """TNO-FCI, shifted right by the lead: mapped to target points, indexed by the target month
        it predicts."""
        return pd.Series(self.slope * self.fci.values + self.intercept, index=self.fci.index + self.lead)

    @property
    def prediction(self):
        """The target path for the lead's worth of months after the last input month."""
        if self.predicted is not None:
            return self.predicted
        return self.shifted[self.shifted.index > self.last_input]

    @property
    def shifted_known(self):
        """TNO-FCI, shifted right by the lead, over the months whose inputs are known, plus the
        first predicted month so it joins the prediction."""
        return self.shifted[self.shifted.index <= self.last_input + 1]

    @property
    def shifted_seen(self):
        """TNO-FCI, shifted right by the lead, over the months the fit saw."""
        return self.shifted[self.shifted.index <= self.last_input]

    @property
    def stretched(self):
        """The predicted months whose value is above every value of TNO-FCI, shifted right by the
        lead, over the months the fit saw."""
        return self.prediction.index[self.prediction > self.shifted_seen.max()]


# What the intro says the model does, and the Method section, for the latest fit's kind.
TERMS = {
    "five weights": "weighs those five series so that",
    "five series": "weighs those five series so that",
    "92 features": "combines features made from those five series, weighted so that",
}
METHOD = {
    "five weights": ("TNO-FCI weighs the five series themselves, each scaled by its average and spread over the "
                     "months the fit saw. The fit tests every lead from 9 to 18 months. For each lead it refits month "
                     "by month on the data known at the time and predicts every month of the lead's path. It scores "
                     "each path against the mean forecast, the ISM PMI's average known at the time, and the first and "
                     "last months weigh most. In that score a miss of 2 points or less costs nothing, and a predicted "
                     "month may match the ISM PMI one month early or late. The fit keeps the lead with the best "
                     "score. A straight line then maps TNO-FCI to ISM PMI points."),
    "five series": ("TNO-FCI weighs the five series themselves, each scaled by its average and spread. The model tests "
                    "every lead from 3 to 18 months and keeps the one where that weighted sum correlates best with "
                    "the ISM PMI that many months later. A straight line then maps TNO-FCI to ISM PMI points."),
    "92 features": ("Alongside the five series, the model uses features built from them: their values 3, 6 and 12 "
                    "months back, momentum, moving averages, volatility, ratios such as gold to silver, and a few "
                    "interaction, squared and seasonal terms. The model tests every lead from 3 to 18 months and "
                    "keeps the one where a weighted sum of the five series correlates best with the ISM PMI that "
                    "many months later. A straight line then maps TNO-FCI to ISM PMI points."),
}


def from_loaded(loaded, output_dir):
    """A page Fit from fci.predict's LoadedFit, with its prediction from the prediction file."""
    if loaded.r2 is None or loaded.generated is None:
        raise StepError(f"{loaded.name} records no R-squared or fit date, which the page states")
    path = predicting.prediction_path(output_dir, loaded.name)
    if not os.path.exists(path):
        raise StepError(f"no prediction file {path}; run fci predict for {loaded.name}")
    try:
        df = pd.read_csv(path)
        predicted = pd.Series(df[TARGET].to_numpy(), index=pd.PeriodIndex(df["month"], freq="M"))
    except (ValueError, KeyError, pd.errors.EmptyDataError) as e:  # DateParseError is a ValueError
        raise StepError(f"{path} isn't a prediction file; run fci predict for {loaded.name} again ({e!r})") from e
    expected = predicting.prediction(loaded)
    if not (predicted.index.equals(expected.index) and np.allclose(predicted.to_numpy(), expected.to_numpy())):
        raise StepError(f"{path} doesn't match its fit; run fci predict for {loaded.name} again")
    return Fit(loaded.lead, loaded.r2, loaded.generated, loaded.tnofci.dropna(), loaded.later,
               loaded.slope, loaded.intercept, loaded.model, predicted, loaded.lag_r2)


def lag_sentence(fit):
    """The lag test, for a fit that has one. The bound rounds up so "or less" stays true."""
    if not fit.lag_r2:
        return ""
    bound = math.ceil(round(max(fit.lag_r2.values()) * 100, 6)) / 100
    return (" Shifted the other way, as if it lagged the ISM PMI, the fit explains almost nothing "
            f"(R² {bound:.2f} or less at every lag from 1 to 12 months).")


def month_name(p):
    return p.strftime("%B %Y")


def joined(months):
    return " and ".join(month_name(p) for p in months)


def nice_range(lo, hi, step):
    return step * np.floor((lo - step / 5) / step), step * np.ceil((hi + step / 5) / step)


# The two chart sizes of the figure pattern. One unit of the wide viewBox is 1px at full width.
SIZES = {
    "wide": dict(w=736, h=360, h_scatter=440, font=14, l=34, r=8, t=30, b=28, line=2.4, thin=1.2, ring=5.2,
                 dot=2.6, recent=4.2),
    "narrow": dict(w=360, h=300, h_scatter=340, font=13, l=30, r=6, t=28, b=26, line=1.9, thin=0.9, ring=4.0,
                   dot=1.7, recent=3.0),
}
LONG_SPAN = 8 * 12  # months; a longer chart draws thin lines
LABEL_GAP = 44      # units at least between year labels
FIFTY = "50: above is expansion, below is contraction"


class Frame:
    """One size of a chart: the plot's edges in viewBox units, and data to units. A scatter is
    taller, with room under its tick labels for the x title."""

    def __init__(self, size, x_lo, x_hi, y_lo, y_hi, scatter=False):
        self.size, self.s = size, SIZES[size]
        self.w, self.h = self.s["w"], self.s["h_scatter" if scatter else "h"]
        self.x0, self.x1 = self.s["l"], self.w - self.s["r"]
        self.y0, self.y1 = self.s["t"], self.h - self.s["b"] - (20 if scatter else 0)
        self.x_lo, self.x_hi, self.y_lo, self.y_hi = x_lo, x_hi, y_lo, y_hi

    def x(self, v):
        return self.x0 + (self.x1 - self.x0) * (v - self.x_lo) / (self.x_hi - self.x_lo)

    def y(self, v):
        return self.y1 - (self.y1 - self.y0) * (v - self.y_lo) / (self.y_hi - self.y_lo)

    def svg(self, title_id, legend_id):
        return (f'<svg class="chart chart-{self.size}" xmlns="http://www.w3.org/2000/svg" '
                f'viewBox="0 0 {self.w} {self.h}" role="img" aria-labelledby="{title_id}" '
                f'aria-describedby="{legend_id}" font-size="{self.s["font"]}">')


def text(x, y, s, anchor=None, weight=None):
    a = f' text-anchor="{anchor}"' if anchor else ""
    w = f' font-weight="{weight}"' if weight else ""
    return f'<text class="f-muted" x="{x:.1f}" y="{y:.1f}"{a}{w}>{html.escape(s)}</text>'


def y_axis(fr, step):
    """A grid line and label at each y tick, and the y title, horizontal above the plot."""
    out = []
    for v in np.arange(fr.y_lo, fr.y_hi + step / 10, step):
        out.append(f'<line class="s-grid" x1="{fr.x0}" x2="{fr.x1}" y1="{fr.y(v):.1f}" y2="{fr.y(v):.1f}"/>')
        out.append(text(fr.x0 - 6, fr.y(v) + fr.s["font"] * 0.32, f"{v:.0f}", anchor="end"))
    out.append(text(0, fr.y0 - 14, "ISM PMI", weight=600))
    return out


def year_axis(fr, start, end):
    """Year labels every 1, 2, 5, 10 or 20 years, the least that keeps LABEL_GAP between them,
    dropping any that would cross the plot's edge; a tick at each January when they sit 6 units
    apart, or else at each label."""
    per_year = 12 * (fr.x1 - fr.x0) / (fr.x_hi - fr.x_lo)
    every = next(n for n in (1, 2, 5, 10, 20) if n * per_year >= LABEL_GAP)
    half = fr.s["font"] * 1.3
    out = []
    for p in pd.period_range(start, end, freq="M"):
        if p.month != 1:
            continue
        x, labelled = fr.x((p - start).n), p.year % every == 0
        if per_year >= 6 or labelled:
            out.append(f'<line class="s-muted" x1="{x:.1f}" x2="{x:.1f}" y1="{fr.y1}" y2="{fr.y1 + 5}"/>')
        if labelled and fr.x0 + half <= x <= fr.x1 - half:
            out.append(text(x, fr.y1 + 19, str(p.year), anchor="middle"))
    return out


class LineChart:
    """Figures 1 to 3: lines over the months start..end, drawn the same way at either size.
    lines: (series, role, dash), where role is a palette key and dash a stroke-dasharray or None.
    band: the first and last month to shade; marker: a month to mark; hollow: points to draw
    hollow, by month."""

    def __init__(self, lines, start, end, band=None, marker=None, hollow=None):
        self.lines = [(s[(s.index >= start) & (s.index <= end)], role, dash) for s, role, dash in lines]
        self.start, self.end, self.band, self.marker = start, end, band, marker
        self.hollow = hollow if hollow is not None and len(hollow) else None
        vmin = min(s.min() for s, _, _ in self.lines if len(s))
        vmax = max(s.max() for s, _, _ in self.lines if len(s))
        self.step = 5 if vmax - vmin < 30 else 10
        self.lo, self.hi = nice_range(vmin, vmax, self.step)
        self.fifty = self.lo < 50 < self.hi

    def svg(self, size, title_id, legend_id):
        span = (self.end - self.start).n
        fr = Frame(size, 0, span, self.lo, self.hi)
        x = lambda p: fr.x((p - self.start).n)
        width = fr.s["thin"] if span > LONG_SPAN else fr.s["line"]
        out = [fr.svg(title_id, legend_id)]
        if self.band:
            x0, x1 = fr.x((self.band[0] - self.start).n - 0.5), fr.x((self.band[1] - self.start).n + 0.5)
            out.append(f'<rect class="f-band" x="{x0:.1f}" y="{fr.y0}" width="{x1 - x0:.1f}" height="{fr.y1 - fr.y0}"/>')
        out += y_axis(fr, self.step)
        if self.fifty:
            out.append(f'<line class="s-muted" x1="{fr.x0}" x2="{fr.x1}" y1="{fr.y(50):.1f}" y2="{fr.y(50):.1f}" stroke-dasharray="2 3"/>')
        out += year_axis(fr, self.start, self.end)
        if self.marker is not None:
            out.append(f'<line class="s-muted" x1="{x(self.marker):.1f}" x2="{x(self.marker):.1f}" '
                       f'y1="{fr.y0}" y2="{fr.y1}" stroke-dasharray="4 4"/>')
        for s, role, dash in self.lines:
            if not len(s):
                continue
            pts = " ".join(f"{x(p):.1f},{fr.y(v):.1f}" for p, v in s.items())
            d = f' stroke-dasharray="{dash}"' if dash else ""
            out.append(f'<polyline class="s-{role}" points="{pts}" fill="none" stroke-width="{width}" '
                       f'stroke-linejoin="round" stroke-linecap="round"{d}/>')
        if self.hollow is not None:
            for p, v in self.hollow.items():
                out.append(f'<circle class="f-bg s-accent" cx="{x(p):.1f}" cy="{fr.y(v):.1f}" '
                           f'r="{fr.s["ring"] - 1:.1f}" stroke-width="{width}"/>')
        out.append("</svg>")
        return "\n".join(out)


# Legend swatches, drawn with the charts' classes so the toggle recolours them.
SWATCH = {
    "ink": '<line class="s-ink" x1="1" x2="23" y1="6" y2="6" stroke-width="2.4"/>',
    "accent": '<line class="s-accent" x1="1" x2="23" y1="6" y2="6" stroke-width="2.4"/>',
    "dashed": '<line class="s-soft" x1="1" x2="23" y1="6" y2="6" stroke-width="2.4" stroke-dasharray="6 4"/>',
    "band": '<rect class="f-band" x="1" y="0" width="22" height="12"/>',
    "fifty": '<line class="s-muted" x1="1" x2="23" y1="6" y2="6" stroke-dasharray="2 3"/>',
    "marker": '<line class="s-muted" x1="12" x2="12" y1="0" y2="12" stroke-dasharray="3 3"/>',
    "hollow": ('<line class="s-accent" x1="1" x2="23" y1="6" y2="6" stroke-width="2.4"/>'
               '<circle class="f-bg s-accent" cx="12" cy="6" r="4" stroke-width="2"/>'),
    "dot": '<circle class="f-muted" cx="12" cy="6" r="2.8" fill-opacity=".6"/>',
    "recent": '<circle class="f-ink" cx="12" cy="6" r="4.2"/>',
    "ring": '<circle class="f-bg s-accent" cx="12" cy="6" r="4.6" stroke-width="1.8"/>',
}


def legend(rows, legend_id):
    """The legend below a plot: rows of (swatch, label markup), the data marks first, then the
    background marks. An empty row is left out."""
    out = [f'<div class="legend" id="{legend_id}">']
    for row in rows:
        if not row:
            continue
        out.append("<ul>" + "".join(
            f'<li><svg class="sw" viewBox="0 0 24 12" aria-hidden="true">{SWATCH[kind]}</svg>'
            f'<span>{label.replace("TNO-FCI", "TNO&#8209;FCI")}</span></li>' for kind, label in row) + "</ul>")
    out.append("</div>")
    return "\n".join(out)


def figure(n, title, chart, rows, note=None):
    """Figure n: its title, the chart at both sizes, its legend and an optional note. chart has
    svg(size, title_id, legend_id)."""
    title_id, legend_id = f"fig{n}-title", f"fig{n}-legend"
    parts = [f'<figure id="figure-{n}">',
             f'<figcaption class="fig-title" id="{title_id}">Figure {n}. {html.escape(title)}</figcaption>',
             chart.svg("wide", title_id, legend_id), chart.svg("narrow", title_id, legend_id),
             legend(rows, legend_id)]
    if note:
        parts.append(f'<p class="fig-note">{note}</p>')
    parts.append("</figure>")
    return "\n".join(parts)


class Scatter:
    """Figure 4: TNO-FCI against the target one lead later, the straight line, the most recent
    months whose target is known, and a ring on the line for each predicted month, at its input
    month's TNO-FCI."""

    def __init__(self, fit):
        self.fit = fit
        self.xs, self.ys = fit.fci[fit.target_later.index], fit.target_later
        self.recent = self.ys.index[-12:]
        self.rings = [(fit.fci[p - fit.lead], v) for p, v in fit.prediction.items()]
        self.x_lo, self.x_hi = nice_range(fit.fci.min(), fit.fci.max(), 1)
        self.y_lo, self.y_hi = nice_range(min(self.ys.min(), fit.prediction.min()),
                                          max(self.ys.max(), fit.prediction.max()), 10)
        span = self.x_hi - self.x_lo
        self.x_step = 1 if span <= 8 else 2.5 if span <= 20 else 5

    def svg(self, size, title_id, legend_id):
        fr = Frame(size, self.x_lo, self.x_hi, self.y_lo, self.y_hi, scatter=True)
        fit, xs, ys = self.fit, self.xs, self.ys
        out = [fr.svg(title_id, legend_id)] + y_axis(fr, 10)
        for v in np.arange(np.ceil(self.x_lo / self.x_step) * self.x_step, self.x_hi + 1e-9, self.x_step):
            out.append(f'<line class="s-muted" x1="{fr.x(v):.1f}" x2="{fr.x(v):.1f}" y1="{fr.y1}" y2="{fr.y1 + 5}"/>')
            out.append(text(fr.x(v), fr.y1 + 19, f"{v:g}", anchor="middle"))
        out.append(text(fr.x1, fr.y1 + 40, f"TNO-FCI, {fit.lead} months before", anchor="end", weight=600))
        for p in ys.index:
            if p not in self.recent:
                out.append(f'<circle class="f-muted" cx="{fr.x(xs[p]):.1f}" cy="{fr.y(ys[p]):.1f}" '
                           f'r="{fr.s["dot"]}" fill-opacity=".45"/>')
        line = lambda v: fit.slope * v + fit.intercept
        out.append(f'<line class="s-accent" x1="{fr.x(self.x_lo):.1f}" y1="{fr.y(line(self.x_lo)):.1f}" '
                   f'x2="{fr.x(self.x_hi):.1f}" y2="{fr.y(line(self.x_hi)):.1f}" stroke-width="2"/>')
        for p in self.recent:
            out.append(f'<circle class="f-ink" cx="{fr.x(xs[p]):.1f}" cy="{fr.y(ys[p]):.1f}" r="{fr.s["recent"]}"/>')
        for x, y in self.rings:
            out.append(f'<circle class="f-bg s-accent" cx="{fr.x(x):.1f}" cy="{fr.y(y):.1f}" '
                       f'r="{fr.s["ring"]}" stroke-width="1.8"/>')
        out.append("</svg>")
        return "\n".join(out)


def table(rows, headers, labelled_by):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table aria-labelledby="{labelled_by}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def palette_vars(p):
    return " ".join(f"--{k}: {v};" for k, v in p.items())


def stretch_note(fit):
    """A paragraph for the stretched months, starting with the † the table links from, or ""
    when there are none."""
    beyond = fit.prediction[fit.stretched]
    if not len(beyond):
        return ""
    past = fit.shifted_seen
    one = len(beyond) == 1
    inputs = beyond.index - fit.lead
    since = (past.index[0] - fit.lead).year
    if len(beyond) <= 2:
        what = (f"The prediction for {joined(beyond.index)} rests on {joined(inputs)}," if one
                else f"The predictions for {joined(beyond.index)} rest on {joined(inputs)},")
    else:
        what = f"{len(beyond)} of these predictions rest on input months"
    return (f'<p id="stretch-note">&dagger; {what} when TNO-FCI rose higher than in any month since {since}. '
            f"The model has never seen readings that high, so {'this number is' if one else 'these numbers are'} "
            f"less reliable than the rest. Take {'it' if one else 'them'} as a sign of strong expansion rather "
            f"than {'a precise value' if one else 'precise values'}.</p>")


def prediction_table(n, title, fit):
    """Table n of the page: the fit's prediction month by month, its stretched months marked with
    a † that links to the stretch note."""
    dagger = ' <a href="#stretch-note" aria-label="see the note on stretched months">&dagger;</a>'
    marked = fit.stretched
    rows = [(month_name(p), f"{v:.1f}" + (dagger if p in marked else "")) for p, v in fit.prediction.items()]
    return (f'<figure class="table"><figcaption class="fig-title" id="table{n}-title">Table {n}. {html.escape(title)}</figcaption>'
            + table(rows, ["Month", "Predicted ISM PMI"], labelled_by=f"table{n}-title") + "</figure>")


def header_sentence(fit, name):
    """One sentence on the prediction's side of 50 and its direction. The direction is measured
    from the first predicted month without a †, or from the first month when all carry one."""
    pred = fit.prediction
    side = ("above 50 in every month" if (pred > 50).all() else
            "below 50 in every month" if (pred < 50).all() else "on both sides of 50")
    plain = pred[~pred.index.isin(fit.stretched)]
    start = (plain if len(plain) else pred).iloc[0]
    change = pred.iloc[-1] - start
    trend = "falling" if change < -1 else "rising" if change > 1 else "level"
    return (f'<p><a href="#current">The {name} prediction</a> has the ISM PMI {side} from '
            f"{month_name(pred.index[0])} to {month_name(pred.index[-1])}, {trend} over that time.</p>")


def render(july, latest, target, out_dir, written):
    """Write index.html for the July 2025 fit and the latest fit, over the target series.
    written is the date the footer says the page was written."""
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        raise StepError(f"can't write the page to {out_dir}: {e.strerror}") from e

    jp, lp = july.prediction, latest.prediction
    pred_above = jp[jp > 50]
    after = target[target.index > july.last_input]
    actual_above = after[after >= 50]
    crossing = (f"The prediction had the ISM PMI back above 50 from {month_name(pred_above.index[0])}; "
                if len(pred_above) else "The prediction kept the ISM PMI below 50; ")
    if len(actual_above):
        crossing += f"the ISM PMI itself crossed 50 in {month_name(actual_above.index[0])}."
    elif len(after):
        crossing += "the ISM PMI itself has stayed below 50 since."
    else:
        crossing += f"there is no ISM PMI after {month_name(july.last_input)} yet to compare."
    leads = sorted({july.lead, latest.lead})
    lead_text = " to ".join(str(n) for n in leads)
    made = latest.generated.to_period("M")
    two_models = july.model == "92 features" and latest.model == "five weights"
    terms = TERMS[latest.model]
    july_model = (" This prediction came from 92 features built from the five series; predictions since come from "
                  "five weights, chosen because they scored better on months the fit hadn't seen." if two_models else "")
    method = METHOD[latest.model] + lag_sentence(latest)
    if two_models:
        method += (" The July 2025 prediction came from a linear model of 92 features built from the same five "
                   "series, such as lags, momentum, moving averages, volatility and ratios; predictions since come "
                   "from five weights, chosen because they scored better on months the fit hadn't seen.")
    since = latest.shifted.index[0].year

    fifty = lambda chart: [("fifty", FIFTY)] if chart.fifty else []
    chart = LineChart([(july.shifted_known, "soft", "6 4"), (target, "ink", None), (jp, "accent", None)],
                      RECENT_START, max(target.index[-1], jp.index[-1]), band=(jp.index[0], jp.index[-1]))
    figure_july = figure(1, "The ISM PMI and the July 2025 prediction", chart, [
        [("ink", "ISM PMI"), ("dashed", f"TNO-FCI, shifted right by {july.lead} months"),
         ("accent", "The July 2025 prediction")],
        ([("band", "Period predicted in July 2025")] if chart.band else []) + fifty(chart)])
    chart = LineChart([(latest.shifted_known, "soft", "6 4"), (target, "ink", None), (lp, "accent", None)],
                      RECENT_START, lp.index[-1], marker=latest.last_input, hollow=lp[latest.stretched])
    stretch_entry = [("hollow", 'Beyond any past TNO-FCI, <a href="#stretch-note">see the note</a>')]
    figure_latest = figure(2, f"The ISM PMI and the {month_name(made)} prediction", chart, [
        [("ink", "ISM PMI"), ("dashed", f"TNO-FCI, shifted right by {latest.lead} months"),
         ("accent", f"The {month_name(made)} prediction")],
        (stretch_entry if chart.hollow is not None else [])
        + ([("marker", f"Last input month, {latest.last_input.strftime('%b %Y')}")] if chart.marker is not None else [])
        + fifty(chart)])
    seen = latest.shifted_seen
    chart = LineChart([(target, "ink", None), (seen, "accent", None)], seen.index[0], latest.last_input)
    figure_history = figure(
        3, f"TNO-FCI, shifted right by {latest.lead} months, against the ISM PMI since {since}", chart,
        [[("ink", "ISM PMI"), ("accent", f"TNO-FCI, shifted right by {latest.lead} months")], fifty(chart)])
    chart = Scatter(latest)
    most_recent = f"The {len(chart.recent)} most recent months"
    figure_scatter = figure(4, f"The ISM PMI against TNO-FCI from {latest.lead} months before", chart, [
        [("dot", "One month"), ("recent", most_recent)],
        [("accent", f"The straight line mapping TNO-FCI to the ISM PMI (in-sample R² {latest.r2:.2f})"),
         ("ring", f"The {month_name(made)} prediction, read off the line")]],
        note=(f"Each dot pairs an ISM PMI data point with the TNO-FCI data point from {latest.lead} months before "
              f"it. The line maps a month's TNO-FCI to the ISM PMI {latest.lead} months ahead. "
              f"{most_recent} show how far real months land from the line."))

    roles = ("ink", "accent", "soft", "band", "grid", "muted")
    chart_css = " ".join(f".s-{r} {{ stroke: var(--{r}); }} .f-{r} {{ fill: var(--{r}); }}" for r in roles)
    chart_css += " .f-bg { fill: var(--bg); }"  # hollow points and rings
    title = f"TNO-FCI: predicting the ISM Manufacturing PMI {lead_text} months ahead"
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="robots" content="noindex">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(DESCRIPTION)}">
<meta property="og:type" content="article">
<meta property="og:title" content="{html.escape(title)}">
<meta property="og:description" content="{html.escape(DESCRIPTION)}">
<link rel="icon" href="{FAVICON}">
<style>
  body {{ margin: 0; }}
  .page {{ {palette_vars(LIGHT)} background: var(--bg); color: var(--ink); min-height: 100vh;
    --serif: {SERIF}; --sans: {SANS}; font: 18px/1.6 var(--serif); }}
  @media (prefers-color-scheme: dark) {{
    .page {{ {palette_vars(DARK)} }}
    #theme:checked ~ .page {{ {palette_vars(LIGHT)} }}
  }}
  @media not (prefers-color-scheme: dark) {{
    #theme:checked ~ .page {{ {palette_vars(DARK)} }}
  }}
  .wrap {{ max-width: 46rem; margin: 0 auto; padding: 3.2rem 1.2rem 2rem; }}
  h1, h2, .lede {{ font-family: var(--sans); }}
  h1 {{ font-size: 1.75rem; line-height: 1.25; margin: 0 0 .5rem; letter-spacing: -.01em; }}
  h2 {{ font-size: 1.2rem; line-height: 1.35; margin: 2.8rem 0 .6rem; }}
  .lede {{ color: var(--muted); margin: 0 0 1.2rem; font-size: 1.05rem; }}
  p {{ margin: 0 0 1rem; }}
  a {{ color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 2px; }}
  a:hover {{ text-decoration-thickness: 2px; }}
  a:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 2px; }}
  figure {{ margin: 1.8rem 0 1.6rem; font-family: var(--sans); break-inside: avoid; }}
  /* The figure pattern: title, the chart at one of two sizes, legend, note. */
  .fig-title {{ color: var(--ink); font-size: .95rem; font-weight: 600; line-height: 1.4; margin-bottom: .5rem; }}
  svg.chart {{ width: 100%; height: auto; display: block; font-family: var(--sans); overflow: visible; }}
  svg.chart text {{ paint-order: stroke; stroke: var(--bg); stroke-width: 3.5px; stroke-linejoin: round; }}
  .chart-narrow {{ display: none !important; }}
  {chart_css}
  .legend {{ margin-top: .7rem; font-size: .85rem; line-height: 1.35; color: var(--muted); }}
  .legend ul {{ list-style: none; margin: 0; padding: 0; display: flex; justify-content: space-between; gap: .4rem 1.6rem; }}
  .legend ul + ul {{ margin-top: .35rem; }}
  .legend li {{ display: flex; align-items: flex-start; gap: .45rem; flex: 0 1 auto; }}
  .legend li:only-child {{ margin-right: auto; }}
  .legend li span {{ text-wrap: balance; }}
  .legend .sw {{ width: 24px; height: 12px; flex: none; overflow: visible; margin-top: .2rem; }}
  .fig-note {{ margin: .7rem 0 0; font-size: .85rem; line-height: 1.5; color: var(--muted); }}
  figure.table {{ margin: 1.2rem 0 1.6rem; }}
  table {{ border-collapse: collapse; font: .9rem/1.5 var(--sans);
    font-variant-numeric: tabular-nums; break-inside: avoid; }}
  th, td {{ padding: .25rem 1.4rem .25rem 0; text-align: left; border-bottom: 1px solid var(--grid); }}
  th {{ font-weight: 600; color: var(--muted); }}
  td:last-child, th:last-child {{ text-align: right; padding-right: 0; }}
  footer {{ max-width: 46rem; margin: 0 auto; padding: 1.2rem 1.2rem 3rem; font: .85rem/1.5 var(--sans); color: var(--muted); }}
  footer p {{ border-top: 1px solid var(--grid); padding-top: 1rem; margin: 0; }}
  #theme {{ position: fixed; top: 0; left: 0; opacity: 0; pointer-events: none; }}
  .theme-label {{ position: fixed; z-index: 10; top: .9rem; left: 1rem; background: var(--bg); cursor: pointer;
    font: .85rem/1.6 var(--sans); color: var(--muted); border: 1px solid var(--grid); border-radius: 999px;
    padding: .15rem .75rem; user-select: none; }}
  .theme-label:hover {{ color: var(--ink); border-color: var(--ink); }}
  #theme:focus-visible ~ .page .theme-label {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
  /* The toggle names the mode a click gives. */
  .to-light {{ display: none; }}
  #theme:checked ~ .page .to-dark {{ display: none; }}
  #theme:checked ~ .page .to-light {{ display: inline; }}
  @media (prefers-color-scheme: dark) {{
    .to-dark {{ display: none; }} .to-light {{ display: inline; }}
    #theme:checked ~ .page .to-dark {{ display: inline; }}
    #theme:checked ~ .page .to-light {{ display: none; }}
  }}
  .vh {{ position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }}
  .github-corner svg {{ position: fixed; z-index: 10; top: 0; right: 0; border: 0; fill: var(--ink); color: var(--bg); }}
  .github-corner:hover .octo-arm {{ animation: octo-wave 560ms ease-in-out; }}
  .github-corner:focus-visible {{ outline: none; }}
  .github-corner:focus-visible svg {{ fill: var(--accent); }}
  @keyframes octo-wave {{ 0%, 100% {{ transform: rotate(0); }} 20%, 60% {{ transform: rotate(-25deg); }} 40%, 80% {{ transform: rotate(10deg); }} }}
  @media (prefers-reduced-motion: reduce) {{ .github-corner:hover .octo-arm {{ animation: none; }} }}
  /* Phones: narrow charts, one legend entry per line, and the overlays scroll away rather than
     cover the charts. */
  @media (max-width: 560px) {{
    .page {{ font-size: 17px; }}
    .wrap {{ padding-top: 4.2rem; }}
    h1 {{ font-size: 1.45rem; }}
    .chart-wide {{ display: none !important; }}
    .chart-narrow {{ display: block !important; }}
    .legend ul {{ flex-direction: column; }}
    .theme-label {{ position: absolute; }}
    .github-corner svg {{ position: absolute; width: 56px; height: 56px; }}
  }}
  @media print {{
    .page, #theme:checked ~ .page {{ {palette_vars(LIGHT)} }}
    .theme-label, .github-corner {{ display: none; }}
    .chart-wide {{ display: block !important; }}
    .chart-narrow {{ display: none !important; }}
    .legend ul {{ flex-direction: row; }}
    .wrap {{ padding-top: 0; }}
    h2 {{ break-after: avoid; }}
  }}
</style></head><body>
<input type="checkbox" id="theme">
<div class="page">
<label for="theme" class="theme-label"><span aria-hidden="true">&#9680; </span><span class="vh">Switch to </span><span class="to-dark">Dark</span><span class="to-light">Light</span><span class="vh"> mode</span></label>
<a href="{CODE_URL}" class="github-corner" aria-label="View source on GitHub"><svg width="80" height="80" viewBox="0 0 250 250" aria-hidden="true"><path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"/><path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"/><path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"/></svg></a>
<main class="wrap">
<h1>{html.escape(title)}</h1>
<p class="lede">A financial conditions index built from five monthly market and policy series.</p>
<p>TNO-FCI is one number per month, built from the Fed funds rate, the US dollar index, gold, oil and silver. A linear model {terms} this month's value tracks the ISM Manufacturing PMI {lead_text} months later. The ISM PMI is a monthly survey of US factory activity; above 50 means expansion.</p>
{header_sentence(latest, month_name(made))}

<p>Inspired by the financial conditions index of Raoul Pal's Global Macro Investor (GMI), which GMI uses to look about nine months ahead of the ISM PMI (<a href="{INSPIRATION_URL}">The GMI Top 5 Weekly Charts That Make You Go Hmmm...</a>, August 2023).</p>

<h2>The July 2025 prediction, and what followed</h2>
<p>On {july.generated.day} {july.generated:%B %Y} the model, fitted on data through {month_name(july.last_input)}, predicted the ISM PMI for {month_name(jp.index[0])} to {month_name(jp.index[-1])}. {crossing}{july_model}</p>
{figure_july}
{prediction_table(1, "The July 2025 prediction, month by month", july)}

<h2 id="current">The {month_name(made)} prediction</h2>
<p>Fitted on {latest.generated.day} {latest.generated:%B %Y} on data through {month_name(latest.last_input)}, with a lead of {latest.lead} months, the model predicts the ISM PMI for {month_name(lp.index[0])} to {month_name(lp.index[-1])}.</p>
{figure_latest}
{prediction_table(2, f"The {month_name(made)} prediction, month by month", latest)}
{stretch_note(latest)}

<h2>How TNO-FCI tracks the ISM PMI {latest.lead} months ahead, since {since}</h2>
{figure_history}
{figure_scatter}
<p>In-sample, the fit has an R² of {latest.r2:.2f} ({july.r2:.2f} in July 2025), so read the prediction as a direction rather than a precise value.</p>

<h2>Method</h2>
<p>{method}</p>

<h2>Data and code</h2>
<p>Monthly series, mostly exported from TradingView: TVC:DXY, TVC:GOLD, TVC:USOIL and TVC:SILVER, plus ECONOMICS:USBCOI (ISM PMI) and ECONOMICS:USINTR (Fed funds rate) through June 2025. From July 2025 the ISM PMI comes from ISM's monthly releases and the Fed funds rate (upper bound of the target range) from FOMC statements. The data isn't redistributed; the <a href="{CODE_URL}">code</a> runs on demo data it generates, or on your own copies.</p>
<p>You can get each series yourself: the dollar index, gold and silver from TradingView, the ISM PMI from ISM, the Fed funds rate from FRED, and the energy prices from EIA. The data isn't offered on request, because the TradingView and ISM licences forbid it. For questions about the method or the results, write to <a href="mailto:economic.struggle220@passmail.net">economic.struggle220@passmail.net</a>.</p>
</main>
<footer><p>ISM® PMI® data: Institute for Supply Management. Page written {written.day} {written:%B %Y}.</p></footer>
</div>
</body></html>
"""
    path = os.path.join(out_dir, "index.html")
    with open(path, "w") as fh:
        fh.write(page)
    return path


def run(paths, july=None, latest=None, out="site", written=None):
    if paths.is_demo:
        raise StepError("no page is built from demo data (g04); run fci page without --demo")
    try:
        written_on = date.fromisoformat(written) if written else date.today()
    except ValueError as e:
        raise StepError(f"--written {written} isn't a date; give it as YYYY-MM-DD") from e
    load = lambda path: from_loaded(predicting.load_fit(path, paths.clean), paths.output)
    july_fit = load(july or paths.july_2025)
    latest_fit = load(latest or predicting.newest_record(paths.fits))
    if latest_fit.model == "five weights" and not latest_fit.lag_r2:  # archived runs have no lag test
        raise StepError("the latest fit records no lag test, which the page states; run fci fit again")
    target = load_clean(paths.clean)[TARGET].dropna()
    return [render(july_fit, latest_fit, target, out, written_on)]
