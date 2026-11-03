"""Seam 1 (g01 spec #2), ported: `fci page` builds the page from earlier steps' files only.

In a temporary copy of the repo: a made-up archived July 2025 run with 92 features, and a
five-weight fit record from the demo data, each with its prediction file from `fci predict`.
The demo data never gets a page of its own (g04); here it only stands in for a real fit.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = shutil.ignore_patterns(".git", ".venv", "venv", "output", "site", "demo", "__pycache__")
JULY = os.path.join("output", "archived_runs", "july")
JULY_LEAD = 9
WRITTEN = "2026-03-02"


def run(cwd, *args):
    return subprocess.run([sys.executable, *args], cwd=cwd, check=True, capture_output=True, text=True)


def write_archived_run(folder, first_month, fci, slope, intercept, method="linear", generated="2025-07-26 15:56:44"):
    """A run folder as the old 3_tno_fci_calculator.py left it: results plus its text report."""
    os.makedirs(folder)
    dates = pd.period_range(first_month, periods=len(fci), freq="M").to_timestamp(how="end").normalize()
    fwd = [intercept + slope * v for v in fci]
    fwd[-JULY_LEAD:] = [None] * JULY_LEAD
    pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "TNO_FCI": fci,
                  f"ISM_{JULY_LEAD}m_Forward": fwd}).to_csv(os.path.join(folder, "fci_results.csv"), index=False)
    with open(os.path.join(folder, "model_params.txt"), "w") as fh:
        fh.write(f"Final Selected Method: {method}\nR-squared: 0.43\nGenerated on: {generated}\n")


class FciPageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = os.path.join(self.tmp, "repo")
        shutil.copytree(ROOT, self.repo, ignore=SKIP)
        p = lambda *parts: os.path.join(self.repo, *parts)
        # The five-weight fit: demo data, moved to where a real fit's files sit.
        run(self.repo, "scripts/make_demo_data.py")
        run(self.repo, "-m", "fci", "prep", "--demo")
        run(self.repo, "-m", "fci", "fit", "--demo")
        shutil.copytree(p("demo", "output"), p("output"))
        (self.record_name,) = os.listdir(p("output", "fits"))
        with open(p("output", "fits", self.record_name)) as fh:
            self.record = json.load(fh)
        # The July 2025 run: inputs through 2023-06, inside the demo target's years.
        write_archived_run(p(JULY), "2021-01", [((i % 7) - 3) / 2 for i in range(30)], 2.0, 50.0)
        run(self.repo, "-m", "fci", "predict", "--fit", JULY)
        run(self.repo, "-m", "fci", "predict")
        self.july_pred = pd.read_csv(p("output", "predictions", "july.csv"))["ism_pmi"]
        self.latest_pred = pd.read_csv(p("output", "predictions", self.record_name[:-5] + ".csv"))["ism_pmi"]

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def page(self):
        run(self.repo, "-m", "fci", "page", "--july", JULY, "--written", WRITTEN)
        with open(os.path.join(self.repo, "site", "index.html")) as fh:
            return fh.read()

    def page_fails(self, *args):
        result = subprocess.run([sys.executable, "-m", "fci", "page", *args], cwd=self.repo,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stderr)
        return result.stderr

    def sections(self, html):
        return {m.group(1): m.group(2) for m in re.finditer(r"<h2[^>]*>(.*?)</h2>(.*?)(?=<h2|</main>)", html, re.S)}

    def test_states_both_predictions_from_their_prediction_files(self):
        html = self.page()
        for v in (self.july_pred.iloc[0], self.july_pred.iloc[-1], self.latest_pred.iloc[0], self.latest_pred.iloc[-1]):
            self.assertIn(f"<td>{v:.1f}</td>", html)

    def test_states_the_fit_date_and_lead_it_used(self):
        html = self.page()
        fitted = pd.Timestamp(self.record["fitted_at"])
        self.assertIn(f"{fitted.day} {fitted:%B %Y}", html)
        self.assertIn(f"a lead of {self.record['lead']} months", html)

    def test_names_both_models_in_the_july_section_and_in_method(self):
        sections = self.sections(self.page())
        july = next(text for title, text in sections.items() if title.startswith("The July 2025 prediction"))
        for text in (july, sections["Method"]):
            self.assertIn("92 features", text)
            self.assertIn("five weights", text)

    def set_lag_r2(self, values):
        """Rewrite the record's lag figures, or drop them when values is None."""
        path = os.path.join(self.repo, "output", "fits", self.record_name)
        record = dict(self.record)
        if values is None:
            del record["lag_in_sample_r2"]
        else:
            record["lag_in_sample_r2"] = {str(lag): v for lag, v in zip(range(1, 13), values)}
        with open(path, "w") as fh:
            json.dump(record, fh)

    def test_method_states_the_largest_lag_r2_from_the_record(self):
        # 0.032 rounds up, so "or less" stays true
        self.set_lag_r2([0.01, 0.0, 0.03, 0.02, 0.032, 0.01, 0.0, 0.0, 0.01, 0.02, 0.01, 0.0])
        method = self.sections(self.page())["Method"]
        self.assertIn("Shifted the other way, as if it lagged the ISM PMI, the fit explains almost nothing "
                      "(R² 0.04 or less at every lag from 1 to 12 months).", method)

    def test_method_describes_how_the_lead_search_grades_a_five_weight_fit(self):
        method = self.sections(self.page())["Method"]
        for words in ("predicts every month of the lead's path", "the first and last months weigh most",
                      "scores each path against the mean forecast, the ISM PMI's average known at the time",
                      "In that score", "keeps the lead with the best score",
                      "a miss of 2 points or less costs nothing", "one month early or late"):
            with self.subTest(words=words):
                self.assertIn(words, method)
        self.assertNotIn("predicts the next month", method)
        self.assertNotIn("guess", method)

    def test_the_july_section_is_the_same_whatever_the_lag_figures(self):
        sections = []
        for top in (0.01, 0.2):
            self.set_lag_r2([0.0] * 11 + [top])
            sections.append(self.sections(self.page()))
        july = [next(text for title, text in s.items() if title.startswith("The July 2025 prediction"))
                for s in sections]
        self.assertEqual(july[0], july[1])
        self.assertNotEqual(sections[0]["Method"], sections[1]["Method"])

    def test_a_record_without_lag_figures_says_to_run_fit(self):
        self.set_lag_r2(None)
        self.assertIn("fci fit", self.page_fails("--july", JULY))

    def test_a_record_with_broken_lag_figures_is_an_error(self):
        self.set_lag_r2([None] * 12)
        self.assertIn("fci fit", self.page_fails("--july", JULY))

    def test_shows_four_figures_and_no_script(self):
        html = self.page()
        self.assertEqual(len(re.findall(r'<figure(?! class="table")[ >]', html)), 4)
        self.assertNotIn("<script", html.lower())

    def figure(self, html, n):
        return re.search(rf'<figure id="figure-{n}">(.*?)</figure>', html, re.S).group(1)

    def charts(self, figure):
        """The figure's wide and narrow SVGs, by size."""
        return {m.group(1): m.group(0) for m in
                re.finditer(r'<svg class="chart chart-(wide|narrow)"[^>]*>.*?</svg>', figure, re.S)}

    def legend(self, figure):
        """The legend's rows, each a list of its entries' text."""
        legend = re.search(r'<div class="legend" id="[^"]+">(.*?)</div>', figure, re.S).group(1)
        return [[re.sub(r"<[^>]+>", "", li) for li in re.findall(r"<li>(.*?)</li>", ul, re.S)]
                for ul in re.findall(r"<ul>(.*?)</ul>", legend, re.S)]

    def set_last_inputs(self, stretched):
        """Rewrite the clean inputs over the latest fit's last lead months of input, and its
        prediction file: TNO-FCI 0 there, so no predicted month is stretched, or with stretched,
        far above any past TNO-FCI in the last input month only. Returns that month."""
        from fci.fit import INPUTS
        through, lead = pd.Period(self.record["inputs_through"], "M"), self.record["lead"]
        for name in INPUTS:
            path = os.path.join(self.repo, "output", "clean", f"{name}.csv")
            df = pd.read_csv(path)
            month = pd.PeriodIndex(df["month"], freq="M")
            norm, weight = self.record["normalization"][name], self.record["weights"][name]
            df.loc[(month > through - lead) & (month <= through), name] = norm["mean"]
            if stretched:
                df.loc[month == through, name] = norm["mean"] + 10 * norm["std"] * (1 if weight > 0 else -1)
            df.to_csv(path, index=False)
        run(self.repo, "-m", "fci", "predict")
        return through

    def titles(self):
        from fci.fit import INPUTS, load_clean
        lead = self.record["lead"]
        first = load_clean(os.path.join(self.repo, "output", "clean"))[INPUTS].dropna().index[0]
        return {1: "The ISM PMI and the July 2025 prediction",
                2: f"The ISM PMI and the {pd.Timestamp(self.record['fitted_at']):%B %Y} prediction",
                3: f"TNO-FCI, shifted right by {lead} months, against the ISM PMI since {(first + lead).year}",
                4: f"The ISM PMI against TNO-FCI from {lead} months before"}

    def test_each_figure_has_a_numbered_title_two_sizes_and_a_legend(self):
        html = self.page()
        for n, title in self.titles().items():
            with self.subTest(figure=n):
                figure = self.figure(html, n)
                self.assertEqual(re.findall(r"<figcaption[^>]*>.*?</figcaption>", figure, re.S),
                                 [f'<figcaption class="fig-title" id="fig{n}-title">Figure {n}. {title}</figcaption>'])
                charts = self.charts(figure)
                self.assertRegex(charts["wide"], r'^<svg [^>]*viewBox="0 0 736 ')
                self.assertRegex(charts["narrow"], r'^<svg [^>]*viewBox="0 0 360 ')
                for chart in charts.values():
                    self.assertIn(f'aria-labelledby="fig{n}-title" aria-describedby="fig{n}-legend"', chart)
                self.assertIn(f'<div class="legend" id="fig{n}-legend">', figure)
                parts = (r'<figcaption|<svg class="chart chart-wide"|<svg class="chart chart-narrow"|<div class="legend"'
                         r'|<p class="fig-note"')
                self.assertEqual(re.findall(parts, figure), ["<figcaption", '<svg class="chart chart-wide"',
                                                             '<svg class="chart chart-narrow"', '<div class="legend"']
                                 + (['<p class="fig-note"'] if n == 4 else []))

    def test_figures_hold_no_text_but_ticks_and_axis_titles(self):
        html = self.page()
        x_title = f"TNO-FCI, {self.record['lead']} months before"
        for n in (1, 2, 3, 4):
            for size, chart in self.charts(self.figure(html, n)).items():
                with self.subTest(figure=n, size=size):
                    self.assertNotIn("<title", chart)
                    texts = re.findall(r"<text[^>]*>(.*?)</text>", chart)
                    titles = ["ISM PMI"] + ([x_title] if n == 4 else [])
                    for title in titles:
                        self.assertEqual(texts.count(title), 1)
                    self.assertEqual([t for t in texts if t not in titles and not re.fullmatch(r"-?\d+(\.\d+)?", t)], [])

    def test_figure_4_rings_each_predicted_month_on_the_line_and_draws_them_last(self):
        figure = self.figure(self.page(), 4)
        for size, chart in self.charts(figure).items():
            with self.subTest(size=size):
                line = re.search(r'<line class="s-accent" x1="([\d.]+)" y1="([\d.]+)" x2="([\d.]+)" y2="([\d.]+)"', chart)
                x1, y1, x2, y2 = map(float, line.groups())
                rings = [(float(x), float(y)) for x, y in
                         re.findall(r'<circle class="f-bg s-accent" cx="([\d.]+)" cy="([\d.]+)"', chart)]
                self.assertEqual(len(rings), len(self.latest_pred))
                for x, y in rings:
                    self.assertAlmostEqual(y, y1 + (y2 - y1) * (x - x1) / (x2 - x1), delta=0.2)
                last = chart[chart.index('<circle class="f-bg s-accent"'):]
                self.assertEqual(re.findall(r"<(\w+)", last), ["circle"] * len(rings))

    def test_figure_4_legend_and_note_name_its_marks_and_r2_in_sample(self):
        figure = self.figure(self.page(), 4)
        lead, made = self.record["lead"], f"{pd.Timestamp(self.record['fitted_at']):%B %Y}"
        self.assertEqual(self.legend(figure), [
            ["One month", "The 12 most recent months"],
            [f"The model, mapping TNO&#8209;FCI to the ISM PMI (in-sample R² {self.record['in_sample_r2']:.2f})",
             f"The {made} prediction, read off the line"]])
        note = re.search(r'<p class="fig-note">(.*?)</p>', figure, re.S).group(1)
        self.assertEqual(note, f"Each dot pairs an ISM PMI data point with the TNO-FCI data point from {lead} months "
                         f"before it. The line is the model. It maps a month's TNO-FCI to the ISM PMI {lead} months "
                         "ahead. The 12 most recent months show how far real months land from the line.")
        self.assertNotIn("R-squared", figure)

    def test_each_legend_names_the_lines_drawn(self):
        html = self.page()
        lead = self.record["lead"]
        made = f"{pd.Timestamp(self.record['fitted_at']):%B %Y}"
        rows = {1: ["ISM PMI", f"TNO&#8209;FCI, shifted right by {JULY_LEAD} months", "The July 2025 prediction"],
                2: ["ISM PMI", f"TNO&#8209;FCI, shifted right by {lead} months", f"The {made} prediction"],
                3: ["ISM PMI", f"TNO&#8209;FCI, shifted right by {lead} months"]}
        for n, row in rows.items():
            with self.subTest(figure=n):
                self.assertEqual(self.legend(self.figure(html, n))[0], row)

    def test_a_legend_names_a_background_mark_only_when_it_is_drawn(self):
        html = self.page()
        through = pd.Period(self.record["inputs_through"], "M")
        marks = [  # entry, how the mark is drawn
            ("Period predicted in July 2025", '<rect class="f-band"'),
            (f"Last input month, {through.strftime('%b %Y')}",'stroke-dasharray="4 4"'),
            ("50: above is expansion, below is contraction", 'stroke-dasharray="2 3"'),
            ("Beyond any past TNO&#8209;FCI, see the note", '<circle class="f-bg s-accent"'),
        ]
        drawn = {1: {0, 2}, 2: {1, 2}, 3: {2}}  # with the demo data, 50 is inside every range
        for n, expected in drawn.items():
            figure = self.figure(html, n)
            entries = [e for row in self.legend(figure)[1:] for e in row]
            for i, (entry, mark) in enumerate(marks):
                with self.subTest(figure=n, entry=entry):
                    for chart in self.charts(figure).values():
                        self.assertEqual(mark in chart, i in expected)
                    self.assertEqual(entry in entries, i in expected)

    def test_the_line_at_50_and_its_entry_go_where_50_is_out_of_range(self):
        # A July 2025 run around 80, and the ISM PMI 30 points higher from Figure 1's first month,
        # so only Figure 1's range leaves out 50.
        high = os.path.join("output", "archived_runs", "high")
        write_archived_run(os.path.join(self.repo, high), "2021-01", [((i % 7) - 3) / 2 for i in range(30)], 2.0, 80.0)
        run(self.repo, "-m", "fci", "predict", "--fit", high)
        path = os.path.join(self.repo, "output", "clean", "ism_pmi.csv")
        df = pd.read_csv(path)
        df.loc[df["month"] >= "2022-07", "ism_pmi"] += 30
        df.to_csv(path, index=False)
        run(self.repo, "-m", "fci", "page", "--july", high, "--written", WRITTEN)
        with open(os.path.join(self.repo, "site", "index.html")) as fh:
            html = fh.read()
        for n, drawn in ((1, False), (2, True), (3, True)):
            figure = self.figure(html, n)
            with self.subTest(figure=n):
                for chart in self.charts(figure).values():
                    self.assertEqual('stroke-dasharray="2 3"' in chart, drawn)
                entries = [e for row in self.legend(figure) for e in row]
                self.assertEqual("50: above is expansion, below is contraction" in entries, drawn)

    def test_hollow_points_mark_the_stretched_months_in_figure_2(self):
        month = self.set_last_inputs(stretched=True)
        figure = self.figure(self.page(), 2)
        for chart in self.charts(figure).values():
            self.assertEqual(chart.count('<circle class="f-bg s-accent"'), 1)
        self.assertIn('Beyond any past TNO&#8209;FCI, <a href="#stretch-note">see the note</a>', figure)
        self.assertEqual(self.legend(figure)[1],
                         ["Beyond any past TNO&#8209;FCI, see the note", f"Last input month, {month.strftime('%b %Y')}",
                          "50: above is expansion, below is contraction"])

    def test_without_a_stretched_month_figure_2_has_no_hollow_points(self):
        self.set_last_inputs(stretched=False)
        figure = self.figure(self.page(), 2)
        self.assertNotIn("<circle", figure.split('<div class="legend"')[0])
        self.assertNotIn("Beyond any past", figure)

    def test_links_only_to_the_code_the_inspiration_and_the_mailto(self):
        html = self.page()
        urls = {u for u in re.findall(r'(?:href|src)="([^"]+)"', html) if not u.startswith(("data:", "#"))}
        self.assertEqual(urls, {"https://github.com/omarzd/fci",
                                "https://raoulpal.substack.com/p/the-gmi-top-5-weekly-charts-that-e8c",
                                "mailto:economic.struggle220@passmail.net"})
        self.assertNotIn("url(", html)
        self.assertNotIn("@import", html)

    def test_a_missing_prediction_file_says_to_run_predict(self):
        os.remove(os.path.join(self.repo, "output", "predictions", "july.csv"))
        self.assertIn("fci predict", self.page_fails("--july", JULY))

    def test_a_prediction_file_that_disagrees_with_its_fit_is_an_error(self):
        path = os.path.join(self.repo, "output", "predictions", "july.csv")
        df = pd.read_csv(path)
        df.loc[0, "ism_pmi"] += 1
        df.to_csv(path, index=False)
        self.assertIn("fci predict", self.page_fails("--july", JULY))

    def test_a_five_series_run_is_described_as_five_series(self):
        plain = os.path.join("output", "archived_runs", "plain")
        write_archived_run(os.path.join(self.repo, plain), "2021-01", [((i % 5) - 2) / 4 for i in range(30)],
                           2.8, 49.0, method="baseline", generated="2023-07-02 10:00:00")
        run(self.repo, "-m", "fci", "predict", "--fit", plain)
        run(self.repo, "-m", "fci", "page", "--july", JULY, "--latest", plain)
        with open(os.path.join(self.repo, "site", "index.html")) as fh:
            html = fh.read()
        self.assertIn("weighs those five series", html)
        self.assertNotIn("momentum", self.sections(html)["Method"])
        self.assertNotIn("five weights", html)

    def test_a_july_fit_from_another_month_says_its_own_month(self):
        other = os.path.join("output", "archived_runs", "other")
        write_archived_run(os.path.join(self.repo, other), "2021-01", [((i % 7) - 3) / 2 for i in range(30)],
                           2.0, 50.0, generated="2023-08-05 10:00:00")
        run(self.repo, "-m", "fci", "predict", "--fit", other)
        run(self.repo, "-m", "fci", "page", "--july", other)
        with open(os.path.join(self.repo, "site", "index.html")) as fh:
            self.assertIn("On 5 August 2023 the model", fh.read())

    def test_a_target_ending_before_the_july_prediction_is_said_so(self):
        path = os.path.join(self.repo, "output", "clean", "ism_pmi.csv")
        df = pd.read_csv(path)
        df[df["month"] <= "2023-06"].to_csv(path, index=False)
        html = self.page()
        self.assertNotIn("has stayed below 50", html)
        self.assertIn("no ISM PMI after", html)

    def test_a_broken_prediction_file_is_an_error(self):
        path = os.path.join(self.repo, "output", "predictions", "july.csv")
        for broken in ("", "month,other\n2023-07,1\n", "month,ism_pmi\nbad,1\n"):
            with open(path, "w") as fh:
                fh.write(broken)
            self.assertIn("fci predict", self.page_fails("--july", JULY))

    def test_an_out_path_that_is_a_file_is_an_error(self):
        with open(os.path.join(self.repo, "taken"), "w") as fh:
            fh.write("x")
        self.assertIn("taken", self.page_fails("--july", JULY, "--out", "taken"))

    def test_data_and_code_names_where_each_series_can_be_had_and_offers_a_contact(self):
        section = self.sections(self.page())["Data and code"]
        text = re.sub(r"<[^>]+>", "", section)
        for provider in ("TradingView", "ISM", "FRED", "EIA"):
            self.assertIn(provider, text)
        self.assertIn("isn't offered on request", text)
        self.assertIn('<a href="mailto:economic.struggle220@passmail.net">', section)

    def test_the_footer_credits_ism_and_says_when_the_page_was_written(self):
        html = self.page()
        footer = re.search(r"<footer>(.*?)</footer>", html, re.S).group(1)
        self.assertEqual(re.sub(r"<[^>]+>", "", footer).strip(),
                         "ISM® PMI® data: Institute for Supply Management. Page written 2 March 2026.")
        self.assertEqual(html.count("ISM® PMI® data"), 1)

    def test_the_head_describes_the_page_for_link_previews(self):
        head = self.page().split("</head>")[0]
        self.assertIn('<meta name="robots" content="noindex">', head)
        title = re.search(r"<title>(.*?)</title>", head).group(1)
        description = re.search(r'<meta name="description" content="([^"]+)">', head).group(1)
        self.assertIn("ISM Manufacturing PMI", description)
        self.assertIn('<meta property="og:type" content="article">', head)
        self.assertIn(f'<meta property="og:title" content="{title}">', head)
        self.assertIn(f'<meta property="og:description" content="{description}">', head)
        self.assertRegex(head, r'<link rel="icon" href="data:image/svg\+xml,[^"]+">')

    def test_the_toggle_names_the_mode_a_click_gives(self):
        html = self.page()
        label = re.search(r'<label for="theme"[^>]*>(.*?)</label>', html, re.S).group(1)
        text = lambda markup: re.sub(r"<[^>]+>", "", markup)
        for mode, other in (("Dark", "light"), ("Light", "dark")):
            with self.subTest(mode=mode):
                # CSS hides the other mode's word, so a reader sees one of the two.
                shown = re.sub(rf'<span class="to-{other}">.*?</span>', "", label)
                self.assertEqual(text(re.sub(r'<span class="vh">.*?</span>', "", shown)), f"&#9680; {mode}")
                self.assertEqual(text(re.sub(r'<span aria-hidden="true">.*?</span>', "", shown)),
                                 f"Switch to {mode} mode")
        self.assertNotRegex(html, r'<input[^>]*id="theme"[^>]*aria-label')

    def current_page(self, tail, intercept=51.0, slope=2.0):
        """The page whose latest fit is a made-up run: 21 input months in a fixed range, then
        `tail`, the nine inputs that make its prediction (July 2023 to March 2024)."""
        shutil.rmtree(os.path.join(self.repo, "output", "archived_runs", "current"), ignore_errors=True)
        run_dir = os.path.join("output", "archived_runs", "current")
        fci = [((i % 7) - 3) / 2 for i in range(21)] + list(tail)
        write_archived_run(os.path.join(self.repo, run_dir), "2021-01", fci, slope, intercept,
                           generated="2023-07-02 10:00:00")
        run(self.repo, "-m", "fci", "predict", "--fit", run_dir)
        run(self.repo, "-m", "fci", "page", "--july", JULY, "--latest", run_dir, "--written", WRITTEN)
        with open(os.path.join(self.repo, "site", "index.html")) as fh:
            return fh.read()

    def header_sentence(self, html):
        link = re.search(r'<p>(<a href="#current">The July 2023 prediction</a> has the ISM PMI .*?\.)</p>', html)
        self.assertIsNotNone(link, "no header sentence")
        return re.sub(r"<[^>]+>", "", link.group(1))

    FALLING = [1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0]
    DAGGER_FIRST = [5.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    def test_the_tables_are_numbered_and_titled(self):
        html = self.page()
        made = pd.Timestamp(self.record["fitted_at"]).strftime("%B %Y")
        figures = re.findall(r'<figure class="table">(.*?)</figure>', html, re.S)
        self.assertEqual(len(figures), 2)
        for n, title, figure in zip((1, 2), ("The July 2025 prediction, month by month",
                                             f"The {made} prediction, month by month"), figures):
            with self.subTest(table=n):
                caption = re.search(r'<figcaption[^>]*id="([^"]+)"[^>]*>(.*?)</figcaption>', figure)
                self.assertEqual(caption.group(2), f"Table {n}. {title}")
                self.assertRegex(figure, rf'<table aria-labelledby="{caption.group(1)}">')

    def test_a_table_title_takes_the_figure_title_style(self):
        figures = re.findall(r'<figure class="table">(.*?)</figure>', self.page(), re.S)
        for n, figure in enumerate(figures, 1):
            with self.subTest(table=n):
                self.assertEqual(re.findall(r"<figcaption[^>]*>", figure),
                                 [f'<figcaption class="fig-title" id="table{n}-title">'])

    def test_a_table_wrapper_carries_the_margin_not_the_table(self):
        css = re.search(r"<style>(.*?)</style>", self.page(), re.S).group(1)
        self.assertRegex(css, r"figure\.table \{[^}]*margin")
        self.assertNotRegex(css, r"(?m)^\s*table \{[^}]*margin")

    def test_a_stretched_month_is_marked_in_table_2_and_the_note(self):
        html = self.current_page(self.DAGGER_FIRST)
        table2 = re.findall(r'<figure class="table">(.*?)</figure>', html, re.S)[1]
        self.assertEqual(table2.count("&dagger;"), 1)
        self.assertEqual(table2.count('<a href="#stretch-note" aria-label="see the note on stretched months">'), 1)
        first_row = re.search(r"<tr><td>July 2023</td><td>(.*?)</td></tr>", table2).group(1)
        self.assertRegex(first_row, r"^61\.0 <a href=\"#stretch-note\" aria-label=\"see the note on stretched months\">&dagger;</a>$")
        note = re.search(r'<p id="stretch-note">(.*?)</p>', html, re.S).group(1)
        self.assertTrue(note.startswith("&dagger; "))

    def test_without_a_stretched_month_there_is_no_dagger_and_no_note(self):
        html = self.current_page(self.FALLING)
        self.assertNotIn("&dagger;", html)
        self.assertNotIn("stretch-note", html)

    def test_the_header_sentence_gives_the_side_of_50_and_the_trend_and_links_to_the_section(self):
        html = self.current_page(self.FALLING)
        self.assertEqual(self.header_sentence(html), "The July 2023 prediction has the ISM PMI above 50 in every "
                         "month from July 2023 to March 2024, falling over that time.")
        self.assertIn('<h2 id="current">The July 2023 prediction</h2>', html)
        # after the paragraph that explains 50, before the inspiration paragraph
        self.assertLess(html.index("above 50 means expansion.</p>"), html.index('<a href="#current">'))
        self.assertLess(html.index('<a href="#current">'), html.index("Inspired by"))

    def test_the_header_sentence_measures_the_trend_from_the_first_month_without_a_dagger(self):
        # 61.0 (stretched), 51.0 ... 52.2: falling from the first month, rising from the second
        html = self.current_page(self.DAGGER_FIRST)
        self.assertTrue(self.header_sentence(html).endswith("rising over that time."))

    def test_with_every_month_stretched_the_trend_is_measured_from_the_first(self):
        html = self.current_page([5.0] * 8 + [3.0])  # 61.0 falling to 57.0, every month a dagger
        self.assertEqual(re.findall(r"&dagger;", html).count("&dagger;"), 9 + 1)
        self.assertTrue(self.header_sentence(html).endswith("falling over that time."))

    def test_the_header_sentence_says_below_or_both_sides_and_level(self):
        for tail, intercept, side, trend in (([1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0], 45.0, "below 50 in every month", "falling"),
                                             ([0.2, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.2], 50.0, "on both sides of 50", "level")):
            with self.subTest(side=side):
                html = self.current_page(tail, intercept)
                self.assertEqual(self.header_sentence(html), f"The July 2023 prediction has the ISM PMI {side} "
                                 f"from July 2023 to March 2024, {trend} over that time.")

    def test_the_headings_and_the_r2_sentence_read_as_ruled(self):
        html = self.page()
        titles = list(self.sections(html))
        lead = self.record["lead"]
        (tracks,) = [t for t in titles if t.startswith("How TNO-FCI tracks")]
        self.assertRegex(tracks, rf"^How TNO-FCI tracks the ISM PMI {lead} months ahead, since \d{{4}}$")
        self.assertNotIn("How it tracks", html)
        text = re.sub(r"<[^>]+>", "", self.sections(html)[tracks])
        self.assertIn("so read the prediction as a direction rather than a precise value.", text)
        self.assertNotIn("guess", html.lower())

    def test_a_written_date_that_isnt_a_date_is_an_error(self):
        self.assertIn("YYYY-MM-DD", self.page_fails("--july", JULY, "--written", "March"))

    def test_demo_data_gets_no_page(self):
        self.assertIn("demo", self.page_fails("--demo"))


if __name__ == "__main__":
    unittest.main()
