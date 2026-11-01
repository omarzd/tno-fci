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
        run(self.repo, "-m", "fci", "page", "--july", JULY)
        with open(os.path.join(self.repo, "site", "index.html")) as fh:
            return fh.read()

    def page_fails(self, *args):
        result = subprocess.run([sys.executable, "-m", "fci", "page", *args], cwd=self.repo,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stderr)
        return result.stderr

    def sections(self, html):
        return {m.group(1): m.group(2) for m in re.finditer(r"<h2>(.*?)</h2>(.*?)(?=<h2>|</div></div>)", html, re.S)}

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
                      "against guessing the ISM PMI's average known at the time",
                      "a miss of 2 points or less costs nothing", "one month early or late"):
            with self.subTest(words=words):
                self.assertIn(words, method)
        self.assertNotIn("predicts the next month", method)

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

    def test_shows_four_inline_charts_and_no_script(self):
        html = self.page()
        self.assertEqual(html.count('<svg class="chart"'), 4)
        self.assertNotIn("<script", html.lower())

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

    def test_demo_data_gets_no_page(self):
        self.assertIn("demo", self.page_fails("--demo"))


if __name__ == "__main__":
    unittest.main()
