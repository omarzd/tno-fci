"""Seam 1 (g01 spec #2): the page step, fed a July 2025 fit and a later fit.

Each synthetic fit's target is an exact straight line of its TNO-FCI, so every number the
page should state is known in advance.
"""
import importlib.util
import os
import re
import shutil
import tempfile
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("build_page", os.path.join(ROOT, "5_build_page.py"))
build_page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_page)

LEAD = 9


def write_fit(folder, first_month, fci, slope, intercept, r2, generated):
    """A fit folder as 3_fit_model.py leaves it: results plus model parameters."""
    os.makedirs(folder)
    dates = pd.period_range(first_month, periods=len(fci), freq="M").to_timestamp(how="end").normalize()
    fwd = [intercept + slope * v for v in fci]
    fwd[-LEAD:] = [None] * LEAD
    pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "TNO_FCI": fci, "ISM_PMI": 50.0,
                  f"ISM_{LEAD}m_Forward": fwd}).to_csv(os.path.join(folder, "fci_results.csv"), index=False)
    with open(os.path.join(folder, "model_params.txt"), "w") as fh:
        fh.write(f"- Lag period: {LEAD} months (fixed)\nR-squared: {r2}\nGenerated on: {generated}\n")


class BuildPageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        t = lambda *p: os.path.join(self.tmp, *p)
        # July 2025 fit: inputs through 2025-06, so it predicts 2025-07 .. 2026-03.
        july_fci = [((i % 7) - 3) / 2 for i in range(30)]
        write_fit(t("july"), "2023-01", july_fci, 2.0, 50.0, 0.33, "2025-07-27 09:57:16")
        self.july_pred = [50.0 + 2.0 * v for v in july_fci[-LEAD:]]
        # October 2026 fit: inputs through 2026-09, so it predicts 2026-10 .. 2027-06.
        oct_fci = [((i % 5) - 2) / 4 for i in range(45)]
        write_fit(t("october"), "2023-01", oct_fci, 3.0, 49.0, 0.33, "2026-10-13 00:31:16")
        self.oct_pred = [49.0 + 3.0 * v for v in oct_fci[-LEAD:]]
        # Actual target: the July 2025 prediction misses by exactly 1.5 points every month.
        months = pd.period_range("2023-01", "2026-09", freq="M")
        actual = {m: 52.0 for m in months}
        for m, p in zip(pd.period_range("2025-07", periods=LEAD, freq="M"), self.july_pred):
            actual[m] = p + 1.5
        pd.DataFrame({"Date": [m.to_timestamp().strftime("%Y-%m-%d") for m in months],
                      "ISM_PMI": [actual[m] for m in months]}).to_csv(t("ismpmi_clean.csv"), index=False)
        self.out = t("site")
        self.html_path = build_page.build_page(t("july"), t("october"), t("ismpmi_clean.csv"), self.out)
        with open(self.html_path) as fh:
            self.html = fh.read()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_states_the_july_2025_prediction(self):
        for v in (self.july_pred[0], self.july_pred[-1]):
            self.assertIn(f"{v:.1f}", self.html)
        self.assertIn("27 July 2025", self.html)

    def test_states_when_prediction_and_pmi_crossed_50(self):
        self.assertIn("back above 50 from", self.html)
        self.assertRegex(self.html, r"crossed 50 in|has not crossed 50")

    def test_states_the_october_2026_prediction(self):
        for v in (self.oct_pred[0], self.oct_pred[-1]):
            self.assertIn(f"<td>{v:.1f}</td>", self.html)
        self.assertIn("June 2027", self.html)

    def test_shows_four_inline_charts(self):
        self.assertEqual(self.html.count('<svg class="chart"'), 4)

    def test_fitted_line_meets_the_prediction(self):
        for chart in re.findall(r'<svg class="chart".*?</svg>', self.html, re.S):
            dashed = re.search(r'class="s-soft" points="([^"]+)"', chart)
            solid = re.search(r'class="s-accent" points="([^"]+)"', chart)
            if dashed and solid:
                self.assertEqual(dashed.group(1).split()[-1], solid.group(1).split()[0])

    def test_links_the_code_from_a_corner_banner(self):
        self.assertRegex(self.html, r'<a href="' + re.escape(build_page.CODE_URL) + r'" class="github-corner"')

    def test_dark_mode_follows_the_system_and_toggles(self):
        self.assertIn("prefers-color-scheme: dark", self.html)
        self.assertIn('<input type="checkbox" id="theme"', self.html)
        self.assertIn('<label for="theme"', self.html)
        self.assertIn("#theme:checked ~ .page", self.html)

    def test_page_ships_no_series_data(self):
        self.assertNotIn("<script", self.html.lower())
        for src in re.findall(r'src="([^"]+)"', self.html):
            self.assertRegex(src, r"\.(svg|png)$")


if __name__ == "__main__":
    unittest.main()
