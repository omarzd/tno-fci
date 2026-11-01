"""Seam 2 (g01 spec #2): `fci fit` on the demo data, in a temporary copy of the repo.

Demo data with a planted lead goes in; the walk-forward lead search must find that lead, and
each fit must leave a record that predict can use. Each test runs in seconds.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = shutil.ignore_patterns(".git", ".venv", "venv", "output", "site", "demo", "__pycache__")
INPUTS = ["fed_rate", "dxy", "gold", "oil", "silver"]


def run(cwd, *args):
    return subprocess.run([sys.executable, *args], cwd=cwd, check=True, capture_output=True, text=True)


class FciFitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = os.path.join(self.tmp, "repo")
        shutil.copytree(ROOT, self.repo, ignore=SKIP)
        self.clean = os.path.join(self.repo, "demo", "output", "clean")
        self.fits = os.path.join(self.repo, "demo", "output", "fits")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def prep(self, *demo_args):
        run(self.repo, "scripts/make_demo_data.py", *demo_args)
        run(self.repo, "-m", "fci", "prep", "--demo")

    def fit(self):
        return run(self.repo, "-m", "fci", "fit", "--demo").stdout

    def records(self):
        return sorted(os.listdir(self.fits))

    def newest_record(self):
        with open(os.path.join(self.fits, self.records()[-1])) as fh:
            return json.load(fh)

    def test_finds_the_planted_lead(self):
        for planted in (10, 12):  # 12 too, so the demo's default isn't the only lead ever checked
            with self.subTest(planted=planted):
                self.prep("--lead", str(planted))
                out = self.fit()
                record = self.newest_record()
                self.assertEqual(record["lead"], planted)
                self.assertEqual(record["lead_chosen_by"], "walk-forward")
                self.assertIn(f"lead {planted} months, out-of-sample score {record['out_of_sample_score']:.3f}", out)

    def test_the_record_holds_date_weights_normalization_lead_and_score(self):
        self.prep()
        self.fit()
        record = self.newest_record()
        datetime.fromisoformat(record["fitted_at"])
        self.assertEqual(record["inputs_through"], "2024-12")  # the demo data's last month
        self.assertEqual(record["target_through"], "2024-12")
        self.assertEqual(sorted(record["weights"]), sorted(INPUTS))
        self.assertEqual(sorted(record["normalization"]), sorted(INPUTS))
        for norm in record["normalization"].values():
            self.assertEqual(sorted(norm), ["mean", "std"])
            self.assertGreater(norm["std"], 0)
        self.assertEqual(sorted(record["line"]), ["intercept", "slope"])
        self.assertEqual(sorted(int(k) for k in record["lead_scores"]), list(range(9, 19)))
        self.assertNotIn("out_of_sample_r2", record)
        self.assertEqual(record["out_of_sample_score"], record["lead_scores"][str(record["lead"])])
        self.assertEqual(record["out_of_sample_score"], max(record["lead_scores"].values()))
        self.assertLess(record["out_of_sample_score"], 1)
        self.assertGreaterEqual(record["graded_origins"]["count"], 24)
        # Every lead is graded on origins whose 18-month path is known, the last of them 18
        # months before the demo data's last month.
        self.assertEqual(record["graded_origins"]["last"], "2023-06")

    def test_the_demo_script_defaults_to_10_and_refuses_leads_outside_9_to_18(self):
        self.assertIn("lead of 10 months", run(self.repo, "scripts/make_demo_data.py").stdout)
        for lead in ("8", "19"):
            with self.subTest(lead=lead):  # not `run`, which raises on the nonzero exit
                result = subprocess.run([sys.executable, "scripts/make_demo_data.py", "--lead", lead],
                                        cwd=self.repo, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("9 to 18", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_the_record_rebuilds_the_fit_from_the_clean_series(self):
        # Everything predict needs is in the record: TNO-FCI from the weights and normalization,
        # then the line to target points.
        self.prep()
        self.fit()
        record = self.newest_record()
        inputs = pd.concat([pd.read_csv(os.path.join(self.clean, f"{s}.csv"), index_col="month")[s]
                            for s in INPUTS], axis=1)
        target = pd.read_csv(os.path.join(self.clean, "ism_pmi.csv"), index_col="month")["ism_pmi"]
        z = (inputs - pd.Series({s: record["normalization"][s]["mean"] for s in INPUTS})) / \
            pd.Series({s: record["normalization"][s]["std"] for s in INPUTS})
        tnofci = (z * pd.Series(record["weights"])).sum(axis=1)
        first, last = record["training_months"]["first"], record["training_months"]["last"]
        tnofci = tnofci.loc[first:last]
        self.assertEqual(len(tnofci), record["training_months"]["count"])
        self.assertAlmostEqual(tnofci.mean(), 0, places=6)
        self.assertAlmostEqual(tnofci.std(ddof=0), 1, places=6)
        later = target.shift(-record["lead"]).loc[first:last]
        predicted = record["line"]["intercept"] + record["line"]["slope"] * tnofci
        r2 = 1 - ((later - predicted) ** 2).sum() / ((later - later.mean()) ** 2).sum()
        self.assertAlmostEqual(r2, record["in_sample_r2"], places=6)

    def test_the_record_holds_the_in_sample_r2_at_lags_1_to_12(self):
        # A lag of k months is the same fit with the target k months before the input month.
        # Least squares with an intercept gives the same R² on raw inputs as on z-scored ones.
        self.prep()
        self.fit()
        record = self.newest_record()
        self.assertEqual(sorted(int(k) for k in record["lag_in_sample_r2"]), list(range(1, 13)))
        inputs = pd.concat([pd.read_csv(os.path.join(self.clean, f"{s}.csv"), index_col="month")[s]
                            for s in INPUTS], axis=1)
        target = pd.read_csv(os.path.join(self.clean, "ism_pmi.csv"), index_col="month")["ism_pmi"]
        for lag in range(1, 13):
            with self.subTest(lag=lag):
                pairs = pd.concat([inputs, target.shift(lag)], axis=1).dropna().to_numpy()
                x, y = np.column_stack([np.ones(len(pairs)), pairs[:, :-1]]), pairs[:, -1]
                residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
                r2 = 1 - (residual ** 2).sum() / ((y - y.mean()) ** 2).sum()
                self.assertAlmostEqual(record["lag_in_sample_r2"][str(lag)], r2, places=6)

    def test_earlier_records_are_kept(self):
        self.prep()
        self.fit()
        first = self.records()
        with open(os.path.join(self.fits, first[0])) as fh:
            before = fh.read()
        self.fit()
        self.assertEqual(len(self.records()), 2)
        with open(os.path.join(self.fits, first[0])) as fh:
            self.assertEqual(before, fh.read())

    def test_falls_back_to_a_9_month_lead_when_the_search_cannot_score_one(self):
        # GMI's lead, kept as the fallback at the manager's word (g04). 100 months leave no room
        # for the search's warm-up, so no lead gets an out-of-sample score.
        self.prep()
        for name in os.listdir(self.clean):
            path = os.path.join(self.clean, name)
            pd.read_csv(path).head(100).to_csv(path, index=False)
        out = self.fit()
        record = self.newest_record()
        self.assertEqual(record["lead"], 9)
        self.assertEqual(record["lead_chosen_by"], "fallback")
        self.assertIsNone(record["out_of_sample_score"])
        self.assertEqual(set(record["lead_scores"].values()), {None})
        self.assertIn("fallback", out)

    def test_no_month_with_all_five_inputs_is_an_error(self):
        self.prep()
        for name, rows in (("fed_rate", slice(0, 150)), ("silver", slice(-150, None))):
            path = os.path.join(self.clean, f"{name}.csv")
            pd.read_csv(path).iloc[rows].to_csv(path, index=False)
        result = subprocess.run([sys.executable, "-m", "fci", "fit", "--demo"], cwd=self.repo,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("too few to fit", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(os.path.exists(self.fits))

    def test_without_clean_series_it_says_to_run_prep(self):
        result = subprocess.run([sys.executable, "-m", "fci", "fit", "--demo"], cwd=self.repo,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("fci prep", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(os.path.exists(self.fits))


if __name__ == "__main__":
    unittest.main()
