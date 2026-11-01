"""Seam 2 (g01 spec #2): `fci predict`, in a temporary copy of the repo.

On demo data it predicts from the newest fit record. Given the archived July 2025 run, it
reproduces the July 2025 prediction the page shows. Each test runs in seconds.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = shutil.ignore_patterns(".git", ".venv", "venv", "output", "site", "demo", "__pycache__")
INPUTS = ["fed_rate", "dxy", "gold", "oil", "silver"]
JULY_2025 = os.path.join("output", "archived_runs", "202507261556")
needs_july = unittest.skipUnless(os.path.isdir(os.path.join(ROOT, JULY_2025)),
                                 "needs the archived July 2025 run, which only the real-data copy holds (ADR 0001)")


def run(cwd, *args):
    return subprocess.run([sys.executable, *args], cwd=cwd, check=True, capture_output=True, text=True)


class FciPredictTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = os.path.join(self.tmp, "repo")
        shutil.copytree(ROOT, self.repo, ignore=SKIP)
        self.out = os.path.join(self.repo, "demo", "output")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def demo_fit(self):
        run(self.repo, "scripts/make_demo_data.py")
        run(self.repo, "-m", "fci", "prep", "--demo")
        run(self.repo, "-m", "fci", "fit", "--demo")
        (name,) = os.listdir(os.path.join(self.out, "fits"))
        with open(os.path.join(self.out, "fits", name)) as fh:
            return name, json.load(fh)

    def prediction(self, name):
        stem = os.path.splitext(name)[0]
        return pd.read_csv(os.path.join(self.out, "predictions", f"{stem}.csv"))

    def test_writes_one_row_per_predicted_month(self):
        name, record = self.demo_fit()
        run(self.repo, "-m", "fci", "predict", "--demo")
        df = self.prediction(name)
        self.assertEqual(list(df.columns), ["month", "ism_pmi"])
        last = pd.Period(record["inputs_through"], "M")
        self.assertEqual(list(df["month"]), [str(last + h) for h in range(1, record["lead"] + 1)])
        self.assertFalse(df["ism_pmi"].isna().any())

    def test_applies_the_record_to_the_clean_series(self):
        name, record = self.demo_fit()
        run(self.repo, "-m", "fci", "predict", "--demo")
        clean = os.path.join(self.out, "clean")
        inputs = pd.concat([pd.read_csv(os.path.join(clean, f"{s}.csv"), index_col="month")[s] for s in INPUTS], axis=1)
        norm = record["normalization"]
        tnofci = sum(record["weights"][s] * (inputs[s] - norm[s]["mean"]) / norm[s]["std"] for s in INPUTS)
        last = pd.Period(record["inputs_through"], "M")
        for month, value in self.prediction(name).itertuples(index=False):
            input_month = str(pd.Period(month, "M") - record["lead"])
            expected = record["line"]["intercept"] + record["line"]["slope"] * tnofci[input_month]
            self.assertAlmostEqual(value, expected, places=9)
            self.assertLessEqual(pd.Period(input_month, "M"), last)

    def test_running_twice_gives_identical_output(self):
        name, _ = self.demo_fit()
        run(self.repo, "-m", "fci", "predict", "--demo")
        path = os.path.join(self.out, "predictions", os.path.splitext(name)[0] + ".csv")
        with open(path, "rb") as fh:
            first = fh.read()
        run(self.repo, "-m", "fci", "predict", "--demo")
        with open(path, "rb") as fh:
            self.assertEqual(first, fh.read())

    @needs_july
    def test_reproduces_the_july_2025_prediction(self):
        shutil.copytree(os.path.join(ROOT, JULY_2025), os.path.join(self.repo, JULY_2025))
        run(self.repo, "-m", "fci", "predict", "--fit", JULY_2025)
        df = pd.read_csv(os.path.join(self.repo, "output", "predictions", "202507261556.csv"))
        # The July 2025 table on the published page
        self.assertEqual(list(df["month"]), [str(pd.Period("2025-07", "M") + i) for i in range(10)])
        self.assertEqual([round(v, 1) for v in df["ism_pmi"]],
                         [52.7, 52.4, 58.0, 52.5, 50.4, 52.8, 54.3, 55.5, 54.1, 54.9])

    def predict_fails(self, *args):
        result = subprocess.run([sys.executable, "-m", "fci", "predict", *args], cwd=self.repo,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stderr)
        return result.stderr

    def test_a_clean_series_ending_before_the_record_is_an_error(self):
        name, record = self.demo_fit()
        gold = os.path.join(self.out, "clean", "gold.csv")
        pd.read_csv(gold).iloc[:-2].to_csv(gold, index=False)
        self.assertIn(record["inputs_through"], self.predict_fails("--demo"))
        self.assertFalse(os.path.exists(os.path.join(self.out, "predictions")))

    def test_a_file_that_is_not_a_fit_record_is_an_error(self):
        self.demo_fit()
        not_json = os.path.join(self.repo, "notes.json")
        with open(not_json, "w") as fh:
            fh.write("not json\n")
        self.assertIn("notes.json", self.predict_fails("--demo", "--fit", not_json))
        no_line = os.path.join(self.repo, "partial.json")
        with open(no_line, "w") as fh:
            json.dump({"lead": 7}, fh)
        self.assertIn("partial.json", self.predict_fails("--demo", "--fit", no_line))

    @needs_july
    def test_an_archived_run_with_an_unnamed_date_column_reads(self):
        folder = os.path.join(self.repo, "run")
        os.makedirs(folder)
        df = pd.read_csv(os.path.join(ROOT, JULY_2025, "tno_fci_results.csv")).rename(columns={"Date": "Unnamed: 0"})
        df.to_csv(os.path.join(folder, "tno_fci_results.csv"), index=False, header=[""] + list(df.columns[1:]))
        run(self.repo, "-m", "fci", "predict", "--fit", folder)
        out = pd.read_csv(os.path.join(self.repo, "output", "predictions", "run.csv"))
        self.assertEqual(round(out["ism_pmi"].iloc[0], 1), 52.7)

    def test_without_a_fit_it_says_to_run_fit(self):
        run(self.repo, "scripts/make_demo_data.py")
        run(self.repo, "-m", "fci", "prep", "--demo")
        result = subprocess.run([sys.executable, "-m", "fci", "predict", "--demo"], cwd=self.repo,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("fci fit", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
