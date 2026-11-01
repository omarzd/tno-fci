"""Seam 2 (g01 spec #2): the fci command on the demo data, in a temporary copy of the repo.

`fci prep` turns the raw exports into one monthly clean series per input series and one for the
target, in output/clean/ (demo/output/clean/ with --demo). Runs in seconds: no fit.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = shutil.ignore_patterns(".git", ".venv", "venv", "output", "site", "demo", "__pycache__")
SERIES = ["ism_pmi", "fed_rate", "dxy", "gold", "oil", "silver"]


def run(cwd, *args):
    return subprocess.run([sys.executable, *args], cwd=cwd, check=True, capture_output=True, text=True)


class FciPrepTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = os.path.join(self.tmp, "repo")
        shutil.copytree(ROOT, self.repo, ignore=SKIP)
        run(self.repo, "scripts/make_demo_data.py")
        self.clean = os.path.join(self.repo, "demo", "output", "clean")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def clean_files(self):
        files = {}
        for name in sorted(os.listdir(self.clean)):
            with open(os.path.join(self.clean, name), "rb") as fh:
                files[name] = fh.read()
        return files

    def test_writes_one_monthly_series_per_input_series_and_the_target(self):
        run(self.repo, "-m", "fci", "prep", "--demo")
        self.assertEqual(sorted(os.listdir(self.clean)), sorted(f"{s}.csv" for s in SERIES))
        for series in SERIES:
            df = pd.read_csv(os.path.join(self.clean, f"{series}.csv"))
            self.assertEqual(list(df.columns), ["month", series])
            self.assertEqual(len(df), 360)  # the demo data's 30 years, one row a month
            self.assertTrue(df["month"].is_unique and df["month"].is_monotonic_increasing)
            self.assertRegex(df["month"].iloc[0], r"^\d{4}-\d{2}$")
            self.assertFalse(df[series].isna().any())

    def raw_path(self, prefix):
        return os.path.join(self.repo, "demo", "data", "raw_all", f"{prefix}, 1M_demo.csv")

    def prep_fails(self):
        result = subprocess.run([sys.executable, "-m", "fci", "prep", "--demo"], cwd=self.repo,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stderr)
        return result.stderr

    def test_takes_the_last_close_of_each_month(self):
        # The month's later row comes first in the file and holds its smallest value, so neither
        # file order nor the month's maximum can stand in for the last close.
        raw = self.raw_path("TVC_GOLD")
        df = pd.read_csv(raw)
        later = df.iloc[[0]].assign(time="1995-01-20", close=1.0)
        pd.concat([later, df]).to_csv(raw, index=False)
        run(self.repo, "-m", "fci", "prep", "--demo")
        gold = pd.read_csv(os.path.join(self.clean, "gold.csv"))
        self.assertEqual(len(gold), 360)
        self.assertEqual(gold.loc[gold["month"] == "1995-01", "gold"].item(), 1.0)

    def test_fills_gaps_of_up_to_three_months_and_leaves_the_rest_of_a_longer_gap_empty(self):
        # As the current fit's monthly step does: every month gets a row, and the last known value
        # carries forward at most 3 months.
        raw = self.raw_path("ECONOMICS_USINTR")
        df = pd.read_csv(raw)
        df = df[~df["time"].isin(["1996-02-01", "1997-01-01", "1997-02-01", "1997-03-01", "1997-04-01"])]
        df.to_csv(raw, index=False)
        run(self.repo, "-m", "fci", "prep", "--demo")
        fed = pd.read_csv(os.path.join(self.clean, "fed_rate.csv")).set_index("month")["fed_rate"]
        self.assertEqual(len(fed), 360)
        self.assertEqual(fed["1996-02"], fed["1996-01"])
        self.assertEqual(list(fed["1997-01":"1997-03"]), [fed["1996-12"]] * 3)
        self.assertTrue(pd.isna(fed["1997-04"]))

    def test_running_twice_gives_identical_output(self):
        run(self.repo, "-m", "fci", "prep", "--demo")
        first = self.clean_files()
        run(self.repo, "-m", "fci", "prep", "--demo")
        self.assertEqual(first, self.clean_files())

    def test_two_exports_of_one_series_is_an_error(self):
        shutil.copy(self.raw_path("TVC_DXY"), self.raw_path("TVC_DXY").replace("1M_demo", "1M_older"))
        self.assertIn("keep one: TVC_DXY, 1M_demo.csv; TVC_DXY, 1M_older.csv", self.prep_fails())

    def test_an_export_without_a_close_column_is_an_error(self):
        with open(self.raw_path("TVC_GOLD"), "w") as fh:
            fh.write("time,open\n1995-01-01,1200\n")
        self.assertIn("TVC_GOLD", self.prep_fails())

    def test_a_failed_prep_leaves_the_last_good_output_untouched(self):
        run(self.repo, "-m", "fci", "prep", "--demo")
        before = self.clean_files()
        # A fresh target export lands beside a stray second dxy export: nothing may be rewritten.
        pd.read_csv(self.raw_path("ECONOMICS_USBCOI")).assign(close=50.0).to_csv(self.raw_path("ECONOMICS_USBCOI"), index=False)
        shutil.copy(self.raw_path("TVC_DXY"), self.raw_path("TVC_DXY").replace("1M_demo", "1M_older"))
        self.prep_fails()
        self.assertEqual(before, self.clean_files())


if __name__ == "__main__":
    unittest.main()
