"""Seam 2 (g01 spec #2): `fci all`, in a temporary copy of the repo.

On the real data it leaves the page; with --demo it stops after the fit and builds no page
(g04). After a full run, each step re-runs alone from the previous step's files.
"""
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = shutil.ignore_patterns(".git", ".venv", "venv", "output", "site", "demo", "__pycache__")
JULY = os.path.join("output", "archived_runs", "202507261556")
HAS_JULY = os.path.isdir(os.path.join(ROOT, JULY))
needs_july = unittest.skipUnless(HAS_JULY, "needs the archived July 2025 run and the real data, "
                                           "which only the real-data copy holds (ADR 0001)")


def run(cwd, *args):
    return subprocess.run([sys.executable, *args], cwd=cwd, check=True, capture_output=True, text=True)


def differences(cmp):
    """Every file added, removed or changed anywhere under the two compared folders."""
    found = cmp.left_only + cmp.right_only + cmp.diff_files
    for sub in cmp.subdirs.values():
        found += differences(sub)
    return found


class FciAllTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = os.path.join(self.tmp, "repo")
        shutil.copytree(ROOT, self.repo, ignore=SKIP)
        if HAS_JULY:
            shutil.copytree(os.path.join(ROOT, JULY), os.path.join(self.repo, JULY))
        self.page = os.path.join(self.repo, "site", "index.html")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    @needs_july
    def test_on_the_real_data_it_leaves_the_page(self):
        run(self.repo, "-m", "fci", "all")
        with open(self.page) as fh:
            html = fh.read()
        self.assertIn("The July 2025 prediction", html)
        self.assertIn("<td>52.7</td>", html)  # the July 2025 table's first month
        self.assertEqual(len(os.listdir(os.path.join(self.repo, "output", "fits"))), 1)

    @needs_july
    def test_each_step_reruns_alone_after_a_full_run(self):
        run(self.repo, "-m", "fci", "all")
        os.remove(self.page)
        run(self.repo, "-m", "fci", "page")
        self.assertTrue(os.path.exists(self.page))
        run(self.repo, "-m", "fci", "predict")
        run(self.repo, "-m", "fci", "prep")
        run(self.repo, "-m", "fci", "fit")
        self.assertEqual(len(os.listdir(os.path.join(self.repo, "output", "fits"))), 2)

    def test_with_demo_it_stops_after_the_fit(self):
        run(self.repo, "scripts/make_demo_data.py", "--lead", "11")
        out = run(self.repo, "-m", "fci", "all", "--demo").stdout
        self.assertIn("lead 11 months", out)
        self.assertEqual(len(os.listdir(os.path.join(self.repo, "demo", "output", "fits"))), 1)
        self.assertFalse(os.path.exists(os.path.join(self.repo, "demo", "output", "predictions")))
        self.assertFalse(os.path.exists(os.path.join(self.repo, "site")))
        # The demo never touches the real data or the real output.
        output = os.path.join(self.repo, "output")
        self.assertEqual(sorted(os.listdir(output)) if os.path.exists(output) else [], ["archived_runs"] if HAS_JULY else [])
        self.assertEqual(differences(filecmp.dircmp(os.path.join(ROOT, "data"), os.path.join(self.repo, "data"))), [])

    def test_a_missing_july_2025_run_stops_before_any_fit(self):
        shutil.rmtree(os.path.join(self.repo, JULY), ignore_errors=True)  # the public copy has none
        result = subprocess.run([sys.executable, "-m", "fci", "all"], cwd=self.repo, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("July 2025", result.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.repo, "output", "fits")))

    def test_the_numbered_scripts_are_gone(self):
        names = os.listdir(ROOT)
        for gone in ("1_collect_data.py", "2_prep_raw_data.py", "3_tno_fci_calculator.py", "5_build_page.py", "lib", "util"):
            self.assertNotIn(gone, names)


if __name__ == "__main__":
    unittest.main()
