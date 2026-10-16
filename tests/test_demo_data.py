"""Seam 1 (g04 spec #10): the README's demo commands, run as a stranger runs them.

Demo data with a planted lead goes in; the fit must report that lead. Runs in a temporary copy
of the repo. The fit searches 92 weights, so this test takes about 4 minutes.
"""
import filecmp
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = shutil.ignore_patterns(".git", ".venv", "venv", "output", "site", "demo", "__pycache__")


def differences(cmp):
    """Every file added, removed or changed anywhere under the two compared folders."""
    found = cmp.left_only + cmp.right_only + cmp.diff_files
    for sub in cmp.subdirs.values():
        found += differences(sub)
    return found


def run(cwd, *args):
    subprocess.run([sys.executable, *args], cwd=cwd, check=True, capture_output=True, text=True)


class DemoDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = os.path.join(self.tmp, "repo")
        shutil.copytree(ROOT, self.repo, ignore=SKIP)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def demo_files(self):
        raw = os.path.join(self.repo, "demo", "data", "raw_all")
        files = {}
        for name in sorted(os.listdir(raw)):
            with open(os.path.join(raw, name)) as fh:
                files[name] = fh.read()
        return files

    def test_same_seed_gives_identical_demo_data(self):
        run(self.repo, "scripts/make_demo_data.py", "--lead", "5", "--seed", "3")
        first = self.demo_files()
        self.assertEqual(len(first), 6)
        run(self.repo, "scripts/make_demo_data.py", "--lead", "5", "--seed", "3")
        self.assertEqual(first, self.demo_files())

    def test_fit_finds_the_planted_lead(self):
        planted = 5  # not the default 7, so the default isn't the only lead ever checked
        run(self.repo, "scripts/make_demo_data.py", "--lead", str(planted))
        run(self.repo, "2_prep_raw_data.py", "--demo")
        run(self.repo, "3_tno_fci_calculator.py", "--demo")

        with open(os.path.join(self.repo, "demo", "output", "model_params.txt")) as fh:
            match = re.search(r"Optimal lag period: (\d+) months", fh.read())
        self.assertIsNotNone(match, "the fit's model parameters name no lead")
        self.assertEqual(int(match.group(1)), planted)
        self.assertTrue(os.listdir(os.path.join(self.repo, "demo", "output", "archived_runs")))

        # The demo never touches the real data or the real output.
        self.assertFalse(os.path.exists(os.path.join(self.repo, "output")))
        self.assertEqual(differences(filecmp.dircmp(os.path.join(ROOT, "data"), os.path.join(self.repo, "data"))), [])
        run(self.repo, "scripts/check_export.py")  # the pre-push check still passes


if __name__ == "__main__":
    unittest.main()
