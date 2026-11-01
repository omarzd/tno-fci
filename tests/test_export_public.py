"""scripts/export_public.py, from a temporary git copy of this repo into a temporary checkout.

The copy commits a made-up fit record and archived run beside the code, as the real-data copy
does (ADR 0002); none of them may reach the public copy (ADR 0001).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = shutil.ignore_patterns(".git", ".venv", "venv", "output", "site", "demo", "__pycache__", "*.egg-info")
NOREPLY = "7592333+omarzd@users.noreply.github.com"
# no global or system git config, so the user's hooks, signing and identity stay out of it
ENV = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
           GIT_AUTHOR_NAME="someone", GIT_AUTHOR_EMAIL="someone@example.com",
           GIT_COMMITTER_NAME="someone", GIT_COMMITTER_EMAIL="someone@example.com")


def git(cwd, *args):
    return subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True, text=True, env=ENV).stdout.strip()


def tree_files(folder):
    found = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        found += [os.path.relpath(os.path.join(dirpath, n), folder) for n in filenames]
    return sorted(found)


class ExportPublicTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.source = os.path.join(self.tmp, "source")
        self.target = os.path.join(self.tmp, "target")
        shutil.copytree(ROOT, self.source, ignore=SKIP)
        for path, text in ((("output", "fits", "20261031-000000-000000.json"), '{"lead": 9}\n'),
                           (("output", "archived_runs", "run", "fci_results.csv"),
                            "Date,TNO_FCI\n" + "".join(f"2020-{m:02d}-28,0.{m}\n" for m in range(1, 13)))):
            os.makedirs(os.path.join(self.source, *path[:-1]), exist_ok=True)
            with open(os.path.join(self.source, *path), "w") as fh:
                fh.write(text)
        git(self.tmp, "init", "-q", self.source)
        git(self.source, "add", "-A")
        git(self.source, "add", "-f", "output")
        git(self.source, "commit", "-q", "-m", "the real-data copy")
        git(self.tmp, "init", "-q", self.target)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def export(self):
        return subprocess.run([sys.executable, os.path.join(self.source, "scripts", "export_public.py"), self.target],
                              capture_output=True, text=True, env=ENV)

    def exported(self):
        result = self.export()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout.strip()

    def test_makes_one_no_reply_commit_with_no_output_and_only_sample_files(self):
        head = self.exported()
        self.assertEqual(git(self.target, "rev-parse", "HEAD"), head)
        self.assertEqual(git(self.target, "log", "--format=%an <%ae> %cn <%ce>"),
                         f"omarzd <{NOREPLY}> omarzd <{NOREPLY}>")
        files = git(self.target, "ls-files").split("\n")
        self.assertFalse([f for f in files if f.startswith("output/")])
        self.assertEqual(sorted(files), tree_files(self.target))
        for name in os.listdir(os.path.join(self.source, "data", "raw_all")):
            with open(os.path.join(self.source, "data", "raw_all", name), "rb") as fh:
                first = b"".join(fh.readlines()[:6])
            with open(os.path.join(self.target, "data", "raw_all", name), "rb") as fh:
                self.assertEqual(fh.read(), first)  # the header and the first 5 data rows, as exported
        for f in files:
            if f.endswith(".csv"):
                with open(os.path.join(self.target, f)) as fh:
                    self.assertLessEqual(sum(1 for line in fh if line.strip()) - 1, 5, f)

    def test_holds_the_code_the_readme_and_the_page_but_not_the_monthly_steps(self):
        self.exported()
        files = git(self.target, "ls-files").split("\n")
        self.assertEqual(sorted({f.split("/")[0] for f in files}),
                         [".gitignore", "LICENSE", "README.md", "data", "docs", "fci", "pyproject.toml", "scripts", "tests"])
        self.assertIn("docs/index.html", files)
        self.assertNotIn("docs/monthly-update.md", files)
        with open(os.path.join(self.target, ".gitignore")) as fh:
            self.assertNotIn("!output/", fh.read())

    def test_the_readme_says_what_tno_means_and_names_no_handle_or_mac_path(self):
        self.exported()
        with open(os.path.join(self.target, "README.md")) as fh:
            readme = fh.read()
        self.assertIn("TNO is the organization authoring this FOSS copy", readme)
        self.assertIn("make_demo_data.py", readme)
        self.assertNotIn("TradingView handle", readme)
        self.assertNotIn("/" + "Users/", readme)

    def test_a_second_export_with_no_code_change_makes_no_commit(self):
        head = self.exported()
        self.assertEqual(self.exported(), head)
        self.assertEqual(git(self.target, "rev-list", "--count", "HEAD"), "1")

    def test_a_second_export_still_runs_the_check(self):
        self.exported()
        git(self.target, "commit", "-q", "--allow-empty", "-m", "by someone else")
        result = self.export()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("someone@example.com", result.stderr)

    def test_a_failed_commit_leaves_a_new_target_empty(self):
        hook = os.path.join(self.target, ".git", "hooks", "pre-commit")
        with open(hook, "w") as fh:
            fh.write("#!/bin/sh\nexit 1\n")
        os.chmod(hook, 0o755)
        result = self.export()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("git commit", result.stderr)
        self.assertEqual(os.listdir(self.target), [".git"])
        self.assertEqual(git(self.target, "ls-files"), "")

    def test_an_ignored_file_where_the_export_writes_is_refused(self):
        with open(os.path.join(self.target, ".gitignore"), "w") as fh:
            fh.write("LICENSE\n")
        with open(os.path.join(self.target, "LICENSE"), "w") as fh:
            fh.write("mine\n")
        git(self.target, "add", ".gitignore")
        git(self.target, "commit", "-q", "-m", "ignore LICENSE")
        result = self.export()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LICENSE", result.stderr)
        with open(os.path.join(self.target, "LICENSE")) as fh:
            self.assertEqual(fh.read(), "mine\n")

    def test_a_dirty_real_data_copy_stops_before_committing(self):
        with open(os.path.join(self.source, "README.md"), "a") as fh:
            fh.write("\nnot committed\n")
        result = self.export()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("uncommitted", result.stderr)
        self.assertEqual(os.listdir(self.target), [".git"])
        self.assertNotEqual(subprocess.run(["git", "-C", self.target, "rev-parse", "-q", "--verify", "HEAD"],
                                           capture_output=True, env=ENV).returncode, 0)

    def test_a_failed_check_stops_before_committing_and_leaves_the_target_as_it_was(self):
        with open(os.path.join(self.target, "notes.md"), "w") as fh:
            fh.write("an old commit\n")
        git(self.target, "add", "notes.md")
        git(self.target, "commit", "-q", "-m", "by someone else")
        before = git(self.target, "rev-parse", "HEAD")
        result = self.export()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("export check", result.stderr)
        self.assertIn("someone@example.com", result.stderr)
        self.assertEqual(git(self.target, "rev-parse", "HEAD"), before)
        self.assertEqual(git(self.target, "status", "--porcelain"), "")
        self.assertEqual(tree_files(self.target), ["notes.md"])

    def test_a_target_that_is_not_a_checkout_is_refused(self):
        plain = os.path.join(self.tmp, "plain")
        os.mkdir(plain)
        self.target = plain
        result = self.export()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("git checkout", result.stderr)
        self.assertEqual(os.listdir(plain), [])

    def test_the_exported_suite_skips_the_archived_run_tests_and_says_why(self):
        self.exported()
        result = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_fci_all", "test_fci_predict"],
                                cwd=os.path.join(self.target, "tests"), capture_output=True, text=True, env=ENV)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("skipped 'needs the archived July 2025 run", result.stderr)


if __name__ == "__main__":
    unittest.main()
