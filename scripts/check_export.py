#!/usr/bin/env python3
"""Run before every push. Fails if this repo would publish licensed data or private things.

- Every CSV under data/ holds at most SAMPLE_ROWS rows: samples show the format, nothing more.
- Only the expected top-level entries exist.
- No text file names a path on the author's Mac.
- Every commit's author and committer email is the GitHub no-reply address.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_ROWS = 5
NOREPLY = "7592333+omarzd@users.noreply.github.com"
ALLOWED = {".git", ".gitignore", "README.md", "requirements.txt", "1_collect_data.py", "2_prep_raw_data.py",
           "3_tno_fci_calculator.py", "5_build_page.py", "lib", "util", "tests", "scripts", "docs", "data",
           # local only, gitignored
           "output", "site", "__pycache__", ".venv", "venv", ".DS_Store"}


def main():
    problems = []
    for entry in sorted(os.listdir(ROOT)):
        if entry not in ALLOWED:
            problems.append(f"unexpected top-level entry: {entry}")
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, "data")):
        for name in filenames:
            if name.endswith(".csv"):
                path = os.path.join(dirpath, name)
                with open(path, errors="replace") as fh:
                    rows = sum(1 for line in fh if line.strip()) - 1
                if rows > SAMPLE_ROWS:
                    problems.append(f"{os.path.relpath(path, ROOT)} has {rows} rows; samples hold at most {SAMPLE_ROWS}")
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".git", "output", "site", "__pycache__", ".venv", "venv")]
        for name in filenames:
            if name.endswith((".py", ".md", ".html", ".txt", ".csv")):
                path = os.path.join(dirpath, name)
                with open(path, errors="replace") as fh:
                    if "/" + "Users/" in fh.read():  # split so this file doesn't match itself
                        problems.append(f"{os.path.relpath(path, ROOT)} names a local home-folder path")
    if os.path.isdir(os.path.join(ROOT, ".git")):
        log = subprocess.run(["git", "-C", ROOT, "log", "--format=%ae %ce"], capture_output=True, text=True).stdout
        for line in log.split("\n"):
            for email in line.split():
                if email != NOREPLY:
                    problems.append(f"a commit carries the email {email}")
    for p in problems:
        print("FAIL:", p)
    if not problems:
        print("export check passed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
