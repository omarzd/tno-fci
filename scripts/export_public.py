#!/usr/bin/env python3
"""Build the public copy of this repo's HEAD and commit it into a public checkout. Never pushes.

    python scripts/export_public.py <public checkout>

The public copy holds the files PUBLIC names, nothing else: no fit records, no predictions, no
archived runs (ADR 0001). Each real export in data/raw_all/ becomes its sample file, the header
and the first SAMPLE_ROWS rows under the same name. .gitignore loses its real-data-only lines.

It writes the public copy into the target, then runs the scripts/check_export.py it just wrote
there. It stops, committing nothing and leaving the target as it was, if this repo or the target
has uncommitted changes, if the check fails, or if any step fails. A tree that matches the
target's HEAD makes no commit, so the public history only grows with the code. On success it
prints the target's HEAD: the new commit, or the old one if nothing changed.
"""
import os
import subprocess
import sys

from check_export import NOREPLY, SAMPLE_ROWS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_FILES = {"LICENSE", "pyproject.toml", ".gitignore", "README.md", "docs/index.html", "docs/.nojekyll"}
PUBLIC_DIRS = ("fci/", "tests/", "scripts/", "data/raw_all/")
GITIGNORE_CUT = "# The real-data copy keeps every fit"
IDENTITY = {"GIT_AUTHOR_NAME": "omarzd", "GIT_AUTHOR_EMAIL": NOREPLY,
            "GIT_COMMITTER_NAME": "omarzd", "GIT_COMMITTER_EMAIL": NOREPLY}


class ExportError(Exception):
    """The export stopped, with a message for the person running it."""


def git(cwd, *args, check=True, env=None):
    result = subprocess.run(["git", "-C", cwd, *args], capture_output=True, env=env)
    if check and result.returncode != 0:
        raise ExportError(f"git {' '.join(args)} in {cwd}: {result.stderr.decode(errors='replace').strip()}")
    return result


def lines(result):
    return [name for name in result.stdout.decode().split("\0") if name]


def is_public(name):
    if name.startswith("data/"):
        return name.startswith("data/raw_all/") and name.endswith(".csv")
    return name in PUBLIC_FILES or name.startswith(PUBLIC_DIRS)


def public_tree():
    """{path: bytes} for every file of HEAD the public copy holds."""
    tree = {}
    for name in filter(is_public, lines(git(ROOT, "ls-tree", "-r", "-z", "--name-only", "HEAD"))):
        content = git(ROOT, "show", f"HEAD:{name}").stdout
        if name.startswith("data/"):
            content = b"".join(content.splitlines(keepends=True)[:SAMPLE_ROWS + 1])
        elif name == ".gitignore":
            content = content.split(GITIGNORE_CUT.encode())[0]
            if b"!output/" in content:
                raise ExportError(f".gitignore keeps output/ files above the line starting {GITIGNORE_CUT!r}")
        tree[name] = content
    return tree


def prune_empty_parents(target, names):
    """Remove the folders that removing names left empty, deepest first."""
    for folder in sorted({os.path.dirname(n) for n in names if os.path.dirname(n)}, key=len, reverse=True):
        while folder:
            path = os.path.join(target, folder)
            if not os.path.isdir(path) or os.listdir(path):
                break
            os.rmdir(path)
            folder = os.path.dirname(folder)


def checked_target(target):
    """The target's absolute path and its HEAD, or None for a new repo; refuses unsafe targets."""
    target = os.path.abspath(target)
    top = git(target, "rev-parse", "--show-toplevel", check=False) if os.path.isdir(target) else None
    if top is None or top.returncode != 0 or os.path.realpath(top.stdout.decode().strip()) != os.path.realpath(target):
        raise ExportError(f"{target} is not the top of a git checkout")
    if os.path.realpath(target) == os.path.realpath(ROOT):
        raise ExportError("the target is this repo")
    for repo, which in ((ROOT, "the real-data copy"), (target, "the target")):
        dirty = git(repo, "status", "--porcelain").stdout.decode()
        if dirty:
            raise ExportError(f"{which} has uncommitted changes:\n{dirty.rstrip()}")
    head = git(target, "rev-parse", "-q", "--verify", "HEAD", check=False)
    return target, head.stdout.decode().strip() if head.returncode == 0 else None


def write_and_check(target, head, tree, old):
    for name in old:
        os.remove(os.path.join(target, name))
    prune_empty_parents(target, old)
    for name, content in tree.items():
        path = os.path.join(target, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(content)
    git(target, "add", "-A")
    check = subprocess.run([sys.executable, os.path.join(target, "scripts", "check_export.py")],
                           capture_output=True, text=True)
    if check.returncode != 0:
        raise ExportError(f"the export check failed in {target}, nothing committed:\n"
                          f"{(check.stdout + check.stderr).rstrip()}")
    if head and git(target, "diff", "--cached", "--quiet", check=False).returncode == 0:
        print(f"no change: the public copy already matches {head}", file=sys.stderr)
        return head
    git(target, "commit", "-q", "-m", "Update to the latest code", env=dict(os.environ, **IDENTITY))
    return git(target, "rev-parse", "HEAD").stdout.decode().strip()


def export(target):
    target, head = checked_target(target)
    tree = public_tree()
    old = lines(git(target, "ls-files", "-z"))
    hidden = [n for n in tree if n not in old and os.path.lexists(os.path.join(target, n))]
    if hidden:
        raise ExportError(f"the target has files git ignores where the export writes: {', '.join(hidden)}")
    try:
        return write_and_check(target, head, tree, old)
    except BaseException:
        # the target was clean, so its HEAD (or, in a new repo, nothing) is all there was
        if head:
            git(target, "reset", "-q", "--hard")
        else:
            git(target, "rm", "-rq", "--cached", "--ignore-unmatch", ".")
            for name in tree:
                if os.path.lexists(os.path.join(target, name)):
                    os.remove(os.path.join(target, name))
        prune_empty_parents(target, list(tree))
        raise


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: export_public.py <public checkout>", file=sys.stderr)
        return 2
    try:
        print(export(argv[0]))
    except ExportError as e:
        print(f"export_public: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
