#!/usr/bin/env python3
"""
Collect class constructors, class methods and module-level functions from a
set of Python repos and save them, repo-by-repo, as JSON.

Run:  python collect_repo_functions.py /abs/path/to/save_dir [--root PATH]
"""

from __future__ import annotations
import ast
import json
import argparse
from pathlib import Path
from typing import List, Tuple

# ──────────────────────────  helpers  ──────────────────────────
def _strip_docstring(node: ast.FunctionDef) -> None:
    """Remove a leading docstring (if present) in-place."""
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        node.body = node.body[1:]


def _unparse(node: ast.AST) -> str:
    """Return source for *node* stripped of comments / docstrings."""
    if isinstance(node, ast.FunctionDef):
        _strip_docstring(node)
    return ast.unparse(node).strip() + "\n"

import tokenize

def _read_source(path: Path) -> str | None:
    """
    Return the text of *path* using the encoding declared in a `# -*- coding: ... -*-`
    line if present, or UTF-8 as a fallback.  
    If we still can’t decode the bytes, return **None** so the caller can skip.
    """
    # 1) Let `tokenize.open()` honour any `coding:` comment (PEP-263)
    try:
        with tokenize.open(path) as fh:
            return fh.read()
    except (SyntaxError, UnicodeDecodeError, tokenize.TokenError):
        pass

    # 2) Try a few common encodings manually
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue

    # 3) Give up – the file is unreadable
    print(f"[skip] {path}: cannot decode with common encodings")
    return None

# ─────────────────── extract_chunks (fixed) ────────────────────
def extract_chunks(path: Path) -> Tuple[List[str], List[str], List[str]]:
    ctors, methods, tops = [], [], []

    source = _read_source(path)
    if source is None or "\x00" in source:
        # ⏩ Skip binary/corrupt files that contain NUL bytes
        print(f"[skip] {path}: contains NULL bytes – probably binary")
        return ctors, methods, tops

    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError) as err:         
        print(f"[skip] {path}: {err}")
        return ctors, methods, tops

    # 3) walk the tree
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            tops.append(_unparse(node))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    (ctors if item.name == "__init__" else methods).append(
                        _unparse(item)
                    )

    return ctors, methods, tops


def file_exists(repo_dir: Path, save_dir: Path) -> bool:
    out_path = save_dir / f"{repo_dir.name}.json"
    if out_path.exists():
        return True
    else:
        return False

# ───────────────────────  new function  ────────────────────────
def create_repo_json(repo_dir: Path, save_dir: Path) -> None:
    """
    Build `<repo>.json` in *save_dir* for a single repository.

    If the file already exists, it is left untouched.
    """
    out_path = save_dir / f"{repo_dir.name}.json"

    ctors_all, methods_all, tops_all = [], [], []
    for py in repo_dir.rglob("*.py"):
        c, m, t = extract_chunks(py)
        ctors_all.extend(c)
        methods_all.extend(m)
        tops_all.extend(t)

    payload = {
        "class_constructors": ctors_all,
        "class_methods": methods_all,
        "non_class_functions": tops_all,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"✔️  Wrote {out_path.relative_to(save_dir.parent)}")


# ───────────────────────────  main  ────────────────────────────
def main() -> None:
    root: Path=Path("/mnt/onetouch/INNOVATE/BENCH_pwc_python_files_from_git")
    save_dir: Path=Path('/mnt/onetouch/INNOVATE/JSON_repos')
    
    if str(save_dir).startswith(str(root)):
        raise ValueError("save_dir must be outside the directory that holds the repos.")

    save_dir.mkdir(parents=True, exist_ok=True)

    for repo in (d for d in root.iterdir() if d.is_dir()):
        if file_exists(repo, save_dir):
            continue
        create_repo_json(repo, save_dir)

    print(f"\nDone – JSON files live in: {save_dir}\n")


if __name__ == "__main__":
    main()
