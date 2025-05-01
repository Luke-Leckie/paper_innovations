#!/usr/bin/env python3
"""
Analyse a set of Python repos and dump one CSV per repo with aggregated
AST / complexity statistics.

Run:  python analyse_repos.py /abs/path/to/save_dir  [--root PATH]
"""

from __future__ import annotations
import argparse
import ast
import json              # only used by _read_source fall-back
import os
from math import log2
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from lib2to3.refactor import RefactoringTool, get_fixers_from_package

# --------------------------------------------------------------------------- #
# -----  shared helpers (copied from your working extractor)  --------------- #
# --------------------------------------------------------------------------- #
import tokenize
RT = RefactoringTool(get_fixers_from_package('lib2to3.fixes'))

def _read_source(path: Path) -> str | None:
    """Return text of *path*, honouring any PEP-263 encoding hint."""
    try:                                 # 1) let tokenize.open() decide
        with tokenize.open(path) as fh:
            return fh.read()
    except (SyntaxError, UnicodeDecodeError, tokenize.TokenError):
        pass

    # 2)  fallback encodings
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    # 3)  give up
    print(f"[skip] {path}: cannot decode with common encodings")
    return None


# --------------------------------------------------------------------------- #
# -----  metric helpers ----------------------------------------------------- #
# --------------------------------------------------------------------------- #
def ast_node_count_and_tree(source: str, filename: str):
    """
    Parse *source* to an AST.  If it is python-2 syntax, auto-convert.
    Returns (node_count, ast_root | None)
    """
    try:
        tree = ast.parse(source, filename=filename)
        return sum(1 for _ in ast.walk(tree)), tree
    except SyntaxError:
        try:
            fixed = RT.refactor_string(source, filename)
            tree2 = ast.parse(str(fixed))
            return sum(1 for _ in ast.walk(tree2)), tree2
        except Exception:
            return len(source.splitlines()), None   # fall back to LOC as volume
    except Exception:
        return 0, None


def ast_max_depth(node):
    if not isinstance(node, ast.AST):
        return 0
    return 1 + max(
        (ast_max_depth(child) for child in ast.iter_child_nodes(node)),
        default=0
    )


def branching_factors(tree):
    child_counts = [len(list(ast.iter_child_nodes(n))) for n in ast.walk(tree)]
    return (
        (sum(child_counts) / len(child_counts)) if child_counts else 0,
        max(child_counts) if child_counts else 0,
    )


def width_profile(tree):
    depth_map = {}

    def dfs(n, d=0):
        depth_map[d] = depth_map.get(d, 0) + 1
        for c in ast.iter_child_nodes(n):
            dfs(c, d + 1)

    dfs(tree)
    if not depth_map:
        return 0, 0
    max_width = max(depth_map.values())
    level = max(depth_map, key=depth_map.get)
    return max_width, level


def node_type_entropy(tree):
    counts = {}
    for n in ast.walk(tree):
        counts[type(n).__name__] = counts.get(type(n).__name__, 0) + 1
    total = sum(counts.values())
    k = len(counts)
    if total > 0 and k > 1:
        ent = -sum((c / total) * log2(c / total) for c in counts.values())
        return ent / log2(k)
    return 0.0


def cyclomatic_complexity(tree):
    decisions = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Assert)
    bool_ops = (ast.And, ast.Or)
    cc = 1
    for n in ast.walk(tree):
        if isinstance(n, decisions):
            cc += 1
        if isinstance(n, ast.BoolOp) and isinstance(n.op, bool_ops):
            cc += len(n.values) - 1
    return cc


def import_metrics(tree):
    nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    unique = set()
    for n in nodes:
        if isinstance(n, ast.Import):
            unique.update(alias.name for alias in n.names)
        else:  # ImportFrom
            mod = ('.' * n.level + n.module) if n.module else ('.' * n.level)
            unique.add(mod)
    return len(nodes), len(unique)


def maintainability_index(volume, cc, loc):
    # Simple MI variant – larger is better
    if volume <= 0 or loc <= 0:
        return 0
    return 171 - 5.2 * log2(volume) - 0.23 * cc - 16.2 * log2(loc)


# --------------------------------------------------------------------------- #
# -----  per-repo analysis -------------------------------------------------- #
# --------------------------------------------------------------------------- #
def analyse_repo(repo_dir: Path) -> pd.DataFrame:
    """
    Scan every *.py* file in *repo_dir*, compute metrics and return
    a single-row DataFrame with aggregates.
    """
    locs, depths, br_avgs, br_maxs, cycs, ents, breadths, mis = \
        ([] for _ in range(8))
    total_nodes = total_ic = total_iu = 0
    n_files = 0

    for py in repo_dir.rglob("*.py"):
        source = _read_source(py)
        if source is None or "\x00" in source:
            continue

        n_files += 1
        loc = len(source.splitlines())
        locs.append(loc)

        # AST + node count
        nodes, tree = ast_node_count_and_tree(source, str(py))
        total_nodes += nodes

        # defaults for unparsable files
        depth = avg_br = max_br = ent = width = 0
        cc = 0
        ic = iu = 0

        if tree:
            depth = ast_max_depth(tree)
            avg_br, max_br = branching_factors(tree)
            width, _ = width_profile(tree)
            ent = node_type_entropy(tree)
            cc = cyclomatic_complexity(tree)
            ic, iu = import_metrics(tree)

        depths.append(depth)
        br_avgs.append(avg_br)
        br_maxs.append(max_br)
        cycs.append(cc)
        ents.append(ent)
        breadths.append(width)
        mis.append(maintainability_index(nodes, cc, loc))
        total_ic += ic
        total_iu += iu

    # -- helper for Σ, μ, max, σ
    def stats(lst: List[float]):
        if not lst:
            return (0, 0, 0, 0)
        arr = np.asarray(lst, dtype=float)
        return arr.sum(), arr.mean(), arr.max(), arr.std(ddof=0)

    d_sum, d_avg, d_max, d_std = stats(depths)
    ba_sum, ba_avg, ba_max, ba_std = stats(br_avgs)
    bm_sum, bm_avg, bm_max, bm_std = stats(br_maxs)
    c_sum,  c_avg,  c_max,  c_std  = stats(cycs)
    e_sum,  e_avg,  e_max,  e_std  = stats(ents)
    l_sum,  l_avg,  l_max,  l_std  = stats(locs)
    mi_sum, mi_avg, mi_max, mi_std = stats(mis)
    brd_sum, brd_avg, brd_max, brd_std = stats(breadths)

    row = {
        'repo': repo_dir.name,
        'n_core_files': n_files,
        # node & import totals
        'node_count_tot': total_nodes,
        'import_count_tot': total_ic,
        'import_unique_tot': total_iu,

        # depth stats
        'depth_sum': d_sum, 'avg_depth': d_avg, 'depth_max': d_max, 'depth_std': d_std,
        # branching-avg stats
        'br_avg_sum': ba_sum, 'avg_branching': ba_avg, 'br_avg_max': ba_max, 'br_avg_std': ba_std,
        # branching-max stats
        'br_max_sum': bm_sum, 'avg_br_max': bm_avg, 'br_max_max': bm_max, 'br_max_std': bm_std,

        # cyclomatic stats
        'cc_sum': c_sum, 'avg_cyclomatic': c_avg, 'cc_max': c_max, 'cc_std': c_std,
        # entropy stats
        'ent_sum': e_sum, 'avg_entropy': e_avg, 'ent_max': e_max, 'ent_std': e_std,

        # LOC stats
        'loc_sum': l_sum, 'avg_loc_per_file': l_avg, 'loc_max': l_max, 'loc_std': l_std,

        # MI stats
        'mi_sum': mi_sum, 'avg_MI': mi_avg, 'mi_max': mi_max, 'mi_std': mi_std,

        # AST breadth stats
        'breadth_sum': brd_sum, 'avg_breadth': brd_avg, 'breadth_max': brd_max, 'breadth_std': brd_std,
    }

    return pd.DataFrame([row])


# --------------------------------------------------------------------------- #
# -----  I/O helpers -------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def csv_exists(repo_dir: Path, save_dir: Path) -> bool:
    return (save_dir / f"{repo_dir.name}.csv").exists()


def write_repo_csv(df: pd.DataFrame, repo_dir: Path, save_dir: Path):
    out_path = save_dir / f"{repo_dir.name}.csv"
    df.to_csv(out_path, index=False)
    print(f"✔️  Wrote {out_path.relative_to(save_dir.parent)}")


# --------------------------------------------------------------------------- #
# -----  main ---------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def main():
    root: Path=Path("/mnt/onetouch/INNOVATE/BENCH_pwc_python_files_from_git")
    save_dir: Path=Path('/mnt/onetouch/INNOVATE/AST_data')

    if str(save_dir).startswith(str(root)):
        raise ValueError("save_dir must be outside the directory that holds the repos.")

    save_dir.mkdir(parents=True, exist_ok=True)

    for repo in (d for d in root.iterdir() if d.is_dir()):
        if csv_exists(repo, save_dir):
            print(f"⏭️  {repo.name}.csv already exists – skipping")
            continue
        df = analyse_repo(repo)
        write_repo_csv(df, repo, save_dir)

    print(f"\nDone – per-repo CSVs live in: {save_dir}\n")


if __name__ == "__main__":
    main()
