#!/usr/bin/env python
"""
Standing consistency check for the `qtft` package.

Reports three classes of rot that repeatedly went unnoticed during development:

  1. **Dead functions** — defined but never referenced from the package, `scripts/`, or any
     notebook code cell.
  2. **Unused imports** — a name imported by a module and never used in it.
  3. **Documented-but-missing names** — a ``qtft`` symbol written in backticks in the README
     or in a docstring that no longer exists, so the docs promise an API the code lacks.

Why AST rather than grep
------------------------
Both failure modes here are invisible to a ``name(`` regex, and both bit this project:

* A function passed *by reference* looks dead — ``executor.submit(_worker, arg)`` never
  writes ``_worker(``. Matching bare names fixes that.
* A dead function looks **alive** when its name appears inside a docstring example or a log
  message, e.g. ``logger.info(f"  plotting.plot_comparison_summary(comparison)")``. That one
  hid a genuinely unused public function through two review passes.

Collecting references from the parsed syntax tree solves both: `ast.Name` / `ast.Attribute`
nodes are real code references, and text inside string literals is never either.

Usage
-----
    python scripts/check_codebase.py            # report
    python scripts/check_codebase.py --quiet    # only problems
    python scripts/check_codebase.py --strict   # exit 1 if anything is found (CI / pre-commit)
"""

from __future__ import annotations

import argparse
import ast
import glob
import io
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Deliberately unreferenced from inside the repo. A library's public API is allowed to have
# no internal callers, so these are not "dead" — anything in qtft/__init__.py's __all__ is
# excluded automatically, and this list covers the rest, each with the reason it is kept.
PUBLIC_API_KEEP = {
    "plot_phased_kinetics": "back-compat alias for plot_kinetics",
    "load_comparison_data": "public counterpart of save_comparison_data (README §9)",
}


def _iter_sources():
    """Yield (label, source_text) for every package module, script and notebook code cell."""
    for path in sorted(glob.glob(os.path.join(REPO, "qtft", "*.py"))) + \
                sorted(glob.glob(os.path.join(REPO, "scripts", "*.py"))):
        yield os.path.relpath(path, REPO), io.open(path, encoding="utf-8").read()
    for path in sorted(glob.glob(os.path.join(REPO, "*.ipynb"))):
        nb = json.load(io.open(path, encoding="utf-8"))
        cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        src = "\n".join("".join(c["source"]) for c in cells)
        yield os.path.relpath(path, REPO), src


def _references(tree: ast.AST) -> set:
    """Every name referenced as *code* (never inside a string literal or docstring).

    Quoted type annotations (``config: "SimulationConfig"``) are the one place a string
    genuinely *is* a reference, so those are parsed and folded in — but only in annotation
    position, never for arbitrary strings, which is what keeps docstring examples from
    resurrecting dead code.
    """
    refs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            refs.add(node.id)
        elif isinstance(node, ast.Attribute):
            refs.add(node.attr)

    for node in ast.walk(tree):
        for ann in _annotation_slots(node):
            if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
                try:
                    refs |= _references(ast.parse(ann.value, mode="eval"))
                except SyntaxError:
                    pass
    return refs


def _annotation_slots(node):
    """Annotation expressions attached to a node (may be quoted strings)."""
    if isinstance(node, ast.arg):
        yield node.annotation
    elif isinstance(node, ast.AnnAssign):
        yield node.annotation
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        yield node.returns


def _exported(tree: ast.AST) -> set:
    """Names listed in ``__all__`` — re-exports, not unused imports."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            for elt in getattr(node.value, "elts", []):
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    out.add(elt.value)
    return out


def _parse(label, src):
    try:
        return ast.parse(src)
    except SyntaxError as exc:                      # a notebook cell with magics, say
        print(f"  ! skipped {label}: {exc.msg}", file=sys.stderr)
        return None


def scan():
    defs, refs, imports, trees = {}, set(), {}, {}
    for label, src in _iter_sources():
        tree = _parse(label, src)
        if tree is None:
            continue
        trees[label] = (tree, src)
        refs |= _references(tree)
        if not label.endswith(".py"):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                defs[node.name] = (label, node.lineno, node.end_lineno - node.lineno + 1)

    # --- 1. dead functions ---
    exported = set()
    if "qtft/__init__.py" in trees:
        exported = _exported(trees["qtft/__init__.py"][0])
    ignored = exported | set(PUBLIC_API_KEEP)
    dead = sorted((v[0], name, v[1], v[2]) for name, v in defs.items()
                  if name not in refs and name not in ignored)

    # --- 2. unused imports (per module) ---
    unused = []
    for label, (tree, _src) in trees.items():
        if not label.endswith(".py"):
            continue
        local = _references(tree) | _exported(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if getattr(node, "module", None) == "__future__":
                    continue          # compiler directive, never "used"
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    bound = (alias.asname or alias.name).split(".")[0]
                    if bound not in local:
                        unused.append((label, node.lineno, bound))
    unused.sort()

    # --- 3. documented-but-missing names ---
    known_prefixes = ("plot_", "get_", "build_", "compute_", "print_", "save_", "load_",
                      "run_", "make_", "convert_", "find_", "weighted_", "count_")
    symbols = set(defs)
    for label, (t, _src) in trees.items():
        if not label.endswith(".py"):
            continue
        for n in ast.walk(t):
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.add(n.name)          # includes methods, not just module-level
            if isinstance(n, ast.arg):
                symbols.add(n.arg)           # parameters get named in docs too
    doc_sources = [("README.md", io.open(os.path.join(REPO, "README.md"), encoding="utf-8").read())]
    for label, (_t, src) in trees.items():
        if label.endswith(".py"):
            doc_sources.append((label, src))
    missing = set()
    for label, text in doc_sources:
        for name in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", text):
            if name.startswith(known_prefixes) and name not in symbols:
                missing.add((label, name))
    return dead, unused, sorted(missing)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quiet", action="store_true", help="print only the problems found")
    p.add_argument("--strict", action="store_true", help="exit 1 if anything is found")
    args = p.parse_args()

    dead, unused, missing = scan()

    if not args.quiet:
        print("=" * 72)
        print("qtft codebase consistency check")
        print("=" * 72)

    print(f"\nDead functions ({len(dead)}) — defined, never referenced in code:")
    for label, name, lineno, size in dead:
        print(f"  {label}:{lineno:<5} {name:<42} {size:>4} lines")
    if not dead:
        print("  none")

    print(f"\nUnused imports ({len(unused)}):")
    for label, lineno, name in unused:
        print(f"  {label}:{lineno:<5} {name}")
    if not unused:
        print("  none")

    print(f"\nDocumented but missing ({len(missing)}) — named in backticks, no such symbol:")
    for label, name in missing:
        print(f"  {label:<34} `{name}`")
    if not missing:
        print("  none")

    total = len(dead) + len(unused) + len(missing)
    print(f"\n{'=' * 72}\n{total} item(s) found.")
    if args.strict and total:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
