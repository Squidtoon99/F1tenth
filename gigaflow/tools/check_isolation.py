#!/usr/bin/env python3
"""Fail if gigaflow runtime modules import training/ or f1tenth_policy."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

FORBIDDEN_PREFIXES = (
    "training",
    "f1tenth_env",
    "f1tenth_sim",
    "qrsac",
    "f1tenth_policy",
    "f1tenth_contract",
)


def _is_forbidden(module: str | None) -> bool:
    if not module:
        return False
    return any(module == p or module.startswith(p + ".") for p in FORBIDDEN_PREFIXES)


def check_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    hits.append(f"{path}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if _is_forbidden(node.module):
                hits.append(f"{path}:{node.lineno}: from {node.module} import ...")
    return hits


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "src" / "gigaflow_f1tenth"
    errors: list[str] = []
    for path in sorted(root.rglob("*.py")):
        errors.extend(check_file(path))
    if errors:
        print("Forbidden imports detected:", file=sys.stderr)
        for line in errors:
            print(line, file=sys.stderr)
        return 1
    print(f"isolation ok: scanned {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
