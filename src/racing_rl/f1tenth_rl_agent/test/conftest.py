"""Make the package root and this test dir importable for both pytest and colcon."""

import os
import sys

_TEST_DIR = os.path.dirname(__file__)
_PKG_ROOT = os.path.abspath(os.path.join(_TEST_DIR, ".."))

for _p in (_PKG_ROOT, _TEST_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
