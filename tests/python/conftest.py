"""Pytest configuration: make the in-repo package + native extension
importable. The extension is built into python/stratakv/ by
-DSTRATAKV_PYTHON=ON; when absent the engine tests SKIP (docs/CI shells
import metadata only) — except under STRATAKV_REQUIRE_NATIVE=1 (used by
CTest, where a missing build is a hard failure, not a silent skip)."""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)


def _native_present() -> bool:
    try:
        import stratakv._core  # noqa: F401
        return True
    except ImportError:
        return False


if os.environ.get("STRATAKV_REQUIRE_NATIVE") == "1":
    if not _native_present():
        pytest.exit(
            "STRATAKV_REQUIRE_NATIVE=1 but stratakv._core is not "
            "built — configure with -DSTRATAKV_PYTHON=ON", returncode=2)

collect_ignore = [] if _native_present() else [
    "test_engine.py",
]
