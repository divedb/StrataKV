"""Connector test configuration: import paths + native-module gate
(mirrors tests/python/conftest.py; STRATAKV_REQUIRE_NATIVE=1 makes a
missing build a hard failure instead of a skip)."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
for path in (os.path.join(REPO_ROOT, "python"),
             os.path.join(REPO_ROOT, "integrations", "vllm", "v0_17"),
             os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)


def _native_present() -> bool:
    try:
        import stratakv._core  # noqa: F401
        return True
    except ImportError:
        return False


if os.environ.get("STRATAKV_REQUIRE_NATIVE") == "1":
    if not _native_present():
        raise SystemExit(
            "STRATAKV_REQUIRE_NATIVE=1 but stratakv._core is not built "
            "— configure with -DSTRATAKV_PYTHON=ON")
else:
    if not _native_present():
        collect_ignore = ["test_connector.py"]
