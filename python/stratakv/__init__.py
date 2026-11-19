"""StrataKV Python package.

Thin layer over the native core exposed through pybind11. The Phase B
surface is introspection only; the Engine facade lands in Phase D (see
roadmap.md and design.md).
"""

from __future__ import annotations

__version__ = "0.1.0"

try:  # pragma: no cover - depends on build configuration
    from stratakv import _core as _native

    version = _native.version
    spec_names = _native.spec_names
    device_kinds = _native.device_kinds
    NATIVE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _native = None
    version = lambda: __version__  # noqa: E731
    spec_names = lambda: []  # noqa: E731
    device_kinds = lambda: []  # noqa: E731
    NATIVE_AVAILABLE = False

__all__ = [
    "NATIVE_AVAILABLE",
    "device_kinds",
    "spec_names",
    "version",
]
