"""sidecaramel.inspect — DEPRECATED alias for :mod:`sidecaramel.blobs`.

The Serato-blob harvester was renamed from ``inspect`` to ``blobs``
in 0.1.0 to avoid shadowing the stdlib ``inspect`` module (which made
``from sidecaramel import inspect`` followed by ``inspect.signature(...)``
a confusion factory).

Importing this module emits ``DeprecationWarning``.  Update callers
to ``from sidecaramel import blobs`` or ``from sidecaramel.blobs import
harvest``.  This shim will be removed in 0.2.0.
"""
import warnings as _warnings
_warnings.warn(
    "sidecaramel.inspect is deprecated and will be removed in 0.2.0; "
    "use sidecaramel.blobs instead (same public API, renamed to avoid "
    "shadowing the stdlib `inspect` module).",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the full public surface of sidecaramel.blobs so existing
# callers continue to work for the deprecation window.
from sidecaramel.blobs import *  # noqa: F401, F403
from sidecaramel import blobs as _blobs
# Forward the symbols the harvest CLI / overview / tags use directly.
from sidecaramel.blobs import (  # noqa: F401
    harvest,
    _robust_b64_decode,
    _unwrap_mp4_envelope,
)

# Make ``sidecaramel.inspect.X`` resolve identically to ``sidecaramel.blobs.X``
# for any attribute we forgot to explicitly re-export.
def __getattr__(name):
    return getattr(_blobs, name)
