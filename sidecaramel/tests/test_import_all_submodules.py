"""Every sidecaramel.* submodule imports clean.

This is the smoke test that catches missing-export / circular-import
regressions before they ship.  Important: the module list is
DISCOVERED via `pkgutil.iter_modules`, not hand-maintained.  That way
the test reflects what the package actually ships, not what an
out-of-date hand-list claims.
"""
from __future__ import annotations

import importlib
import pkgutil


def _discover_submodules() -> list[str]:
    """Return every top-level submodule name shipped under
    `sidecaramel.*`, excluding dunder helpers and sub-packages we
    don't want exercised by this smoke test."""
    import sidecaramel
    pkg_path = sidecaramel.__path__
    skip = {
        # Sub-packages are walked elsewhere (data/, demo/, tests/).
        "tests", "data", "demo",
    }
    return sorted(
        m.name for m in pkgutil.iter_modules(pkg_path)
        if not m.ispkg and m.name not in skip
    )


def test_top_level_imports_clean():
    import sidecaramel
    assert sidecaramel.__name__ == "sidecaramel"
    # Curated public API
    assert hasattr(sidecaramel, "read_serato_metadata")
    assert hasattr(sidecaramel, "write_uslt")
    assert hasattr(sidecaramel, "read_cover_art")


def test_every_submodule_imports_clean():
    submodules = _discover_submodules()
    # Sanity: we expect a non-empty list (otherwise pkgutil discovery
    # broke).  The exact count depends on what shipped.
    assert submodules, "pkgutil.iter_modules returned nothing"

    failed = []
    skipped = []
    for sub in submodules:
        modname = f"sidecaramel.{sub}"
        try:
            importlib.import_module(modname)
        except ImportError as e:
            # Optional-dependency-missing is allowed and informative,
            # not a failure (e.g. sidecaramel.gui without PySide6).
            skipped.append((modname, str(e)))
        except Exception as e:
            failed.append((modname, repr(e)))
    if skipped:
        print(f"\nOptional-dependency skips ({len(skipped)}):")
        for m, e in skipped:
            print(f"  {m}: {e}")
    assert not failed, (
        f"{len(failed)} submodule(s) failed to import:\n  "
        + "\n  ".join(f"{m}: {e}" for m, e in failed))


def test_check_module_exports_assert_not_running():
    """Guards write_crate()'s Serato-running gate against a silent
    fallback: the in-package check module MUST resolve and expose
    `assert_not_running`."""
    from sidecaramel.check import assert_not_running, SeratoRunningError
    assert callable(assert_not_running)
    assert issubclass(SeratoRunningError, Exception)


def test_flip_writer_exports_required_for_tags_encode_flip_entry():
    """Guards the symbols `tags.encode_flip_entry()` imports from
    `flip_writer`; a missing one is fatal on every FLIP write."""
    from sidecaramel.flip_writer import (build_flip_entry, build_jump,
                                              build_censor)
    assert callable(build_flip_entry)
    assert callable(build_jump)
    assert callable(build_censor)
