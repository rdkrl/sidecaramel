"""Regression tests for the crate-writer volume-path handling.

Two bugs found in the 0.1.0 external review, fixed here and locked in:

1. **External-volume `ptrk` duplication** — the crate writer stored
   the per-track path with a bare ``lstrip("/")`` instead of stripping
   the crate's own volume mount root.  A track on a non-boot volume
   therefore round-tripped to a duplicated
   ``/Volumes/DJDrive/Volumes/DJDrive/…`` on read.

2. **`.scrate` clobber** — ``write_crate`` accepted ``.scrate`` paths
   and serialised an ordinary static-crate payload, destroying the
   Smart Crate's rule tags.  ``.scrate`` writes are now refused.
"""
import pytest

from sidecaramel.db import (
    _denormalize_crate_path,
    _normalize_path_to_volume,
    write_crate,
    CrateWriteError,
)


def test_external_volume_ptrk_strips_mount_root():
    abs_p = "/Volumes/DJDrive/Music/Artist - Track.mp3"
    rel = _denormalize_crate_path(abs_p, "/Volumes/DJDrive")
    assert rel == "Music/Artist - Track.mp3"
    # The read-side normaliser must reconstruct the ORIGINAL absolute
    # path — not the duplicated /Volumes/DJDrive/Volumes/DJDrive/… that
    # the 0.1.0 bare-lstrip produced.
    assert _normalize_path_to_volume(rel, "/Volumes/DJDrive") == abs_p


def test_system_volume_ptrk_drops_leading_slash():
    abs_p = "/Users/dj/Music/Artist - Track.mp3"
    rel = _denormalize_crate_path(abs_p, "/")
    assert rel == "Users/dj/Music/Artist - Track.mp3"
    assert _normalize_path_to_volume(rel, "/") == abs_p


def test_denormalize_passes_relative_input_through():
    # An already-relative path is left untouched (no double-strip).
    assert (_denormalize_crate_path("Music/x.mp3", "/Volumes/DJDrive")
            == "Music/x.mp3")


def test_scrate_write_is_refused(tmp_path):
    target = tmp_path / "Smart.scrate"
    with pytest.raises(CrateWriteError):
        write_crate(str(target),
                    ["/Users/dj/Music/x.mp3"],
                    confirm=True,
                    serato_known_closed=True)
    # The Smart Crate must be left untouched.
    assert not target.exists()


def test_static_crate_write_still_accepted(tmp_path):
    target = tmp_path / "Static.crate"
    ok = write_crate(str(target),
                     ["/Users/dj/Music/x.mp3"],
                     confirm=True,
                     serato_known_closed=True)
    assert ok is True
    assert target.exists() and target.stat().st_size > 0
