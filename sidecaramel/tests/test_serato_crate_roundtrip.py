"""Real-crate round-trip against a Serato-written `.crate` file.

`read_crate` → `write_crate` reproduces a genuine Serato crate
byte-for-byte: the TLV envelope (`vrsn` / `osrt` sort / `ovct` columns /
`otrk` → `ptrk`), the volume-relative path convention, and the column
spec all re-serialise exactly. It is the crate-level counterpart of the
Markers2/BeatGrid/stems byte-parity checks.

The crate carries real user paths, so it is NOT committed; it lives on
the `gold-dropbox` branch under `roundtrip/crates/` (git-ignored), and
this test skips when it is absent (main, CI, a normal clone).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sidecaramel.db import read_crate, write_crate

CRATE = (Path(__file__).parent.parent / "roundtrip" / "crates"
         / "Sidecaramel.crate")

pytestmark = pytest.mark.skipif(
    not CRATE.exists(),
    reason="real Serato .crate (gold-dropbox roundtrip/crates/) not present")


def test_real_crate_write_roundtrip(tmp_path):
    """Decode a real Serato crate, re-serialise it through the public
    writer, and assert the bytes come back identical."""
    real = CRATE.read_bytes()
    r = read_crate(str(CRATE))
    assert r is not None, "real crate failed to decode"

    # Decoded content sanity — concrete values from the real crate.
    assert r["version"] == "1.0/Serato ScratchLive Crate"
    assert r["sort_column"] == "bpm"
    col_names = [c["name"] for c in r["columns"]]
    assert "video track" in col_names          # the crate carries a video column
    assert len(r["track_paths"]) >= 1

    out = tmp_path / "Sidecaramel.crate"
    cols = [(c["name"], c["width"]) for c in r["columns"]]
    assert write_crate(
        str(out), r["track_paths"],
        sort_column=r["sort_column"], sort_reverse=r["sort_reverse"],
        columns=cols, version_str=r["version"],
        confirm=True, serato_known_closed=True) is True
    assert out.read_bytes() == real, \
        "crate did not re-serialise byte-for-byte"
