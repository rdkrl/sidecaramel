"""write_crate → read_crate roundtrip preserves track
paths, sort column + reverse flag, and column display spec exactly.
Also covers Quirk B: the flat `sort_column` / `sort_reverse` aliases
must match what we wrote.
"""
from __future__ import annotations

from pathlib import Path


def test_crate_content_roundtrip(serato_crate_path: Path):
    from sidecaramel.db import write_crate, read_crate

    # Absolute paths used here because read_crate normalises
    # volume-relative ptrk to volume-rooted absolute.  Using
    # absolutes ensures the roundtrip compares byte-for-byte.
    tracks = [
        "/Music/Beginner - Ahnma.m4a",
        "/Music/Outkast - Hey Ya.mp3",
        "/Music/Tribe - Scenario.flac",
    ]
    ok = write_crate(
        str(serato_crate_path),
        tracks,
        sort_column="song",
        sort_reverse=False,
        columns=[
            ("song", "200"),
            ("artist", "180"),
            ("bpm", "60"),
        ],
        confirm=True,
        allow_serato_running=True,
    )
    assert ok is True
    assert serato_crate_path.exists()

    read = read_crate(str(serato_crate_path))
    assert read is not None
    assert read["track_paths"] == tracks
    # Nested form (legacy)
    assert read["sort"] == {"column": "song", "reverse": False}
    # Flat alias form (Quirk B fix)
    assert read["sort_column"] == "song"
    assert read["sort_reverse"] is False
    # Columns in order
    col_names = [c["name"] for c in read["columns"]]
    assert col_names == ["song", "artist", "bpm"]


def test_crate_sort_reverse_roundtrip(serato_crate_path: Path):
    from sidecaramel.db import write_crate, read_crate
    write_crate(
        str(serato_crate_path),
        ["/a.mp3", "/b.mp3"],
        sort_column="bpm",
        sort_reverse=True,
        confirm=True,
        allow_serato_running=True,
    )
    read = read_crate(str(serato_crate_path))
    assert read["sort_column"] == "bpm"
    assert read["sort_reverse"] is True
    assert read["sort"]["reverse"] is True


def test_write_crate_requires_confirm(serato_crate_path: Path):
    """confirm=True gate must raise without it."""
    from sidecaramel.db import write_crate, CrateWriteError
    import pytest
    with pytest.raises(CrateWriteError):
        write_crate(str(serato_crate_path), ["/a.mp3"], confirm=False,
                     allow_serato_running=True)
    assert not serato_crate_path.exists()
