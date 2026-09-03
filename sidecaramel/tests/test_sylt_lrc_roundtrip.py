"""SYLT (ID3 timed lyrics) and `.lrc` sidecar read/write roundtrips."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_sylt_write_read_roundtrip(minimal_mp3_with_id3: Path):
    from sidecaramel.lyrics import (write_sylt, read_synced_lyrics,
                                          read_embedded_lyrics)

    pairs = [(0, "Some lyrics"), (2000, "right on time"),
              (5500, "and one more line")]
    ok = write_sylt(str(minimal_mp3_with_id3), pairs, confirm=True)
    assert ok is True

    got = read_synced_lyrics(str(minimal_mp3_with_id3))
    assert got == pairs

    plain = read_embedded_lyrics(str(minimal_mp3_with_id3))
    assert plain is not None
    assert "Some lyrics" in plain
    assert "right on time" in plain


def test_sylt_write_unsorted_input_sorts_on_write(
        minimal_mp3_with_id3: Path):
    from sidecaramel.lyrics import write_sylt, read_synced_lyrics

    pairs = [(5000, "later"), (0, "first")]
    write_sylt(str(minimal_mp3_with_id3), pairs, confirm=True)
    assert read_synced_lyrics(str(minimal_mp3_with_id3)) == \
        [(0, "first"), (5000, "later")]


def test_sylt_requires_confirm(minimal_mp3_with_id3: Path):
    from sidecaramel.lyrics import write_sylt

    with pytest.raises(RuntimeError):
        write_sylt(str(minimal_mp3_with_id3), [(0, "x")])


def test_sylt_empty_pairs_returns_false(minimal_mp3_with_id3: Path):
    from sidecaramel.lyrics import write_sylt

    assert write_sylt(str(minimal_mp3_with_id3), [], confirm=True) is False


def test_sylt_unsupported_container_returns_false(tmp_path: Path):
    from sidecaramel.lyrics import write_sylt

    fake = tmp_path / "fixture.flac"
    fake.write_bytes(b"not a real flac")
    assert write_sylt(str(fake), [(0, "x")], confirm=True) is False


def test_lrc_sidecar_write_read_roundtrip(minimal_mp3_with_id3: Path):
    from sidecaramel.lyrics import (write_lrc_sidecar, read_lrc_sidecar,
                                          lrc_sidecar_path)

    pairs = [(0, "Some lyrics"), (2000, "right on time")]
    ok = write_lrc_sidecar(str(minimal_mp3_with_id3), pairs, confirm=True)
    assert ok is True

    sidecar = Path(lrc_sidecar_path(str(minimal_mp3_with_id3)))
    assert sidecar.is_file()
    assert sidecar.suffix == ".lrc"
    assert sidecar.stem == minimal_mp3_with_id3.stem

    got = read_lrc_sidecar(str(minimal_mp3_with_id3))
    assert got == pairs


def test_lrc_sidecar_write_from_text(minimal_mp3_with_id3: Path):
    from sidecaramel.lyrics import write_lrc_sidecar, read_lrc_sidecar

    text = "[00:00.00]Some lyrics\n[00:02.00]right on time\n"
    ok = write_lrc_sidecar(str(minimal_mp3_with_id3), text, confirm=True)
    assert ok is True

    got = read_lrc_sidecar(str(minimal_mp3_with_id3))
    assert got == [(0, "Some lyrics"), (2000, "right on time")]


def test_lrc_sidecar_missing_returns_none(minimal_mp3_with_id3: Path):
    from sidecaramel.lyrics import read_lrc_sidecar

    assert read_lrc_sidecar(str(minimal_mp3_with_id3)) is None


def test_lrc_sidecar_requires_confirm(minimal_mp3_with_id3: Path):
    from sidecaramel.lyrics import write_lrc_sidecar

    with pytest.raises(RuntimeError):
        write_lrc_sidecar(str(minimal_mp3_with_id3), [(0, "x")])


def test_lrc_sidecar_empty_returns_false(minimal_mp3_with_id3: Path):
    from sidecaramel.lyrics import write_lrc_sidecar

    assert write_lrc_sidecar(str(minimal_mp3_with_id3), [],
                               confirm=True) is False
    assert write_lrc_sidecar(str(minimal_mp3_with_id3), "   ",
                               confirm=True) is False
