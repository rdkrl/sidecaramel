"""lyrics + cover-art write/read with MIME assertion."""
from __future__ import annotations

from pathlib import Path


# A 1×1 transparent PNG (minimum valid; useful as a fake cover-art).
_PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
    b"\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc\xfa\xcf\x00\x00\x00\x02\x00\x01"
    b"\xe5'\xde\xfc"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_uslt_roundtrip(minimal_mp3_with_id3: Path):
    from sidecaramel.lyrics import (write_uslt, read_id3_lyrics,
                                          read_embedded_lyrics)
    body = "[00:01.00]Hello\n[00:02.00]World\n"
    ok = write_uslt(str(minimal_mp3_with_id3), body,
                     lang="eng", desc="", confirm=True)
    assert ok is True

    got = read_id3_lyrics(str(minimal_mp3_with_id3))
    assert got is not None
    assert "Hello" in got
    assert "World" in got

    # Higher-level reader returns the same body
    got2 = read_embedded_lyrics(str(minimal_mp3_with_id3))
    assert got2 is not None
    assert "Hello" in got2


def test_cover_art_roundtrip(minimal_mp3_with_id3: Path):
    from sidecaramel.art import (write_cover_art, read_cover_art)
    ok = write_cover_art(str(minimal_mp3_with_id3),
                          _PNG_1X1, mime="image/png", confirm=True)
    assert ok is True

    art = read_cover_art(str(minimal_mp3_with_id3))
    assert art is not None
    mime, data = art
    assert mime == "image/png"
    assert data == _PNG_1X1


def test_cover_art_mime_autodetect(minimal_mp3_with_id3: Path):
    """No explicit mime → sniffed from magic bytes."""
    from sidecaramel.art import write_cover_art, read_cover_art
    write_cover_art(str(minimal_mp3_with_id3), _PNG_1X1, confirm=True)
    mime, _data = read_cover_art(str(minimal_mp3_with_id3))
    assert mime == "image/png"
