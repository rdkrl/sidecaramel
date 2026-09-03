"""Cross-container Markers2 and USLT roundtrip tests against real
ffmpeg-generated fixtures.

These cover the containers that synthetic-only tests miss: WAV,
AIFF, and OGG.  The corresponding writers were broken in earlier
revisions (bare `ID3(path)` for WAV/AIFF corrupted the RIFF/FORM
header; `FLAC(path)` for OGG silently failed) — these tests fence
against regressions.

Skipped automatically when ffmpeg is not on PATH.
"""
from __future__ import annotations

from pathlib import Path


# ---- container-integrity smoke tests ----------------------------

def test_wav_markers2_write_preserves_riff_header(ffmpeg_wav: Path):
    """Writing a Serato GEOB to a WAV must NOT destroy the RIFF
    container.  The reviewer-flagged bug: bare ID3(path).save(path)
    prepended ID3v2 in front of the RIFF header on non-MP3 files."""
    import sidecaramel.tags as t

    assert ffmpeg_wav.read_bytes()[:4] == b"RIFF"
    ok = t.write_serato_geob(str(ffmpeg_wav), "Serato Markers2",
                                b"\x01\x01PAYLOAD", confirm=True)
    assert ok is True
    assert ffmpeg_wav.read_bytes()[:4] == b"RIFF", \
        "WAV RIFF header destroyed by the writer"

    blobs = t.harvest_serato_blobs(str(ffmpeg_wav))
    found = {desc: payload for desc, payload, _ in blobs}
    assert "Serato Markers2" in found
    assert found["Serato Markers2"][:9] == b"\x01\x01PAYLOAD"


def test_aiff_markers2_write_preserves_form_header(ffmpeg_aiff: Path):
    import sidecaramel.tags as t

    assert ffmpeg_aiff.read_bytes()[:4] == b"FORM"
    ok = t.write_serato_geob(str(ffmpeg_aiff), "Serato Markers2",
                                b"\x01\x01PAYLOAD", confirm=True)
    assert ok is True
    assert ffmpeg_aiff.read_bytes()[:4] == b"FORM", \
        "AIFF FORM header destroyed by the writer"

    blobs = t.harvest_serato_blobs(str(ffmpeg_aiff))
    found = {desc: payload for desc, payload, _ in blobs}
    assert "Serato Markers2" in found


def test_ogg_vorbis_blob_writes_and_roundtrips(ffmpeg_ogg: Path):
    """OGG (Vorbis) was silently failing on every write: the writer
    used mutagen.flac.FLAC which cannot open Ogg containers and the
    exception was swallowed."""
    import sidecaramel.tags as t

    head = ffmpeg_ogg.read_bytes()[:4]
    assert head == b"OggS"

    ok = t.write_serato_vorbis_blob(str(ffmpeg_ogg),
                                      "Serato Markers2",
                                      b"\x01\x01PAYLOAD",
                                      confirm=True)
    assert ok is True, "OGG Vorbis blob write should now succeed"

    # File still a valid Ogg container after write
    assert ffmpeg_ogg.read_bytes()[:4] == b"OggS"

    blobs = t.harvest_serato_blobs(str(ffmpeg_ogg))
    found = {desc: payload for desc, payload, _ in blobs}
    assert "Serato Markers2" in found


def test_flac_vorbis_blob_writes_and_roundtrips(ffmpeg_flac: Path):
    import sidecaramel.tags as t

    assert ffmpeg_flac.read_bytes()[:4] == b"fLaC"

    ok = t.write_serato_vorbis_blob(str(ffmpeg_flac),
                                      "Serato Markers2",
                                      b"\x01\x01PAYLOAD",
                                      confirm=True)
    assert ok is True

    blobs = t.harvest_serato_blobs(str(ffmpeg_flac))
    found = {desc: payload for desc, payload, _ in blobs}
    assert "Serato Markers2" in found


# ---- USLT lyrics cross-container --------------------------------

def test_uslt_wav_roundtrip(ffmpeg_wav: Path):
    from sidecaramel.lyrics import (read_embedded_lyrics, write_uslt)

    assert ffmpeg_wav.read_bytes()[:4] == b"RIFF"
    assert write_uslt(str(ffmpeg_wav), "test lyrics",
                        confirm=True) is True
    assert ffmpeg_wav.read_bytes()[:4] == b"RIFF"
    assert read_embedded_lyrics(str(ffmpeg_wav)) == "test lyrics"


def test_uslt_aiff_roundtrip(ffmpeg_aiff: Path):
    from sidecaramel.lyrics import (read_embedded_lyrics, write_uslt)

    assert ffmpeg_aiff.read_bytes()[:4] == b"FORM"
    assert write_uslt(str(ffmpeg_aiff), "test lyrics",
                        confirm=True) is True
    assert ffmpeg_aiff.read_bytes()[:4] == b"FORM"
    assert read_embedded_lyrics(str(ffmpeg_aiff)) == "test lyrics"


def test_uslt_ogg_roundtrip(ffmpeg_ogg: Path):
    from sidecaramel.lyrics import (read_embedded_lyrics, write_uslt)

    assert write_uslt(str(ffmpeg_ogg), "test lyrics",
                        confirm=True) is True
    # OGG round-trips via Vorbis LYRICS comment.
    assert read_embedded_lyrics(str(ffmpeg_ogg)) == "test lyrics"


def test_uslt_flac_roundtrip(ffmpeg_flac: Path):
    from sidecaramel.lyrics import (read_embedded_lyrics, write_uslt)

    assert write_uslt(str(ffmpeg_flac), "test lyrics",
                        confirm=True) is True
    assert read_embedded_lyrics(str(ffmpeg_flac)) == "test lyrics"


def test_uslt_m4a_roundtrip(ffmpeg_m4a: Path):
    from sidecaramel.lyrics import (read_embedded_lyrics, write_uslt)

    assert write_uslt(str(ffmpeg_m4a), "test lyrics",
                        confirm=True) is True
    assert read_embedded_lyrics(str(ffmpeg_m4a)) == "test lyrics"
