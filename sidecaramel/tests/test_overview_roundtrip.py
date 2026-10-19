"""Serato Overview blob — export → strip → rewrite → byte identity.

The strongest shape of round-trip available for this format: take a
blob, export it into a DIFFERENT encoding (plain text, nothing packed),
strip every Serato blob out of a copy of the file, rebuild the blob
from that text alone, write it back, and require byte identity.

That is stricter than "the writer preserved what it found": it proves
the decode/encode pair is lossless through a foreign representation,
so nothing survives merely by never having been touched.

The text form is the cube reading of `overview_palette`: per byte its
digits (a, b, c) and whether it is flagged (>= 216).
"""
from __future__ import annotations

import json
import os

import pytest

from sidecaramel.overview_palette import (byte_from_description,
                                          describe_byte)
from sidecaramel.tags import (harvest_serato_blobs, write_serato_geob,
                              wipe_serato_geobs)

HEADER = bytes([0x01, 0x05])
N_COLS, ROWS_PER_COL = 240, 16
PAYLOAD = N_COLS * ROWS_PER_COL          # 3840, blob is 3842 with header


def _export_plain(blob: bytes) -> str:
    """Blob → plain text: each byte becomes its cube digits and flag."""
    doc = {"header": list(blob[:len(HEADER)]),
           "rows": [{"digits": describe_byte(b)["digits"],
                     "flagged": describe_byte(b)["flagged"]}
                    for b in blob[len(HEADER):]]}
    return json.dumps(doc)


def _rebuild(plain: str) -> bytes:
    doc = json.loads(plain)
    return bytes(doc["header"]) + bytes(byte_from_description(row)
                                        for row in doc["rows"])


def _blob_covering_every_byte() -> bytes:
    """A payload that contains all 256 byte values, so the round-trip is
    tested over the whole space and not just what one track happens to
    produce."""
    body = bytes((i % 256) for i in range(PAYLOAD))
    assert len(set(body)) == 256
    return HEADER + body


# ── the full chain ───────────────────────────────────────────────────

def test_export_strip_rewrite_is_byte_identical(minimal_mp3_with_id3):
    path = str(minimal_mp3_with_id3)
    blob = _blob_covering_every_byte()

    # 1. real-shaped data in the file, alongside companions that must
    #    also disappear in the strip step.
    write_serato_geob(path, "Serato Overview", blob, confirm=True)
    write_serato_geob(path, "Serato Analysis", b"\x02\x01", confirm=True)
    write_serato_geob(path, "Serato BeatGrid",
                      b"\x01\x00" + b"\x11\x22\x33\x44", confirm=True)
    before = {d: raw for d, raw, _s in harvest_serato_blobs(path)}
    assert before["Serato Overview"] == blob

    # 2. export into a different encoding
    plain = _export_plain(before["Serato Overview"])
    assert "\\x" not in plain and len(plain) > 10 * len(blob)

    # 3. strip the copy completely
    wipe_serato_geobs(path, confirm=True)
    assert harvest_serato_blobs(path) == [] or not [
        d for d, _r, _s in harvest_serato_blobs(path)]

    # 4. rebuild from the text alone and write it back
    rebuilt = _rebuild(plain)
    write_serato_geob(path, "Serato Overview", rebuilt, confirm=True)

    # 5. byte identity
    after = {d: raw for d, raw, _s in harvest_serato_blobs(path)}
    assert after["Serato Overview"] == blob, (
        f"{sum(1 for a, b in zip(rebuilt, blob) if a != b)} bytes differ")


def test_strip_really_removes_everything(minimal_mp3_with_id3):
    """The strip step is load-bearing for the test above — if it left
    the original blob in place, a broken rewrite would still 'pass'."""
    path = str(minimal_mp3_with_id3)
    for name in ("Serato Overview", "Serato Analysis", "Serato Autotags"):
        write_serato_geob(path, name, b"\x01\x02\x03", confirm=True)
    assert len(harvest_serato_blobs(path)) == 3
    wipe_serato_geobs(path, confirm=True)
    assert [d for d, _r, _s in harvest_serato_blobs(path)] == []


# ── Golden: against a real Serato-written file ───────────────────────
# Opt-in via SIDECARAMEL_TEST_AUDIO=/path/to/track-with-serato-tags.
# The blob a synthetic fixture carries is one this package produced;
# only a Serato-written file can show whether the decode model holds
# for bytes Serato actually emits.

_REAL_AUDIO = os.environ.get("SIDECARAMEL_TEST_AUDIO", "")
_skip_real = pytest.mark.skipif(
    not (_REAL_AUDIO and os.path.isfile(_REAL_AUDIO)),
    reason="set SIDECARAMEL_TEST_AUDIO to a Serato-tagged audio file",
)


def _real_overview() -> bytes:
    blobs = {d: raw for d, raw, _s in harvest_serato_blobs(_REAL_AUDIO)}
    if "Serato Overview" not in blobs:
        pytest.skip("file carries no Serato Overview blob")
    return blobs["Serato Overview"]


@_skip_real
def test_real_serato_overview_survives_the_chain(tmp_path):
    """Export → strip → rewrite on a blob Serato itself wrote."""
    import shutil
    ext = os.path.splitext(_REAL_AUDIO)[1].lower()
    if ext not in (".mp3", ".wav", ".aif", ".aiff"):
        pytest.skip("the strip step (wipe_serato_geobs) is ID3-only")
    work = tmp_path / ("work" + ext)
    shutil.copy2(_REAL_AUDIO, work)

    blobs = {d: raw for d, raw, _s in harvest_serato_blobs(str(work))}
    if "Serato Overview" not in blobs:
        pytest.skip("file carries no Serato Overview blob")
    original = blobs["Serato Overview"]
    assert original[:2] == HEADER, "unexpected Overview header"
    assert len(original) - len(HEADER) - PAYLOAD in (0, 1)

    plain = _export_plain(original)
    wipe_serato_geobs(str(work), confirm=True)
    assert [d for d, _r, _s in harvest_serato_blobs(str(work))] == []

    write_serato_geob(str(work), "Serato Overview", _rebuild(plain),
                      confirm=True)
    got = {d: raw for d, raw, _s in harvest_serato_blobs(str(work))}
    assert got["Serato Overview"] == original


@_skip_real
def test_real_overview_stays_inside_the_cube():
    """Every value Serato writes is a cube byte, background, or a
    flagged cell in the centreline row 8."""
    body = _real_overview()[len(HEADER):len(HEADER) + PAYLOAD]
    flagged_rows = {i % ROWS_PER_COL for i, v in enumerate(body) if v >= 216}
    assert flagged_rows <= {8}, sorted(flagged_rows)
    assert {v for v in body if v >= 216} <= {223}


@_skip_real
def test_real_blob_is_column_major_not_row_major():
    """A BMP is row-major; this is not. Column-major leaves the outer
    rows as background — row-major does not."""
    from sidecaramel.overview_palette import blob_to_pixels, is_background
    blob = _real_overview()
    assert blob[:2] != b"BM", "unexpectedly a BMP file header"

    body = blob[len(HEADER):len(HEADER) + PAYLOAD]
    col = [[body[c * 16 + r] for c in range(240)] for r in range(16)]
    row = [[body[r * 240 + c] for c in range(240)] for r in range(16)]

    def outer_background(grid):
        edge = grid[0] + grid[15]
        return sum(1 for v in edge if is_background(v)) / len(edge)

    assert outer_background(col) > outer_background(row)
    assert blob_to_pixels(blob) == col


# ── The lossless layer ───────────────────────────────────────────────

def test_describe_byte_is_lossless_over_the_whole_space():
    for b in range(256):
        assert byte_from_description(describe_byte(b)) == b, f"byte {b}"


def test_blob_pixels_roundtrip():
    from sidecaramel.overview_palette import blob_to_pixels, pixels_to_blob
    blob = _blob_covering_every_byte()
    px = blob_to_pixels(blob)
    assert len(px) == 16 and len(px[0]) == 240
    assert pixels_to_blob(px, header=blob[:2]) == blob


def test_blob_pixels_ignore_a_trailing_byte():
    """Some M4A payloads carry one byte after the 3840; it is not part
    of the image."""
    from sidecaramel.overview_palette import blob_to_pixels
    blob = _blob_covering_every_byte()
    assert blob_to_pixels(blob + b"\x00") == blob_to_pixels(blob)
