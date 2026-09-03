"""Serato Overview blob — export → strip → rewrite → byte identity.

The strongest shape of round-trip available for this format: take a
blob, export it into a DIFFERENT encoding (plain text, nothing packed),
strip every Serato blob out of a copy of the file, rebuild the blob
from that text alone, write it back, and require byte identity.

That is stricter than "the writer preserved what it found": it proves
the decode/encode pair is lossless through a foreign representation,
so nothing survives merely by never having been touched.

Two rebuild paths are exercised, and both must reproduce the input:
  * `rgb332`  — the bit packing (3 bits bass, 3 mid, 2 treble)
  * `tiers`   — the semantic band-tier model of `overview_palette`
"""
from __future__ import annotations

import json
import os

import pytest

from sidecaramel.overview_palette import decompose_mix_byte, mix_band_tiers
from sidecaramel.tags import (harvest_serato_blobs, write_serato_geob,
                              wipe_serato_geobs)

HEADER = bytes([0x01, 0x05])
N_COLS, ROWS_PER_COL = 240, 16
PAYLOAD = N_COLS * ROWS_PER_COL          # 3840, blob is 3842 with header


def _export_plain(blob: bytes) -> str:
    """Blob → plain text. No packed byte survives: each one becomes its
    band decomposition plus its RGB332 triple, and a literal only where
    the byte is a constant (silence / centerline) rather than a mix."""
    doc = {"header": list(blob[:len(HEADER)]), "rows": []}
    for b in blob[len(HEADER):]:
        tiers = [[ch, t] for ch, t in decompose_mix_byte(b)]
        row = {"tiers": tiers,
               "rgb332": [(b >> 5) & 7, (b >> 2) & 7, b & 3]}
        if not tiers:
            row["literal"] = b
        doc["rows"].append(row)
    return json.dumps(doc)


def _rebuild(plain: str, *, via: str) -> bytes:
    doc = json.loads(plain)
    out = bytearray(bytes(doc["header"]))
    for row in doc["rows"]:
        if via == "rgb332":
            r, g, b = row["rgb332"]
            out.append(((r & 7) << 5) | ((g & 7) << 2) | (b & 3))
        else:
            kw = {f"{ch}_tier": t for ch, t in row["tiers"]}
            out.append(mix_band_tiers(**kw) if kw else row["literal"])
    return bytes(out)


def _blob_covering_every_byte() -> bytes:
    """A payload that contains all 256 byte values, so the round-trip is
    tested over the whole space and not just what one track happens to
    produce."""
    body = bytes((i % 256) for i in range(PAYLOAD))
    assert len(set(body)) == 256
    return HEADER + body


# ── the model itself ─────────────────────────────────────────────────

def test_every_band_mix_byte_maps_back_exactly():
    """Whatever `decompose_mix_byte` claims to understand, it must be
    able to rebuild. A byte it decomposes but cannot reproduce would be
    a silently wrong colour on every render."""
    decomposed = failed = 0
    for b in range(256):
        tiers = decompose_mix_byte(b)
        if not tiers:
            continue
        decomposed += 1
        kw = {f"{ch}_tier": t for ch, t in tiers}
        if mix_band_tiers(**kw) != b:
            failed += 1
    assert decomposed >= 64, "band model covers implausibly few bytes"
    assert failed == 0, f"{failed} of {decomposed} band bytes do not rebuild"


# ── the full chain ───────────────────────────────────────────────────

@pytest.mark.parametrize("via", ["rgb332", "tiers"])
def test_export_strip_rewrite_is_byte_identical(minimal_mp3_with_id3, via):
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
    rebuilt = _rebuild(plain, via=via)
    write_serato_geob(path, "Serato Overview", rebuilt, confirm=True)

    # 5. byte identity
    after = {d: raw for d, raw, _s in harvest_serato_blobs(path)}
    assert after["Serato Overview"] == blob, (
        f"via {via}: "
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


@_skip_real
@pytest.mark.parametrize("via", ["rgb332", "tiers"])
def test_real_serato_overview_survives_the_chain(tmp_path, via):
    """Export → strip → rewrite on a blob Serato itself wrote."""
    import shutil
    work = tmp_path / "work.mp3"
    shutil.copy2(_REAL_AUDIO, work)

    blobs = {d: raw for d, raw, _s in harvest_serato_blobs(str(work))}
    if "Serato Overview" not in blobs:
        pytest.skip("file carries no Serato Overview blob")
    original = blobs["Serato Overview"]
    assert original[:2] == HEADER, "unexpected Overview header"
    assert len(original) == len(HEADER) + PAYLOAD

    plain = _export_plain(original)
    wipe_serato_geobs(str(work), confirm=True)
    assert [d for d, _r, _s in harvest_serato_blobs(str(work))] == []

    write_serato_geob(str(work), "Serato Overview",
                      _rebuild(plain, via=via), confirm=True)
    got = {d: raw for d, raw, _s in harvest_serato_blobs(str(work))}
    assert got["Serato Overview"] == original


@_skip_real
def test_real_overview_byte_vocabulary_is_covered_or_literal():
    """Every byte value Serato writes must either decompose into band
    tiers or be carried as a literal. This test does not require the
    band model to explain them — it records how much of the real
    vocabulary it does explain, and fails only if a byte would be lost
    entirely."""
    blobs = {d: raw for d, raw, _s in harvest_serato_blobs(_REAL_AUDIO)}
    if "Serato Overview" not in blobs:
        pytest.skip("file carries no Serato Overview blob")
    body = blobs["Serato Overview"][len(HEADER):]
    vocabulary = sorted(set(body))
    modelled = [b for b in vocabulary if decompose_mix_byte(b)]
    # The round-trip must not lose anything…
    rebuilt = _rebuild(_export_plain(blobs["Serato Overview"]), via="tiers")
    assert rebuilt == blobs["Serato Overview"]
    # …and the coverage is reported, not asserted at a threshold.
    print(f"\n  real byte vocabulary: {len(vocabulary)} values, "
          f"{len(modelled)} with a band decomposition "
          f"({100 * len(modelled) / len(vocabulary):.0f} %)")


# ── The lossless layer ───────────────────────────────────────────────

def test_describe_byte_is_lossless_over_the_whole_space():
    """`decompose_mix_byte` legitimately returns nothing for most
    bytes. `describe_byte` must still round-trip all 256, or every
    consumer has to invent its own fallback — which is exactly how a
    byte gets silently replaced by zero."""
    from sidecaramel.overview_palette import (byte_from_description,
                                              describe_byte)
    for b in range(256):
        assert byte_from_description(describe_byte(b)) == b, f"byte {b}"


def test_rgb332_pair_is_bijective():
    from sidecaramel.overview_palette import byte_to_rgb332, rgb332_to_byte
    seen = set()
    for b in range(256):
        r, g, bl = byte_to_rgb332(b)
        assert 0 <= r <= 7 and 0 <= g <= 7 and 0 <= bl <= 3
        assert rgb332_to_byte(r, g, bl) == b
        seen.add((r, g, bl))
    assert len(seen) == 256


@_skip_real
def test_real_blob_shows_a_vertical_gradient():
    """The measurement that undercuts the per-band reading of a byte:
    a byte's fields depend on WHERE in the column it sits.

    R and G rise toward the centreline — checked on two Serato-written
    files and true in both. B is NOT asserted: it falls toward the
    centre in one file (1.33 vs 1.41) and rises steeply in the other
    (2.49 vs 0.12), so it tracks the track's content, not the
    geometry. An earlier version of this test required B to fall and
    passed only because it had seen a single file."""
    import statistics
    blobs = {d: raw for d, raw, _s in harvest_serato_blobs(_REAL_AUDIO)}
    if "Serato Overview" not in blobs:
        pytest.skip("file carries no Serato Overview blob")
    body = blobs["Serato Overview"][len(HEADER):]

    def fields(distance):
        vals = [body[c * ROWS_PER_COL + r]
                for c in range(N_COLS) for r in range(ROWS_PER_COL)
                if min(abs(r - 7), abs(r - 8)) == distance and
                body[c * ROWS_PER_COL + r] != 1]
        if not vals:
            return None
        return (statistics.mean((v >> 5) & 7 for v in vals),
                statistics.mean((v >> 2) & 7 for v in vals),
                statistics.mean(v & 3 for v in vals))

    centre, far = fields(0), fields(4)
    assert centre and far
    assert centre[0] > far[0] + 0.5, "R does not rise toward the centre"
    assert centre[1] > far[1] + 1.0, "G does not rise toward the centre"


# ── The 7-bit reading ────────────────────────────────────────────────

def test_rgb222_is_lossless():
    from sidecaramel.overview_palette import byte_to_rgb222, rgb222_to_byte
    for b in range(256):
        assert rgb222_to_byte(*byte_to_rgb222(b)) == b, f"byte {b}"


def test_rgb222_turns_the_common_values_into_greys():
    """The reason to prefer 2-2-2 over 3-3-2: it makes the dominant
    vocabulary grey, which is what a waveform overview is built from.
    Under 3-3-2 the same bytes are not neutral at all."""
    from sidecaramel.overview_palette import byte_to_rgb222, byte_to_rgb332
    for byte, level in ((1, 0), (43, 1), (85, 2), (127, 3)):
        _m, r, g, b, _lo = byte_to_rgb222(byte)
        assert (r, g, b) == (level, level, level), (
            f"byte {byte} is not grey level {level} under 2-2-2")
    # …and 85 is emphatically not neutral under 3-3-2.
    assert len(set(byte_to_rgb332(85))) == 3


@_skip_real
def test_marker_bit_is_confined_to_the_centreline():
    """Bit 7 is not a colour bit: in real output it only appears in the
    rows around the centreline. If that ever stops holding, the 7-bit
    reading needs revisiting."""
    blobs = {d: raw for d, raw, _s in harvest_serato_blobs(_REAL_AUDIO)}
    if "Serato Overview" not in blobs:
        pytest.skip("file carries no Serato Overview blob")
    body = blobs["Serato Overview"][len(HEADER):]
    marked_rows = {i % ROWS_PER_COL for i, b in enumerate(body) if b & 0x80}
    assert marked_rows, "no marker bits at all — unexpected"
    assert marked_rows <= {6, 7, 8, 9}, (
        f"marker bit outside the centreline band: {sorted(marked_rows)}")
    share = sum(1 for b in body if b & 0x80) / len(body)
    assert share < 0.05, f"{share:.1%} of bytes carry the marker bit"


# ── The blob as a bitmap ─────────────────────────────────────────────

def test_blob_pixels_roundtrip():
    from sidecaramel.overview_palette import blob_to_pixels, pixels_to_blob
    blob = _blob_covering_every_byte()
    px = blob_to_pixels(blob)
    assert len(px) == 16 and len(px[0]) == 240
    assert pixels_to_blob(px, header=blob[:2]) == blob


@_skip_real
def test_real_blob_is_column_major_not_row_major():
    """A BMP is row-major; this is not. Column-major puts the
    centreline at rows 7/8 and leaves the outer rows empty — row-major
    does neither."""
    from sidecaramel.overview_palette import blob_to_pixels
    blobs = {d: raw for d, raw, _s in harvest_serato_blobs(_REAL_AUDIO)}
    if "Serato Overview" not in blobs:
        pytest.skip("file carries no Serato Overview blob")
    blob = blobs["Serato Overview"]
    assert blob[:2] != b"BM", "unexpectedly a BMP file header"

    body = blob[len(HEADER):]
    col = [[body[c * 16 + r] for c in range(240)] for r in range(16)]
    row = [[body[r * 240 + c] for c in range(240)] for r in range(16)]

    def outer_silence(grid):
        edge = grid[0] + grid[15]
        return sum(1 for v in edge if v == 1) / len(edge)

    assert outer_silence(col) > 0.9
    assert outer_silence(col) > outer_silence(row) + 0.2
    # …and the public helper agrees with the column-major reading.
    assert blob_to_pixels(blob) == col


# ---- the measured band model ---------------------------------------

def test_g_never_below_r_in_generated_bytes():
    """The invariant that made the model: G >= R held in 4058 of 4058
    measured cells, so the encoder must never emit a byte that breaks
    it — not for any input, including out-of-range ones."""
    from sidecaramel.overview_palette import byte_from_bands
    for bright in (0.0, 0.1, 0.5, 0.9, 1.0, -1.0, 2.0):
        for mid in (0.0, 0.25, 0.5, 1.0, 3.0):
            for treb in (0.0, 0.33, 0.67, 1.0):
                b = byte_from_bands(bright, mid, treb)
                assert 0 <= b <= 255
                assert ((b >> 2) & 7) >= ((b >> 5) & 7), (bright, mid, treb, b)


def test_bands_from_byte_skips_what_carries_no_colour():
    from sidecaramel.overview_palette import (bands_from_byte,
                                              OVERVIEW_BACKGROUND_BYTE)
    assert bands_from_byte(OVERVIEW_BACKGROUND_BYTE) is None
    assert bands_from_byte(128) is None, "pure marker byte"
    assert bands_from_byte(223) is not None, "223 fits the colour ramp"
    d = bands_from_byte(85)
    assert set(d) == {"brightness", "mid_tilt", "treble_tilt"}
    assert all(0.0 <= v <= 1.0 for v in d.values())


def test_treble_tilt_is_monotone_in_the_low_two_bits():
    """b0 tracks the treble share; the accessor must not scramble the
    order it was measured in."""
    from sidecaramel.overview_palette import bands_from_byte
    base = 0b010_100_00          # R2 G5, vary only B
    tilts = [bands_from_byte(base | b)["treble_tilt"] for b in range(4)]
    assert tilts == sorted(tilts) and tilts[0] < tilts[-1]


@_skip_real
def test_real_blob_obeys_g_ge_r():
    """Golden: every non-background, non-marker cell of a real
    Serato-written blob keeps G >= R."""
    blobs = {d: raw for d, raw, _s in harvest_serato_blobs(_REAL_AUDIO)}
    blob = blobs.get("Serato Overview")
    if not blob:
        pytest.skip("file carries no Serato Overview blob")
    body = blob[2:]
    checked = bad = 0
    for v in body:
        if v == 1 or (v & 0x80 and v != 223):
            continue
        checked += 1
        if ((v >> 2) & 7) < ((v >> 5) & 7):
            bad += 1
    assert checked > 100, "blob too empty to conclude anything"
    assert bad == 0, f"{bad} of {checked} cells break G >= R"
