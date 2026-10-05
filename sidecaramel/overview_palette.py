"""sidecaramel.overview_palette — Overview-blob byte vocabulary.

THE BLOB IS A BITMAP. 3840 payload bytes = 240 columns x 16 rows, one
byte per pixel, after a 2-byte header (`01 05`; header and size as
documented in github.com/Holzhaus/serato-tags). Some M4A files carry
one further trailing byte. Scan order is COLUMN-major: one 16-byte
column per time slice, row 0 first. Row 0 is the TOP row; the image is
not mirrored or flipped.

Orientation is measured, not inferred: Serato sends the same overview
to a Rane Performer as the `song_overview_bitmap` datagram (8 chunks,
row-major 16 x 240). Against the file blob of the same track it is
byte-identical in 99.92 % of cells, column-major, with no flip; the
only differences are background (file 1, MIDI 0) and 3 cells where
the file has 223 and the datagram 7. Rows are not mirror images of
each other (row r equals row 15-r in 32 % of cells near the centre
across 93 tracks), so a renderer must not mirror one half.

EACH BYTE INDEXES A 6x6x6 COLOUR CUBE:

    v = 36*a + 6*b + c        with a, b, c in 0..5
    R = 51*a,  G = 51*b,  B = 51*c

    a = bass, b = mid, c = treble

Measured on 93 Serato-analysed tracks: 97 distinct values, every one
below 216 except 223. Calibration tones put the digits on the axes:
an amplitude sweep at 60 Hz peaks at (5, 1, 1), at 1 kHz at (1, 4, 1),
at 8 kHz at (1, 1, 4). Two- and three-tone mixes combine per digit and
independently (60 Hz + 10 kHz -> (3, 0, 3); 60 Hz + 1 kHz -> (3, 3, 0);
all three -> (3, 3, 3)).

BACKGROUND is 1 in most files and 0 in some (two library tracks, one
generated sweep; the Rane datagram always uses 0). Both are treated as
background. 1 also decodes as cube colour (0, 0, 1); it is background
by convention, not by arithmetic.

FLAGGED CELLS: values >= 216 lie outside the cube. Only 223 has been
observed, almost always in row 8. 223 = 216 + 7, and the Rane datagram
carries 7 = (0, 1, 1) in those cells, so the reading used here is
"flag + base colour". Hypothesis — what the flag marks is not
established; it appears in quiet columns and also 240 times in a
full-scale 4 kHz sine.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ---- Layout ---------------------------------------------------------

OVERVIEW_WIDTH = 240            # columns = time slices
OVERVIEW_HEIGHT = 16            # rows, row 0 at the top
OVERVIEW_HEADER = b"\x01\x05"
OVERVIEW_PAYLOAD = OVERVIEW_WIDTH * OVERVIEW_HEIGHT     # 3840


def blob_payload(blob: bytes) -> bytes:
    """The 3840 pixel bytes of an Overview blob, without header and
    without any trailing byte. Raises ValueError when too short."""
    start = len(OVERVIEW_HEADER)
    body = bytes(blob[start:start + OVERVIEW_PAYLOAD])
    if len(body) != OVERVIEW_PAYLOAD:
        raise ValueError(f"Overview blob too short: {len(blob)} bytes")
    return body


def blob_to_pixels(blob: bytes) -> List[List[int]]:
    """Overview blob -> `pixels[row][column]`, row 0 at the top.

    Lossless: `pixels_to_blob` restores the input given the same
    header (a trailing byte past the payload is not part of the image
    and is not carried).
    """
    body = blob_payload(blob)
    return [[body[c * OVERVIEW_HEIGHT + r] for c in range(OVERVIEW_WIDTH)]
            for r in range(OVERVIEW_HEIGHT)]


def pixels_to_blob(pixels: List[List[int]],
                   header: bytes = OVERVIEW_HEADER) -> bytes:
    """`pixels[row][column]` -> Overview blob. Inverse of
    `blob_to_pixels`."""
    out = bytearray(header)
    for c in range(OVERVIEW_WIDTH):
        for r in range(OVERVIEW_HEIGHT):
            out.append(int(pixels[r][c]) & 0xFF)
    return bytes(out)


# ---- The cube -------------------------------------------------------

CUBE_SIZE = 6
CUBE_STEP = 51                   # 255 // (CUBE_SIZE - 1)
CUBE_MAX = CUBE_SIZE ** 3        # 216 — first value outside the cube

BACKGROUND_BYTES = (0, 1)
FLAG_OFFSET = CUBE_MAX           # 223 = FLAG_OFFSET + 7


def is_background(byte: int) -> bool:
    """True for the two observed background values, 0 and 1."""
    return int(byte) in BACKGROUND_BYTES


def split_flag(byte: int) -> Tuple[bool, int]:
    """`(flagged, base)`: values >= 216 are a flag on top of the cube
    value `byte - 216`; everything below is unflagged.
    `split_flag(223) == (True, 7)`."""
    v = int(byte) & 0xFF
    if v >= FLAG_OFFSET:
        return True, v - FLAG_OFFSET
    return False, v


def byte_to_digits(byte: int) -> Tuple[int, int, int]:
    """Cube digits `(a, b, c)` of a byte, each 0..5. A flagged byte
    yields the digits of its base value."""
    _flagged, v = split_flag(byte)
    return v // 36, (v // 6) % CUBE_SIZE, v % CUBE_SIZE


def digits_to_byte(a: int, b: int, c: int) -> int:
    """Pack cube digits into a byte. Digits are clamped to 0..5."""
    def clamp(x: int) -> int:
        return max(0, min(CUBE_SIZE - 1, int(x)))
    return 36 * clamp(a) + 6 * clamp(b) + clamp(c)


def byte_to_cube(byte: int) -> Tuple[int, int, int]:
    """Display colour `(R, G, B)` in 0..255 of a byte. A flagged byte
    shows its base colour, as the Rane datagram does. Background is a
    rendering decision and is not applied here."""
    a, b, c = byte_to_digits(byte)
    return a * CUBE_STEP, b * CUBE_STEP, c * CUBE_STEP


def cube_to_byte(r: int, g: int, b: int) -> int:
    """Nearest cube byte for an `(R, G, B)` colour. Inverse of
    `byte_to_cube` for unflagged bytes."""
    def level(x: int) -> int:
        return (int(x) + CUBE_STEP // 2) // CUBE_STEP
    return digits_to_byte(level(r), level(g), level(b))


def describe_byte(byte: int) -> Dict[str, object]:
    """Full description of an Overview byte, lossless over all 256
    values: `byte_from_description(describe_byte(v)) == v`."""
    v = int(byte) & 0xFF
    flagged, _base = split_flag(v)
    return {"byte": v,
            "digits": list(byte_to_digits(v)),
            "rgb": list(byte_to_cube(v)),
            "flagged": flagged,
            "background": is_background(v)}


def byte_from_description(desc: Dict[str, object]) -> int:
    """Rebuild a byte from `describe_byte()`."""
    a, b, c = desc["digits"]
    return digits_to_byte(a, b, c) + (FLAG_OFFSET if desc["flagged"] else 0)


# ---- Band tiers -> digits (encoder side) ----------------------------
# The encoder measures each band as a tier 0..7 (see
# `overview_encode._amp_to_tier`). Each band drives exactly one digit:
# bass -> a, mid -> b, treble -> c, independently. The tables are the
# own-axis digit of the earlier calibration bytes (bass 36/72/108/144/
# 180, mid 6/12/18/24/30, treble 2/3/4), made monotone; treble tiers
# above 2 were never observed and stay at the last measured digit.

BAND_TIER_DIGIT: Dict[str, Tuple[int, ...]] = {
    "bass":   (1, 2, 2, 3, 3, 4, 4, 5),
    "mid":    (1, 2, 3, 3, 4, 4, 4, 5),
    "treble": (2, 3, 4, 4, 4, 4, 4, 4),
}


def band_tier_to_digit(band: str, tier: Optional[int]) -> int:
    """Digit for a band at `tier`; None (band absent) -> 0."""
    if tier is None:
        return 0
    table = BAND_TIER_DIGIT[band]
    return table[max(0, min(len(table) - 1, int(tier)))]


def mix_band_tiers(bass_tier: Optional[int] = None,
                   mid_tier: Optional[int] = None,
                   treble_tier: Optional[int] = None) -> Optional[int]:
    """Encode per-band tiers into one cube byte, one digit per band.

    Returns None when every band is absent, so the caller decides how
    silence is drawn.
    """
    if bass_tier is None and mid_tier is None and treble_tier is None:
        return None
    return digits_to_byte(band_tier_to_digit("bass", bass_tier),
                          band_tier_to_digit("mid", mid_tier),
                          band_tier_to_digit("treble", treble_tier))
