"""sidecaramel.overview_palette — Overview-blob byte vocabulary.

Two layers, and the difference matters:

  * `byte_to_rgb332` / `rgb332_to_byte` — the BIT LAYOUT. Exact,
    total, bijective over all 256 values. Use this whenever a byte
    has to survive a round-trip.
  * the (channel, tier) LUT below — an INTERPRETATION of what a byte
    means, built from calibration sweeps. Partial by construction:
    it covers 21 of 256 values and returns nothing for the rest.

Measured against a Serato-written Overview blob (one track, 240x16,
2026-08-13): the file used **27 distinct byte values, of which the LUT
explains 13**. Frequent real bytes with no LUT entry include 85 (573x),
42 (255x), 86 (164x) and 122 (124x).

The same measurement casts doubt on the LUT's premise. If a byte
encoded only "which band at which amplitude", its value would not
depend on WHERE in the column it sits. It does — strongly. Mean RGB332
fields by distance from the centreline (rows 7/8):

    distance   0      1      2      3      4      5-7
    R       2.58-2.83  2.15   1.81   1.53   1.23   ~1.2
    G          4.85    4.87   4.00   2.97   2.34   ~2.2
    B          1.40    1.44   1.80   2.40   2.60   ~2.6

R and G rise toward the centre — a vertical intensity gradient of a
rendered bar, not a per-band code. Non-silence cells per row fall off
the same way (234/240 at the centre, 2 at distance 7), which is the
bar height varying per column.

B does NOT behave consistently with geometry: measured on a second
Serato-written file it rises steeply toward the centre (2.49 vs 0.12)
where the first one has it roughly flat (1.33 vs 1.41). B follows the
MATERIAL, and the next block says what it follows.

WHERE THE COLOUR IS. Measured across both files, 4058 non-background,
non-marker cells:

  * **G >= R in 4058 of 4058 cells — 100.0 %.** Under a free RGB332
    that would hold about half the time. R and G are not independent:
    R is a brightness floor, G rides on top of it.
  * `g0 = G - R` takes few values (mostly 0, 1, 3) and tracks the MID
    share of the column.
  * `b0 = B` (bits 1-0) tracks the TREBLE share. Holding g0 at 1 and
    dropping b0 from 3 to 2 costs 0.15 of treble share in BOTH files
    independently (0.504 -> 0.352, and 0.484 -> 0.377).
  * amplitude is NOT in the byte — it is bar height. Height vs peak
    correlates 0.86 and 0.64; the R field alone does not (-0.03,
    -0.40).

Bytes also decompose as `base + 36 * tier`: 36 is (1, 1, 0) in RGB332,
so a tier step raises R and G together and leaves B alone. That is the
same statement as "R is a floor and G rides on it", arrived at from
the arithmetic instead of the fields. 27 distinct bytes collapse to 12
bases; 17 collapse to 8.

`byte_from_bands()` and `bands_from_byte()` below implement this.

Treat the (channel, tier) layer as a working hypothesis with known
gaps, and the RGB332 layer as the ground truth.

Historic structure of the LUT, kept for reference:

    8 amp-tiers x 3 channels + 1 silence

Each channel has its own byte vocabulary with ZERO overlap:

    bass    bytes: 36, 72, 79, 108, 115, 144, 151, 180  (8 distinct)
    mid     bytes: 6, 12, 18, 24, 49, 55, 60, ...        (7-8 distinct)
    treble  bytes: 2, 3, 4, 44, 45, ...                  (5-8 distinct)
    silence:       1
    centerline:    42, 43 (upper), 223 (lower-peak)

When multiple bands are simultaneously active in the same time-bin /
row, the per-band bytes composite via per-channel bitwise OR:

    bass-byte 108 (01101100) | treble-byte 3 (00000011) = 111 = 01101111
    bass-byte 108 | treble-byte 2 = 110 = 01101110
"""
from __future__ import annotations

from typing import Optional, Tuple, List, Dict


# ---- Channel vocabularies -----------------------------------------

# Bass channel — 8 amplitude tiers (0..7).  Even-indexed tiers are
# "base" bytes (B=0); odd-indexed are "edge" bytes (with B=3 bleed).
# Pattern: R=G in lockstep, growing by 1 per tier.
BASS_TIERS: Dict[int, int] = {
    0:  36,   # R=1 G=1 B=0     tier 1
    1:  79,   # R=2 G=1 B=3     edge variant
    2:  72,   # R=2 G=2 B=0     base
    3: 115,   # R=3 G=4 B=3     edge
    4: 108,   # R=3 G=3 B=0     base — default full mid-amp
    5: 151,   # R=4 G=5 B=3     edge
    6: 144,   # R=4 G=4 B=0     base — peak normal
    7: 180,   # R=5 G=5 B=0     base — over-amplitude
    # Higher tiers seen at very-end of amp range:
    # 216 (R=6 G=6), 252 (R=7 G=7) — extreme over-amp, rare
}

# Mid channel — 8 tiers, G grows from 1 to 7, R stays ≤1, B bleeds at
# even tiers.
MID_TIERS: Dict[int, int] = {
    0:  6,    # R=0 G=1 B=2
    1: 12,    # R=0 G=3 B=0     edge
    2: 18,    # R=0 G=4 B=2     base
    3: 49,    # R=1 G=4 B=1     mix-edge variant
    4: 24,    # R=0 G=6 B=0     base — high-mid
    5: 55,    # R=1 G=5 B=3     edge with R bleed
    6: 60,    # R=1 G=7 B=0     base — full mid
    7: 30,    # R=0 G=7 B=2     hypothetical extreme — not observed
}

# Treble channel — fewer observed because cmcal sweeps don't reach the
# upper tiers in pure-treble.  Lower-tier bytes confirmed; upper tiers
# are educated guesses extrapolating the bit pattern.
TREBLE_TIERS: Dict[int, int] = {
    0:  2,    # R=0 G=0 B=2     low treble
    1:  3,    # R=0 G=0 B=3     peak treble (B is only 2 bits!)
    2:  4,    # R=0 G=1 B=0     edge (1-bit G bleed)
    3: 44,    # R=1 G=3 B=0     mix-edge
    4: 45,    # R=1 G=3 B=1     edge
    # Tiers 5-7 not reliably observed in cmcal — extrapolated bytes
    # would be in {35, 67, 99} range (R=1 G=0 B=3, R=2 G=0 B=3, ...).
}

# Inverse maps: byte → (channel, tier)
BYTE_TO_CHANNEL_TIER: Dict[int, Tuple[str, int]] = {}
for tier, b in BASS_TIERS.items():
    BYTE_TO_CHANNEL_TIER[b] = ("bass", tier)
for tier, b in MID_TIERS.items():
    BYTE_TO_CHANNEL_TIER[b] = ("mid", tier)
for tier, b in TREBLE_TIERS.items():
    BYTE_TO_CHANNEL_TIER[b] = ("treble", tier)

# Special markers
SILENCE_BYTE = 1
CENTERLINE_UPPER = 43   # offset 7 at low-amp
CENTERLINE_LOWER = 223  # offset 8 at low-amp
CENTERLINE_ALT = 42     # observed alternate at offset 8 in some
                          # tracks (Common, etc.) — uppermost row
                          # marker variant


# ---- Bit layout: the lossless layer ---------------------------------
# RGB332: bits 7..5 = R, bits 4..2 = G, bits 1..0 = B. Total and
# bijective — every byte decomposes and every triple recomposes.

def byte_to_rgb332(byte: int) -> Tuple[int, int, int]:
    """Split a byte into its (R, G, B) fields. Exact for all 256."""
    b = int(byte) & 0xFF
    return ((b >> 5) & 7, (b >> 2) & 7, b & 3)


def rgb332_to_byte(r: int, g: int, b: int) -> int:
    """Pack (R, G, B) fields back into a byte. Inverse of
    `byte_to_rgb332` for r,g in 0..7 and b in 0..3."""
    return ((int(r) & 7) << 5) | ((int(g) & 7) << 2) | (int(b) & 3)


def describe_byte(byte: int) -> Dict[str, object]:
    """Full description of an Overview byte, LOSSLESS by construction.

    Always carries `rgb332`, so `byte_from_description()` can rebuild
    the byte whatever the interpretation layer knows. `tiers` is the
    (channel, tier) reading where one exists and an empty list where
    it does not — an empty list is not an error, it means the LUT has
    no entry for this value.

    `role` names the special values: silence, centerline-upper,
    centerline-lower, centerline-alt.
    """
    b = int(byte) & 0xFF
    roles = {SILENCE_BYTE: "silence",
             CENTERLINE_UPPER: "centerline-upper",
             CENTERLINE_LOWER: "centerline-lower",
             CENTERLINE_ALT: "centerline-alt"}
    marker, r2, g2, b2, low = byte_to_rgb222(b)
    return {"byte": b,
            "rgb332": byte_to_rgb332(b),
            "rgb222": {"marker": marker, "r": r2, "g": g2, "b": b2,
                       "low_bit": low},
            "tiers": [list(ct) for ct in decompose_mix_byte(b)],
            "role": roles.get(b)}


def byte_from_description(desc: Dict[str, object]) -> int:
    """Rebuild a byte from `describe_byte()`. Uses the bit layout, so
    it cannot lose values the (channel, tier) layer does not model."""
    r, g, b = desc["rgb332"]
    return rgb332_to_byte(r, g, b)


# ---- The blob IS a bitmap -------------------------------------------
# 3840 payload bytes = 240 x 16 pixels, one byte per pixel. Not a BMP
# *file* — no `BM` magic, no DIB header, no padded scanlines — and the
# scan order is the other one: COLUMN-major, one 16-byte column per
# time slice, top row first.
#
# Measured on a real blob, column-major against row-major:
#
#                          col-major   row-major
#     top/bottom symmetry     0.52        0.38
#     outer rows silent       0.98        0.72
#     rows 0/15 identical     0.97        0.56
#
# The first column shows it directly: bytes 0..15 of the payload are
# `01 01 01 01 01 01 01 2b df 01 01 01 01 01 01 01` — the centreline
# pair 0x2b/0xdf sitting exactly at rows 7 and 8.

OVERVIEW_WIDTH = 240
OVERVIEW_HEIGHT = 16


def blob_to_pixels(blob: bytes) -> List[List[int]]:
    """Payload of an Overview blob → `pixels[row][column]`.

    Skips the 2-byte blob header. Lossless: `pixels_to_blob` restores
    the input given the same header.
    """
    body = blob[2:] if len(blob) == 2 + OVERVIEW_WIDTH * OVERVIEW_HEIGHT \
        else blob
    return [[body[c * OVERVIEW_HEIGHT + r] for c in range(OVERVIEW_WIDTH)]
            for r in range(OVERVIEW_HEIGHT)]


def pixels_to_blob(pixels: List[List[int]],
                   header: bytes = b"\x01\x05") -> bytes:
    """`pixels[row][column]` → Overview blob. Inverse of
    `blob_to_pixels`."""
    out = bytearray(header)
    for c in range(OVERVIEW_WIDTH):
        for r in range(OVERVIEW_HEIGHT):
            out.append(int(pixels[r][c]) & 0xFF)
    return bytes(out)


# ---- 7-bit reading: bit 7 is not colour -----------------------------
# The evidence for this is MEASURED, not architectural. An earlier
# version of this comment argued that MIDI SysEx forces 7-bit data and
# that therefore bit 7 cannot be colour. That argument is withdrawn:
# the device definitions' own datagram templates contain bytes above
# 0x7F — `..1B0501` ends in the BGRA default `C1 86 00 FF`, in all four
# inMusic files. So `report_length` counts LOGICAL record bytes and a
# packing layer sits between the record and the SysEx wire; 8-bit
# values pass through it fine. Cue colours, 32-bit times and the
# overview blob itself all carry high bytes.
#
# What stands is the measurement below: in a real blob the high bit is
# set on 78 of 3840 bytes and every one of them sits in the centreline
# band. That is a marker, not a colour bit — regardless of transport.
#
# All four carry the overview the same way, and the type name is the
# point:
#
#     userio     song_overview_bitmap
#     condition  song_overview_fully_rendered == completed
#     data_type  "8Bit Image"          <- an 8-bit INDEXED IMAGE
#     datagram   report_length 517 = 1 template + 2 frame-number
#                + 2 data-size + 512 payload bytes, multiextent
#
# So the blob is not "waveform data that happens to be bytes" — the
# transport treats it as a paletted bitmap, which is what the 240x16
# column-major layout above already implies.
#
# TWO SEPARATE PATHS, and only the first one is this blob. The same
# devices also carry a live scrolling waveform, which is a different
# datagram with different lifetime and different inputs:
#
#     overview   song_overview_bitmap   "8Bit Image"       517 B,
#                chunked, sent once when rendering completes
#     scroller   (no userio of its own)  "Screen Waveform" 2170 B,
#                whole strip resent per playhead update, driven by
#                deck_audio_data + zoom_level, with loop_start_point /
#                loop_end_point / loop_*_edit_mode as overlay context
#
# The scroller is zoom-dependent and rebuilt from decoded audio, so it
# cannot be this blob. (`Screen Scroll Position` is a third thing
# entirely — library list scrolling, 80 bits.)
#
# Pioneer has both paths too, with its own type names: `CDJ Image` for
# the overview and `CDJ Heightmap Extend 2000NX2` for the scroller.
# The CDJs are `<hid>` devices, 64-byte reports; the inMusic gear is
# `<midi>`. Neither transport constrains the record to 7 bits — see
# the withdrawal above.
#
# Measured on a Serato-written blob (240x16): 3762 of 3840 payload
# bytes are <= 127. All 78 exceptions sit in rows 6-9, i.e. the
# centreline band, and take only three values (128 x70, 223 x7,
# 129 x1). Bit 7 therefore reads as a MARKER, not a colour bit, and
# the colour payload is 7 bits wide.
#
# Within those 7 bits the per-bit statistics pair up: bits 6, 4 and 2
# fall as a cell moves away from the centreline while bits 5, 3 and 1
# rise. That is the signature of three 2-BIT FIELDS whose value moves
# between 2 (binary 10) and 1 (binary 01) — not of 3-3-2 packing.
#
# Reading them as R=bits 6-5, G=bits 4-3, B=bits 2-1 turns the
# dominant vocabulary into greys, which is what an overview should be
# made of and what 3-3-2 does NOT produce:
#
#     byte   7 bits    2-2-2        3-3-2 (for comparison)
#        1   0000001   R0 G0 B0     R0 G0 B1
#       43   0101011   R1 G1 B1     R1 G2 B3
#       85   1010101   R2 G2 B2     R2 G5 B1     <- 573x, grey
#      127   1111111   R3 G3 B3     R3 G7 B3
#      121   1111001   R3 G3 B0     R3 G6 B1     <- centreline, yellow
#
# Bit 0 is left over. It is heavily biased to 1 on the frequent values
# (43/42, 85/84, 79/78 differ only in it) and shows no clean ramp with
# distance, so it is carried as `low_bit` rather than guessed at.
#
# TRANSPARENCY: a pink key (R3 G0 B3 = byte 102/103) is a plausible
# candidate for "draw nothing", but it does NOT occur in the measured
# file — the value acting as background there is 1 (R0 G0 B0), which
# fills the padding rows 0/1/14/15. `OVERVIEW_BACKGROUND_BYTE` names
# what was observed; the pink hypothesis is left untested rather than
# asserted.

MARKER_BIT = 0x80
OVERVIEW_BACKGROUND_BYTE = 1          # observed background / no-data
PINK_KEY_CANDIDATE = 0b1100110        # R3 G0 B3 — hypothesis, unobserved


def byte_to_rgb222(byte: int) -> Tuple[int, int, int, int, int]:
    """Split a byte into `(marker, r, g, b, low_bit)`.

    marker  = bit 7 (centreline flag, not colour)
    r, g, b = 2-bit fields at bits 6-5, 4-3, 2-1
    low_bit = bit 0, unexplained, carried verbatim

    Total and lossless — see `rgb222_to_byte`.
    """
    v = int(byte) & 0xFF
    return ((v >> 7) & 1, (v >> 5) & 3, (v >> 3) & 3, (v >> 1) & 3, v & 1)


def rgb222_to_byte(marker: int, r: int, g: int, b: int,
                   low_bit: int = 0) -> int:
    """Inverse of `byte_to_rgb222`, exact for all 256 values."""
    return (((int(marker) & 1) << 7) | ((int(r) & 3) << 5)
            | ((int(g) & 3) << 3) | ((int(b) & 3) << 1)
            | (int(low_bit) & 1))


# ---- The measured band model ----------------------------------------
# See the module docstring for the measurement this rests on. These two
# are deliberately NOT presented as exact: `bands_from_byte` returns
# tilts on 0..1, not band energies, and the round-trip through
# `byte_from_bands` is lossy because g0 and b0 are coarse. Anything
# that must reproduce a byte goes through `describe_byte` /
# `byte_from_description` instead.

G0_LEVELS = 5          # observed g0 range 0..5, concentrated on 0/1/3
B0_LEVELS = 3          # b0 is 2 bits


def bands_from_byte(byte: int) -> Optional[Dict[str, float]]:
    """Byte -> `{"brightness", "mid_tilt", "treble_tilt"}` on 0..1.

    Returns None for the background byte and for the marker bytes,
    which carry no colour. `brightness` is the R floor; amplitude
    proper is bar height, not this.
    """
    v = int(byte) & 0xFF
    if v == OVERVIEW_BACKGROUND_BYTE:
        return None
    if v & MARKER_BIT and v != 223:
        return None
    r, g, b = (v >> 5) & 7, (v >> 2) & 7, v & 3
    return {"brightness": r / 7.0,
            "mid_tilt": min(1.0, max(0, g - r) / G0_LEVELS),
            "treble_tilt": b / B0_LEVELS}


def byte_from_bands(brightness: float, mid_tilt: float,
                    treble_tilt: float) -> int:
    """Inverse direction, quantised to what the format can hold.

    `G >= R` is enforced, because it held in 4058 of 4058 measured
    cells and a byte that breaks it is not one Serato writes.
    """
    r = max(0, min(7, round(float(brightness) * 7)))
    g0 = max(0, min(G0_LEVELS, round(float(mid_tilt) * G0_LEVELS)))
    b = max(0, min(3, round(float(treble_tilt) * B0_LEVELS)))
    g = min(7, r + g0)                      # G >= R, always
    return ((r & 7) << 5) | ((g & 7) << 2) | (b & 3)


# ---- Encoder: (channel, tier) → byte --------------------------------

def channel_tier_to_byte(channel: str, tier: int) -> Optional[int]:
    """Return the byte value for a (channel, tier) pair.

    Args:
        channel: 'bass' | 'mid' | 'treble'
        tier:    0..7 amplitude tier

    Returns the byte value, or None if the (channel, tier) pair has no
    documented byte (e.g. treble tier 5-7 not observed in cmcal).
    """
    tables = {
        "bass":   BASS_TIERS,
        "mid":    MID_TIERS,
        "treble": TREBLE_TIERS,
    }
    return tables.get(channel, {}).get(int(tier))


# ---- Decoder: byte → (channel, tier) --------------------------------

def byte_to_channel_tier(byte: int) -> Optional[Tuple[str, int]]:
    """Return (channel, tier) for a Serato Overview byte.

    Returns None for byte 1 (silence), 42/43 (centerline-upper),
    223 (centerline-lower) — those are markers, not channel-tier codes.
    """
    if byte in (SILENCE_BYTE, CENTERLINE_UPPER, CENTERLINE_LOWER,
                 CENTERLINE_ALT):
        return None
    if byte in BYTE_TO_CHANNEL_TIER:
        return BYTE_TO_CHANNEL_TIER[byte]
    # Unknown byte — may be a MIX byte (per-channel-OR of two or three
    # band bytes).  Try to decompose.
    decomp = decompose_mix_byte(byte)
    if decomp:
        # Return the dominant channel (highest tier)
        best = max(decomp, key=lambda ct: ct[1])
        return best
    return None


def decompose_mix_byte(byte: int) -> List[Tuple[str, int]]:
    """Decompose a mix-byte into its constituent (channel, tier) pairs.

    Mix bytes are bitwise OR of two or three single-channel bytes.
    e.g. byte 111 = 0x6F = 108 | 3 = (bass tier 4) | (treble tier 1)

    Returns a list of (channel, tier) pairs, EMPTY when the LUT has no
    reading for this byte — which is the common case: measured against
    a real Serato blob, 14 of the 27 byte values it used decompose to
    nothing. An empty result is not an error and must not be treated
    as "this byte is zero". Anything that has to reproduce the byte
    afterwards must go through `describe_byte()` /
    `byte_from_description()`, which carry the bit layout as well.
    """
    if byte in BYTE_TO_CHANNEL_TIER:
        return [BYTE_TO_CHANNEL_TIER[byte]]
    decomp: List[Tuple[str, int]] = []
    # Try all single-channel-byte pairs and triples to find combinations
    # that OR to the target.
    all_singles = [
        ("bass",   t, b) for t, b in BASS_TIERS.items()
    ] + [
        ("mid",    t, b) for t, b in MID_TIERS.items()
    ] + [
        ("treble", t, b) for t, b in TREBLE_TIERS.items()
    ]
    # Pair check
    for c1, t1, b1 in all_singles:
        for c2, t2, b2 in all_singles:
            if c1 == c2:
                continue
            if (b1 | b2) == byte:
                # Found a 2-channel decomposition
                return [(c1, t1), (c2, t2)]
    # Triple check
    for c1, t1, b1 in all_singles:
        for c2, t2, b2 in all_singles:
            for c3, t3, b3 in all_singles:
                if c1 == c2 or c1 == c3 or c2 == c3:
                    continue
                if (b1 | b2 | b3) == byte:
                    return [(c1, t1), (c2, t2), (c3, t3)]
    return decomp


# ---- Mix encoder: combine band-tiers into one byte ------------------

def mix_band_tiers(bass_tier: Optional[int] = None,
                    mid_tier: Optional[int]  = None,
                    treble_tier: Optional[int] = None) -> int:
    """Encode a per-band (tier) tuple into one Serato byte.

    None tier → channel is absent at this position (= 0 contribution).

    Returns:
        Byte value (1..255).  Returns SILENCE_BYTE (=1) if all tiers
        are None or all bytes resolve to 0.
    """
    parts = []
    if bass_tier is not None:
        b = channel_tier_to_byte("bass", bass_tier)
        if b: parts.append(b)
    if mid_tier is not None:
        b = channel_tier_to_byte("mid", mid_tier)
        if b: parts.append(b)
    if treble_tier is not None:
        b = channel_tier_to_byte("treble", treble_tier)
        if b: parts.append(b)
    if not parts:
        return SILENCE_BYTE
    out = 0
    for p in parts:
        out |= p
    return out
