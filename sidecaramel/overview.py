"""sidecaramel.overview — Overview-blob waveform renderer.

Three rendering modes:

  grayscale  — safe fallback: byte = darkness ramp.  Always
               correct-enough for "show me the rough waveform shape".
  color      — column-spread-aware HLS hue: hue derived from
               per-column non-silence-byte-count (a proxy for
               frequency-band activity); lightness from per-pixel
               byte value.
  alpha      — HSL hue ramp blue → cyan → green → yellow → orange →
               red across the byte range, plus a low-saturation
               stair for bytes 1..4 (silence + low-amplitude residue).

Layout constants (Holzhaus-verified):
    blob = 3842 bytes total
    [0:2]      version header (typically 0x01 0x05)
    [2:3842]   240 chunks × 16 bytes each
                each chunk = 1 vertical column of the waveform
                offsets 0..2 + 13..15 are padding (silence)
                offsets 3..12 are the actual amplitude window
                offset 4..7 in particular drives the bright pixels
                "G2 geometry": 4 top + off-white center + 4 mirror
"""
from __future__ import annotations

import colorsys
from typing import List, Optional, Tuple


# --- Layout constants ------------------------------------------------

OVERVIEW_BLOB_HEADER = 2
OVERVIEW_CHUNK_SIZE = 16
OVERVIEW_NUM_CHUNKS = 240

# G2 (active waveform) geometry: 4 top + off-white centerline + 4 mirror
# = 9 wide per column.
OVERVIEW_WIDTH_G2 = 9
TOP_OFFSETS = (4, 5, 6, 7)
OFF_WHITE = (235, 230, 215)
OVERVIEW_PADDING_TOP = 3
OVERVIEW_PADDING_BOTTOM = 3

# Default backgrounds for color modes
BG_BLACK = (0, 0, 0)
PINK_RGB = (255, 0, 255)


# --- Internal helpers ------------------------------------------------

def _alpha_blend(fg: Tuple[int, int, int],
                  bg: Tuple[int, int, int],
                  alpha: float) -> Tuple[int, int, int]:
    return (int(fg[0] * alpha + bg[0] * (1 - alpha)),
            int(fg[1] * alpha + bg[1] * (1 - alpha)),
            int(fg[2] * alpha + bg[2] * (1 - alpha)))


def _column_spread(chunk: bytes) -> int:
    """Count STRONG-SIGNAL bytes (>4) in this chunk's active rows
    (offsets 3..12).  Bytes 0..4 are the pink/alpha-stair zone and
    don't count as frequency-band evidence."""
    count = 0
    for o in range(OVERVIEW_PADDING_TOP,
                    OVERVIEW_CHUNK_SIZE - OVERVIEW_PADDING_BOTTOM):
        if chunk[o] > 4:
            count += 1
    return count


def _spread_to_hue(spread: int) -> float:
    """Map column-spread (0..10) → hue (degrees, 0=red, 240=blue).

    Hypothesis (iteration 5): spread = vertical reach of the active
    amplitude column = frequency-band proxy.
      0..1 active → blue (240°)   treble / sharp transient
      2..3 active → cyan/green (180..150°)
      4..5 active → green/yellow (120..90°)
      6..7 active → orange (60..30°)
      8..10 active → red (0°)      bass / sustained low-freq
    """
    s = max(0, min(10, spread))
    return (10 - s) / 10.0 * 240.0


def _byte_lightness(b: int) -> float:
    """Map byte intensity to lightness 0.20..0.65 (skipping alpha-
    stair bytes 0..4)."""
    if b <= 4:
        return 0.0
    t = (b - 5) / (255 - 5)
    return 0.20 + 0.45 * t


def _render_color_pixel(chunk: bytes, offset: int,
                          spread: int,
                          bg: Tuple[int, int, int] = BG_BLACK
                          ) -> Tuple[int, int, int]:
    """Color-mode pixel: alpha-stair for bytes 1..4, HLS for 5..255."""
    b = chunk[offset]
    if b == 0:
        return bg
    if 1 <= b <= 4:
        # 4-level alpha-stair: byte 1 = fully opaque pink,
        # byte 4 = fully transparent
        alpha = (4 - b) / 3.0
        return _alpha_blend(PINK_RGB, bg, alpha)
    hue = _spread_to_hue(spread)
    L = _byte_lightness(b)
    r, g, bl = colorsys.hls_to_rgb(hue / 360, L, 1.0)
    return (int(r * 255), int(g * 255), int(bl * 255))


def _alpha_palette() -> List[Tuple[int, int, int]]:
    """HSL hue ramp blue→red across byte range 5..255, plus pink-
    alpha-stair for bytes 1..4.

    Used by the `alpha` rendering mode.  Returns a 256-entry palette
    suitable for 8-bit indexed images.
    """
    pal: List[Tuple[int, int, int]] = []
    for b in range(256):
        if b == 0:
            pal.append(BG_BLACK)
            continue
        if 1 <= b <= 4:
            alpha = (4 - b) / 3.0
            pal.append(_alpha_blend(PINK_RGB, BG_BLACK, alpha))
            continue
        t = (b - 5) / (255 - 5)
        # blue 240° → cyan → green → yellow → orange → red 0°
        hue = (1 - t) * 240.0
        r, g, bl = colorsys.hls_to_rgb(hue / 360, 0.45, 1.0)
        pal.append((int(r * 255), int(g * 255), int(bl * 255)))
    return pal


def _serato_palette_calibrated() -> List[Tuple[int, int, int]]:
    """Hand-tuned byte → RGB lookup table for the Overview blob's
    256-byte vocabulary.

    Each anchor byte is mapped to the RGB colour of its dominant
    frequency-band combination, with additive band mixing:

        bass-only         → red
        mid-only          → green
        treble-only       → blue
        bass+mid          → yellow (additive R+G)
        bass+treble       → magenta (additive R+B)
        mid+treble        → cyan (additive G+B)
        bass+mid+treble   → white
        bass-dominant+mid → orange
        broadband         → grey

    Unobserved bytes fall back to nearest-anchor by numerical
    distance.
    """
    # Anchor table: byte_val → RGB
    anchors = {
        0: (0, 0, 0),
        1: (0, 0, 0),                # silence/padding
        2: (40, 140, 230),           # treble darker
        3: (40, 140, 255),           # treble bright blue
        6: (240, 200, 30),           # bass+mid faint yellow
        7: (0, 220, 200),            # mid+treble cyan
        12: (60, 220, 60),           # mid green darker
        14: (0, 220, 200),           # mid+treble cyan
        18: (60, 220, 60),           # mid bright green
        20: (0, 220, 200),           # mid+treble cyan
        21: (0, 220, 200),           # mid+treble cyan
        36: (180, 180, 180),         # broadband grey
        37: (255, 0, 255),           # bass+treble magenta
        38: (255, 0, 255),           # bass+treble magenta
        42: (240, 200, 30),          # bass+mid yellow
        43: (200, 200, 200),         # centerline grey-white
        44: (255, 0, 255),           # bass+treble magenta
        48: (240, 200, 30),          # bass+mid yellow
        49: (240, 200, 30),          # bass+mid yellow
        50: (0, 220, 200),           # mid+treble cyan
        73: (255, 0, 255),           # bass+treble magenta
        74: (255, 0, 255),           # bass+treble magenta
        78: (240, 200, 30),          # bass+mid yellow
        79: (220, 220, 220),         # bass+mid+treble white
        80: (255, 0, 255),           # bass+treble magenta
        84: (240, 200, 30),          # bass+mid yellow
        85: (240, 200, 30),          # bass+mid yellow
        86: (220, 220, 220),         # bass+mid+treble white
        108: (255, 30, 0),           # bass red
        110: (255, 0, 255),          # bass+treble magenta
        111: (255, 0, 255),          # bass+treble magenta
        114: (240, 120, 0),          # bass-dominant+mid orange
        120: (240, 200, 30),         # bass+mid yellow
        122: (220, 220, 220),        # bass+mid+treble white
        126: (240, 200, 30),         # bass+mid yellow
        129: (220, 220, 220),        # bass+mid+treble white
        144: (255, 30, 0),           # bass deep red
        150: (240, 120, 0),          # bass-dominant+mid orange
        156: (240, 120, 0),          # bass-dominant+mid orange
        223: (255, 230, 100),        # peak limiter cream
    }
    pal: List[Tuple[int, int, int]] = []
    anchor_keys = sorted(anchors.keys())
    for b in range(256):
        if b in anchors:
            pal.append(anchors[b])
            continue
        # Nearest-observed-byte fallback by numerical distance.
        nearest = min(anchor_keys, key=lambda k: abs(k - b))
        pal.append(anchors[nearest])
    return pal


def _serato_hue_palette() -> List[Tuple[int, int, int]]:
    """Anchored byte → RGB lookup table.

    Maps each byte value to an RGB colour via linear interpolation
    between anchor points tuned for visual contrast across the
    bass / mid / treble frequency bands:

        byte 1   → dim dark-blue   (silence floor)
        byte 2   → medium blue     (treble darker)
        byte 3   → bright blue     (treble peak)
        byte 18  → bright green    (mid peak)
        byte 43  → grey-white      (centerline marker)
        byte 78  → yellow          (low-mid)
        byte 108 → red-orange      (bass)
        byte 144 → deep red        (bass peak)
        byte 223 → white-yellow    (peak / limiter marker)

    Convention: bass = RED, mid = GREEN, treble = BLUE — an
    obvious additive RGB-to-band mapping.  Unlike `rgb332` which
    exposes the raw bit structure, `serato_hue` is the user-facing
    default render mode.
    """
    # Anchor distribution: ROYGBIV-style spectral gradient
    #   RED → ORANGE → YELLOW → GREEN → CYAN → BLUE
    #   ↑ (bass / high byte)             ↑ (treble / low byte)
    # In sawtooth/harmonic-rich content, multiple bytes co-occur per
    # column and the aggregate looks "white-ish" — but that's
    # mixing, not a green-less palette.  The base LUT keeps green.
    #
    # Anchor 1 set to pure black: byte 1 is the silence/padding
    # marker (fills offsets 0,1,14,15 of every chunk + all rows of
    # silent sections).  Rendering it as black makes empty cells
    # vanish so the waveform stands clean on a black bg.
    anchors = [
        (0,   (0,   0,   0)),     # background
        (1,   (0,   0,   0)),     # silence padding — INVISIBLE
        (2,   (0,   80,  220)),   # high treble medium blue
        (3,   (40,  140, 255)),   # treble bright blue (10kHz)
        (8,   (60,  220, 200)),   # cyan-teal transition
        (18,  (60,  220, 60)),    # mid GREEN (1kHz pure)
        (43,  (200, 200, 200)),   # centerline grey-white
        (78,  (240, 200, 30)),    # low-mid YELLOW (250Hz)
        (108, (240, 120, 0)),     # bass orange (~80Hz)
        (130, (250, 60,  0)),     # bass red-orange
        (144, (255, 30,  0)),     # bass deep RED (60Hz peak)
        (223, (255, 230, 100)),   # peak limiter cream-yellow
        (255, (255, 255, 255)),   # max white
    ]
    pal: List[Tuple[int, int, int]] = []
    for b in range(256):
        # Find bracketing anchors and interpolate.
        chosen = anchors[-1][1]
        for i in range(len(anchors) - 1):
            b0, c0 = anchors[i]
            b1, c1 = anchors[i + 1]
            if b0 <= b <= b1:
                t = (b - b0) / max(1, b1 - b0)
                chosen = (int(c0[0] + (c1[0] - c0[0]) * t),
                           int(c0[1] + (c1[1] - c0[1]) * t),
                           int(c0[2] + (c1[2] - c0[2]) * t))
                break
        pal.append(chosen)
    return pal


def _rgb332_palette() -> List[Tuple[int, int, int]]:
    """RGB332 byte → RGB colour mapping.

    Bit layout:

        bit 7..5  (top 3 bits)     → R channel = BASS intensity  (0..7)
        bit 4..2  (middle 3 bits)  → G channel = MID intensity   (0..7)
        bit 1..0  (bottom 2 bits)  → B channel = TREBLE intensity (0..3)

    Per-band byte signatures:

        bass-dominant   → bytes 108, 144  (R=3-4 dominant, G=mid, B=0)
        low-mid         → bytes 78, 114    (R+G both present, low B)
        mid-dominant    → byte  18         (G=4 dominant)
        treble          → byte   2         (B=2)
        treble peak     → byte   3         (B=3)

    Channel intensities are upscaled to 0..255 by simple
    multiplication.  R/G channels (3 bits) scale by 36 (= 255/7);
    B channel (2 bits) scales by 85 (= 255/3).
    """
    pal: List[Tuple[int, int, int]] = []
    for b in range(256):
        if b == 0:
            pal.append(BG_BLACK)
            continue
        r = (b >> 5) & 0b111
        g = (b >> 2) & 0b111
        bl = b & 0b11
        pal.append((r * 36, g * 36, bl * 85))
    return pal


def _hsl_palette() -> List[Tuple[int, int, int]]:
    """Pure HSL hue ramp blue→red across the FULL byte range 1..255.

    No pink-alpha-stair, no column-spread heuristic.  This isolates
    the hue-mapping hypothesis: if the resulting render matches
    Serato's own output on bytes 1..4 too, then the alpha-stair was
    never the right model.  If bytes 1..4 produce visibly-wrong
    colors, the pink-alpha-stair hypothesis stands.

    Saturation = 1.0 (full).  Lightness = 0.45 (constant — byte
    encodes hue only).
    """
    pal: List[Tuple[int, int, int]] = []
    for b in range(256):
        if b == 0:
            pal.append(BG_BLACK)
            continue
        t = (b - 1) / (255 - 1)
        # blue 240° → cyan → green → yellow → orange → red 0°
        hue = (1 - t) * 240.0
        r, g, bl = colorsys.hls_to_rgb(hue / 360, 0.45, 1.0)
        pal.append((int(r * 255), int(g * 255), int(bl * 255)))
    return pal


# --- Public API ------------------------------------------------------

def grayscale_palette() -> List[Tuple[int, int, int]]:
    """256-entry grayscale palette.  Bytes 0..1 = near-white
    silence-bg, bytes 2..255 = linear ramp to near-black."""
    pal: List[Tuple[int, int, int]] = []
    for b in range(256):
        if b <= 1:
            pal.append((245, 245, 245))
        else:
            v = 240 - int(220 * (b / 255))
            v = max(0, min(255, v))
            pal.append((v, v, v))
    return pal


def render_overview(blob: bytes,
                       out_path: str,
                       *,
                       mode: str = "grayscale",
                       scale: int = 4,
                       bg: Tuple[int, int, int] = BG_BLACK) -> bool:
    """Render a Serato Overview blob to a BMP file.

    Args:
        blob:     raw bytes of the `Serato Overview` GEOB / atom /
                   Vorbis blob (must be ≥ 3842 bytes).
        out_path: target BMP path.
        mode:     "grayscale" | "color" | "alpha".
                   - grayscale: safe fallback, byte → darkness ramp
                   - color: column-spread-aware HLS (experimental)
                   - alpha: HSL hue ramp + pink-as-alpha stair
                            (experimental)
        scale:    nearest-neighbor upscale factor (default 4).
        bg:       background RGB (color mode only).

    Returns True on success.  Returns False if Pillow is missing
    or the blob is malformed.
    """
    try:
        from PIL import Image
    except ImportError:
        return False
    if not blob or len(blob) < (OVERVIEW_BLOB_HEADER
                                   + OVERVIEW_NUM_CHUNKS
                                   * OVERVIEW_CHUNK_SIZE):
        return False
    body = blob[OVERVIEW_BLOB_HEADER:
                OVERVIEW_BLOB_HEADER + OVERVIEW_NUM_CHUNKS
                * OVERVIEW_CHUNK_SIZE]

    if mode == "color":
        img = Image.new("RGB", (OVERVIEW_WIDTH_G2,
                                OVERVIEW_NUM_CHUNKS), bg)
        px = img.load()
        for c in range(OVERVIEW_NUM_CHUNKS):
            chunk = body[c * OVERVIEW_CHUNK_SIZE:
                          (c + 1) * OVERVIEW_CHUNK_SIZE]
            spread = _column_spread(chunk)
            for i, o in enumerate(TOP_OFFSETS):
                px[i, c] = _render_color_pixel(chunk, o, spread, bg)
            px[4, c] = OFF_WHITE
            for i, o in enumerate(reversed(TOP_OFFSETS)):
                px[5 + i, c] = _render_color_pixel(chunk, o, spread,
                                                      bg)
    elif mode == "alpha":
        palette = _alpha_palette()
        img = Image.new("P", (OVERVIEW_CHUNK_SIZE, OVERVIEW_NUM_CHUNKS))
        flat: List[int] = []
        for r, g, b in palette:
            flat.extend([r, g, b])
        while len(flat) < 256 * 3:
            flat.extend([0, 0, 0])
        img.putpalette(flat)
        img.putdata(bytes(body))
    elif mode == "hsl":
        palette = _hsl_palette()
        img = Image.new("P", (OVERVIEW_CHUNK_SIZE, OVERVIEW_NUM_CHUNKS))
        flat = []
        for r, g, b in palette:
            flat.extend([r, g, b])
        while len(flat) < 256 * 3:
            flat.extend([0, 0, 0])
        img.putpalette(flat)
        img.putdata(bytes(body))
    elif mode == "column_tornado":
        # Per-column tornado oscillation around centerline — applies
        # the audio-tornado concept to the blob.  Each column gets
        # ONE color (dominant byte via palette LUT) and the bar
        # oscillates around centerline with phase θ(c) = 2π * turns
        # * c/240.  Squint → collapses to the regular overview shape.
        import math as _m
        palette = _serato_palette_calibrated()
        img = Image.new("P", (OVERVIEW_CHUNK_SIZE, OVERVIEW_NUM_CHUNKS))
        flat = []
        for r, g, b in palette:
            flat.extend([r, g, b])
        while len(flat) < 256 * 3:
            flat.extend([0, 0, 0])
        img.putpalette(flat)
        from collections import Counter
        new_body = bytearray(len(body))
        center = OVERVIEW_CHUNK_SIZE // 2  # = 8
        turns = 30  # rotation count over the blob's 240 columns
        for ci in range(OVERVIEW_NUM_CHUNKS):
            chunk = body[ci*16:(ci+1)*16]
            active = [b for b in chunk[2:14]
                       if b not in (0, 1, 43, 223)]
            if not active:
                continue
            volume = len(active)
            most_common = Counter(active).most_common(1)[0][0]
            theta = 2 * _m.pi * turns * (ci / OVERVIEW_NUM_CHUNKS)
            amp = min(7, max(1, volume // 2))
            y_off = int(amp * _m.sin(theta))
            for off in range(OVERVIEW_CHUNK_SIZE):
                row_orig = chunk[off]
                if off in (0, 1, 14, 15):
                    new_body[ci*16 + off] = 0
                    continue
                if off == center and row_orig in (43, 223):
                    new_body[ci*16 + off] = row_orig
                    continue
                # Inside oscillation band?
                if y_off >= 0 and center <= off <= center + y_off:
                    new_body[ci*16 + off] = most_common
                elif y_off < 0 and center + y_off <= off <= center:
                    new_body[ci*16 + off] = most_common
                else:
                    new_body[ci*16 + off] = 0
        img.putdata(bytes(new_body))
    elif mode == "column_agg":
        # this "treat values per column (Timestamp) vol + freq"
        # strategy.  For each column (= 1 timestamp):
        #   - count active-row bytes > 1 → VOLUME (bar height)
        #   - take MODE byte value over active rows → BAND-MIX color
        # Then render as 1-column-tall bars centered on the canvas,
        # COLOR from calibrated LUT for that representative byte.
        # Result: per-timestamp colored amplitude bar like a normal
        # waveform display, but recovered FROM THE BLOB.
        palette = _serato_palette_calibrated()
        # We're rebuilding the body manually as a 240×16 image where
        # each column's pixels reflect its band-mix bar.
        img = Image.new("P", (OVERVIEW_CHUNK_SIZE, OVERVIEW_NUM_CHUNKS))
        flat = []
        for r, g, b in palette:
            flat.extend([r, g, b])
        while len(flat) < 256 * 3:
            flat.extend([0, 0, 0])
        img.putpalette(flat)
        # Reshape body (240 chunks × 16) and aggregate per chunk
        from collections import Counter
        new_body = bytearray(len(body))
        center = OVERVIEW_CHUNK_SIZE // 2  # = 8 (centerline offset)
        for ci in range(OVERVIEW_NUM_CHUNKS):
            chunk = body[ci*16:(ci+1)*16]
            # Active rows = offsets 2..13 (skip 0/1/14/15 padding).
            # Drop universal-bg bytes (1, 43, 223) to pick the
            # band-content bytes.
            active = [b for b in chunk[2:14]
                       if b not in (0, 1, 43, 223)]
            if not active:
                # All padding/silence → render whole column as bg.
                continue
            # VOLUME: how many active rows are filled (0..12).
            volume = len(active)
            # BAND-MIX: most common byte (= the dominant freq mix).
            most_common = Counter(active).most_common(1)[0][0]
            # Render: mirror bar around centerline (offset 8) using
            # half_h proportional to volume.  Pixels within bar →
            # most_common byte; outside → byte 0 (= black bg).
            half_h = min(7, max(1, volume // 2))
            for off in range(OVERVIEW_CHUNK_SIZE):
                # Reuse the chunk's existing structure markers
                # (byte 43/223 stay on centerline if they were there)
                row_orig = chunk[off]
                if off == center and row_orig in (43, 223):
                    new_body[ci*16 + off] = row_orig
                elif abs(off - center) <= half_h and off not in (0, 1, 14, 15):
                    new_body[ci*16 + off] = most_common
                else:
                    new_body[ci*16 + off] = 0  # bg black
        img.putdata(bytes(new_body))
    elif mode == "serato_palette":
        palette = _serato_palette_calibrated()
        img = Image.new("P", (OVERVIEW_CHUNK_SIZE, OVERVIEW_NUM_CHUNKS))
        flat = []
        for r, g, b in palette:
            flat.extend([r, g, b])
        while len(flat) < 256 * 3:
            flat.extend([0, 0, 0])
        img.putpalette(flat)
        img.putdata(bytes(body))
    elif mode == "rgb332":
        palette = _rgb332_palette()
        img = Image.new("P", (OVERVIEW_CHUNK_SIZE, OVERVIEW_NUM_CHUNKS))
        flat = []
        for r, g, b in palette:
            flat.extend([r, g, b])
        while len(flat) < 256 * 3:
            flat.extend([0, 0, 0])
        img.putpalette(flat)
        img.putdata(bytes(body))
    elif mode == "serato_hue":
        palette = _serato_hue_palette()
        img = Image.new("P", (OVERVIEW_CHUNK_SIZE, OVERVIEW_NUM_CHUNKS))
        flat = []
        for r, g, b in palette:
            flat.extend([r, g, b])
        while len(flat) < 256 * 3:
            flat.extend([0, 0, 0])
        img.putpalette(flat)
        img.putdata(bytes(body))
    else:
        palette = grayscale_palette()
        img = Image.new("P", (OVERVIEW_CHUNK_SIZE, OVERVIEW_NUM_CHUNKS))
        flat = []
        for r, g, b in palette:
            flat.extend([r, g, b])
        while len(flat) < 256 * 3:
            flat.extend([0, 0, 0])
        img.putpalette(flat)
        img.putdata(bytes(body))

    # Serato displays the waveform with time running LEFT → RIGHT.
    # The blob is column-major (240 chunks of 16 bytes each), and
    # PIL Image.new((16, 240)) + putdata(body) lays chunks out
    # TOP → BOTTOM.  Rotate 90° counter-clockwise so chunk-0 lands
    # on the left edge (= track start), matching Serato's own
    # render orientation.  Without this, drum transients appear in
    # reverse time order.
    img = img.transpose(Image.Transpose.ROTATE_90)

    if scale != 1:
        w, h = img.size
        img = img.resize((w * scale, h * scale),
                          resample=Image.NEAREST)
    try:
        if mode != "color":
            img = img.convert("RGB")
        img.save(out_path, "BMP")
        return True
    except Exception:
        return False


def render_overview_for_path(audio_path: str,
                                out_path: str,
                                *,
                                mode: str = "grayscale",
                                scale: int = 4) -> bool:
    """Convenience: harvest the Overview blob from an audio file and
    render it via `render_overview`."""
    from sidecaramel.blobs import harvest
    for desc, payload, _src in harvest(audio_path):
        if desc == "Serato Overview" and payload:
            return render_overview(payload, out_path,
                                     mode=mode, scale=scale)
    return False


# --- Reference comparison --------------------------------------------

def compare_against_reference(blob: bytes,
                                 reference_image_path: str,
                                 *,
                                 mode: str = "color",
                                 ) -> Optional[dict]:
    """Render `blob` via the chosen mode and pixel-diff against a
    reference image.

    Args:
        blob:        Serato Overview blob bytes.
        reference_image_path: PNG/JPG of Serato's own render of the
                                same track's Overview.  Will be
                                resized to match the rendered output.
        mode:        rendering mode to compare (same options as
                      `render_overview`).

    Returns a dict:
        {
          "mode":             str,
          "mean_rgb_error":   float (0..255),
          "by_spread_band":   {band_label: {"n": int,
                                              "mean_rgb_error": float}}
          "by_byte_zone":     {"alpha_stair": ..., "color_5plus": ...,
                                "silence_0":   ...},
          "diff_image_path":  str  (saved heatmap, same base as ref)
        }
    or None if Pillow / blob is missing.

    Use the returned dict to iterate hypotheses — high error in
    `alpha_stair` zone means the pink-stair encoding is off, high
    error in a single `by_spread_band` value means the hue-mapping
    needs adjustment for that band.
    """
    try:
        from PIL import Image
        import os
    except ImportError:
        return None
    if not blob or len(blob) < (OVERVIEW_BLOB_HEADER
                                   + OVERVIEW_NUM_CHUNKS
                                   * OVERVIEW_CHUNK_SIZE):
        return None

    # Render the candidate into memory.
    rendered_path = os.path.splitext(reference_image_path)[0] + (
        f".sidecaramel_{mode}.bmp")
    if not render_overview(blob, rendered_path,
                              mode=mode, scale=1):
        return None
    cand = Image.open(rendered_path).convert("RGB")
    ref = Image.open(reference_image_path).convert("RGB")
    ref = ref.resize(cand.size, resample=Image.NEAREST)

    body = blob[OVERVIEW_BLOB_HEADER:
                OVERVIEW_BLOB_HEADER + OVERVIEW_NUM_CHUNKS
                * OVERVIEW_CHUNK_SIZE]

    spread_buckets = {
        "0-1 (treble)":  [],
        "2-3 (mid-low)": [],
        "4-5 (mid)":     [],
        "6-7 (mid-bass)": [],
        "8-10 (bass)":   [],
    }
    zone_buckets = {
        "silence_0":   [],
        "alpha_stair": [],
        "color_5plus": [],
    }

    cw, ch = cand.size
    chunks_per_col = max(1, ch // OVERVIEW_NUM_CHUNKS)
    chunks_per_row = max(1, cw // OVERVIEW_CHUNK_SIZE)
    total_err = 0.0
    n_total = 0

    for c in range(OVERVIEW_NUM_CHUNKS):
        chunk = body[c * OVERVIEW_CHUNK_SIZE:
                      (c + 1) * OVERVIEW_CHUNK_SIZE]
        spread = _column_spread(chunk)
        if spread <= 1:
            band_label = "0-1 (treble)"
        elif spread <= 3:
            band_label = "2-3 (mid-low)"
        elif spread <= 5:
            band_label = "4-5 (mid)"
        elif spread <= 7:
            band_label = "6-7 (mid-bass)"
        else:
            band_label = "8-10 (bass)"

        for offset in range(OVERVIEW_CHUNK_SIZE):
            b = chunk[offset]
            xp = offset * chunks_per_row
            yp = c * chunks_per_col
            if xp >= cw or yp >= ch:
                continue
            cr, cg, cb = cand.getpixel((xp, yp))
            rr, rg, rb = ref.getpixel((xp, yp))
            err = ((cr - rr) ** 2 + (cg - rg) ** 2
                     + (cb - rb) ** 2) ** 0.5
            total_err += err
            n_total += 1
            spread_buckets[band_label].append(err)
            if b == 0:
                zone_buckets["silence_0"].append(err)
            elif 1 <= b <= 4:
                zone_buckets["alpha_stair"].append(err)
            else:
                zone_buckets["color_5plus"].append(err)

    # Heatmap (per-pixel error magnitude)
    heatmap = Image.new("L", cand.size)
    hp = heatmap.load()
    for x in range(cw):
        for y in range(ch):
            cr, cg, cb = cand.getpixel((x, y))
            rr, rg, rb = ref.getpixel((x, y))
            e = ((cr - rr) ** 2 + (cg - rg) ** 2
                   + (cb - rb) ** 2) ** 0.5
            hp[x, y] = min(255, int(e * 1.2))
    heat_path = (os.path.splitext(reference_image_path)[0]
                  + f".sidecaramel_{mode}.diff_heatmap.png")
    heatmap.save(heat_path)

    def _summary(samples):
        if not samples:
            return {"n": 0, "mean_rgb_error": 0.0}
        return {"n": len(samples),
                "mean_rgb_error": round(sum(samples) / len(samples), 2)}

    return {
        "mode": mode,
        "mean_rgb_error": round(total_err / max(1, n_total), 2),
        "by_spread_band": {k: _summary(v)
                            for k, v in spread_buckets.items()},
        "by_byte_zone": {k: _summary(v)
                          for k, v in zone_buckets.items()},
        "diff_image_path": heat_path,
        "rendered_image_path": rendered_path,
    }
