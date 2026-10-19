"""sidecaramel.overview — Overview-blob waveform renderer.

Renders the blob as what it is: a 240 x 16 bitmap whose bytes index a
6x6x6 colour cube (see `sidecaramel.overview_palette` for the format
and the measurements behind it).

    column  = time slice, track start on the left
    row 0   = top row; nothing is mirrored, rotated or flipped
    colour  = R/G/B = 51 * (a, b, c) for v = 36a + 6b + c
    0 and 1 = background
    >= 216  = flagged cell, drawn in its base colour (v - 216) unless a
              flag colour is given
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from sidecaramel.overview_palette import (OVERVIEW_HEIGHT, OVERVIEW_WIDTH,
                                          blob_payload, byte_to_cube,
                                          is_background, split_flag)

RGB = Tuple[int, int, int]
BG_BLACK: RGB = (0, 0, 0)


def cell_colour(byte: int, *, background: RGB = BG_BLACK,
                flag_colour: Optional[RGB] = None) -> RGB:
    """Display colour of one Overview byte."""
    if is_background(byte):
        return background
    if flag_colour is not None and split_flag(byte)[0]:
        return flag_colour
    return byte_to_cube(byte)


def overview_palette(*, background: RGB = BG_BLACK,
                     flag_colour: Optional[RGB] = None) -> List[RGB]:
    """256-entry palette, index = byte value."""
    return [cell_colour(v, background=background, flag_colour=flag_colour)
            for v in range(256)]


def overview_image(blob: bytes, *, background: RGB = BG_BLACK,
                   flag_colour: Optional[RGB] = None):
    """The blob as a 240 x 16 RGB `PIL.Image`, row 0 at the top.

    Raises ImportError without Pillow and ValueError for a short blob.
    """
    from PIL import Image
    body = blob_payload(blob)
    pal = overview_palette(background=background, flag_colour=flag_colour)
    img = Image.new("RGB", (OVERVIEW_WIDTH, OVERVIEW_HEIGHT))
    img.putdata([pal[body[c * OVERVIEW_HEIGHT + r]]
                 for r in range(OVERVIEW_HEIGHT)
                 for c in range(OVERVIEW_WIDTH)])
    return img


def _save(img, out_path: str) -> None:
    """Save in the format the extension names; BMP when it names none."""
    from PIL import Image
    ext = os.path.splitext(out_path)[1].lower()
    fmt = None if ext in Image.registered_extensions() else "BMP"
    img.save(out_path, fmt)


def render_overview(blob: bytes,
                    out_path: str,
                    *,
                    scale: int = 4,
                    background: RGB = BG_BLACK,
                    flag_colour: Optional[RGB] = None) -> bool:
    """Render a Serato Overview blob to an image file.

    Args:
        blob:        raw `Serato Overview` payload (>= 3842 bytes).
        out_path:    target path; the extension picks the format
                     (.png, .bmp, ...), BMP when it names none.
        scale:       nearest-neighbour upscale factor (default 4 ->
                     960 x 64).
        background:  colour for background bytes 0 and 1.
        flag_colour: colour for flagged bytes (>= 216); None draws
                     their base colour.

    Returns True on success, False when Pillow is missing, the blob is
    malformed or the file cannot be written.
    """
    try:
        from PIL import Image
        img = overview_image(blob, background=background,
                             flag_colour=flag_colour)
    except (ImportError, ValueError):
        return False
    if scale != 1:
        img = img.resize((OVERVIEW_WIDTH * scale, OVERVIEW_HEIGHT * scale),
                         resample=Image.NEAREST)
    try:
        _save(img, out_path)
        return True
    except Exception:
        return False


def render_overview_for_path(audio_path: str,
                             out_path: str,
                             **kwargs) -> bool:
    """Harvest the Overview blob from an audio file and render it via
    `render_overview` (keyword arguments pass through)."""
    from sidecaramel.blobs import harvest
    for desc, payload, _src in harvest(audio_path):
        if desc == "Serato Overview" and payload:
            return render_overview(payload, out_path, **kwargs)
    return False


# --- Reference comparison --------------------------------------------

def compare_against_reference(blob: bytes,
                              reference_image_path: str) -> Optional[dict]:
    """Render `blob` at 240 x 16 and pixel-diff it against a reference
    image of the same overview (e.g. a crop of Serato's display),
    resized to 240 x 16.

    Returns
        {"mean_rgb_error":      float (0..441),
         "by_cell":             {"background" | "colour" | "flagged":
                                 {"n": int, "mean_rgb_error": float}},
         "rendered_image_path": str,
         "diff_image_path":     str}
    or None when Pillow is missing or the blob is malformed. Both
    images are written next to the reference.
    """
    try:
        from PIL import Image
        cand = overview_image(blob)
    except (ImportError, ValueError):
        return None
    body = blob_payload(blob)
    ref = (Image.open(reference_image_path).convert("RGB")
           .resize(cand.size, resample=Image.BOX))

    base = os.path.splitext(reference_image_path)[0]
    rendered_path = base + ".sidecaramel_cube.png"
    heat_path = base + ".sidecaramel_cube.diff_heatmap.png"
    cand.save(rendered_path)

    buckets = {"background": [], "colour": [], "flagged": []}
    heat = Image.new("L", cand.size)
    for r in range(OVERVIEW_HEIGHT):
        for c in range(OVERVIEW_WIDTH):
            v = body[c * OVERVIEW_HEIGHT + r]
            a, b = cand.getpixel((c, r)), ref.getpixel((c, r))
            err = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
            kind = ("background" if is_background(v)
                    else "flagged" if split_flag(v)[0] else "colour")
            buckets[kind].append(err)
            heat.putpixel((c, r), min(255, int(err)))
    heat.save(heat_path)

    def summary(samples):
        return {"n": len(samples),
                "mean_rgb_error": round(sum(samples) / len(samples), 2)
                if samples else 0.0}

    every = [e for s in buckets.values() for e in s]
    return {"mean_rgb_error": summary(every)["mean_rgb_error"],
            "by_cell": {k: summary(v) for k, v in buckets.items()},
            "rendered_image_path": rendered_path,
            "diff_image_path": heat_path}
