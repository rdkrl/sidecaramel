"""sidecaramel.audio_render — audio-direct big-waveform renderer.

Per-column short-time FFT on the audio file, with bass / mid /
treble band-energies rendered as additive R / G / B vertical bars.
Bar height tracks peak amplitude.

Unlike the byte-LUT modes in `sidecaramel.overview` which decode the
embedded Overview blob, this module recomputes the visualisation
from the source audio.  Result: a full-resolution waveform for any
track, including ones that have never been analysed by any DJ
software.

Dependencies (optional — module import will raise if missing):
    numpy, scipy, soundfile  + ffmpeg in $PATH for non-PCM formats

CLI:
    python -m sidecaramel.audio_render <audio> --out big.bmp \\
        [--width 1800] [--height 240]
"""
from __future__ import annotations

import argparse
import io
import math
import os
import subprocess
import sys
from typing import Tuple


# Band edges (Hz) — conventional bass / mid / treble split.
BASS_LO, BASS_HI = 20.0, 200.0
MID_LO, MID_HI = 200.0, 2000.0
TREBLE_LO, TREBLE_HI = 2000.0, 20000.0


def _decode_audio(audio_path: str,
                    target_sr: int = 22050,
                    stereo: bool = False
                    ) -> Tuple["numpy.ndarray", int]:
    """Decode any audio file to PCM at `target_sr` Hz.

    Tries soundfile first (handles WAV/FLAC/AIFF/OGG natively);
    falls back to ffmpeg-via-stdin for MP3/M4A/AAC.

    Returns: (samples, sr).  If stereo=True, shape is (N, 2)
    (channels L, R).  Else mono shape (N,).
    """
    import numpy as np
    data = None
    try:
        import soundfile as sf
        try:
            data, sr = sf.read(audio_path, dtype="float32",
                                  always_2d=True)
        except Exception:
            data = None
    except ImportError:
        pass
    if data is None:
        # ffmpeg → raw WAV on stdout → soundfile via BytesIO.
        n_ch = 2 if stereo else 1
        cmd = ["ffmpeg", "-loglevel", "error", "-i", audio_path,
               "-ar", str(target_sr), "-ac", str(n_ch), "-f", "wav",
               "pipe:1"]
        result = subprocess.run(cmd, capture_output=True, check=True)
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(result.stdout),
                              dtype="float32", always_2d=True)
    # Now data is (N, channels)
    if stereo:
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)  # mono → duplicate
        elif data.shape[1] > 2:
            data = data[:, :2]  # take first 2 channels
    else:
        if data.shape[1] >= 2:
            data = data.mean(axis=1)
        else:
            data = data[:, 0]
    if sr != target_sr:
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(sr), int(target_sr))
            if data.ndim == 1:
                data = resample_poly(data, target_sr // g, sr // g)
            else:
                data = np.stack([resample_poly(data[:, c],
                                                    target_sr // g,
                                                    sr // g)
                                   for c in range(data.shape[1])],
                                  axis=1)
            sr = target_sr
        except ImportError:
            # scipy not available — fall back to numpy linear-interp
            # resample.  Lower quality than polyphase resample_poly
            # but adequate for waveform-render purposes.
            n_out = int(round(len(data) * target_sr / sr))
            old_idx = np.linspace(0, len(data) - 1, n_out)
            old_lo = old_idx.astype("int64")
            old_hi = np.clip(old_lo + 1, 0, len(data) - 1)
            frac = (old_idx - old_lo).astype("float32")
            if data.ndim == 1:
                data = data[old_lo] * (1 - frac) + data[old_hi] * frac
            else:
                # per-channel lerp
                out = np.empty((n_out, data.shape[1]), dtype="float32")
                for c in range(data.shape[1]):
                    out[:, c] = (data[old_lo, c] * (1 - frac)
                                  + data[old_hi, c] * frac)
                data = out
            sr = target_sr
    return data.astype("float32"), sr


def _band_energies_per_column(samples,
                                sr: int,
                                n_columns: int,
                                fft_window: int = 2048):
    """Return (bass, mid, treble, peak) energy arrays, each length
    `n_columns`.

    Each column samples a non-overlapping segment of audio and runs
    an FFT to extract per-band energy.  `peak` is the max absolute
    amplitude in the column's segment (drives bar HEIGHT in the
    render).
    """
    import numpy as np
    n_samples = len(samples)
    bass = np.zeros(n_columns, dtype="float32")
    mid = np.zeros(n_columns, dtype="float32")
    treble = np.zeros(n_columns, dtype="float32")
    peak = np.zeros(n_columns, dtype="float32")

    # Frequency bins
    freqs = np.fft.rfftfreq(fft_window, d=1.0 / sr)
    bass_mask = (freqs >= BASS_LO) & (freqs < BASS_HI)
    mid_mask = (freqs >= MID_LO) & (freqs < MID_HI)
    treble_mask = (freqs >= TREBLE_LO) & (freqs < TREBLE_HI)

    win = np.hanning(fft_window).astype("float32")

    # Per-column audio slice WIDTH (used for tight peak detection,
    # smaller than the FFT window so transients pop out instead of
    # being smeared across adjacent columns).
    col_slice = max(1, n_samples // n_columns)

    for i in range(n_columns):
        start = int(i / n_columns * n_samples)
        end = min(start + fft_window, n_samples)
        seg = samples[start:end]
        if len(seg) < fft_window:
            seg = np.concatenate([seg,
                                     np.zeros(fft_window - len(seg),
                                                dtype="float32")])
        # PEAK amplitude — use the column's own audio slice (not the
        # FFT window) so kick transients show as tall spikes instead
        # of being smeared.
        slice_end = min(start + col_slice, n_samples)
        peak[i] = float(np.max(np.abs(samples[start:slice_end])))
        # Hann-windowed FFT for clean spectrum.
        spec = np.abs(np.fft.rfft(seg * win))
        bass[i] = float(spec[bass_mask].sum())
        mid[i] = float(spec[mid_mask].sum())
        treble[i] = float(spec[treble_mask].sum())
    return bass, mid, treble, peak


def band_energies_per_column(samples,
                                sr: int,
                                n_columns: int,
                                fft_window: int = 2048):
    """Public-API wrapper around `_band_energies_per_column`.

    Returns `(bass, mid, treble, peak)` as numpy arrays, each of
    length `n_columns`.  See `_band_energies_per_column` for the
    full algorithm description.
    """
    return _band_energies_per_column(samples, sr, n_columns,
                                         fft_window)


def render_big_waveform(audio_path: str,
                          out_path: str,
                          *,
                          width: int = 1800,
                          height: int = 240,
                          bg=(0, 0, 0),
                          bass_boost: float = 1.0,
                          show_beatgrid: bool = False,
                          dynamics: float = 1.6,
                          oversample: int = 1,
                          axis_deg: int = 135) -> bool:
    """Render `audio_path` as an audio-direct big-deck waveform.

    Per column: bar height ∝ peak amplitude; bar color = additive
    (R = bass band, G = mid band, B = treble band).  Mirror-
    symmetric around the vertical centerline of the image.

    **`axis_deg`** picks the vinyl-physical groove-axis rotation
    (the docstring "Vinyl-physical waveform render — 45/45 Westrex
    cartridge analog").  Only four distinct values exist because
    abs-amplitude rendering folds 180°-rotated views onto each
    other:

      -   0° → top=|lateral|=|(L+R)/2|, bottom=|vertical|=|(L-R)/2|
              (sum / diff projection — mono needle vs. stereo bounce)
      -  45° → top=|L|/√2,            bottom=|R|/√2
              (per-channel)
      -  90° → top=|vertical|,        bottom=|lateral|
              (sum/diff flipped)
      - 135° → top=|R|/√2,            bottom=|L|/√2
              (per-channel flipped; **default** — matches the demo
               and `render_big_waveform_png`, for B-deck pairing)

    Any other angle is rounded to the nearest of these four.

    Returns True on success.  Requires numpy + soundfile + ffmpeg
    (for non-PCM input).
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return False

    if not os.path.isfile(audio_path):
        return False

    samples, sr = _decode_audio(audio_path, stereo=True)
    if len(samples) < 1024:
        return False

    # this oversample-then-compress strategy: render at N× target
    # width with per-column FFT, then downsample with high-quality
    # resampling so adjacent columns blend smoothly through
    # additive band-color transitions (waveform-line look instead
    # of blocky bars).
    render_w = int(width * max(1, oversample))

    # ── Vinyl-axis projection (45/45 Westrex) ──
    # Pick top/bottom channels per `axis_deg`.  Bands + peak are
    # computed on the projected channels so the colour/amp behaviour
    # remains correct for every angle.
    L_raw = samples[:, 0]
    R_raw = samples[:, 1]
    snapped = min((0, 45, 90, 135),
                  key=lambda v: abs(v - (axis_deg % 180)))
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    if snapped == 0:
        # top = |lateral| = |(L+R)/2|   bottom = |vertical| = |(L-R)/2|
        top_ch = (L_raw + R_raw) * 0.5
        bot_ch = (L_raw - R_raw) * 0.5
    elif snapped == 90:
        # top = |vertical|, bottom = |lateral|
        top_ch = (L_raw - R_raw) * 0.5
        bot_ch = (L_raw + R_raw) * 0.5
    elif snapped == 135:
        # top = |R|/√2, bottom = |L|/√2  (per-channel flipped)
        top_ch = R_raw * inv_sqrt2
        bot_ch = L_raw * inv_sqrt2
    else:  # 45° — top = |L|/√2, bottom = |R|/√2  (per-channel)
        top_ch = L_raw * inv_sqrt2
        bot_ch = R_raw * inv_sqrt2

    # Process top (upper half) and bottom (lower half) separately.
    bass_L, mid_L, treble_L, peak_L = _band_energies_per_column(
        top_ch, sr, render_w)
    bass_R, mid_R, treble_R, peak_R = _band_energies_per_column(
        bot_ch, sr, render_w)
    # Render-internal height is full final height (no oversample).
    width_final = width
    width = render_w
    render_h = height

    # Normalize bands using 99th percentile (cross-channel) so L and
    # R are comparable, then dynamics gamma EXPANDS peak contrast.
    # gamma > 1 = quiet stays quiet, peaks dominate → dynamic look.
    def _norm_band(x, p99):
        if p99 < 1e-9:
            return np.zeros_like(x)
        return np.clip(x / p99, 0.0, 1.0)
    bass_p99 = float(np.percentile(np.concatenate([bass_L, bass_R]), 99))
    mid_p99 = float(np.percentile(np.concatenate([mid_L, mid_R]), 99))
    treble_p99 = float(np.percentile(np.concatenate([treble_L, treble_R]), 99))
    bass_Ln = _norm_band(bass_L, bass_p99) ** 0.75
    bass_Rn = _norm_band(bass_R, bass_p99) ** 0.75
    mid_Ln = _norm_band(mid_L, mid_p99) ** 0.75
    mid_Rn = _norm_band(mid_R, mid_p99) ** 0.75
    treble_Ln = _norm_band(treble_L, treble_p99) ** 0.75
    treble_Rn = _norm_band(treble_R, treble_p99) ** 0.75
    # Peak normalization: tighter (99.5 pct, clipped) + dynamics gamma.
    peak_p995 = float(np.percentile(np.concatenate([peak_L, peak_R]), 99.5))
    if peak_p995 < 1e-9:
        peak_p995 = 1e-9
    peak_Ln = np.clip(peak_L / peak_p995, 0.0, 1.0) ** dynamics
    peak_Rn = np.clip(peak_R / peak_p995, 0.0, 1.0) ** dynamics

    # Reserve top + bottom black header/footer strips for grid
    # numbers and cue triangles — keeps them out of the waveform
    # body so colored bars don't interfere with annotations and the
    # vertical grid lines stand out against the black margins.
    header_h = max(28, int(height * 0.10))
    footer_h = max(18, int(height * 0.06))
    body_top = header_h
    body_bot = height - footer_h
    body_h = body_bot - body_top

    # Build canvas.  Body section: top half = L channel bar growing
    # UP from centerline; bottom half = R channel growing DOWN.
    img = np.zeros((height, width, 3), dtype="uint8")
    img[:, :] = bg
    center = body_top + body_h // 2

    # Per-column RGB colors per channel.  bass_boost scales R.
    RL = np.clip(bass_Ln * 255.0 * bass_boost, 0, 255).astype("uint8")
    GL = (mid_Ln * 255).astype("uint8")
    BL = (treble_Ln * 255).astype("uint8")
    RR = np.clip(bass_Rn * 255.0 * bass_boost, 0, 255).astype("uint8")
    GR = (mid_Rn * 255).astype("uint8")
    BR = (treble_Rn * 255).astype("uint8")
    # Bar pixel-heights per channel — constrained to body section
    # (header/footer black margins reserved for grid annotations).
    half_body = body_h // 2 - 2
    bar_L = (peak_Ln * half_body).astype("int32")
    bar_R = (peak_Rn * half_body).astype("int32")

    for x in range(width):
        # L (top half) — bar grows UP from center
        hL = int(bar_L[x])
        if hL > 0:
            y0 = center - hL
            img[y0:center, x, 0] = int(RL[x])
            img[y0:center, x, 1] = int(GL[x])
            img[y0:center, x, 2] = int(BL[x])
        # R (bottom half) — bar grows DOWN from center
        hR = int(bar_R[x])
        if hR > 0:
            y1 = center + hR
            img[center:y1, x, 0] = int(RR[x])
            img[center:y1, x, 1] = int(GR[x])
            img[center:y1, x, 2] = int(BR[x])

    # ── DOWNSAMPLE FIRST — so beatgrid annotations get drawn on
    # the final (post-LANCZOS) image with pixel-sharp widths.
    pil_pre = Image.fromarray(img, mode="RGB")
    if width != width_final:
        pil_pre = pil_pre.resize((width_final, render_h), Image.LANCZOS)
    img = np.asarray(pil_pre).copy()
    # Re-bind width to final width for annotation coordinates.
    width = width_final

    # Beatgrid overlay:
    #   • Manual anchor downbeats (positions matching BeatGrid blob
    #     markers) → RED
    #   • Other downbeats (interpolated every-4th) → WHITE
    #   • Regular beats → light grey, thinner
    #   • At high zoom: quarter-beat ticks at top + bottom edges
    #   • Bar numbers drawn as text on each downbeat
    #   • Cue points → coloured triangles dropping from the top
    if show_beatgrid:
        try:
            from sidecaramel.tags import (read_serato_metadata,
                                              serato_beat_grid_seconds,
                                              _parse_serato_beatgrid,
                                              harvest_serato_blobs)
            meta = read_serato_metadata(audio_path)
            duration = len(samples) / sr
            beats = serato_beat_grid_seconds(meta, duration)
            if beats:
                bpm = (meta or {}).get("bpm") or 120.0
                beat_period = 60.0 / bpm
                px_per_beat = width / duration * beat_period
                # Get raw anchor positions from the BeatGrid markers
                # — these are the user-set or analyzer-confirmed
                # downbeat anchors.
                anchor_positions = []
                for raw_desc, raw_payload, _ in harvest_serato_blobs(audio_path):
                    if raw_desc == "Serato BeatGrid":
                        bg = _parse_serato_beatgrid(raw_payload)
                        if bg:
                            anchor_positions = [m[0] for m in bg.get("markers", [])]
                        break
                # Determine downbeat anchor for COUNTING bars.
                cues = (meta or {}).get("cues") or []
                downbeat_t = (anchor_positions[0]
                                if anchor_positions
                                else (cues[0].get("pos_ms", 0) / 1000.0
                                       if cues else beats[0]))

                # Line widths in FINAL-image pixels — drawn after
                # downsample so 1 px = 1 px on screen.
                lw_anchor = 5
                lw_db = 3
                lw_be = 1
                scale_factor = 1  # for font sizing below

                # Colors
                RED = np.array([255, 30, 30], dtype="float32")
                WHITE = np.array([235, 235, 235], dtype="float32")
                GREY = np.array([130, 130, 130], dtype="float32")
                TICK_GREY = np.array([180, 180, 180], dtype="float32")

                def paint_line(x_px, col, alpha, half_w,
                                  y0=0, y1=None):
                    if y1 is None:
                        y1 = height
                    x0 = max(0, x_px - half_w)
                    x1 = min(width, x_px + half_w + 1)
                    src = img[y0:y1, x0:x1, :].astype("float32")
                    img[y0:y1, x0:x1, :] = (
                        src * (1 - alpha) + col[None, None, :] * alpha
                    ).astype("uint8")

                # Anchor positions for fast lookup
                set(round(p, 3) for p in anchor_positions)

                for bt in beats:
                    x = int(bt / duration * width)
                    if x < 0 or x >= width:
                        continue
                    n_from_downbeat = round((bt - downbeat_t) / beat_period)
                    is_downbeat = (n_from_downbeat % 4 == 0)
                    # Anchor = closest to an anchor_position within
                    # half-a-beat tolerance.
                    is_anchor = any(abs(bt - a) < beat_period * 0.5
                                       for a in anchor_positions)
                    if is_anchor:
                        paint_line(x, RED, 1.0, lw_anchor // 2)
                    elif is_downbeat:
                        paint_line(x, WHITE, 0.95, lw_db // 2)
                    else:
                        paint_line(x, GREY, 0.6, lw_be // 2)
                    # Quarter-beat ticks at top/bottom edges (only
                    # if zoomed in enough that they'd be readable).
                    if px_per_beat >= 30:
                        tick_h = max(4, int(render_h * 0.06))
                        for q in (0.25, 0.5, 0.75):
                            qx = int((bt + q * beat_period) / duration * width)
                            if 0 <= qx < width:
                                paint_line(qx, TICK_GREY, 0.75, max(1, lw_be // 3))
                                # but ONLY in the top + bottom strips
                                img[tick_h:render_h - tick_h, qx, :] = bg

                # Bar numbers (text on each downbeat top-edge).  Only
                # if there's enough horizontal room per bar (≥ 60 px).
                px_per_bar = px_per_beat * 4
                if px_per_bar >= 60:
                    try:
                        from PIL import ImageDraw, ImageFont
                        pil_overlay = Image.fromarray(img, mode="RGB")
                        draw = ImageDraw.Draw(pil_overlay)
                        # Use default font; scale by oversample so
                        # text survives LANCZOS downsample.
                        font_size = max(12, int(scale_factor * 18))
                        try:
                            font = ImageFont.truetype(
                                "/usr/share/fonts/truetype/dejavu/"
                                "DejaVuSans-Bold.ttf", font_size)
                        except Exception:
                            font = ImageFont.load_default()
                        cur_bar = 1
                        for bt in beats:
                            x = int(bt / duration * width)
                            if x < 0 or x >= width:
                                continue
                            n_from_downbeat = round((bt - downbeat_t) / beat_period)
                            if n_from_downbeat % 4 != 0:
                                continue
                            text = str(cur_bar)
                            cur_bar += 1
                            draw.text((x + 2, 2), text,
                                        fill=(220, 220, 220),
                                        font=font)
                        img = np.asarray(pil_overlay)
                    except Exception:
                        pass

                # Cue points as triangles dropping from the top in
                # their own color.
                if cues:
                    try:
                        from PIL import ImageDraw
                        pil_overlay = Image.fromarray(img, mode="RGB")
                        draw = ImageDraw.Draw(pil_overlay)
                        tri_h = max(10, int(render_h * 0.10))
                        tri_w = max(8, int(scale_factor * 12))
                        for cue in cues:
                            cx = int(cue["pos_ms"] / 1000.0 / duration * width)
                            if cx < 0 or cx >= width:
                                continue
                            color = tuple(cue.get("color", (255, 255, 0)))
                            # Triangle pointing DOWN, apex at top edge
                            draw.polygon(
                                [(cx, tri_h),
                                  (cx - tri_w // 2, 0),
                                  (cx + tri_w // 2, 0)],
                                fill=color,
                                outline=(255, 255, 255))
                        img = np.asarray(pil_overlay)
                    except Exception:
                        pass
        except Exception:
            pass  # No beatgrid available, render without

    # (Downsample already happened above before beatgrid; save now.)
    Image.fromarray(img, mode="RGB").save(out_path)
    return True


def render_spiral_flatten(audio_path: str,
                             out_path: str,
                             *,
                             size: int = 1800,
                             turns: int = 20,
                             width: int = 3600,
                             height: int = 600,
                             bass_boost: float = 1.4,
                             dynamics: float = 1.7,
                             lr_phase_shift: float = math.pi / 2,
                             mono_mix: bool = False) -> bool:
    """Render the audio as a TORNADO around the horizontal centerline
    axis, then orthographically project to 2D.

    "wie ein tornado um die achse auf der mittellinie
    von links nach rechts kreisen und dass dann flach projeziert zur
    fast gleichen waveform werden"

    Geometry:
      - X axis = time (linear, left to right)
      - Centerline = horizontal Y = height/2
      - At each x, audio amplitude R(x) rotates around the centerline
        axis with phase θ(x) progressing linearly through `turns`
        turns over the full track
      - Visible y(x) = R(x) * sin(θ(x))
      - Color = additive R/G/B from bass/mid/treble at time x

    Projection drops the depth (z = R*cos(θ)) → 2D image looks like
    a sinusoidal modulation of the regular waveform.  The "fast
    gleiche" similarity: at θ ≈ π/2 (peak of rotation), the
    projection equals the regular waveform; in between, partial
    cancellation produces a beating envelope.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return False
    if not os.path.isfile(audio_path):
        return False
    samples, sr = _decode_audio(audio_path, stereo=True)
    len(samples) / sr
    n_cols = width
    if mono_mix:
        # Sum L+R to mono BEFORE FFT.  Both channels share a single
        # set of band-energies → tornado collapses to 2 strands
        # (front + back) instead of 4 (L-front/back + R-front/back).
        mono = samples.mean(axis=1)
        bass_L, mid_L, treble_L, peak_L = _band_energies_per_column(
            mono, sr, n_cols)
        bass_R, mid_R, treble_R, peak_R = (bass_L, mid_L,
                                              treble_L, peak_L)
    else:
        bass_L, mid_L, treble_L, peak_L = _band_energies_per_column(
            samples[:, 0], sr, n_cols)
        bass_R, mid_R, treble_R, peak_R = _band_energies_per_column(
            samples[:, 1], sr, n_cols)
    def _norm(x, p):
        v = float(np.percentile(x, p))
        if v < 1e-9:
            return np.zeros_like(x)
        return np.clip(x / v, 0.0, 1.0) ** 0.75
    bass_p99 = float(np.percentile(np.concatenate([bass_L, bass_R]), 99))
    mid_p99 = float(np.percentile(np.concatenate([mid_L, mid_R]), 99))
    treble_p99 = float(np.percentile(np.concatenate([treble_L, treble_R]), 99))
    peak_p995 = float(np.percentile(np.concatenate([peak_L, peak_R]), 99.5))
    def _bn(x, p99): return np.clip(x / max(p99, 1e-9), 0, 1) ** 0.75
    bass_Ln, bass_Rn = _bn(bass_L, bass_p99), _bn(bass_R, bass_p99)
    mid_Ln, mid_Rn = _bn(mid_L, mid_p99), _bn(mid_R, mid_p99)
    treble_Ln, treble_Rn = _bn(treble_L, treble_p99), _bn(treble_R, treble_p99)
    peak_Ln = np.clip(peak_L / max(peak_p995, 1e-9), 0, 1) ** dynamics
    peak_Rn = np.clip(peak_R / max(peak_p995, 1e-9), 0, 1) ** dynamics
    R_L = np.clip(bass_Ln * 255 * bass_boost, 0, 255)
    G_L = (mid_Ln * 255)
    B_L = (treble_Ln * 255)
    R_R = np.clip(bass_Rn * 255 * bass_boost, 0, 255)
    G_R = (mid_Rn * 255)
    B_R = (treble_Rn * 255)

    out = np.zeros((height, width, 3), dtype="float32")
    center_y = height // 2
    max_amp_px = (height // 2) - 4

    # Tornado with FOUR STRANDS as POINT-PLOTTED HELIX:
    #
    #   • L-front  phase θ            y = R_L·sin(θ),  z = R_L·cos(θ)
    #   • L-back   phase θ+π          (= other side of L-helix)
    #   • R-front  phase θ+lr_shift
    #   • R-back   phase θ+lr_shift+π
    #
    # Previous bug: each strand was rendered as a centerline-to-y
    # FILLED BAR — that mirror-fill cancels the helix's asymmetry
    # and the winding became invisible.  Real fix: plot each strand
    # as a thin POINT-LINE at y = center + amp·sin(θ_strand) and
    # let cos(θ_strand) drive z-order + brightness.  Back strand
    # at θ+π → its y is genuinely the negation of front's y BUT
    # now both are 2-px points (not filled lobes), so the helix
    # reads as four interleaving sinusoids over the L→R axis.
    #
    # If you want the old filled-lobe look back, pass mode='ribbon'
    # via an opt-in flag (TODO if requested).
    POINT_HALF = 1   # ±1px = 3-px-tall stroke per strand
    for x in range(width):
        theta = 2 * math.pi * turns * (x / width)
        theta_R = theta + lr_phase_shift
        amp_L = float(peak_Ln[x]) * max_amp_px
        amp_R = float(peak_Rn[x]) * max_amp_px
        col_L = np.array([float(R_L[x]), float(G_L[x]), float(B_L[x])])
        col_R = np.array([float(R_R[x]), float(G_R[x]), float(B_R[x])])
        strands = [
            (theta,          amp_L, col_L),   # L-front
            (theta + math.pi, amp_L, col_L),   # L-back
            (theta_R,        amp_R, col_R),   # R-front
            (theta_R + math.pi, amp_R, col_R), # R-back
        ]
        # Sort back→front (smaller cos = farther).  Front strands
        # paint last → occlude back strands at the same y.
        strands.sort(key=lambda s: math.cos(s[0]))
        for s_theta, amp_px, col in strands:
            s_sin = math.sin(s_theta)
            s_cos = math.cos(s_theta)
            y_pt = int(center_y + amp_px * s_sin)
            # Depth shading: z=+1 (front) full bright, z=-1 (back)
            # dim + low-alpha so back-strands sit BEHIND visually.
            depth_factor = 0.35 + 0.65 * ((s_cos + 1) * 0.5)
            alpha = 0.55 + 0.45 * ((s_cos + 1) * 0.5)
            painted = col * depth_factor
            y0 = max(0, y_pt - POINT_HALF)
            y1 = min(height, y_pt + POINT_HALF + 1)
            if y1 <= y0:
                continue
            cur = out[y0:y1, x, :]
            out[y0:y1, x, :] = cur * (1 - alpha) + painted * alpha

    out = np.clip(out, 0, 255).astype("uint8")
    Image.fromarray(out, mode="RGB").save(out_path)
    return True


def render_big_waveform_png(audio_path: str,
                              *,
                              width: int = 1800,
                              height: int = 240,
                              bg=(0, 0, 0),
                              bass_boost: float = 1.4,
                              show_beatgrid: bool = False,
                              dynamics: float = 1.7,
                              oversample: int = 1,
                              axis_deg: int = 135) -> bytes | None:
    """In-memory variant of `render_big_waveform` — returns PNG bytes
    (None on failure).  Default `axis_deg=135` matches the demo's
    vinyl-physical per-channel-flipped view: top=|R|/√2, bottom=|L|/√2.
    """
    try:
        import tempfile
        import os as _os
    except ImportError:
        return None
    fd, tmp = tempfile.mkstemp(suffix=".png", prefix="sidecaramel_bigwave_")
    _os.close(fd)
    try:
        ok = render_big_waveform(
            audio_path, tmp,
            width=width, height=height, bg=bg,
            bass_boost=bass_boost, show_beatgrid=show_beatgrid,
            dynamics=dynamics, oversample=oversample,
            axis_deg=axis_deg,
        )
        if not ok:
            return None
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        try:
            _os.unlink(tmp)
        except OSError:
            pass


def render_spiral_flatten_png(audio_path: str,
                                 *,
                                 turns: int = 20,
                                 width: int = 3600,
                                 height: int = 600,
                                 bass_boost: float = 1.4,
                                 dynamics: float = 1.7,
                                 lr_phase_shift: float = math.pi / 2,
                                 mono_mix: bool = False) -> bytes | None:
    """In-memory variant of `render_spiral_flatten` — returns PNG
    bytes instead of writing to a file.  Same tornado geometry, same
    locked defaults (quadrature shift, bass boost, dynamics gamma).
    Returns None when audio decode / PIL is unavailable.
    """
    try:
        import tempfile
        import os as _os
    except ImportError:
        return None
    # Render to a tempfile then read back — the underlying function
    # only takes a path, and the I/O cost is negligible compared to
    # the FFT work above.
    fd, tmp = tempfile.mkstemp(suffix=".png", prefix="sidecaramel_tornado_")
    _os.close(fd)
    try:
        ok = render_spiral_flatten(
            audio_path, tmp,
            turns=turns, width=width, height=height,
            bass_boost=bass_boost, dynamics=dynamics,
            lr_phase_shift=lr_phase_shift, mono_mix=mono_mix,
        )
        if not ok:
            return None
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        try:
            _os.unlink(tmp)
        except OSError:
            pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="sidecaramel-render",
        description="Render an audio file as a big deck-waveform via "
                    "direct per-column FFT with additive R/G/B band "
                    "coloring (bass / mid / treble).  Independent of "
                    "the Overview blob."
    )
    p.add_argument("audio", help="Audio file path (WAV/FLAC/MP3/M4A/AIFF/OGG)")
    p.add_argument("--out", required=True,
                     help="Output image path (BMP or PNG).")
    p.add_argument("--width", type=int, default=1800,
                     help="Output width in pixels (default 1800).")
    p.add_argument("--height", type=int, default=240,
                     help="Output height in pixels (default 240).")
    p.add_argument("--bass-boost", type=float, default=1.0,
                     help="Multiplier for R/bass channel saturation.")
    p.add_argument("--beatgrid", action="store_true",
                     help="Overlay Serato beatgrid markers "
                          "(downbeat orange, beats grey).")
    p.add_argument("--dynamics", type=float, default=1.6,
                     help="Peak gamma >1 expands dynamic range; "
                          "quiet stays quiet, peaks dominate "
                          "(default 1.6).")
    p.add_argument("--oversample", type=int, default=1,
                     help="Render at N× width then LANCZOS-down to "
                          "target.  Smoothes color transitions, "
                          "produces waveform-line look (default 1 "
                          "= flat bars; 4-8 = soft Serato-look).")
    p.add_argument("--spiral", action="store_true",
                     help="Render as a 3D-helix-flattened spiral "
                          "(track winds inward from outer edge).")
    p.add_argument("--spiral-turns", type=int, default=20,
                     help="Number of spiral turns (default 20).")
    p.add_argument("--lr-phase-shift", type=float,
                     default=math.pi / 2,
                     help="Phase offset between L and R viewing "
                          "angles in radians (default π/2 ≈ 1.5708, "
                          "= quadrature, default locked).  "
                          "0 = L/R synchronized; π = anti-phase.")
    p.add_argument("--mono-mix", action="store_true",
                     help="Sum L+R to mono BEFORE FFT.  Tornado "
                          "renders as 2 strands instead of 4 — "
                          "cleaner look without L/R asymmetry.")
    args = p.parse_args(argv)
    if args.spiral:
        ok = render_spiral_flatten(args.audio, args.out,
                                       width=args.width,
                                       height=args.height,
                                       turns=args.spiral_turns,
                                       bass_boost=args.bass_boost,
                                       dynamics=args.dynamics,
                                       lr_phase_shift=args.lr_phase_shift,
                                       mono_mix=args.mono_mix)
    else:
        ok = render_big_waveform(args.audio, args.out,
                                    width=args.width, height=args.height,
                                    bass_boost=args.bass_boost,
                                    show_beatgrid=args.beatgrid,
                                    dynamics=args.dynamics,
                                    oversample=args.oversample)
    if not ok:
        print(f"render failed: {args.audio}", file=sys.stderr)
        return 1
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
