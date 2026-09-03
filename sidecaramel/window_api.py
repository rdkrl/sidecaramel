"""sidecaramel.window_api — windowed-waveform interface for live
overlays (external overlay HTML L-view consumer).

Provides an interface for scrolling the waveform with roughly
2-3 bars on screen, alongside the lyric tiers and structure
analysis in the formerly-L view.

Two consumption modes:

1. **`render_window_png(...)`** — returns a finished PNG (bytes
   ready to base64-embed in `<img src="data:image/png;...">` or
   write to a file).  Use this for the simplest integration: server
   re-renders on every scroll, the overlay just shows the image.

2. **`compute_window_data(...)`** — returns a JSON-friendly dict of
   per-column (bass, mid, treble, peak) energies + beat / cue lists
   in the visible window.  Use this if the consumer wants to draw via HTML5
   canvas / SVG in the overlay for sub-frame-accurate scroll
   animation without re-fetching PNGs.

Both functions take a `center_time_sec` so the consumer can slide
the window left/right freely.  At the maintainer's stated 2-3 bars visible:
`window_seconds = bars_visible * 60.0 / bpm * 4`.

Cache layer (`_band_cache`) memoizes the full-track per-column FFT
arrays per audio_path so repeated window-calls during scroll are
fast (< 5 ms after the initial decode).
"""
from __future__ import annotations

import base64
import io
import math
import os
import threading
from typing import Optional, Tuple, List, Dict, Any


# Audio decode + FFT cache.  Key = audio_path; value = dict with
# bass/mid/treble/peak numpy arrays + sr + n_columns + duration.
_band_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()

# Max distinct tracks we keep in memory.  ~5 MB per track.
_CACHE_MAX = 8


def _ensure_full_track_bands(audio_path: str,
                                native_columns: int = 8000
                                ) -> Optional[Dict[str, Any]]:
    """Compute (or fetch from cache) the full-track per-column FFT
    band-energy arrays at high resolution.  Subsequent windowed
    views slice into these arrays."""
    with _cache_lock:
        cached = _band_cache.get(audio_path)
        if cached and cached.get("n_columns", 0) >= native_columns:
            return cached
    try:
        import numpy as np
        from sidecaramel.audio_render import (_decode_audio,
                                                  _band_energies_per_column)
    except ImportError:
        return None
    samples, sr = _decode_audio(audio_path, stereo=False)
    if len(samples) < 1024:
        return None
    bass, mid, treble, peak = _band_energies_per_column(samples, sr,
                                                            native_columns)
    duration = len(samples) / sr
    entry = {
        "sr": sr,
        "duration": float(duration),
        "n_columns": native_columns,
        "bass": bass,
        "mid": mid,
        "treble": treble,
        "peak": peak,
    }
    with _cache_lock:
        # LRU-ish eviction: drop oldest if over limit
        if len(_band_cache) >= _CACHE_MAX:
            _band_cache.pop(next(iter(_band_cache)))
        _band_cache[audio_path] = entry
    return entry


def _slice_window(entry: Dict[str, Any],
                    center_time_sec: float,
                    window_seconds: float,
                    out_columns: int
                    ) -> Optional[Dict[str, Any]]:
    """Slice the full-track arrays to a window and resample to
    out_columns via linear interpolation.

    When the requested window
    extends BEFORE 0 or BEYOND duration, the corresponding output
    columns get amplitude 0 instead of being stretched-filled with
    in-bounds audio.  This preserves absolute time-position mapping
    (x=W/2 = center, regardless of edges).
    """
    try:
        import numpy as np
    except ImportError:
        return None
    duration = entry["duration"]
    n_total = entry["n_columns"]
    half = window_seconds / 2.0
    # Un-clamped requested window — may extend before 0 or beyond
    # duration; that's intentional.
    req_start = center_time_sec - half
    req_end   = center_time_sec + half
    if req_end <= req_start:
        return None

    # Map absolute-time → source-column-index (sec_per_col = duration/n_total)
    req_start_idx = (req_start / duration) * n_total
    req_end_idx   = (req_end   / duration) * n_total
    src_idx = np.linspace(req_start_idx, req_end_idx, out_columns)

    # Out-of-bounds mask: columns whose source index is outside the
    # track's [0, n_total) range → render as silence.
    oob_mask = (src_idx < 0) | (src_idx >= n_total)
    src_idx_clipped = np.clip(src_idx, 0, n_total - 1)
    src_lo = src_idx_clipped.astype("int32")
    src_hi = np.clip(src_lo + 1, 0, n_total - 1)
    frac = src_idx_clipped - src_lo

    def _lerp(arr):
        out = arr[src_lo] * (1 - frac) + arr[src_hi] * frac
        # Zero out-of-bounds columns AFTER lerp
        if oob_mask.any():
            out = out.copy()
            out[oob_mask] = 0.0
        return out

    return {
        # UN-clamped window bounds — beats/cues x_frac uses these for
        # correct absolute-time positioning even when window extends
        # past track edges.
        "win_start_sec": float(req_start),
        "win_end_sec":   float(req_end),
        "bass":   _lerp(entry["bass"]),
        "mid":    _lerp(entry["mid"]),
        "treble": _lerp(entry["treble"]),
        "peak":   _lerp(entry["peak"]),
    }


def _norm_arrays(slice_data: Dict[str, Any], dynamics: float = 1.7
                  ) -> Dict[str, "numpy.ndarray"]:
    """Normalize band arrays to 0..1 with gamma."""
    import numpy as np
    def _n(x, p, g):
        v = float(np.percentile(x, p))
        if v < 1e-9:
            return np.zeros_like(x)
        return np.clip(x / v, 0, 1) ** g
    return {
        "bass_n": _n(slice_data["bass"], 99, 0.75),
        "mid_n": _n(slice_data["mid"], 99, 0.75),
        "treble_n": _n(slice_data["treble"], 99, 0.75),
        "peak_n": _n(slice_data["peak"], 99.5, dynamics),
    }


def render_window_png(audio_path: str,
                        center_time_sec: float,
                        window_seconds: float,
                        *,
                        width: int = 1200,
                        height: int = 200,
                        bass_boost: float = 1.4,
                        dynamics: float = 1.7,
                        show_centerline: bool = True,
                        min_amp_floor: float = 0.0
                        ) -> Optional[bytes]:
    """Render the audio window as a PNG.  Returns raw PNG bytes
    (suitable for writing to disk OR base64-embedding into an
    `<img>` element).

    Cached: the full-track FFT runs once per audio_path; subsequent
    calls with different center_time_sec are sub-millisecond.

    Args:
        min_amp_floor: 0.0..1.0, minimum peak amplitude that gets
            rendered as a visible bar.  Default 0.0 = silent regions
            render fully-black.  Set e.g. 0.02 for overview-style
            "barely-visible thin bars in silent areas" look.  Useful
            to distinguish "silent intro" from "no audio at all"
            (= before track start / after track end).

    Returns None on decode failure or if numpy/Pillow missing.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    entry = _ensure_full_track_bands(audio_path)
    if not entry:
        return None
    slice_data = _slice_window(entry, center_time_sec, window_seconds, width)
    if not slice_data:
        return None
    norm = _norm_arrays(slice_data, dynamics=dynamics)
    bass_n = norm["bass_n"]
    mid_n = norm["mid_n"]
    treble_n = norm["treble_n"]
    peak_n = norm["peak_n"]

    # Compute in-bounds mask BEFORE applying floor, so before-/after-
    # track-edge columns stay TRULY black even when min_amp_floor>0.
    duration = entry["duration"]
    half = window_seconds / 2.0
    req_start = center_time_sec - half
    req_end   = center_time_sec + half
    abs_times = np.linspace(req_start, req_end, width)
    in_bounds = (abs_times >= 0.0) & (abs_times < duration)

    if min_amp_floor > 0.0:
        # Lift in-bounds silent areas to floor; out-of-bounds stay 0
        peak_n = np.where(in_bounds,
                             np.maximum(peak_n, min_amp_floor),
                             peak_n)

    R = np.clip(bass_n * 255 * bass_boost, 0, 255).astype("uint8")
    G = (mid_n * 255).astype("uint8")
    B = (treble_n * 255).astype("uint8")
    img = np.zeros((height, width, 3), dtype="uint8")
    center = height // 2
    half_body = center - 2
    bar_half = (peak_n * half_body).astype("int32")
    # For floor-lifted columns where R+G+B are all zero (truly silent
    # in-bounds audio), use a subtle grey so the bar is visible.
    fallback_color = (50, 50, 50)
    for x in range(width):
        h = int(bar_half[x])
        if h <= 0:
            continue
        y0 = max(0, center - h)
        y1 = min(height, center + h)
        r_v, g_v, b_v = int(R[x]), int(G[x]), int(B[x])
        if r_v + g_v + b_v == 0:
            r_v, g_v, b_v = fallback_color
        img[y0:y1, x, 0] = r_v
        img[y0:y1, x, 1] = g_v
        img[y0:y1, x, 2] = b_v
    if show_centerline:
        img[center, :, :] = (60, 60, 60)
    pil = Image.fromarray(img, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def compute_window_data(audio_path: str,
                         center_time_sec: float,
                         window_seconds: float,
                         *,
                         columns: int = 240,
                         include_beats: bool = True,
                         include_cues: bool = True,
                         dynamics: float = 1.7
                         ) -> Optional[Dict[str, Any]]:
    """Return a JSON-friendly dict describing the windowed view's
    per-column band energies + beats + cues within the window.
    Consumer (e.g. an overlay via canvas) does its own rendering.

    Schema:
    ```
    {
      "audio_path":  str,
      "center_sec":  float,
      "window_sec":  [start, end],
      "columns":     int,
      "duration_sec": float,
      "bass":        [N floats 0..1],
      "mid":         [N floats 0..1],
      "treble":      [N floats 0..1],
      "peak":        [N floats 0..1],
      "beats": [{                          # if include_beats
          "time_sec": float,
          "is_downbeat": bool,
          "is_anchor": bool,
          "x_frac": float                  # 0..1 within window
      }, ...],
      "cues":  [{                          # if include_cues
          "idx": int, "time_sec": float,
          "color_rgb": [r,g,b],
          "label": str,
          "x_frac": float
      }, ...],
      "bpm":   float | None,
    }
    ```
    """
    try:
        import numpy as np
    except ImportError:
        return None
    entry = _ensure_full_track_bands(audio_path)
    if not entry:
        return None
    slice_data = _slice_window(entry, center_time_sec, window_seconds, columns)
    if not slice_data:
        return None
    norm = _norm_arrays(slice_data, dynamics=dynamics)
    win_start = slice_data["win_start_sec"]
    win_end = slice_data["win_end_sec"]
    win_dur = win_end - win_start
    if win_dur < 1e-6:
        return None
    result = {
        "audio_path": audio_path,
        "center_sec": float(center_time_sec),
        "window_sec": [float(win_start), float(win_end)],
        "columns": columns,
        "duration_sec": entry["duration"],
        "bass": norm["bass_n"].tolist(),
        "mid": norm["mid_n"].tolist(),
        "treble": norm["treble_n"].tolist(),
        "peak": norm["peak_n"].tolist(),
        "bpm": None,
        "beats": [],
        "cues": [],
    }
    if not (include_beats or include_cues):
        return result
    try:
        from sidecaramel.tags import (read_serato_metadata,
                                          serato_beat_grid_seconds,
                                          harvest_serato_blobs,
                                          _parse_serato_beatgrid)
    except ImportError:
        return result
    meta = read_serato_metadata(audio_path)
    if meta:
        result["bpm"] = meta.get("bpm")
        if include_beats and result["bpm"]:
            beats = serato_beat_grid_seconds(meta, entry["duration"])
            anchors = []
            for desc, payload, _ in harvest_serato_blobs(audio_path):
                if desc == "Serato BeatGrid":
                    bg = _parse_serato_beatgrid(payload)
                    if bg:
                        anchors = [m[0] for m in bg.get("markers", [])]
                    break
            beat_period = 60.0 / result["bpm"]
            cues_list = meta.get("cues") or []
            downbeat_t = (anchors[0] if anchors
                              else (cues_list[0]["pos_ms"] / 1000.0
                                     if cues_list else
                                     (beats[0] if beats else 0.0)))
            for bt in beats:
                if bt < win_start or bt > win_end:
                    continue
                n_from = round((bt - downbeat_t) / beat_period)
                is_down = (n_from % 4) == 0
                is_anchor = any(abs(bt - a) < beat_period * 0.5
                                   for a in anchors)
                result["beats"].append({
                    "time_sec": float(bt),
                    "is_downbeat": bool(is_down),
                    "is_anchor": bool(is_anchor),
                    "x_frac": float((bt - win_start) / win_dur),
                })
        if include_cues:
            for cue in (meta.get("cues") or []):
                t = cue["pos_ms"] / 1000.0
                if t < win_start or t > win_end:
                    continue
                result["cues"].append({
                    "idx": cue["idx"],
                    "time_sec": float(t),
                    "color_rgb": list(cue.get("color", (255, 255, 0))),
                    "label": cue.get("label") or "",
                    "x_frac": float((t - win_start) / win_dur),
                })
    return result


def seconds_for_bars(bars: float, bpm: float,
                       beats_per_bar: int = 4) -> float:
    """Convenience: convert (bars, bpm) to window-seconds."""
    if bpm <= 0:
        return 0.0
    return bars * beats_per_bar * 60.0 / bpm


def render_window_png_base64(audio_path: str,
                                center_time_sec: float,
                                window_seconds: float,
                                **kwargs) -> Optional[str]:
    """Same as render_window_png but returns a base64 data-URI
    string ready for `<img src="...">` embedding."""
    png = render_window_png(audio_path, center_time_sec,
                                window_seconds, **kwargs)
    if not png:
        return None
    return "data:image/png;base64," + base64.b64encode(png).decode()


# =====================================================================
# Self-test
# =====================================================================

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python -m sidecaramel.window_api <audio> "
                "[center_sec=0] [window_sec=2.0]")
        sys.exit(2)
    path = sys.argv[1]
    center = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    win = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    data = compute_window_data(path, center, win)
    if not data:
        print("compute_window_data failed", file=sys.stderr)
        sys.exit(1)
    print(json.dumps({
        "audio_path": data["audio_path"],
        "center_sec": data["center_sec"],
        "window_sec": data["window_sec"],
        "columns": data["columns"],
        "bpm": data["bpm"],
        "n_beats_in_window": len(data["beats"]),
        "n_cues_in_window": len(data["cues"]),
        "peak_sample": data["peak"][:8],
    }, indent=2))
