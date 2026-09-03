"""sidecaramel.loops_extract — extract loop regions as audio files.

Reads all Serato loop markers from a track and writes each one as a
separate audio file (WAV by default).  Never modifies the source
audio — output goes to a target directory.

CLI:
    python -m sidecaramel loops-extract <audio> [--out-dir DIR]
                                            [--format wav|flac]

Programmatic:
    from sidecaramel.loops_extract import extract_loops
    paths = extract_loops("track.mp3", out_dir="/tmp/loops/")
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional


def _safe_filename(s: str, maxlen: int = 60) -> str:
    """Sanitize a label for use as a filename component."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_.")
    return s[:maxlen] or "loop"


def _ffmpeg_extract(audio_path: str, out_path: str,
                      start_sec: float, duration_sec: float,
                      *, out_format: str = "wav") -> bool:
    """Use ffmpeg to extract a time-range into a new audio file.

    Defensive: never overwrites the source audio.  Returns True on
    success.
    """
    if Path(audio_path).resolve() == Path(out_path).resolve():
        raise RuntimeError(
            f"out_path equals source audio_path — refusing to "
            f"overwrite source: {audio_path}")
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", audio_path,
        "-t", f"{duration_sec:.3f}",
        "-c:a", "pcm_s16le" if out_format == "wav" else "flac",
        out_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0 and os.path.isfile(out_path)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def extract_loops(audio_path: str,
                   out_dir: Optional[str] = None,
                   *,
                   out_format: str = "wav") -> List[str]:
    """Extract all Serato loops from `audio_path` as separate files.

    Args:
        audio_path: source audio file (read-only).
        out_dir:    directory to write loop files into.  If None, uses
                      `<audio_path's parent>/<basename>_loops/`.
        out_format: 'wav' (default, lossless PCM) or 'flac'.

    Returns:
        List of written file paths (one per extracted loop).  Returns
        empty list if track has no loops or ffmpeg is unavailable.

    Raises:
        FileNotFoundError if audio_path doesn't exist.
        RuntimeError      if out_dir resolves to audio_path's parent
                            AND the basename collides (defensive).
    """
    from sidecaramel.tags import read_serato_metadata

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(audio_path)

    if out_dir is None:
        ap = Path(audio_path)
        out_dir = str(ap.parent / f"{ap.stem}_loops")
    os.makedirs(out_dir, exist_ok=True)

    meta = read_serato_metadata(audio_path) or {}
    loops = meta.get("loops") or []
    if not loops:
        return []

    written: List[str] = []
    audio_ext = Path(audio_path).suffix
    for loop in loops:
        idx = loop.get("idx", 0)
        label = loop.get("label") or f"loop{idx}"
        start_sec = (loop.get("start_sec")
                       if loop.get("start_sec") is not None
                       else (loop.get("start_ms") or 0) / 1000.0)
        end_sec   = (loop.get("end_sec")
                       if loop.get("end_sec") is not None
                       else (loop.get("end_ms") or 0) / 1000.0)
        duration = end_sec - start_sec
        if duration <= 0:
            continue
        safe_label = _safe_filename(label)
        ext = "wav" if out_format == "wav" else "flac"
        out_name = f"loop_{idx:02d}_{safe_label}.{ext}"
        out_path = str(Path(out_dir) / out_name)
        ok = _ffmpeg_extract(audio_path, out_path,
                                start_sec, duration,
                                out_format=out_format)
        if ok:
            written.append(out_path)
    return written


# ---- CLI integration ----------------------------------------------

def _cli_main(args) -> int:
    paths = extract_loops(args.path,
                              out_dir=args.out_dir,
                              out_format=args.format)
    if not paths:
        print(f"no loops extracted from {args.path}")
        return 1
    print(f"extracted {len(paths)} loops:")
    for p in paths:
        print(f"  {p}")
    return 0
