"""sidecaramel.flip_render — bake a Serato CENSOR FLIP into new
audio.

2026-06-08: "render existing censor metadata flip to (vocal
only-reverse-censored auf restlichen stems normal nur an den
entscheidenden stellen audio degrading. <keep original ext> new file"

Pipeline:

    1. Read Markers2 FLIP entries from `audio_path` → list of
       CENSOR+JUMP action pairs → derive word windows
       [reverse_target_s .. jump_to_s].
    2. Load stems via sidecaramel.stems.extract_all_stems (requires
       .serato-stems sidecar).  If unavailable: fall back to full-
       audio processing without stem-isolation.
    3. Per word-window:
         vocals     → reverse-play in [reverse_target .. jump_to]
                       (matches what Serato would do live)
         drums/bass/harmony →
                       gentle audio-degrading (lowpass shelf +
                       short volume duck) — listener notices "etwas
                       passiert hier" without losing the groove
    4. Mix stems back together → write to `<basename>.CLEAN.<ext>`
       sibling (preserves original extension per the maintainer).

audio-write policy: writes to NEW file only.  Source untouched.
Serato-running check: Serato-check (read-only on audio is safe; warns if
           Serato is alive but does not block — pure read of audio
           doesn't fight Serato).
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict


@dataclass
class CensorWindow:
    """One word-window derived from a CENSOR+JUMP pair in a FLIP."""
    flip_slot: int
    flip_name: str
    reverse_target_s: float   # CENSOR action's reverse target (= word start)
    trigger_s: float           # CENSOR action's trigger (~ word mid)
    jump_to_s: float           # JUMP action's destination (= word end)

    @property
    def window(self) -> Tuple[float, float]:
        """[start_s, end_s] = [reverse_target, jump_to]."""
        return (self.reverse_target_s, self.jump_to_s)


def extract_censor_windows(audio_path: str) -> List[CensorWindow]:
    """Parse Markers2 from `audio_path` and decode every CENSOR+JUMP
    pair from each FLIP entry into a `CensorWindow`."""
    from sidecaramel.tags import harvest_serato_blobs
    from sidecaramel.flip_writer import decode_outer, parse_inner

    out: List[CensorWindow] = []
    inner = None
    for desc, payload, _ in harvest_serato_blobs(audio_path):
        if desc == "Serato Markers2":
            try:
                inner, _ = decode_outer(payload)
            except Exception:
                inner = None
            break
    if inner is None:
        return out
    entries = parse_inner(inner)
    for e in entries:
        if e.type != "FLIP" or len(e.body) < 7:
            continue
        # FLIP body layout (PROJECT FACTS):
        #   u8 NUL_prefix, u8 slot, u8 flag, cstring name,
        #   u8 loop, u32 BE action_count, actions...
        slot = e.body[1]
        cur = 3
        nul = e.body.find(b"\x00", cur)
        name = (e.body[cur:nul].decode("utf-8", "ignore")
                if nul >= 0 else "")
        cur = (nul + 1) if nul >= 0 else cur
        if cur + 5 > len(e.body):
            continue
        cur += 1  # loop byte
        action_count = struct.unpack(">I", e.body[cur:cur + 4])[0]
        cur += 4
        # Decode actions sequentially, then pair CENSOR with the
        # NEXT JUMP (build_word_bleep emits them in order).
        actions: List[dict] = []
        for _ in range(action_count):
            if cur + 5 > len(e.body):
                break
            aid = e.body[cur]
            alen = struct.unpack(">I", e.body[cur + 1:cur + 5])[0]
            payload = e.body[cur + 5:cur + 5 + alen]
            cur += 5 + alen
            if aid == 0 and alen == 16:
                a, b = struct.unpack(">dd", payload)
                actions.append({"type": "jump", "from_s": a, "to_s": b})
            elif aid == 1 and alen == 24:
                t, r, s = struct.unpack(">ddd", payload)
                actions.append({"type": "censor",
                                  "trigger_s": t,
                                  "reverse_target_s": r,
                                  "speed": s})
        # Pair CENSOR+JUMP
        i = 0
        while i < len(actions) - 1:
            a, b = actions[i], actions[i + 1]
            if a["type"] == "censor" and b["type"] == "jump":
                out.append(CensorWindow(
                    flip_slot=int(slot),
                    flip_name=name,
                    reverse_target_s=float(a["reverse_target_s"]),
                    trigger_s=float(a["trigger_s"]),
                    jump_to_s=float(b["to_s"]),
                ))
                i += 2
            else:
                i += 1
    return out


# =====================================================================
# Per-stem effects
# =====================================================================

def _reverse_in_place(samples, sr: int, start_s: float, end_s: float):
    """Reverse the [start_s..end_s] audio window in place."""
    i0 = max(0, int(start_s * sr))
    i1 = min(len(samples), int(end_s * sr))
    if i1 > i0:
        samples[i0:i1] = samples[i0:i1][::-1]


def _degrade_in_place(samples, sr: int, start_s: float, end_s: float):
    """Apply 'something happening here' effect to instrumental stem:
    short volume duck + low-pass roll-off via simple moving-average
    so the listener notices an artifact at the cuss-word position.
    """
    import numpy as np
    i0 = max(0, int(start_s * sr))
    i1 = min(len(samples), int(end_s * sr))
    if i1 <= i0:
        return
    seg = samples[i0:i1].astype("float32")
    n = len(seg)
    # 1) lowpass via moving average (kernel = 8 samples at high SR
    #    ≈ ~5.5 kHz cutoff, audible muffle)
    if n > 16:
        K = 8
        kernel = np.ones(K, dtype="float32") / K
        seg = np.convolve(seg, kernel, mode="same").astype("float32")
    # 2) duck — Hanning-window-shaped volume fade -10dB centred
    ramp = 0.55 + 0.45 * (1 - np.hanning(n)).astype("float32")
    seg = seg * ramp
    samples[i0:i1] = seg


# =====================================================================
# Audio decode / encode helpers
# =====================================================================

def _decode_to_array(audio_path: str, sr: int = 44100
                       ) -> Tuple["numpy.ndarray", int]:
    """Decode any audio to float32 stereo at `sr`.  Falls back to ffmpeg
    pipe if soundfile can't read directly."""
    import io
    import subprocess
    import numpy as np
    try:
        import soundfile as sf
        data, file_sr = sf.read(audio_path, dtype="float32",
                                   always_2d=True)
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        if file_sr != sr:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(file_sr), int(sr))
            data = np.stack([resample_poly(data[:, c], sr // g,
                                                file_sr // g)
                                for c in range(data.shape[1])], axis=1)
        return data.astype("float32"), sr
    except Exception:
        cmd = ["ffmpeg", "-loglevel", "error", "-i", audio_path,
                "-ar", str(sr), "-ac", "2", "-f", "wav", "pipe:1"]
        out = subprocess.run(cmd, capture_output=True, check=True)
        import soundfile as sf
        data, file_sr = sf.read(io.BytesIO(out.stdout),
                                   dtype="float32", always_2d=True)
        return data.astype("float32"), sr


def _write_audio(out_path: str, samples, sr: int) -> None:
    """Write samples to out_path.  Uses ffmpeg for non-WAV output to
    preserve original container (MP3, FLAC, M4A, etc.)."""
    import io
    import subprocess
    import soundfile as sf
    ext = os.path.splitext(out_path)[1].lower().lstrip(".")
    if ext in ("wav", "aiff", "aif", "flac", "ogg"):
        # Native soundfile write
        subtype = "PCM_16"
        if ext == "flac":
            subtype = "PCM_16"
        sf.write(out_path, samples, sr, subtype=subtype,
                   format=ext.upper() if ext != "aif" else "AIFF")
        return
    # MP3, M4A, MP4 etc. → write to temp WAV then ffmpeg-encode
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        sf.write(tmp.name, samples, sr, subtype="PCM_16",
                   format="WAV")
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
                "-i", tmp.name, out_path]
        subprocess.run(cmd, check=True)
    finally:
        try: os.unlink(tmp.name)
        except OSError: pass


# =====================================================================
# Main render
# =====================================================================

def render_baked_censor(audio_path: str,
                          out_path: Optional[str] = None,
                          *,
                          use_stems: bool = True,
                          target_sr: int = 44100) -> Tuple[str, dict]:
    """Bake the audio file's CENSOR FLIPs into a new audio file.

    Returns (out_path, stats_dict).  Stats include
    {windows, stems_used, sr, duration_sec}.
    """
    try:
        import numpy as np
    except ImportError:
        raise RuntimeError("flip_render needs numpy + scipy + "
                              "soundfile + ffmpeg")
    windows = extract_censor_windows(audio_path)
    if not windows:
        raise RuntimeError(
            f"no CENSOR FLIPs found in {audio_path!r}.  Add a FLIP "
            "first via python -m sidecaramel.cuss_flip --write.")

    if out_path is None:
        stem, ext = os.path.splitext(audio_path)
        out_path = stem + ".CLEAN" + ext

    stem_paths: Dict[str, Path] = {}
    if use_stems:
        from sidecaramel.stems import extract_all_stems
        try:
            stem_paths = extract_all_stems(audio_path,
                                              log=lambda m: print(f"  {m}"))
        except Exception as e:
            print(f"  [warn] stem extraction failed: {e}")
            stem_paths = {}

    stats = {
        "windows": [w.window for w in windows],
        "n_windows": len(windows),
        "stems_used": list(stem_paths.keys()),
        "sr": target_sr,
        "out_path": out_path,
    }

    # Case A: stems available → process per-stem then mix
    if stem_paths:
        print(f"  per-stem processing ({len(stem_paths)} stems): "
                f"{list(stem_paths.keys())}")
        mixed = None
        for name, path in stem_paths.items():
            data, _ = _decode_to_array(str(path), sr=target_sr)
            for w in windows:
                s, e = w.window
                if name == "vocals":
                    _reverse_in_place(data[:, 0], target_sr, s, e)
                    _reverse_in_place(data[:, 1], target_sr, s, e)
                else:
                    _degrade_in_place(data[:, 0], target_sr, s, e)
                    _degrade_in_place(data[:, 1], target_sr, s, e)
            if mixed is None:
                mixed = data.copy()
            else:
                # Pad / truncate to match length
                n = min(len(mixed), len(data))
                mixed = mixed[:n] + data[:n]
        # Normalize headroom — additive 4 stems can clip
        if mixed is not None:
            peak = float(np.max(np.abs(mixed)))
            if peak > 0.99:
                mixed = (mixed / peak) * 0.95
            _write_audio(out_path, mixed, target_sr)
            stats["duration_sec"] = float(len(mixed) / target_sr)
            return out_path, stats

    # Case B: no stems → process full audio (vocal-only reverse not
    # possible; do degrade-only on the whole window)
    print(f"  no stems available — degrading full audio at windows")
    data, _ = _decode_to_array(audio_path, sr=target_sr)
    for w in windows:
        s, e = w.window
        _reverse_in_place(data[:, 0], target_sr, s, e)
        _reverse_in_place(data[:, 1], target_sr, s, e)
    _write_audio(out_path, data, target_sr)
    stats["duration_sec"] = float(len(data) / target_sr)
    return out_path, stats


# =====================================================================
# Generic FLIP renderer
# =====================================================================

def _has_video_stream(path: str) -> bool:
    """ffprobe check whether `path` has a video stream."""
    import subprocess
    try:
        r = subprocess.run(
            ["ffprobe", "-loglevel", "error",
              "-select_streams", "v:0",
              "-show_entries", "stream=codec_type",
              "-of", "csv=p=0", path],
            capture_output=True, timeout=10, text=True)
        return "video" in r.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _build_flip_timeline(flip: dict, duration_s: float
                          ) -> List[Tuple[float, float, str]]:
    """Convert a FLIP dict's actions into output-time segments.

    Returns list of (src_start_s, src_end_s, mode) where mode is
    'forward' or 'reverse'.  Glued together in order produces the
    output audio.

    Looped FLIPs are flattened to one pass (looped flips play as
    loops that run once).  Non-looped behave the same — single
    pass through the actions.
    """
    actions = flip.get("actions") or []
    # If no actions, just play the full source unchanged
    if not actions:
        return [(0.0, duration_s, "forward")]

    segments: List[Tuple[float, float, str]] = []
    cursor = 0.0
    for act in actions:
        t = act.get("type")
        if t == "jump":
            from_s = act["from_s"]
            to_s   = act["to_s"]
            # Play forward from cursor up to from_s, then jump to to_s
            if from_s > cursor:
                segments.append((cursor, from_s, "forward"))
            cursor = to_s
        elif t == "censor":
            # CENSOR convention: trigger_s = LATER time (where to
            # start reversing FROM), reverse_target_s = EARLIER time.
            # Pattern in real Serato: CENSOR is paired with a
            # following JUMP — but a bare CENSOR means "reverse-
            # play from trigger backwards to reverse_target".
            trigger = act["trigger_s"]
            reverse_target = act["reverse_target_s"]
            # Play forward up to reverse_target (if cursor < reverse_target)
            if reverse_target > cursor:
                segments.append((cursor, reverse_target, "forward"))
            # Reverse segment: reverse_target..trigger played backward
            segments.append((reverse_target, trigger, "reverse"))
            cursor = trigger
        # unknown action type → ignore
    # Tail: from cursor to end of track
    if cursor < duration_s:
        segments.append((cursor, duration_s, "forward"))
    return segments


def _render_segments_to_audio(samples, sr: int,
                                segments: List[Tuple[float, float, str]]):
    """Concatenate audio per timeline segments into a single array."""
    import numpy as np
    out_parts = []
    n_total = len(samples)
    for s_start, s_end, mode in segments:
        a = max(0, int(s_start * sr))
        b = min(n_total, int(s_end * sr))
        if b <= a:
            continue
        chunk = samples[a:b]
        if mode == "reverse":
            chunk = chunk[::-1]
        out_parts.append(chunk)
    if not out_parts:
        return samples[:0]
    return np.concatenate(out_parts, axis=0)


def _copy_nonserato_tags(src: str, dst: str) -> None:
    """Copy mutagen tags from src → dst, omitting Serato-specific
    GEOBs / atoms / Vorbis comments.
    """
    try:
        import mutagen
        from mutagen.id3 import ID3
        from mutagen.mp4 import MP4
        from mutagen.flac import FLAC
    except ImportError:
        return
    # Try each container type
    try:
        src_obj = mutagen.File(src)
        dst_obj = mutagen.File(dst)
        if src_obj is None or dst_obj is None:
            return
        # ID3 path (MP3 / WAV / AIFF)
        if hasattr(src_obj, "tags") and isinstance(src_obj.tags,
                                                       (ID3, type(None))):
            if src_obj.tags is None:
                return
            for key, frame in src_obj.tags.items():
                # Skip Serato GEOBs
                desc = getattr(frame, "desc", "") or ""
                if isinstance(desc, str) and desc.startswith("Serato"):
                    continue
                # Skip other GEOBs from Serato sandbox if any
                if (hasattr(dst_obj, "tags")
                        and dst_obj.tags is not None):
                    dst_obj.tags[key] = frame
            dst_obj.save()
            return
        # MP4 path
        if isinstance(src_obj, MP4):
            for key, val in (src_obj.tags or {}).items():
                if "com.serato.dj" in key:
                    continue
                dst_obj.tags[key] = val
            dst_obj.save()
            return
        # FLAC / Vorbis path
        if isinstance(src_obj, FLAC):
            for key, vals in src_obj.tags or []:
                if key.lower().startswith("serato_"):
                    continue
                dst_obj[key] = vals
            dst_obj.save()
            return
    except Exception:
        pass


def render_flip(audio_path: str,
                  flip: dict,
                  out_path: Optional[str] = None,
                  *,
                  censor_with_stems: bool = False,
                  preserve_video: bool = True,
                  target_sr: int = 44100) -> Tuple[str, dict]:
    """Render a single Serato FLIP into a new file.

    Args:
        audio_path: source audio/video file (READ-ONLY).
        flip: parsed FLIP dict from
            `parse_serato_markers2_full(...)['flips'][i]`.  Must
            contain `actions` list (see decoder above).
        out_path: target path.  Defaults to
            `<basename>.flip_<slot>_<name>.<ext>`.
        censor_with_stems: if True AND the FLIP contains CENSOR
            actions AND a `.serato-stems` sidecar exists, only
            censor the vocals stem and remux with the other 3
            stems.  See `render_baked_censor()` semantics.
        preserve_video: if source has a video stream, copy it
            through unchanged and only edit the audio track.
        target_sr: sample rate to decode audio at.

    Returns (out_path, stats_dict).
    """
    import numpy as np
    import subprocess

    if out_path is None:
        stem, ext = os.path.splitext(audio_path)
        slot = flip.get("slot", 0)
        name_safe = "".join(c if c.isalnum() else "_"
                              for c in (flip.get("name") or "flip"))[:32]
        out_path = f"{stem}.flip_{slot:02d}_{name_safe}{ext}"

    # Get duration via ffprobe
    try:
        r = subprocess.run(
            ["ffprobe", "-loglevel", "error",
              "-show_entries", "format=duration",
              "-of", "csv=p=0", audio_path],
            capture_output=True, timeout=10, text=True)
        duration_s = float(r.stdout.strip() or 0)
    except Exception:
        duration_s = 0.0

    segments = _build_flip_timeline(flip, duration_s)

    # Censor-with-stems mode
    has_censor = any(a.get("type") == "censor"
                       for a in (flip.get("actions") or []))
    has_video = preserve_video and _has_video_stream(audio_path)

    if censor_with_stems and has_censor:
        # Bake stems-aware censor (vocals-only reverse) using
        # existing render_baked_censor.  Drops video.
        return render_baked_censor(audio_path, out_path,
                                       use_stems=True,
                                       target_sr=target_sr)

    # Generic timeline render: decode → splice segments → write
    samples, sr = _decode_to_array(audio_path, sr=target_sr)
    spliced = _render_segments_to_audio(samples, sr, segments)

    if has_video:
        # Render audio to temp WAV, then mux with original video
        import tempfile
        tmp_wav = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False).name
        try:
            _write_audio(tmp_wav, spliced, sr)
            # Use ffmpeg to mux: video copy + new audio
            cmd = ["ffmpeg", "-loglevel", "error", "-y",
                    "-i", audio_path,
                    "-i", tmp_wav,
                    "-map", "0:v",
                    "-map", "1:a",
                    "-c:v", "copy",
                    "-c:a", "aac" if out_path.lower().endswith(".mp4")
                                  or out_path.lower().endswith(".m4a")
                              else "libmp3lame",
                    out_path]
            r = subprocess.run(cmd, capture_output=True, timeout=300)
            if r.returncode != 0:
                # Fallback: write audio only
                _write_audio(out_path, spliced, sr)
        finally:
            try: os.remove(tmp_wav)
            except Exception: pass
    else:
        _write_audio(out_path, spliced, sr)

    # Copy non-Serato tags
    _copy_nonserato_tags(audio_path, out_path)

    return out_path, {
        "slot":     flip.get("slot"),
        "name":     flip.get("name"),
        "loop":     flip.get("loop"),
        "actions":  len(flip.get("actions") or []),
        "segments": len(segments),
        "duration_in":  duration_s,
        "duration_out": float(len(spliced) / sr),
        "video":    has_video,
        "sr":       sr,
        "out_path": out_path,
    }


def render_all_flips(audio_path: str,
                       out_dir: Optional[str] = None,
                       *,
                       censor_with_stems: bool = False,
                       preserve_video: bool = True,
                       target_sr: int = 44100) -> List[Tuple[str, dict]]:
    """Render every FLIP in `audio_path` into separate files.

    Returns list of (out_path, stats_dict) tuples — one per FLIP.
    """
    from sidecaramel.tags import (harvest_serato_blobs,
                                       parse_serato_markers2_full)

    ref_blob = None
    for desc, payload, _src in harvest_serato_blobs(audio_path):
        if desc == "Serato Markers2":
            ref_blob = bytes(payload); break
    if ref_blob is None:
        return []
    parsed = parse_serato_markers2_full(ref_blob) or {}
    flips = parsed.get("flips") or []
    if not flips:
        return []

    if out_dir is None:
        ap = Path(audio_path)
        out_dir = str(ap.parent / f"{ap.stem}_flips")
    os.makedirs(out_dir, exist_ok=True)

    src_ext = os.path.splitext(audio_path)[1]
    results = []
    for flip in flips:
        slot = flip.get("slot", 0)
        name_safe = "".join(c if c.isalnum() else "_"
                              for c in (flip.get("name") or "flip"))[:32]
        loop_tag = "_loop" if flip.get("loop") else ""
        out_path = str(Path(out_dir) /
                        f"flip_{slot:02d}_{name_safe}{loop_tag}{src_ext}")
        try:
            written, stats = render_flip(
                audio_path, flip, out_path,
                censor_with_stems=censor_with_stems,
                preserve_video=preserve_video,
                target_sr=target_sr)
            results.append((written, stats))
        except Exception as e:
            results.append((out_path, {"error": repr(e)}))
    return results


# =====================================================================
# CLI
# =====================================================================

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m sidecaramel.flip_render",
        description=(
            "Bake a track's Serato CENSOR FLIP into new audio.  "
            "Vocal stem gets reverse-censored; other stems get "
            "audio-degrading at the same word position.  Writes a "
            "<name>.CLEAN.<ext> sibling (audio-write policy — source untouched)."))
    p.add_argument("tracks", nargs="+", help="audio file(s) with Markers2 FLIPs")
    p.add_argument("--no-stems", action="store_true",
                     help="skip stem extraction; reverse full audio "
                          "(falls back to whole-mix processing)")
    p.add_argument("--sr", type=int, default=44100,
                     help="target sample rate (default 44100)")
    args = p.parse_args(argv)
    ok = 0
    for t in args.tracks:
        print(f"=== {os.path.basename(t)} ===")
        try:
            wins = extract_censor_windows(t)
            if not wins:
                print(f"  no CENSOR FLIPs — skipping")
                continue
            print(f"  {len(wins)} CENSOR windows from FLIPs:")
            for w in wins[:5]:
                print(f"    slot {w.flip_slot} '{w.flip_name}': "
                        f"{w.reverse_target_s:>6.2f}s .. {w.jump_to_s:>6.2f}s")
            if len(wins) > 5:
                print(f"    … +{len(wins)-5} more")
            out, stats = render_baked_censor(
                t, use_stems=not args.no_stems, target_sr=args.sr)
            print(f"  wrote {out}  ({stats['duration_sec']:.1f}s, "
                    f"stems={stats['stems_used']})")
            ok += 1
        except Exception as e:
            print(f"  ERROR: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
