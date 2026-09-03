"""sidecaramel.verify — cross-checks the decoders against ground truth.

For each sample track:
  • duration from mutagen vs BeatGrid terminal_position (sanity)
  • Autotags BPM vs BeatGrid terminal BPM (display vs analytic)
  • Cue point positions land inside [0, duration]
  • FLIP jump targets land inside [0, duration]

2026-05-12. Verification pass.
"""
from __future__ import annotations

import base64
import os
import struct
import sys

from mutagen import File as MFile


def get_blob(path: str, descriptor: str) -> bytes | None:
    f = MFile(path)
    if f is None or f.tags is None:
        return None
    for k in f.tags.keys():
        ks = str(k)
        if ks.startswith("GEOB:") and descriptor in ks:
            return bytes(f.tags[k].data)
        if "com.serato.dj" in ks:
            v = f.tags[k]
            raw = bytes(v[0] if isinstance(v, list) else v)
            decoded = base64.b64decode(raw + b"==", validate=False)
            if descriptor.encode() in decoded:
                idx = decoded.find(descriptor.encode())
                nul = decoded.find(b"\x00", idx)
                return decoded[nul + 1:]
    return None


def beatgrid_terminal(bg: bytes) -> tuple[int, float, float]:
    nm = struct.unpack(">I", bg[2:6])[0]
    off = 6 + (nm - 1) * 8
    pos, bpm = struct.unpack(">ff", bg[off:off + 8])
    return nm, pos, bpm


def autotags_bpm(at: bytes) -> float:
    # bytes: u8 major, u8 minor, ascii bpm, NUL, …
    # (no NUL between version and bpm)
    return float(at[2:].split(b"\x00")[0])


def cue_positions_ms(markers2_raw: bytes) -> list[int]:
    # outer: u8 u8 \0 base64 \0…
    payload = markers2_raw[2:]
    while payload and payload[0] == 0:
        payload = payload[1:]
    end = payload.find(b"\x00")
    inner = base64.b64decode(
        payload[:end].replace(b"\n", b"") + b"==", validate=False)
    positions = []
    cur = 2
    while cur < len(inner):
        nul = inner.find(b"\x00", cur)
        if nul < 0:
            break
        entry_type = inner[cur:nul].decode("ascii", "replace")
        cur = nul + 1
        if not entry_type:
            break
        elen = struct.unpack(">I", inner[cur:cur + 4])[0]
        cur += 4
        body = inner[cur:cur + elen]
        cur += elen
        if entry_type == "CUE" and len(body) >= 6:
            positions.append(struct.unpack(">I", body[2:6])[0])
    return positions


def flip_jumps(markers2_raw: bytes) -> list[tuple[float, float]]:
    # Same outer unwrap
    payload = markers2_raw[2:]
    while payload and payload[0] == 0:
        payload = payload[1:]
    end = payload.find(b"\x00")
    inner = base64.b64decode(
        payload[:end].replace(b"\n", b"") + b"==", validate=False)
    jumps = []
    cur = 2
    while cur < len(inner):
        nul = inner.find(b"\x00", cur)
        if nul < 0:
            break
        entry_type = inner[cur:nul].decode("ascii", "replace")
        cur = nul + 1
        if not entry_type:
            break
        elen = struct.unpack(">I", inner[cur:cur + 4])[0]
        cur += 4
        body = inner[cur:cur + elen]
        cur += elen
        if entry_type != "FLIP":
            continue
        # skip header: 1 NUL + 1 slot + cstring name + 1 enabled
        # + 1 looped + 4 action_count
        bcur = 0
        if body[bcur] == 0:
            bcur += 1
        bcur += 1  # slot
        n = body.find(b"\x00", bcur)
        bcur = n + 1
        bcur += 2  # enabled + looped
        action_count = struct.unpack(
            ">I", body[bcur:bcur + 4])[0]
        bcur += 4
        for _ in range(action_count):
            a_id = body[bcur]; bcur += 1
            a_len = struct.unpack(
                ">I", body[bcur:bcur + 4])[0]
            bcur += 4
            payload2 = body[bcur:bcur + a_len]
            bcur += a_len
            if a_id == 0 and a_len >= 16:
                jumps.append(struct.unpack(">dd", payload2[:16]))
    return jumps


# Audio files to verify are supplied as CLI arguments.  Run:
#     python -m sidecaramel.verify <track1> [<track2> ...]
# Each path must point to a Serato-analyzed audio file (MP3 / WAV /
# AIFF / MP4 / M4A / FLAC / OGG).

OK = "✓"
BAD = "✗"


def main() -> int:
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage: python -m sidecaramel.verify <audio> "
                "[<audio> ...]", file=_sys.stderr)
        print("Each path must be a Serato-analyzed audio file.",
                file=_sys.stderr)
        return 2
    print("Serato decoder verification")
    print("=" * 60)
    fails = 0
    for path in _sys.argv[1:]:
        label = os.path.basename(path)
        print()
        print(label)
        print("-" * len(label))
        if not os.path.isfile(path):
            print(f"  {BAD} FILE NOT FOUND")
            fails += 1
            continue
        f = MFile(path)
        dur = f.info.length
        bg = get_blob(path, "Serato BeatGrid")
        at = get_blob(path, "Serato Autotags")
        m2 = get_blob(path, "Serato Markers2")
        if not bg or not at:
            print(f"  {BAD} missing required blob")
            fails += 1
            continue
        nm, t_pos, t_bpm = beatgrid_terminal(bg)
        at_bpm = autotags_bpm(at)
        print(f"  duration:        {dur:.2f} s")
        print(f"  beatgrid markers: {nm}  "
              f"(non-terminal: {nm - 1})")
        print(f"  terminal position: {t_pos:.3f} s  "
              + (OK if t_pos <= dur else BAD)
              + f"  (must be <= duration)")
        if t_pos > dur:
            fails += 1
        print(f"  autotags BPM:    {at_bpm:.2f}")
        print(f"  beatgrid BPM:    {t_bpm:.3f}")
        print(f"  agreement:       "
              + (OK if abs(at_bpm - t_bpm) < 2.0 else
                 "Δ %.2f (display rounding or live tempo)"
                 % abs(at_bpm - t_bpm)))
        if m2:
            cues = cue_positions_ms(m2)
            print(f"  cue points:      {len(cues)}")
            in_range = [c for c in cues if c <= dur * 1000]
            if cues:
                bad = [c for c in cues if c > dur * 1000]
                print(f"  cues in track:   "
                      f"{len(in_range)}/{len(cues)}  "
                      + (OK if not bad else BAD))
                if bad:
                    fails += 1
            jumps = flip_jumps(m2)
            if jumps:
                bad = [j for j in jumps
                        if j[0] > dur or j[1] > dur
                        or j[0] < 0 or j[1] < 0]
                print(f"  FLIP jumps:      "
                      f"{len(jumps)}, all in [0,{dur:.0f}]: "
                      + (OK if not bad else
                         f"{BAD} ({len(bad)} OOB)"))
                if bad:
                    fails += 1
    print()
    print("=" * 60)
    print(f"Verification: {'PASS' if fails == 0 else f'{fails} fails'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
