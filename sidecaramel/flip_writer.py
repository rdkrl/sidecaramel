"""sidecaramel.flip_writer — splice a censor FLIP into an MP3's Serato
Markers2 blob.

Preserves COLOR / BPMLOCK / TRACKCOLOR / existing CUE / LOOP entries
that already live in Markers2; injects (or replaces) a single FLIP
entry containing user-supplied CENSOR or JUMP actions.

Verified Markers2 layout:

  OUTER (lives in ID3 GEOB 'Serato Markers2'):
    u8 1, u8 1                       major.minor (1.1)
    base64( inner )                  the Markers2 inner stream
    NUL padding                      to original GEOB size

  INNER:
    u8 1, u8 1                       inner version
    repeated entries:
        cstring  entry_type
        u32 BE   entry_length
        <entry_length bytes>         entry body
    final NUL                        end-of-stream

  FLIP entry body (verified against user's hand-crafted Serato FLIP):
    u8 NUL prefix
    u8 slot_index                    0..5
    u8 flag                          0 or 1 (= "slot occupied" / writer
                                      version marker; emit 1)
    cstring name                     UTF-8, may be empty
    u8 loop                          1 = loop the FLIP on playback;
                                      0 = one-shot (default)
    u32 BE action_count
    action_count × action

  CENSOR action (id=1, payload_size=24):
    u8 1, u32 BE 24,
    f64 BE trigger_time_s            the LATER timestamp (= where the
                                      DJ's finger would hit censor)
    f64 BE reverse_target_s          the EARLIER timestamp (= where the
                                      reverse playback lands)
    f64 BE speed                     always -1.0 (reverse at 1x)

  JUMP action (id=0, payload_size=16):
    u8 0, u32 BE 16,
    f64 BE from_s, f64 BE to_s

  To cleanly bleep a word at [start_s, end_s], emit a CENSOR followed
  by a JUMP:
    CENSOR (mid_s, start_s, -1.0)    reverse mid → start
    JUMP   (start_s, end_s)          then skip past the rest of the word
  See `build_word_bleep()` below.

Usage (CLI):
    python -m sidecaramel.flip_writer \\
        --in  /path/to/track.mp3 \\
        --out /path/to/track_censored.mp3 \\
        --slot 0 --name CLEAN \\
        --censor 31.50:32.10  \\
        --censor 55.20:55.80  \\
        ...
"""
from __future__ import annotations

import argparse
import base64
import os
import shutil
import struct
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ============================================================
# Inner Markers2 parser/serialiser
# ============================================================

@dataclass
class M2Entry:
    type: str          # "COLOR", "CUE", "FLIP", etc.
    body: bytes


def parse_inner(inner: bytes) -> List[M2Entry]:
    """Parse the inner Markers2 stream into a list of entries."""
    out: List[M2Entry] = []
    cur = 0
    if len(inner) < 2:
        return out
    # version u8 u8 then entries
    cur = 2
    while cur < len(inner):
        nul = inner.find(b"\x00", cur)
        if nul < 0:
            break
        typ = inner[cur:nul].decode("ascii", "replace")
        cur = nul + 1
        if not typ:
            break
        if cur + 4 > len(inner):
            break
        elen = struct.unpack(">I", inner[cur:cur + 4])[0]
        cur += 4
        body = inner[cur:cur + elen]
        cur += elen
        out.append(M2Entry(type=typ, body=body))
    return out


def serialize_inner(entries: List[M2Entry]) -> bytes:
    """Serialise entries back to the inner Markers2 stream.

    Layout: u8 1, u8 1, then each entry as
        cstring type, u32 BE length, body
    then a trailing NUL byte to mark end-of-stream."""
    out = bytearray(b"\x01\x01")
    for e in entries:
        out += e.type.encode("ascii") + b"\x00"
        out += struct.pack(">I", len(e.body))
        out += e.body
    out += b"\x00"   # end-of-stream sentinel
    return bytes(out)


def merge_markers2_inner(existing_inner: bytes,
                          *,
                          cues: Optional[List[dict]] = None,
                          loops: Optional[List[dict]] = None,
                          color_rgb=None,
                          bpm_lock: Optional[bool] = None,
                          flips: Optional[List[dict]] = None,
                          ) -> bytes:
    """Preservation-oriented CUE / LOOP / COLOR / BPMLOCK / FLIP splice.

    The cue/loop sibling of `write_flip_any`'s byte-exact preservation:
    parse the EXISTING inner Markers2 stream into a raw `M2Entry` list,
    replace ONLY the entry types the caller explicitly supplies, and
    re-serialise everything else — AUTOGAIN, unknown / future entries,
    and (unless overridden) BPMLOCK and FLIPs — VERBATIM.

    Override semantics — ``None`` means "leave whatever exists untouched":

        cues / loops / flips  not None  → replace ALL entries of that type
        color_rgb             not None  → replace the COLOR entry
        bpm_lock              not None  → replace the BPMLOCK entry

    An explicit empty list (e.g. ``cues=[]``) clears that type; ``None``
    keeps the existing entries.  This is what makes a cue write
    non-destructive: a locked grid (BPMLOCK), AUTOGAIN, and any
    unknown/future Serato entry survive a cue/loop rewrite.

    Returns the new inner stream — feed to ``tags.pack_serato_markers2``.
    """
    from sidecaramel.tags import build_markers2_inner
    existing = parse_inner(existing_inner)
    override = set()
    if cues is not None:
        override.add("CUE")
    if loops is not None:
        override.add("LOOP")
    if color_rgb is not None:
        override.add("COLOR")
    if bpm_lock is not None:
        override.add("BPMLOCK")
    if flips is not None:
        override.add("FLIP")
    # Encode the replacement entries through the canonical builder, then
    # peel them back to raw M2Entry so they rejoin the preserved stream
    # via the same serialiser (no duplicated framing logic).
    fresh = [e for e in parse_inner(build_markers2_inner(
                cues=cues, loops=loops, color_rgb=color_rgb,
                bpm_lock=bpm_lock, flips=flips))
             if e.type in override]
    # Serato's real entry order (verified against genuine blobs):
    # COLOR, CUE, LOOP, BPMLOCK, FLIP.
    canon = ("COLOR", "CUE", "LOOP", "BPMLOCK", "FLIP")
    out: List[M2Entry] = []
    for t in canon:
        src = fresh if t in override else existing
        out += [e for e in src if e.type == t]
    # Preserve every non-canonical existing entry (AUTOGAIN, unknown
    # FUTUREX, …) verbatim, after the canonical block.
    out += [e for e in existing if e.type not in canon]
    return serialize_inner(out)


# ============================================================
# Outer Markers2 wrapper
# ============================================================

def decode_outer(blob: bytes) -> Tuple[bytes, int]:
    """Return (inner_bytes, original_blob_size).

    Outer format: u8 1, u8 1, base64(inner), NUL padding.
    Some Serato writes have a trailing extra base64 char that breaks
    `1 mod 4`; we handle that by trying trim {0,1,2,3}.
    """
    payload = blob[2:]
    while payload and payload[0] == 0:
        payload = payload[1:]
    end = payload.find(b"\x00")
    if end < 0:
        end = len(payload)
    b64 = (payload[:end]
            .replace(b"\n", b"")
            .replace(b"\r", b""))
    for trim in (0, 1, 2, 3):
        chunk = b64[:len(b64) - trim] if trim else b64
        if len(chunk) % 4 == 1:
            continue
        pad = (-len(chunk)) % 4
        try:
            return (base64.b64decode(chunk + b"=" * pad,
                                       validate=False),
                    len(blob))
        except Exception:
            continue
    raise ValueError("could not decode Markers2 outer base64")


def encode_outer(inner: bytes, target_size: int = 0,
                  line_width: int = 72) -> bytes:
    """Pack `inner` into the Markers2 GEOB blob format.

    Single point of truth: the body is delegated to
    `sidecaramel.tags.pack_serato_markers2` so the line_width / NUL-pad
    behaviour lives in exactly ONE place.  This function is kept as an
    alias for callers that reach for it here.
    """
    from sidecaramel.tags import pack_serato_markers2
    return pack_serato_markers2(inner, target_size=target_size,
                                  line_width=line_width)


# ============================================================
# FLIP entry builders
# ============================================================

def build_jump(from_s: float, to_s: float) -> bytes:
    """JUMP action bytes: id=0, len=16, two f64 BE doubles."""
    return (b"\x00"
             + struct.pack(">I", 16)
             + struct.pack(">dd", from_s, to_s))


def build_censor(trigger_s: float, reverse_target_s: float,
                  speed: float = -1.0) -> bytes:
    """CENSOR action bytes: id=1, len=24, three f64 BE doubles.

    IMPORTANT field order (verified against a Serato-saved manual
    FLIP): `trigger_s` is the LATER timestamp (where the censor
    finger fires), `reverse_target_s` is the EARLIER timestamp (where
    the reverse playback lands). Speed is always -1.0.
    """
    return (b"\x01"
             + struct.pack(">I", 24)
             + struct.pack(">ddd", trigger_s, reverse_target_s,
                            speed))


def build_word_bleep(word_start_s: float, word_end_s: float,
                       reverse_fraction: float = 1.0,
                       head_pad_s: float = 0.02) -> List[bytes]:
    """Return [CENSOR, JUMP] action bytes that cleanly bleep an
    audio span [word_start_s, word_end_s] in Serato.

     (rev 2): "die müssen mit wortbeginn triggern
    und das wort lang halten" — censor MUST fire at word START
    (not word-mid) and play in reverse for the FULL word duration
    so the listener never hears the cuss-word forwards.

    Mechanic (Serato CENSOR + JUMP convention):
      1. Forward playback proceeds normally up to `trigger`.
      2. At `trigger`, CENSOR fires → audio plays in reverse from
         `trigger` back to `reverse_target` (= pre-word audio in
         our case, taking `word_duration` seconds × |speed|=1.0).
         The listener hears the audio LEADING UP TO the cuss-word,
         played backwards.  The word itself never goes through the
         output forward.
      3. Once playback reaches `reverse_target`, JUMP fires →
         playhead skips forward to `word_end_s`, past the cuss-word.

    `reverse_fraction` (default 1.0 = full word) controls HOW LONG
    the reverse-effect lasts as a multiple of the word duration:
      • 1.0 → reverse for `word_duration` seconds (default)
      • 0.5 → reverse for half a word duration (older behaviour;
              produced an audible first-half of the cuss-word, BAD)

    `head_pad_s` (default 20ms) shifts the trigger slightly AFTER
    word_start to avoid clicking into the preceding word's tail
    (Serato's manual FLIPs do this with ~20 ms of pad).
    """
    duration = max(word_end_s - word_start_s, 0.05)
    # Trigger AT the word start (with tiny head pad to avoid the
    # preceding word's tail click).
    trigger = word_start_s + head_pad_s
    # Reverse target = `reverse_fraction × duration` seconds BEFORE
    # the trigger.  With default 1.0, reverse plays for the full
    # word's worth of audio (using pre-word content backwards).
    reverse_target = trigger - duration * reverse_fraction
    return [
        build_censor(trigger, reverse_target, -1.0),
        build_jump(reverse_target, word_end_s),
    ]


def build_flip_entry(*, slot: int,
                       name: str,
                       loop: bool,
                       actions: List[bytes],
                       flag: int = 1) -> M2Entry:
    """Build a FLIP M2Entry.

    `loop`: True = repeat the FLIP after the last action; False =
            play once and stop (default user choice).
    `flag`: writer-version / slot-occupied byte (we emit 1, matching
            every observed named FLIP in the user's library except
            Eddie Johns' older-format outlier).
    `actions`: list of raw action byte chunks (use `build_jump`,
               `build_censor`, or `build_word_bleep`).
    """
    body = bytearray()
    body += b"\x00"                        # NUL prefix
    body += bytes([slot & 0xFF])           # slot
    body += bytes([flag & 0xFF])           # flag
    body += name.encode("utf-8") + b"\x00"  # cstring name
    body += b"\x01" if loop else b"\x00"    # loop flag
    body += struct.pack(">I", len(actions))  # action_count
    for a in actions:
        body += a
    return M2Entry(type="FLIP", body=bytes(body))


# ============================================================
# CLI / main
# ============================================================

def _parse_window(spec: str) -> Tuple[float, float]:
    a, b = spec.split(":")
    return (float(a), float(b))


def write_flip_any(audio_path: str,
                       *, slot: int, name: str, loop: bool,
                       bleeps: List[Tuple[float, float]],
                       reverse_fraction: float = 0.5,
                       head_pad_s: float = 0.02,
                       replace_slot: bool = True,
                       confirm: bool = False) -> bool:
    """Container-agnostic single-channel FLIP writer for ALL
    supported audio types (MP3, WAV, AIFF, MP4/M4A/M4V, FLAC/OGG).

    "ja, alle types, ein kanal".

    Pipeline (decoupling policy — re-uses existing infrastructure end-to-end):
        1. Read existing Markers2 blob via harvest_serato_blobs
        2. decode_outer → inner bytes (Pro 1-mod-4-quirk handled)
        3. parse_inner → list of M2Entry (RAW bytes per entry, byte-
           exact preservation of cues / loops / color / bpm_lock /
           other FLIPs)
        4. build_flip_entry(...) for the new FLIP body
        5. Splice (replace_slot=True) or append the new FLIP into the
           entry list
        6. serialize_inner → new inner stream
        7. write_serato_markers2_any(path, inner, confirm=True) →
           container-aware dispatch (ID3 GEOB / MP4 atom / FLAC Vorbis)

    `bleeps`: list of (word_start_s, word_end_s) windows.  Each is
              converted to a CENSOR+JUMP pair via build_word_bleep().
    `confirm`: audio-write confirm-flag — must be True.

    Returns True on success, False if no Markers2 blob existed AND
    we couldn't seed an empty one (rare; only happens on tracks
    Serato never analyzed).
    """
    if not confirm:
        raise RuntimeError(
            "write_flip_any requires confirm=True (audio-write policy); "
            f"path={audio_path!r}")
    from sidecaramel.tags import (harvest_serato_blobs,
                                       write_serato_markers2_any)

    # 1. Read existing Markers2 blob (if any)
    existing_inner = None
    for desc, payload, _ in harvest_serato_blobs(audio_path):
        if desc == "Serato Markers2":
            try:
                existing_inner, _ = decode_outer(payload)
            except Exception:
                existing_inner = None
            break

    # 2. Parse to entries (or seed an empty header if no blob existed)
    if existing_inner is None:
        existing_inner = b"\x01\x01"
    entries = parse_inner(existing_inner)

    # 3. Build the new FLIP entry's body bytes
    bleep_actions: List[bytes] = []
    for (a, b) in bleeps:
        bleep_actions.extend(build_word_bleep(
            float(a), float(b),
            reverse_fraction=reverse_fraction,
            head_pad_s=head_pad_s))
    new_flip_body = build_flip_entry(
        slot=int(slot),
        name=str(name or ""),
        loop=bool(loop),
        actions=bleep_actions,
        flag=1,
    ).body
    new_entry = M2Entry(type="FLIP", body=new_flip_body)

    # 4. Splice into entry list — replace_slot=True overwrites any
    # existing FLIP with the same slot index; False appends as a
    # new FLIP entry.  Slot index lives at body[1] per the format docs.
    if replace_slot:
        out_entries: List[M2Entry] = []
        replaced = False
        for e in entries:
            if e.type == "FLIP" and len(e.body) >= 2 and e.body[1] == slot:
                out_entries.append(new_entry)
                replaced = True
            else:
                out_entries.append(e)
        if not replaced:
            out_entries.append(new_entry)
        entries = out_entries
    else:
        entries.append(new_entry)

    # 5. Serialize back to inner stream
    new_inner = serialize_inner(entries)

    # 6. Container-aware write — write_serato_markers2_any wants the
    # INNER (not outer-packed) and dispatches per file extension.
    return write_serato_markers2_any(audio_path, new_inner,
                                          confirm=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Inject a censor FLIP into the Serato Markers2 "
                    "blob of any container (MP3, WAV, AIFF, MP4/M4A, "
                    "FLAC, OGG).  Each --bleep window becomes a "
                    "CENSOR+JUMP pair.")
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True,
                     help="Output path; if different from --in we "
                          "shutil.copy2 first and write the FLIP into "
                          "the copy (audio-write policy safe).")
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--name", default="")
    ap.add_argument("--loop", action="store_true",
                     help="Set the FLIP's loop flag (default: off).")
    ap.add_argument("--bleep", action="append", default=[],
                     metavar="WORD_START:WORD_END",
                     help="Bleep a word span (seconds). "
                          "Repeatable. Each becomes CENSOR+JUMP.")
    ap.add_argument("--reverse-fraction", type=float, default=0.5,
                     help="What fraction of the word to reverse "
                          "(default 0.5; the rest is JUMP'd over).")
    ap.add_argument("--head-pad", type=float, default=0.02,
                     help="Pad in seconds to push the reverse target "
                          "after word_start (default 0.02).")
    ap.add_argument("--append", action="store_true",
                     help="Append as a new FLIP even if slot is busy "
                          "(default: replace slot if it exists)")
    args = ap.parse_args()

    bleeps = [_parse_window(s) for s in args.bleep]
    if not bleeps:
        print("error: no --bleep windows given")
        return 2

    # Copy src → dst first (audio-write policy: leave src untouched if a
    # separate dst is requested).
    if os.path.abspath(args.src) != os.path.abspath(args.dst):
        shutil.copy2(args.src, args.dst)
    write_flip_any(
        args.dst,
        slot=args.slot, name=args.name, loop=args.loop,
        bleeps=bleeps,
        reverse_fraction=args.reverse_fraction,
        head_pad_s=args.head_pad,
        replace_slot=not args.append,
        # CLI invocation IS user confirmation; library callers must
        # opt in via the kwarg.
        confirm=True)
    print(f"OK: wrote FLIP to {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
