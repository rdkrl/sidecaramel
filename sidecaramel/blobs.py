"""sidecaramel.blobs — decode every Serato blob mutagen can see on a track.

Reads an audio file (MP3, M4A/MP4, FLAC, AIFF, WAV/OGG) with mutagen,
finds every Serato-owned tag (GEOB frame, MP4 ----:com.serato.dj atom,
Vorbis SERATO_* base64 comment, AIFF GEOB-in-ID3 chunk) and dumps each
one with both a hex preview and a structured decode.

Supported blob decoders:

  • Serato Analysis     - version stamp ("2.1")
  • Serato Autotags     - version + bpm/autogain/gainDb (ASCII, NUL sep)
  • Serato BeatGrid     - version + non-terminal beat markers + final
                          BPM marker + 1-byte footer
  • Serato Markers_     - 14 cue/loop slots, color-encoded
                          (MP3/AIFF format with base64+0x80 mask;
                           MP4 format unwrapped)
  • Serato Markers2     - base64 wrapper around a stream of
                          tagged entries:
                            COLOR, BPMLOCK, CUE, LOOP, FLIP, TRACKCOLOR
  • Serato Overview     - 16-row waveform thumbnail, 1 byte per cell
  • Serato Offsets_     - tail of int64 sample offsets (VBR alignment)
  • Serato RelVolAd     - relative-volume / EQ adjust block (rare)
  • Serato VideoAssoc   - reference to an associated video file
  • Serato FLIP         - encoded as a "FLIP" entry inside Markers2

For each container we also note where the blob *lives*:
  MP3/AIFF: GEOB:"<name>"               (ID3 v2.3/2.4)
  MP4/M4A:  ----:com.serato.dj:<name>   (freeform atom)
  FLAC/OGG: SERATO_<NAME>=<base64...>   (Vorbis comment;
                                          name uppercased)

Usage:
    python -m sidecaramel.blobs /path/to/track.mp3
    python -m sidecaramel.blobs file1.mp3 file2.m4a ...

Researches Serato's on-track metadata
so we can preserve / round-trip it from consumer workflows.
"""
from __future__ import annotations

import base64
import binascii
import os
import struct
import sys
from typing import Any, Dict, List, Tuple


# ============================================================
# Tag-name canonicalisation across containers
# ============================================================
# These are the eleven Serato GEOB descriptors we expect to see.
SERATO_GEOB_NAMES = (
    "Serato Analysis",
    "Serato Autotags",
    "Serato BeatGrid",
    "Serato Markers_",
    "Serato Markers2",
    "Serato Offsets_",
    "Serato Overview",
    "Serato RelVolAd",
    "Serato VideoAssoc",
    # Newer / rarer:
    "Serato Playcount",
    "Serato StemSet",
)

# Vorbis-comment key for each ID3 GEOB descriptor. Lowercase of the
# value after "Serato ", prefixed with "SERATO_".  e.g.
#   "Serato BeatGrid"  -> "SERATO_BEATGRID"
def _vorbis_key(geob_name: str) -> str:
    short = geob_name.replace("Serato ", "")
    return "SERATO_" + short.replace("_", "").upper()


# ============================================================
# Pretty-printer helpers
# ============================================================

def _hex_preview(b: bytes, n: int = 48) -> str:
    head = b[:n]
    asc = "".join(chr(x) if 32 <= x < 127 else "." for x in head)
    return f"{head.hex(' ')}  |{asc}|"


def _print(title: str) -> None:
    print()
    print("─" * 72)
    print(title)
    print("─" * 72)


# ============================================================
# Tag harvesting
# ============================================================

def harvest(path: str) -> List[Tuple[str, bytes, str]]:
    """Return list of (name, raw_bytes, source_label) for every Serato
    blob on the file. `name` is canonical (e.g. "Serato BeatGrid")."""
    from mutagen import File as _MFile  # lazy

    f = _MFile(path)
    if f is None:
        return []
    blobs: List[Tuple[str, bytes, str]] = []
    if f.tags is None:
        return blobs

    # ---- MP3 / AIFF (ID3) ----
    # ID3.tags is a dict-like keyed by frame id, "GEOB:<desc>" for our
    # multi-instance binary frames. Each value is a GEOB() with .data.
    try:
        from mutagen.id3 import ID3
        if isinstance(f.tags, ID3):
            for key in f.tags.keys():
                if not str(key).startswith("GEOB:"):
                    continue
                fr = f.tags[key]
                desc = getattr(fr, "desc", "?")
                if not desc.startswith("Serato"):
                    continue
                blobs.append((desc, bytes(fr.data),
                              f"ID3 GEOB '{desc}' "
                              f"mime={fr.mime!r} fname={fr.filename!r}"))
            return blobs
    except Exception:
        pass

    # ---- MP4 / M4A ----
    try:
        from mutagen.mp4 import MP4
        if isinstance(f.tags, MP4) or hasattr(f.tags, "items"):
            mp4_found = False
            for k, v in f.tags.items():
                ks = str(k)
                if "com.serato.dj" not in ks:
                    continue
                mp4_found = True
                # k is like "----:com.serato.dj:beatgrid"
                # mp4 atom values are list[bytes].  The bytes are
                # ALWAYS the ASCII base64 encoding of an envelope:
                #   "application/octet-stream\0\0Serato <Name>\0<binary>"
                # … so the canonical descriptor lives INSIDE the
                # decoded envelope, not in the atom name.
                name_lc = ks.split(":")[-1].lower()
                fallback = {
                    "analysis":     "Serato Analysis",
                    "analysisflags":"Serato AnalysisFlags",
                    "autgain":      "Serato Autotags",
                    "autotags":     "Serato Autotags",
                    "beatgrid":     "Serato BeatGrid",
                    "markersv2":    "Serato Markers2",
                    "markers":      "Serato Markers_",
                    "markers_":     "Serato Markers_",
                    "offsets":      "Serato Offsets_",
                    "overview":     "Serato Overview",
                    "relvoladj":    "Serato RelVolAd",
                    "relvolad":     "Serato RelVolAd",
                    "videoassoc":   "Serato VideoAssoc",
                }.get(name_lc, f"Serato {name_lc}")
                items = v if isinstance(v, list) else [v]
                for item in items:
                    raw_envelope = bytes(item)
                    canon, payload = _unwrap_mp4_envelope(
                        raw_envelope, fallback)
                    blobs.append((canon, payload,
                                  f"MP4 atom {ks!r}  "
                                  f"envelope={len(raw_envelope)}B  "
                                  f"payload={len(payload)}B"))
            if mp4_found:
                return blobs
    except Exception:
        pass

    # ---- FLAC / OGG (Vorbis comments) ----
    try:
        # Vorbis-comment keys are case-insensitive; values are str.
        items = []
        if hasattr(f.tags, "items"):
            items = list(f.tags.items())
        for k, v in items:
            ks = str(k).upper()
            if not ks.startswith("SERATO_"):
                continue
            short = ks[len("SERATO_"):]
            canon_map = {
                "ANALYSIS":    "Serato Analysis",
                "AUTOGAIN":    "Serato Autotags",
                "AUTOTAGS":    "Serato Autotags",
                "BEATGRID":    "Serato BeatGrid",
                "MARKERS":     "Serato Markers_",
                "MARKERS2":    "Serato Markers2",
                "MARKERSV2":   "Serato Markers2",
                "MARKERS_V2":  "Serato Markers2",  # 
                "OFFSETS":     "Serato Offsets_",
                "OVERVIEW":    "Serato Overview",
                "PLAYCOUNT":   "Serato Playcount",
                "RELVOL":      "Serato RelVolAd",  # 
                "RELVOLAD":    "Serato RelVolAd",
                "VIDEOASSOC":  "Serato VidAssoc",
                "VIDEO_ASSOC": "Serato VidAssoc",  # 
                "VIDASSOC":    "Serato VidAssoc",
            }
            canon = canon_map.get(short, f"Serato {short.title()}")
            # FLAC stores value as ASCII; sometimes a list.
            if isinstance(v, list):
                for item in v:
                    raw = _decode_vorbis_blob(item)
                    blobs.append((canon, raw,
                                  f"FLAC Vorbis {ks!r}"))
            else:
                raw = _decode_vorbis_blob(v)
                blobs.append((canon, raw, f"FLAC Vorbis {ks!r}"))
        return blobs
    except Exception:
        pass

    return blobs


_B64_ALPHABET_RE_SI: Any = None  # lazy compiled regex; see _b64_alphabet_clean


def _b64_alphabet_clean(b64_bytes: bytes) -> bytes:
    """Strip any byte not in the standard / URL-safe base64 alphabet.

    Serato Pro sometimes emits stray newlines / spaces inside the
    base64 text of an MP4 atom; this drops them so the trim-fallback
    decoder downstream sees a clean stream.
    """
    global _B64_ALPHABET_RE_SI
    if _B64_ALPHABET_RE_SI is None:
        import re as _re
        _B64_ALPHABET_RE_SI = _re.compile(rb"[^A-Za-z0-9+/=_-]")
    return _B64_ALPHABET_RE_SI.sub(b"", b64_bytes)


def _robust_b64_decode(b64_bytes: bytes,
                         prefer_prefix: bytes = b"",
                         fallback_any: bool = True
                         ) -> bytes:
    """Decode base64 bytes tolerantly.  SPOT for Serato MP4-atom
    "length ≡ 1 mod 4" writer-quirk handling — also used by
    `sidecaramel.tags._robust_b64_decode` via re-export.

    Strategy: try trim 0, 1, 2, 3 trailing chars in order; for
    each, pad to a multiple of 4 with `=` and call
    `base64.b64decode(validate=False)`.

    Selection logic:
      • `prefer_prefix` empty (default) → return the FIRST
        non-empty successful decode.
      • `prefer_prefix` non-empty → return the FIRST decode
        whose bytes start with that prefix.  If none match AND
        `fallback_any` is True, return the LONGEST successful
        decode that doesn't match the prefix.
      • All decodes throw → return b"".

    Skips trim variants that land at length ≡ 1 mod 4 (would
    need 3 `=` chars which base64 rejects).

    Lives here rather than in the tags module so that `harvest()`
    can use it too — the MP4 Markers2 path needs the same tolerant
    decode.
    """
    if not b64_bytes:
        return b""
    preferred_match = b""
    fallback = b""
    for _trim in (0, 1, 2, 3):
        chunk = (b64_bytes[:len(b64_bytes) - _trim]
                  if _trim else b64_bytes)
        if not chunk:
            continue
        pad = (-len(chunk)) % 4
        if pad == 3:
            continue
        try:
            dec = base64.b64decode(
                chunk + b"=" * pad, validate=False)
        except Exception:
            continue
        if not dec:
            continue
        if not prefer_prefix:
            return dec
        if dec.startswith(prefer_prefix):
            preferred_match = dec
            break
        if fallback_any and len(dec) > len(fallback):
            fallback = dec
    if preferred_match:
        return preferred_match
    return fallback if fallback_any else b""


def _unwrap_mp4_envelope(
        raw: bytes, fallback_name: str) -> Tuple[str, bytes]:
    """MP4 Serato atom value is ASCII base64 of:
         "application/octet-stream\\x00\\x00Serato <Name>\\x00<binary>"

    Uses `_robust_b64_decode` to handle Serato Pro's
    "length ≡ 1 mod 4" writer-quirk (trim-{0,1,2,3} fallback).
    Returns (canonical_name, binary_payload).

    previously used naive
    `base64.b64decode(raw + b"==", validate=False)` which silently
    failed on quirk-files (e.g. a sample MP4 file) and returned
    the raw b64-text — downstream consumers like
    `sidecaramel.tags._markers2_blob_for_path`, `read_loops`,
    `read_track_color`, `read_bpm_lock` then saw garbage.
    """
    cleaned = _b64_alphabet_clean(raw)
    # First prefer a decode that starts with the envelope marker.
    decoded = _robust_b64_decode(
        cleaned,
        prefer_prefix=b"application/octet-stream",
        fallback_any=True)
    if not decoded:
        # Decode failed entirely — fall back to raw atom bytes so
        # callers that handle their own format (e.g. analysisflags
        # which is sometimes raw, not base64) still get a chance.
        return (fallback_name, raw)
    if not decoded.startswith(b"application/octet-stream"):
        # Some short atoms (analysisflags) are JUST the version
        # bytes base64-encoded with no MIME envelope.  Return the
        # longest decode the robust decoder produced.
        return (fallback_name, decoded)
    # Strip the MIME header and find the Serato descriptor.
    # "application/octet-stream" is 24 chars + two NULs = 26
    rest = decoded[24:]
    # Sometimes 2 NUL bytes, sometimes more
    while rest and rest[0] == 0:
        rest = rest[1:]
    # Now we should have "Serato <Name>\0<binary>"
    nul = rest.find(b"\x00")
    if nul < 0:
        return (fallback_name, rest)
    desc = rest[:nul].decode("ascii", errors="replace")
    payload = rest[nul + 1:]
    return (desc or fallback_name, payload)


def _decode_vorbis_blob(value: str) -> bytes:
    """Decode a Serato Vorbis-comment value into the raw blob bytes.

    Real Serato files store the value as base64-text with
    newline-wrap every 72 chars.  Decoded, it contains the envelope:

        application/octet-stream\\x00\\x00Serato <Name>\\x00<blob>

    We base64-decode (handling Serato Pro's "length ≡ 1 mod 4"
    writer-quirk via `_robust_b64_decode`), then strip the envelope
    so the returned bytes are JUST the raw blob — parallel to
    `_unwrap_mp4_envelope`.  Callers downstream (read_serato_metadata,
    parse_serato_markers2_full, etc.) get the same payload shape
    regardless of source container.
    """
    if isinstance(value, bytes):
        value = value.decode("latin-1", errors="replace")
    # Strip newlines/carriage-returns (Serato writes line-wrapped b64)
    clean = value.replace("\n", "").replace("\r", "").encode("ascii",
                                                                 "replace")
    # Use the robust decoder to handle Serato Pro's 1-mod-4 quirk.
    decoded = _robust_b64_decode(
        clean,
        prefer_prefix=b"application/octet-stream",
        fallback_any=True)
    if not decoded:
        return b""
    # Unwrap the envelope if present.  Some old/short blobs (e.g.
    # SERATO_PLAYCOUNT, which is actually plain text "11", not base64)
    # won't have the envelope; return the raw decode in that case.
    if decoded.startswith(b"application/octet-stream"):
        rest = decoded[24:]
        while rest and rest[0] == 0:
            rest = rest[1:]
        nul = rest.find(b"\x00")
        if nul >= 0:
            # Skip the "Serato <Name>" descriptor + NUL
            return rest[nul + 1:]
        return rest
    return decoded


def _robust_b64_decode_v_compat():
    # Module-init time helper — `_robust_b64_decode` is defined later
    # in this module, so we ensure the forward-reference resolves at
    # function-call time, not import time.  Python resolves it
    # naturally at call site — this stub is just a marker that the
    # ordering matters.
    pass


# ============================================================
# Decoders
# ============================================================
# Reference notes inline.  Authoritative-ish sources:
#   github.com/Holzhaus/serato-tags  (community-reverse-engineered)
#   github.com/jbrunis/serato-tools
# Anything marked "OBSERVED" below was read off real Serato-
# written files here, not taken from those sources.


def decode_analysis(b: bytes) -> Dict[str, Any]:
    """ANALYSIS is just `<MAJOR><MINOR>` two bytes — the analyser
    version that touched the file. Common values: 02 01 ("v2.1")."""
    if len(b) < 2:
        return {"raw": b.hex(), "note": "too short"}
    return {
        "version": f"{b[0]}.{b[1]}",
        "extra_bytes": b[2:].hex(" ") if len(b) > 2 else "(none)",
    }


def decode_autotags(b: bytes) -> Dict[str, Any]:
    """AUTOTAGS = `<MAJOR><MINOR>\\x00<bpm>\\x00<autoGain>\\x00<gainDb>\\x00`
    All three values are ASCII strings of floats.  e.g.
       b"\\x01\\x01\\x00128.00\\x000.000\\x00-3.157\\x00"."""
    if len(b) < 3:
        return {"raw": b.hex()}
    major, minor = b[0], b[1]
    # Skip the version-trailing NUL
    body = b[3:] if b[2] == 0 else b[2:]
    parts = body.split(b"\x00")
    fields = [p.decode("ascii", errors="replace") for p in parts if p]
    out = {"version": f"{major}.{minor}", "fields": fields}
    if len(fields) >= 3:
        try:
            out["bpm"] = float(fields[0])
            out["autoGain"] = float(fields[1])
            out["gainDb"] = float(fields[2])
        except ValueError:
            pass
    return out


def decode_beatgrid(b: bytes) -> Dict[str, Any]:
    """BEATGRID layout (verified against tracks ):
        u8  major, u8 minor                       (1, 0)
        u32 BE   num_markers           (INCLUDES the terminal marker)
        repeated (num_markers - 1) times:           # non-terminal
            f32 BE  position_seconds
            u32 BE  beats_till_next_marker
        if num_markers >= 1, terminal marker:
            f32 BE  position_seconds
            f32 BE  bpm
        u8       trailing_footer (often 0x00 or random)
    A flat-tempo track has num_markers = 1 (just the terminal)."""
    if len(b) < 6:
        return {"raw": b.hex()}
    major, minor = b[0], b[1]
    num_markers = struct.unpack(">I", b[2:6])[0]
    out: Dict[str, Any] = {
        "version": f"{major}.{minor}",
        "num_markers_total": num_markers,
        "num_non_terminal": max(num_markers - 1, 0),
    }
    off = 6
    non_term = []
    for i in range(max(num_markers - 1, 0)):
        if off + 8 > len(b):
            out["truncated_at"] = i
            break
        pos = struct.unpack(">f", b[off:off + 4])[0]
        bt = struct.unpack(">I", b[off + 4:off + 8])[0]
        # Keep the exact f32 value (no round()) so re-encoding via
        # `struct.pack(">f", position_s)` reproduces Serato's bytes
        # byte-for-byte — the golden write-roundtrip depends on it.
        non_term.append({"position_s": pos,
                          "beats_till_next": bt})
        off += 8
    out["non_terminal_markers"] = non_term
    if num_markers >= 1 and off + 8 <= len(b):
        t_pos, t_bpm = struct.unpack(">ff", b[off:off + 8])
        # Exact f32 values (unrounded) for byte-exact re-encode.
        out["terminal_marker"] = {"position_s": t_pos,
                                    "bpm": t_bpm}
        off += 8
    out["footer_byte"] = b[off:off + 1].hex() if off < len(b) else None
    out["trailing"] = b[off + 1:].hex() if off + 1 < len(b) else "(none)"
    return out


def decode_markers_v1(b: bytes, container: str) -> Dict[str, Any]:
    """MARKERS_  (the older "v1" cue-point block).

    On MP3/AIFF Serato base64-encodes the binary AND masks the
    high bit of each byte. We carry the raw bytes through for the
    container the caller knows about.

    Structure after any unmasking:
       u8 major (2), u8 minor (5)
       14 cue entries  (22 bytes each)
       14 loop entries (21 bytes each)
       1 byte color terminator

    Cue entry (22 B):
       1B    record-type   (0 if cue is empty, else 1?)
       4B BE position_ms                       (millis from start)
       1B    "field separator" 0x7F
       4B BE loop_position_ms                  (only used for loops)
       1B    0x7F
       1B    color_index (palette idx 0..7)
       1B    flags
       4B    rgb 0RRGGBB
       1B    NUL
       4B    NUL  (label terminator + 4-byte int?)
       Actually layout drifts a lot between Serato versions; this
       decoder reports the raw 22-byte slices so you can eyeball.
    """
    out: Dict[str, Any] = {
        "container": container,
        "length": len(b),
        "head_hex_raw": b[:32].hex(" "),
        "tail_hex_raw": b[-8:].hex(" "),
    }
    if len(b) < 4:
        return out
    # NB: on MP3/AIFF, each cue/loop record's u32 fields are
    # 7-bit-safe encoded (5 bytes per logical u32, with the high
    # nibble redistributed into a 5th byte). We don't attempt to
    # invert that here — Markers2 carries the same data in clean
    # form and is canonical on modern Serato. We DO decode the
    # outer envelope (version + record count + RGB bytes) which
    # survives the encoding unchanged.
    out["note"] = ("MP3/AIFF Markers_ uses Serato's 5-byte-per-u32 "
                    "7-bit-safe scheme. Markers2 carries the same "
                    "data in plain form — use it for round-tripping.")
    out["major"] = b[0]
    out["minor"] = b[1]
    # Common Markers_ outer layout (the encoded record bytes
    # themselves are NOT touched here — only counted):
    #   u8 major (2), u8 minor (5)
    #   u32 BE  num_entries          (14 on every track we've seen)
    #   num_entries × 22 bytes        encoded cue/loop record
    #   trailing bytes:               4-byte track color or footer
    if len(b) >= 6:
        out["num_entries"] = struct.unpack(">I", b[2:6])[0]
    expected_body = (out.get("num_entries") or 0) * 22
    out["expected_body_bytes"] = expected_body
    out["records_byte_count"] = max(len(b) - 6, 0)
    out["trailing_bytes"] = max(len(b) - 6 - expected_body, 0)
    # Show one sample raw record (slot 0):
    if len(b) >= 28:
        out["sample_record_slot0_raw"] = b[6:28].hex(" ")
    return out


def decode_markers2(b: bytes) -> Dict[str, Any]:
    """MARKERS2 — the modern Serato cue-point format.

    Outer wrapper:
       u8  major, u8 minor  (1, 1)
       1B  NUL
       <base64 of inner stream>
       <NULs to pad>

    The inner stream after base64-decode is itself
       u8 major, u8 minor  (1, 1)
       repeated entries:
           CSTRING  entry_type   (e.g. "COLOR", "CUE", "LOOP",
                                  "FLIP", "BPMLOCK", "TRACKCOLOR")
           u32 BE   entry_length (bytes that follow, not including
                                  this u32 nor the cstring)
           <entry_length bytes>  entry_body
       Stream ends at NUL terminator or first all-NUL run.
    """
    out: Dict[str, Any] = {
        "outer_length": len(b),
        "outer_head": b[:8].hex(" "),
    }
    if len(b) < 2:
        return out
    out["outer_version"] = f"{b[0]}.{b[1]}"
    # Find the base64 payload: skip the version + NUL, then base64 chars
    # run until NUL padding.
    # Practically: first NUL after offset 2 starts the payload.
    payload_start = 2
    while payload_start < len(b) and b[payload_start] == 0:
        payload_start += 1
    # Payload ends at first NUL
    end = b.find(b"\x00", payload_start)
    if end < 0:
        end = len(b)
    b64 = b[payload_start:end]
    try:
        # Serato pads base64 lines with newlines every 72 chars.
        # Tolerate any whitespace.
        inner = base64.b64decode(
            b64.replace(b"\n", b"").replace(b"\r", b"") + b"==",
            validate=False)
    except binascii.Error as e:
        out["b64_error"] = repr(e)
        return out
    out["inner_length"] = len(inner)
    if len(inner) < 2:
        return out
    out["inner_version"] = f"{inner[0]}.{inner[1]}"
    cursor = 2
    entries: List[Dict[str, Any]] = []
    while cursor < len(inner):
        # Read cstring
        nul = inner.find(b"\x00", cursor)
        if nul < 0:
            break
        entry_type = inner[cursor:nul].decode("ascii",
                                                errors="replace")
        cursor = nul + 1
        if not entry_type:
            # End-of-stream marker
            break
        if cursor + 4 > len(inner):
            break
        (entry_len,) = struct.unpack(">I", inner[cursor:cursor + 4])
        cursor += 4
        body = inner[cursor:cursor + entry_len]
        cursor += entry_len
        entry = {"type": entry_type, "length": entry_len,
                 "body_hex": body.hex(" ") if len(body) <= 64
                              else body[:64].hex(" ") + " ..."}
        try:
            entry["decoded"] = _decode_markers2_entry(entry_type, body)
        except Exception as e:
            entry["decode_error"] = repr(e)
        entries.append(entry)
    out["entries"] = entries
    return out


def _decode_markers2_entry(typ: str, body: bytes) -> Dict[str, Any]:
    """Per-type body decoders for Markers2 entries."""
    if typ == "COLOR":
        # Track color override: 1 NUL + 3 RGB bytes
        if len(body) >= 4:
            return {"rgb": "#{:02x}{:02x}{:02x}".format(
                    body[1], body[2], body[3])}
    if typ == "BPMLOCK":
        # Just one byte: 0 unlocked, 1 locked.
        return {"locked": bool(body[0]) if body else None}
    if typ == "TRACKCOLOR":
        if len(body) >= 4:
            return {"rgb": "#{:02x}{:02x}{:02x}".format(
                    body[1], body[2], body[3])}
    if typ == "CUE":
        # Layout (15 bytes):
        #   u8   NUL
        #   u8   index (0..7 visible cue button)
        #   u32 BE position_ms
        #   u8   NUL
        #   3B   rgb
        #   u8   NUL u8 NUL
        #   cstring name (utf-8)
        if len(body) < 13:
            return {"too_short": True}
        idx = body[1]
        pos_ms = struct.unpack(">I", body[2:6])[0]
        rgb = "#{:02x}{:02x}{:02x}".format(body[7], body[8], body[9])
        name_start = 12
        name_end = body.find(b"\x00", name_start)
        if name_end < 0:
            name_end = len(body)
        name = body[name_start:name_end].decode("utf-8",
                                                 errors="replace")
        return {"index": idx, "position_ms": pos_ms,
                "rgb": rgb, "name": name}
    if typ == "LOOP":
        # Layout — Serato Markers2 community spec (Holzhaus serato-tags /
        # Mixxx SeratoMarkers2LoopEntry):
        #   u8   NUL
        #   u8   index
        #   u32 BE start_ms
        #   u32 BE end_ms
        #   u32 BE 0xFFFFFFFF (constant)
        #   4B   color "ARGB" (00 R G B), so RGB is at 15..17
        #   u8   NUL
        #   u8   locked (bool) at 19
        #   cstring name at 20
        if len(body) < 20:
            return {"too_short": True}
        idx = body[1]
        start_ms = struct.unpack(">I", body[2:6])[0]
        end_ms = struct.unpack(">I", body[6:10])[0]
        rgb = "#{:02x}{:02x}{:02x}".format(body[15], body[16],
                                              body[17])
        locked_flag = body[19]
        name_start = 20
        name_end = body.find(b"\x00", name_start)
        if name_end < 0:
            name_end = len(body)
        name = body[name_start:name_end].decode("utf-8",
                                                 errors="replace")
        return {"index": idx, "start_ms": start_ms,
                "end_ms": end_ms, "rgb": rgb,
                "locked": bool(locked_flag), "name": name}
    if typ == "FLIP":
        return _decode_flip(body)
    return {"unknown_type": typ, "len": len(body)}


def _decode_flip(body: bytes) -> Dict[str, Any]:
    """FLIP entry (inside Markers2). A FLIP is a recorded sequence
    of cue jumps and censor (reverse-playback) actions that Serato
    can play back like a sample.

    VERIFIED layout (Samy Deluxe Pt.2, Eddie Johns, K.I.Z., Godzilla,
    plus 116 other tracks across the user's library):
        u8       NUL prefix
        u8       slot_index               (0–5)
        u8       flag                      (0 or 1; possibly a
                                             "has_name" hint but the
                                             cstring NUL is what
                                             actually delimits)
        cstring  name                      (UTF-8, may be empty)
        u8       enabled
        u32 BE   action_count
        action_count × action:
            u8       action_id
            u32 BE   payload_size
            payload                        (action_id-dependent)

    Action types we've seen in the wild (decoded key names match the
    encoder in `sidecaramel.flip_writer` so a decoded FLIP feeds
    straight back into `tags.encode_flip_entry`):
        id 0 = JUMP     payload_size 16   f64 from_s, f64 to_s
        id 1 = CENSOR   payload_size 24   f64 trigger_s,
                                            f64 reverse_target_s,
                                            f64 speed (-1.0 = reverse)

    Holzhaus' published spec mentions u8 looped + f64 total_length_s
    between enabled and action_count — neither survives in the
    Serato versions writing these tracks. The math
    (header + count × {21, 29}) matches every observed file exactly.
    """
    out: Dict[str, Any] = {"raw_len": len(body)}
    if len(body) < 4:
        return out
    cur = 0
    if body[cur] == 0:
        cur += 1
    if cur < len(body):
        out["slot_index"] = body[cur]; cur += 1
    if cur < len(body):
        out["flag_byte"] = body[cur]; cur += 1
    nul = body.find(b"\x00", cur)
    if nul < 0:
        return out | {"trunc_name": True}
    out["name"] = body[cur:nul].decode("utf-8", errors="replace")
    cur = nul + 1
    if cur >= len(body):
        return out
    # The byte right after the cstring NUL terminator is the FLIP's
    # `loop` flag (1 = loop the recorded sequence, 0 = one-shot).
    # It is NOT an "enabled" flag — an easy and tempting misread.
    out["loop"] = bool(body[cur]); cur += 1
    if cur + 4 > len(body):
        return out
    action_count = struct.unpack(">I", body[cur:cur + 4])[0]
    cur += 4
    out["action_count"] = action_count
    actions = []
    for i in range(action_count):
        if cur + 5 > len(body):
            actions.append({"truncated_at": i})
            break
        a_id = body[cur]; cur += 1
        a_len = struct.unpack(">I", body[cur:cur + 4])[0]
        cur += 4
        if a_len > 1024 or cur + a_len > len(body):
            actions.append({"truncated_payload_at": i,
                             "claimed_len": a_len})
            break
        payload = body[cur:cur + a_len]
        cur += a_len
        a = {"action_id": a_id, "payload_len": a_len}
        # Emit the SAME action vocabulary the encoder consumes
        # (`sidecaramel.flip_writer` / `tags.encode_flip_entry`):
        # JUMP -> from_s/to_s, CENSOR -> trigger_s/reverse_target_s/speed.
        # Values are the exact f64s (no round()) so a decoded FLIP
        # re-encodes byte-for-byte.
        if a_id == 0 and a_len == 16:
            from_s, to_s = struct.unpack(">dd", payload)
            a["type"] = "JUMP"
            a["from_s"] = from_s
            a["to_s"] = to_s
        elif a_id == 1 and a_len == 24:
            trigger_s, reverse_target_s, speed = struct.unpack(">ddd", payload)
            a["type"] = "CENSOR"
            a["trigger_s"] = trigger_s
            a["reverse_target_s"] = reverse_target_s
            a["speed"] = speed
        else:
            a["type"] = f"UNKNOWN_id={a_id}"
            a["payload_hex"] = payload.hex(" ")
        actions.append(a)
    out["actions"] = actions
    out["bytes_consumed"] = cur
    out["bytes_left"] = len(body) - cur
    return out


def decode_overview(b: bytes) -> Dict[str, Any]:
    """OVERVIEW = waveform thumbnail (the squiggly line Serato draws
    in the track-list row).
       u8  major (1), u8 minor (5)
       16 rows × N columns of bytes, where each byte is the
       per-column amplitude for that frequency band (0..255).
    N is roughly 240 for a 3-min track; Serato uses ~240 columns no
    matter the track length, then stretches.
    """
    out: Dict[str, Any] = {"length": len(b)}
    if len(b) < 2:
        return out
    out["version"] = f"{b[0]}.{b[1]}"
    body = b[2:]
    # The header is followed by another NUL or two; the row data
    # length is len(body) - some_header.  We treat everything from
    # offset 16 onward as rows, but most Serato writers also leave
    # 14 NULs after the version.
    skip = 0
    while skip < len(body) and body[skip] == 0 and skip < 32:
        skip += 1
    grid = body[skip:]
    # Serato uses 16 rows fixed; columns = len(grid) // 16.
    rows = 16
    cols = len(grid) // rows
    out["rows"] = rows
    out["cols"] = cols
    out["grid_bytes"] = len(grid)
    out["header_skip"] = skip
    if cols > 0:
        # Sample: first & last column per row, and overall avg amp.
        firsts = [grid[i * cols] for i in range(rows)]
        lasts = [grid[i * cols + cols - 1] for i in range(rows)]
        out["sample_first_col"] = firsts
        out["sample_last_col"] = lasts
        out["mean_amp"] = round(sum(grid) / len(grid), 2)
        out["max_amp"] = max(grid)
    return out


def decode_offsets(b: bytes) -> Dict[str, Any]:
    """OFFSETS_ — Serato's per-frame seek table for MP3 / VBR files.

    OBSERVED layout (a 19,807-byte blob on a 320 kbps MP3 / 4'17"):
       u8 major (1), u8 minor (2)
       ASCII zero-padded "%012.6f\\0"  total_samples        (e.g.
                                            "000000320000.000000")
       ASCII zero-padded "%012.6f\\0"  sample_rate          (e.g.
                                            "000000044100.000000")
       u32 BE  ?? (looks like total byte length or first frame ofs)
       4B      'viz\\0' or similar chunk tag (?)
       u32 BE  num_frames
       num_frames × 2B  (per-frame byte deltas? MP3 has a small set
                          of frame sizes depending on padding bit;
                          this looks like a packed delta table)
    Decoder dumps the ASCII headers + sniffs the rest.  Caveat:
    this layout was reverse-engineered from one MP3, not the
    public Serato spec; treat anything past the ASCII fields as
    best-effort."""
    out: Dict[str, Any] = {"length": len(b)}
    if len(b) < 4:
        return out
    out["version"] = f"{b[0]}.{b[1]}"
    cur = 2
    # ASCII header 1: total samples
    n1 = b.find(b"\x00", cur)
    if 0 < n1 - cur <= 32:
        out["ascii_field_1"] = b[cur:n1].decode("ascii", "replace")
        cur = n1 + 1
    # ASCII header 2: sample rate
    n2 = b.find(b"\x00", cur)
    if 0 < n2 - cur <= 32:
        out["ascii_field_2"] = b[cur:n2].decode("ascii", "replace")
        cur = n2 + 1
    # Field 1 turns out to be the audio bitrate in bps (320000 for
    # a 320 kbps MP3); field 2 is the sample rate (44100, 48000, …).
    try:
        out["bitrate_bps"] = float(out.get("ascii_field_1", "nan"))
        out["sample_rate_hz"] = float(out.get("ascii_field_2", "nan"))
    except (TypeError, ValueError):
        pass
    out["binary_section_offset"] = cur
    binary = b[cur:]
    out["binary_section_len"] = len(binary)
    if len(binary) >= 8:
        out["binary_head_hex"] = binary[:24].hex(" ")
    if len(binary) >= 8:
        # u32 + 4-byte tag + u32 + ...
        n_tag = struct.unpack(">I", binary[0:4])[0]
        tag = binary[4:8]
        out["sniff_first_u32"] = n_tag
        out["sniff_tag_4B"] = tag.decode("ascii", "replace")
    if len(binary) >= 12:
        out["sniff_second_u32"] = struct.unpack(
            ">I", binary[8:12])[0]
    # Per-frame deltas — count repetition of byte values to confirm
    if len(binary) >= 24:
        from collections import Counter
        pair_counter = Counter()
        for i in range(12, min(len(binary) - 1, 12 + 2000), 2):
            pair_counter[binary[i:i + 2]] += 1
        out["most_common_2B_pairs"] = [
            (p.hex(), c) for p, c in pair_counter.most_common(5)]
    return out


def decode_relvolad(b: bytes) -> Dict[str, Any]:
    """RELVOLAD — relative-volume adjustment / EQ block (10-band).
    Layout we've seen on user files:
       u8 major, u8 minor
       repeated: 4B BE float gain per band  (some versions 16 floats)
    Best-effort: dump as floats and let the user judge."""
    out: Dict[str, Any] = {"length": len(b)}
    if len(b) < 2:
        return out
    out["version"] = f"{b[0]}.{b[1]}"
    body = b[2:]
    floats = []
    for i in range(0, len(body) - 3, 4):
        (g,) = struct.unpack(">f", body[i:i + 4])
        floats.append(round(g, 5))
    out["band_floats"] = floats
    return out


def decode_videoassoc(b: bytes) -> Dict[str, Any]:
    """VIDEOASSOC — associates a video file with the audio track
    (Serato Video plugin). Looks like a UTF-16-BE path with a
    short header. Display the printable runs."""
    txt = b.decode("utf-16-be", errors="ignore")
    return {"length": len(b),
            "printable_utf16be": "".join(c for c in txt
                                          if c.isprintable())}


def decode_analysisflags(b: bytes) -> Dict[str, Any]:
    """ANALYSISFLAGS (MP4-only, 9 bytes after base64-decode):
       u8  major (0)
       u8  minor (0)
       ... (6 NUL bytes)
       u8  flags_bitfield
    Bitfield values seen:
       0x39 = 0b00111001 - analyzed + beatgrid + autotags + ?? + ??
       0x01 = analyzed only
    """
    out = {"length": len(b), "hex": b.hex(" ")}
    if len(b) >= 9:
        out["version"] = f"{b[0]}.{b[1]}"
        out["flags_byte"] = f"0x{b[-1]:02x}"
        out["flags_bits"] = f"0b{b[-1]:08b}"
    return out


DECODERS = {
    "Serato Analysis":      decode_analysis,
    "Serato AnalysisFlags": decode_analysisflags,
    "Serato Autotags":      decode_autotags,
    "Serato BeatGrid":      decode_beatgrid,
    "Serato Markers_":      lambda b: decode_markers_v1(b, "?"),
    "Serato Markers2":      decode_markers2,
    "Serato Overview":      decode_overview,
    "Serato Offsets_":      decode_offsets,
    "Serato RelVolAd":      decode_relvolad,
    "Serato VideoAssoc":    decode_videoassoc,
}


# ============================================================
# Main
# ============================================================

def container_of(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".mp3": "MP3", ".aif": "AIFF", ".aiff": "AIFF",
        ".m4a": "MP4", ".mp4": "MP4", ".flac": "FLAC",
        ".ogg": "OGG", ".wav": "WAV",
    }.get(ext, "?")


def inspect_one(path: str) -> None:
    print()
    print("=" * 72)
    print(f"TRACK  {path}")
    print("=" * 72)
    if not os.path.isfile(path):
        print("  not a file")
        return
    sz = os.path.getsize(path)
    print(f"  container : {container_of(path)}")
    print(f"  size      : {sz:,} bytes")

    try:
        from mutagen import File as _MFile
        m = _MFile(path)
        if m is not None:
            print(f"  duration  : "
                  f"{getattr(getattr(m, 'info', None), 'length', '?')!r} s")
            tag_class = type(m.tags).__name__ if m.tags else "none"
            print(f"  tag class : {tag_class}")
    except Exception as e:
        print(f"  mutagen open failed: {e!r}")
        return

    blobs = harvest(path)
    if not blobs:
        print("  (no Serato blobs found)")
        return
    print(f"  serato blobs: {len(blobs)}  "
          f"({', '.join(sorted({n for n, _, _ in blobs}))})")

    for name, raw, src in blobs:
        _print(f"{name}   [{len(raw)} bytes]  via {src}")
        print(f"  hex head : {_hex_preview(raw, 64)}")
        dec_fn = DECODERS.get(name)
        if dec_fn is None:
            print("  (no decoder)")
            continue
        # Patch container-aware Markers_:
        if name == "Serato Markers_":
            dec = decode_markers_v1(raw, container_of(path))
        else:
            dec = dec_fn(raw)
        _pretty(dec, indent=2)


def _pretty(obj: Any, indent: int = 0) -> None:
    sp = " " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                print(f"{sp}{k}:")
                _pretty(v, indent + 2)
            else:
                vs = repr(v)
                if len(vs) > 200:
                    vs = vs[:200] + "..."
                print(f"{sp}{k}: {vs}")
    elif isinstance(obj, list):
        if not obj:
            print(f"{sp}(empty)")
            return
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                # Compact one-line if small
                keys = list(item.keys())
                small = all(not isinstance(item[k],
                                              (dict, list))
                            for k in keys)
                if small and len(item) <= 6:
                    print(f"{sp}[{i}] " + "  ".join(
                        f"{k}={item[k]!r}" for k in keys))
                else:
                    print(f"{sp}[{i}]:")
                    _pretty(item, indent + 4)
            else:
                print(f"{sp}- {item!r}")


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    for path in argv[1:]:
        try:
            inspect_one(path)
        except Exception as e:
            print(f"ERROR on {path}: {e!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
