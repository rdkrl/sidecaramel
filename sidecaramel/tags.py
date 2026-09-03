"""sidecaramel.tags — Serato Markers2 + Markers_ V1 +
BeatGrid + Autotags + Analysis + Overview + RelVolAd +
cue-color palette + stems wrappers + DB-V2 wrappers.
Container-agnostic dispatchers (`*_any`) supporting
MP3 / WAV / AIFF (ID3 GEOB) / MP4 / M4A / M4V (freeform atoms) /
FLAC / OGG / OPUS (Vorbis comments).
"""
from __future__ import annotations

import base64
import os
import re
import struct
from typing import List, Optional, Tuple

# mutagen-side imports — kept lazy where they raise on missing
# install paths, eager where they're cheap.
from mutagen.id3 import ID3NoHeaderError
from mutagen.mp4 import MP4, MP4StreamInfoError
from mutagen.flac import FLAC
from mutagen import MutagenError



# Shared confirm-gate.  Both Serato writers and lyric writers
# import this single function.
from sidecaramel.confirm_gate import _require_confirm  # noqa: F401

# =====================================================================
# Serato BeatGrid blob — modern Serato Pro format
# =====================================================================

def _parse_serato_beatgrid(data: bytes) -> Optional[dict]:
    """Decode the Serato BeatGrid blob.

    Format (modern Serato Pro, verified across a real Serato library):
        bytes 0..1   uint16 BE   version (0x0100)
        bytes 2..5   uint32 BE   count = TOTAL number of markers
        per marker (8 bytes):
            • markers[0..count-2] (non-terminal):
                float32 BE  position in seconds
                uint32  BE  beats_until_next_marker
            • markers[count-1] (terminal):
                float32 BE  position
                float32 BE  BPM at that point
        trailing byte                 1-byte footer (BPM-lock flag)

    Total bytes = 6 + 8*count + 1.

    Returns
        {"bpm": float,
         "markers": [(t_sec, beats_to_next), ...],   # NON-terminal
         "terminal_t": float}
    or None on parse failure / sanity-check fail.
    """
    if not data or len(data) < 6 + 8:
        return None
    try:
        n_total = struct.unpack(">I", data[2:6])[0]
        if n_total < 1 or n_total > 10000:
            return None
        markers = []
        p = 6
        for _ in range(n_total - 1):
            if p + 8 > len(data):
                return None
            t = struct.unpack(">f", data[p:p + 4])[0]
            beats = struct.unpack(">I", data[p + 4:p + 8])[0]
            markers.append((float(t), int(beats)))
            p += 8
        if p + 8 > len(data):
            return None
        term_t = struct.unpack(">f", data[p:p + 4])[0]
        bpm = struct.unpack(">f", data[p + 4:p + 8])[0]
        if bpm <= 0 or bpm > 300:
            return None
        return {
            "bpm": float(bpm),
            "markers": markers,
            "terminal_t": float(term_t),
        }
    except Exception:
        return None


# =====================================================================
# Base64 decode tolerance — Serato Pro writer quirk (SPOT re-export)
# =====================================================================
# Serato Pro writes Markers2 base64 streams whose length can land at
# "length ≡ 1 mod 4", which a naive decoder rejects silently — the
# symptom is Markers2-MP4 loops that simply never parse.  The
# trim-{0,1,2,3} tolerant decoder handles it.
#
# Single owner of "decode a Serato base64 stream" is
# `sidecaramel.blobs` (the lowest-level Serato module); every
# call-site — the outer envelope decode in `_unwrap_mp4_envelope`,
# the inner Markers2 base64 in `_parse_serato_markers2`, and any
# future Serato-blob reader — goes through it.  This module
# re-exports the symbol so `tags` callers can reach it directly.
from sidecaramel.blobs import _robust_b64_decode  # noqa: F401


# =====================================================================
# Serato Markers2 blob — cues + loops + FLIP entries
# =====================================================================

def _parse_serato_markers2(data: bytes) -> List[dict]:
    """Decode Serato Markers2 (cue points + loops).

    Outer wrapper: 2-byte version + base64 of inner stream + trailing
    NULs.  Inner stream: u8 1, u8 1, then zero or more
    (entry_type_cstring, u32 BE length, payload) records.

    For 'CUE' payloads (≥13 bytes):
        payload[1]   = idx
        payload[2..5]= pos_ms (u32 BE)
        payload[7..9]= RGB
        payload[12..]= zero-terminated ASCII label

    Returns list of {"idx", "pos_ms", "pos_sec", "color", "label"}.
    """
    if not data or len(data) < 2:
        return []
    try:
        b64 = bytearray()
        for byte in data[2:]:
            if byte == 0:
                break
            if byte == 0x0A:
                continue
            b64.append(byte)
        # use shared _robust_b64_decode helper
        # which handles Serato Pro's "length ≡ 1 mod 4" padding
        # quirk.  Returns first non-empty successful decode.
        dec = _robust_b64_decode(bytes(b64))
        if not dec:
            return []
    except Exception:
        return []

    cues = []
    p = 2
    while p < len(dec) - 4:
        te = dec.find(0, p)
        if te < 0 or te == p:
            break
        try:
            entry_type = dec[p:te].decode("latin-1")
        except Exception:
            break
        p = te + 1
        if p + 4 > len(dec):
            break
        entry_len = ((dec[p] << 24) | (dec[p + 1] << 16)
                      | (dec[p + 2] << 8) | dec[p + 3])
        p += 4
        if entry_len <= 0 or p + entry_len > len(dec):
            break
        payload = dec[p:p + entry_len]
        p += entry_len

        if entry_type == "CUE" and len(payload) >= 13:
            idx = payload[1]
            pos_ms = ((payload[2] << 24) | (payload[3] << 16)
                      | (payload[4] << 8) | payload[5])
            r, g, b = payload[7], payload[8], payload[9]
            label = ""
            for j in range(12, len(payload)):
                if payload[j] == 0:
                    break
                label += chr(payload[j])
            cues.append({
                "idx": idx,
                "pos_ms": pos_ms,
                "pos_sec": pos_ms / 1000.0,
                "color": (r, g, b),
                "label": label,
            })
    cues.sort(key=lambda c: c["idx"])
    return cues


# =====================================================================
# Multi-container Serato metadata reader
# =====================================================================

# Used by the trim-{0,1,2,3} fallback inside MP4 unwrap (Serato writer
# quirk: some MP4 atoms ship with 1 stray trailing base64 char).
_B64_ALPHABET_RE = re.compile(rb"[^A-Za-z0-9+/=]")


def read_serato_metadata(audio_path: str) -> Optional[dict]:
    """Load BPM, beatgrid markers, cue points, and saved loops from a
    Serato-tagged audio file.

    Returns::

        {
          "bpm": float | None,
          "beat_markers": [(t_sec, beats_to_next), ...],
          "terminal_t": float | None,
          "cues": [{idx, pos_ms, pos_sec, color, label}, ...],
          "loops": [{idx, start_ms, end_ms, start_sec, end_sec,
                      color, label, locked}, ...],
          "flips": [{slot, flag, name, loop, action_count, actions}, ...],
          "first_cue": float | None,
          "autotags": {bpm, gain, gain_db} | None,   # autogain
          "track_color": (r, g, b) | None,
          "bpm_lock": bool,                          # grid lock
        }

    or ``None`` when the file is a supported container but has no
    Serato tags.

    Supported containers:
        MP3 / WAV       → ID3 GEOB frames
        MP4 / M4A / M4V → ``----:com.serato.dj:*`` freeform atoms
        FLAC / OGG      → Vorbis ``SERATO_*`` comments
        AIFF            → ID3 GEOB frames

    Raises:
        :class:`FileNotFoundError`: when ``audio_path`` does not
            exist on disk.  A typo'd path being indistinguishable
            from "no metadata" is the kind of footgun that costs
            users an afternoon.

    Returns ``None`` (does not raise) when the extension is
    unsupported, when the file exists but holds no Serato blobs, or
    when blob parsing fails internally.
    """
    if not audio_path or not os.path.exists(audio_path):
        # Explicit nonexistent-path raise — distinct from
        # "exists but no metadata" (None) below.  Reader silent-fail
        # for the latter is documented in README "Known
        # limitations"; raising for the former is the only
        # correctness-positive default.
        raise FileNotFoundError(audio_path)
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in (".mp3", ".wav", ".mp4", ".m4a", ".m4v",
                    ".flac", ".ogg", ".oga", ".opus", ".aif", ".aiff"):
        return None

    beatgrid = None
    cues: List[dict] = []
    loops: List[dict] = []
    flips: List[dict] = []
    autotags = None
    track_color = None
    bpm_lock = False

    # Unified path: blobs.harvest() handles every container
    # (ID3 GEOB for MP3/WAV/AIFF, ----:com.serato.dj:* atoms for
    # MP4/M4A/M4V with envelope-unwrap, SERATO_* Vorbis comments for
    # FLAC/OGG).  The base64 trim-{0,1,2,3} fallback lives in
    # `blobs._robust_b64_decode`, so Serato Pro's "length ≡ 1 mod 4"
    # writer-quirk files come through this path correctly — a naive
    # decoder drops them silently.
    try:
        for desc, payload, _src in harvest_serato_blobs(audio_path):
            if not payload:
                continue
            if desc == "Serato BeatGrid":
                parsed = _parse_serato_beatgrid(payload)
                if parsed:
                    beatgrid = parsed
            elif desc == "Serato Markers2":
                full = parse_serato_markers2_full(payload)
                if full.get("cues"):
                    cues = full["cues"]
                if full.get("loops"):
                    loops = full["loops"]
                # FLIP entries (cue-jump sequences) live in Markers2 too;
                # parse_serato_markers2_full already decodes them, just
                # surface them here so callers don't re-parse the blob.
                if full.get("flips"):
                    flips = full["flips"]
                # track-color + bpm/grid-lock also live in Markers2 — surface
                # them so callers porting tags between libraries do not
                # have to re-parse the blob.
                if full.get("color") is not None:
                    track_color = full["color"]
                bpm_lock = bool(full.get("bpm_lock"))
            elif desc == "Serato Autotags":
                # autogain / bpm / gain_db — read-from-file so callers can carry
                # autogain (the build/write side already existed).
                try:
                    autotags = parse_serato_autotags(payload)
                except Exception:
                    autotags = None
    except Exception:
        return None

    if (not beatgrid and not cues and not loops and not flips
            and not autotags):
        return None

    bpm = beatgrid["bpm"] if beatgrid else None
    markers = beatgrid["markers"] if beatgrid else []
    terminal_t = beatgrid["terminal_t"] if beatgrid else None
    first_cue = cues[0]["pos_sec"] if cues else None

    return {
        "bpm": bpm,
        "beat_markers": markers,
        "terminal_t": terminal_t,
        "cues": cues,
        "loops": loops,
        "flips": flips,
        "first_cue": first_cue,
        "autotags": autotags,        # {bpm, gain (autogain), gain_db} | None
        "track_color": track_color,  # (r,g,b) | None
        "bpm_lock": bpm_lock,        # grid lock
    }


def read_autotags_from_file(audio_path: str) -> Optional[dict]:
    """Read just the Serato Autotags (autogain) blob from a file → {bpm, gain,
    gain_db} | None.  Thin convenience wrapper around harvest + parse_serato_
    autotags (the parse-bytes/build/write side already existed; this adds the
    read-from-file side)."""
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)
    try:
        for desc, payload, _src in harvest_serato_blobs(audio_path):
            if desc == "Serato Autotags" and payload:
                return parse_serato_autotags(payload)
    except Exception:
        return None
    return None



# =====================================================================
# Beat grid expansion
# =====================================================================

def serato_beat_grid_seconds(serato_meta: Optional[dict],
                              audio_duration_sec: Optional[float] = None
                              ) -> List[float]:
    """Expand Serato's sparse non-terminal markers into a dense list
    of beat timestamps in seconds, covering the whole track.

    Serato stores a marker every time the BPM or beat spacing
    changes. Between two markers, beats are evenly distributed.
    After the last non-terminal marker, beats continue at the
    terminal BPM until the end of the audio (or a best-effort
    estimate if duration is unknown).

    Returns a sorted list of float seconds. Empty list if no
    metadata.
    """
    if not serato_meta:
        return []
    bpm = serato_meta.get("bpm")
    # Accept both the internal key ("markers", from
    # _parse_serato_beatgrid directly) and the public key
    # ("beat_markers", from read_serato_metadata).
    markers = (serato_meta.get("beat_markers")
                or serato_meta.get("markers") or [])
    terminal_t = serato_meta.get("terminal_t")
    if not bpm:
        return []

    beats: List[float] = []
    for i, (t, n_beats) in enumerate(markers):
        if n_beats <= 0:
            continue
        next_t = (markers[i + 1][0]
                   if (i + 1) < len(markers) else terminal_t)
        if next_t is None or next_t <= t:
            continue
        span = next_t - t
        step = span / n_beats
        for k in range(n_beats):
            beats.append(t + k * step)

    if terminal_t is not None:
        tail_start = terminal_t
        tail_end = (audio_duration_sec if audio_duration_sec
                    else tail_start + 900.0)
        beat_period = 60.0 / bpm
        t = tail_start
        while t < tail_end:
            beats.append(t)
            t += beat_period

    beats.sort()
    return beats



# =====================================================================
# Serato Autotags blob — BPM + Gain + GainDB
# =====================================================================

def parse_serato_autotags(data: bytes) -> Optional[dict]:
    """Parse the Serato Autotags blob.

    Format: u16 version + null-terminated ASCII strings:
        strings[0] = BPM
        strings[1] = AutoGain
        strings[2] = GainDB

    Returns {"bpm": float, "gain": float, "gain_db": float} (only
    keys that decoded successfully) or None.

    Used by `read_serato_geob_tags` debug-panel reader.
    """
    if not data or len(data) < 4:
        return None
    try:
        pos = 2  # skip version
        strings: List[str] = []
        while pos < len(data) and len(strings) < 3:
            end = data.find(b"\x00", pos)
            if end < 0:
                end = len(data)
            try:
                s = data[pos:end].decode("latin-1", errors="ignore")
            except Exception:
                s = ""
            strings.append(s)
            pos = end + 1
        out: dict = {}
        if strings and strings[0]:
            try:
                out["bpm"] = round(float(strings[0]), 2)
            except (ValueError, TypeError):
                pass
        if len(strings) >= 2 and strings[1]:
            try:
                out["gain"] = round(float(strings[1]), 3)
            except (ValueError, TypeError):
                pass
        if len(strings) >= 3 and strings[2]:
            try:
                out["gain_db"] = round(float(strings[2]), 3)
            except (ValueError, TypeError):
                pass
        return out if out else None
    except Exception:
        return None



# =====================================================================
# Serato GEOB / atom dispatcher — full debug-panel shape
# =====================================================================
# Distinct from `read_serato_metadata` (alignment-math contract):
# this one returns the rich shape the GUI debug panel expects, with
# autotags + dict-shape cues + dict-shape beatgrid markers.
# Both readers share the same low-level parsers — no duplication.

def read_serato_geob_tags(path: str) -> dict:
    """Read all Serato GEOB / freeform-atom tags from an audio file
    and return them in the GUI-debug-panel shape.

    Returns:
        {
          "bpm":      float | None,    # from Autotags
          "gain_db":  float | None,    # from Autotags
          "cues":     [{index, position_s, color_hex, label}, ...] | None,
          "beatgrid": {markers: [{position_s, beats_till_next}],
                        terminal_pos_s, terminal_bpm} | None,
        }

    Empty dict on parse failure.  Each top-level key is independently-
    optional (null-safe convention for tags).

    This parallels `read_serato_metadata` but returns the dict-shape
    contract used by the GUI debug panel.  Both share
    the same canonical low-level parsers + autotags parser.
    """
    out = {"bpm": None, "gain_db": None, "cues": None,
           "beatgrid": None}
    meta_canon = read_serato_metadata(path)
    # Pull cues + beatgrid via the canonical reader (multi-container).
    if meta_canon:
        if meta_canon.get("cues"):
            out["cues"] = [
                {
                    "index": c["idx"],
                    "position_s": c["pos_sec"],
                    "color_hex": "#{:02x}{:02x}{:02x}".format(*c["color"]),
                    "label": c["label"],
                }
                for c in meta_canon["cues"]
            ]
        if meta_canon.get("bpm") is not None:
            out["beatgrid"] = {
                "markers": [
                    {"position_s": t, "beats_till_next": b}
                    for (t, b) in meta_canon.get("beat_markers", [])
                ],
                "terminal_pos_s": meta_canon.get("terminal_t"),
                "terminal_bpm": meta_canon.get("bpm"),
            }
            out["bpm"] = meta_canon["bpm"]

    # Autotags is GEOB-specific (MP3/WAV).  Read it directly to get
    # the Gain DB and BPM-from-Autotags (which can differ from the
    # BeatGrid terminal BPM if the user re-analysed after a manual
    # grid edit).
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp3", ".wav", ".aif", ".aiff"):
        try:
            # Use the container-aware opener so WAV/AIFF resolve the
            # ID3 chunk inside the RIFF/FORM envelope rather than
            # reading from the bare file start.
            c = _open_id3_container(path)
            id3 = c.tags
            for frame in id3.getall("GEOB"):
                desc = getattr(frame, "desc", "") or ""
                if desc != "Serato Autotags":
                    continue
                data = getattr(frame, "data", b"") or b""
                parsed = parse_serato_autotags(data)
                if parsed:
                    if parsed.get("bpm") is not None and out["bpm"] is None:
                        out["bpm"] = parsed["bpm"]
                    if parsed.get("gain_db") is not None:
                        out["gain_db"] = parsed["gain_db"]
                break
        except (ID3NoHeaderError, MutagenError, Exception):
            pass

    return out


# =====================================================================
# Multi-container Serato blob harvester — re-export of blobs
# =====================================================================
# Single entry point for low-level blob harvesting (MP3 GEOB, MP4
# freeform atom, FLAC Vorbis SERATO_*, AIFF GEOB).  Wraps
# `sidecaramel.blobs.harvest()` so callers depend only on this module.

def harvest_serato_blobs(audio_path: str) -> list:
    """Return a list of (descriptor, payload_bytes, source_label)
    tuples for every Serato blob in the file, across all containers.

    Thin wrapper around the low-level cross-container harvester
    (see ``blobs.harvest()``).  Returns ``[]`` if the file has no
    blobs or cannot be opened.
    """
    from . import blobs as si
    try:
        return si.harvest(audio_path)
    except Exception:
        return []



def _open_id3_container(path: str):
    """Return a mutagen container whose `.tags` is the ID3-like and
    whose `.save()` persists the file with ID3 in the correct
    envelope: bare ID3v2 for MP3, RIFF `id3 ` chunk for WAV, FORM
    `ID3 ` chunk for AIFF.

    Raises whatever mutagen raises if the file is malformed or
    unreadable.  Caller is expected to wrap in try/except.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        from mutagen.wave import WAVE
        c = WAVE(path)
        if c.tags is None:
            c.add_tags()
        return c
    if ext in (".aif", ".aiff"):
        from mutagen.aiff import AIFF
        c = AIFF(path)
        if c.tags is None:
            c.add_tags()
        return c
    # MP3 (or anything else with bare ID3v2 at file start)
    from mutagen.id3 import ID3, ID3NoHeaderError
    try:
        id3 = ID3(path)
    except ID3NoHeaderError:
        id3 = ID3()

    class _MP3IDContainer:
        def __init__(self, id3_tags, path):
            self.tags = id3_tags
            self._path = path

        def save(self):
            self.tags.save(self._path)
    return _MP3IDContainer(id3, path)


def wipe_serato_geobs(path: str, *, confirm: bool = False) -> int:
    """Delete every Serato-* GEOB frame from an ID3-tagged audio file
    (MP3 / WAV / AIFF).  Returns the number of frames removed.  Returns
    0 for non-ID3 containers or if the file cannot be opened.

    Caller MUST pass `confirm=True`; without it `RuntimeError` is
    raised before any write.
    """
    _require_confirm(confirm, "wipe_serato_geobs", path)
    try:
        c = _open_id3_container(path)
    except Exception:
        return 0
    tags = c.tags
    removed = 0
    for key in list(tags.keys()):
        ks = str(key)
        if ks.startswith("GEOB:") and "Serato" in ks:
            try:
                tags.delall(key)
                removed += 1
            except Exception:
                continue
    if removed:
        try:
            c.save()
        except Exception:
            return 0
    return removed


def write_serato_geob(path: str, descriptor: str, payload: bytes,
                       *, confirm: bool = False) -> bool:
    """Write a single Serato GEOB frame to an ID3-tagged audio file
    (MP3 / WAV / AIFF).

    `descriptor` is the GEOB description string (e.g.
    "Serato BeatGrid", "Serato Markers2", "Serato Autotags").
    `payload` is the raw blob bytes.

    Returns True on success.  Silently returns False for other
    container types or when mutagen rejects the file.

    Caller MUST pass `confirm=True`; without it `RuntimeError` is
    raised before any write.
    """
    _require_confirm(confirm, "write_serato_geob", path)
    try:
        from mutagen.id3 import GEOB
    except ImportError:
        return False
    try:
        c = _open_id3_container(path)
    except Exception:
        return False
    tags = c.tags
    try:
        # Wipe any existing GEOB with this exact descriptor first
        for key in list(tags.keys()):
            ks = str(key)
            if ks.startswith("GEOB:") and descriptor in ks:
                tags.delall(key)
        tags.add(GEOB(encoding=0, mime="application/octet-stream",
                       filename="", desc=descriptor, data=payload))
        c.save()
        return True
    except Exception:
        return False


def pack_serato_markers2(inner: bytes, target_size: int = 0,
                          line_width: int = 72) -> bytes:
    """Pack a Markers2 inner stream into its outer GEOB blob format.

    Serato wraps the base64 with newlines every `line_width` chars
    (observed: 72), then NUL-pads to `target_size` to preserve the
    original GEOB byte count.

    Pure function — no I/O.
    """
    b64 = base64.b64encode(inner)
    if line_width > 0:
        chunks = [b64[i:i + line_width]
                  for i in range(0, len(b64), line_width)]
        b64_lines = b"\n".join(chunks)
    else:
        b64_lines = b64
    out = bytearray(b"\x01\x01")
    out += b64_lines
    if target_size > 0 and len(out) < target_size:
        out += b"\x00" * (target_size - len(out))
    return bytes(out)


def write_serato_markers2(path: str, inner: bytes,
                           target_size: int = 0,
                           line_width: int = 72,
                           *, confirm: bool = False) -> bool:
    """Pack `inner` into the Markers2 outer blob format and write it
    as a "Serato Markers2" GEOB frame on the given MP3/WAV file.

    Confirm-flag check: caller MUST pass `confirm=True`.

    This is the shared writer for FLIP / drum-5pack workflows — that
    is, for entry types V1 does not carry.

    DO NOT write cues or loops through this function. Serato honours
    cues and loops only when the V1 `Serato Markers_` blob and the V2
    `Serato Markers2` blob BOTH exist and agree; writing V2 alone
    leaves V1 stale and Serato may keep showing the old positions.
    Use `write_serato_markers_full_mp3` / `_mp4`, which emit the pair
    in a single save.
    """
    _require_confirm(confirm, "write_serato_markers2", path)
    outer = pack_serato_markers2(inner, target_size=target_size,
                                   line_width=line_width)
    return write_serato_geob(path, "Serato Markers2", outer,
                                confirm=True)



# =====================================================================
# Serato Markers2 ENCODE side — entry payload encoders + inner-stream
# aggregator + MP4 freeform-atom writer + container dispatcher
# =====================================================================
#   - a downstream consumer needs to WRITE
#     Markers2 CUE+LOOP entries to MP4 freeform atoms (this existing
#     `write_serato_markers2` is ID3-only).
#   - All encoders below are PURE FUNCTIONS — no IO, no audio touch.
#     The single audio-touching primitive is
#     `write_serato_markers2_mp4`, which goes through `_require_confirm`
#     like every other audio-write in this module.
#   - Byte-layouts come from `parse_serato_markers2_full` (read-side
#     SPOT) so the parse→encode→parse roundtrip is exact.
#   - LOOP locked-flag is INVERTED at the wire level:
#     `locked=True`  → flag u32 = 0x00000000
#     `locked=False` → flag u32 = 0xFFFFFFFF
#     (matches read-side check `locked = (locked_flag != 0xFFFFFFFF)`.)
#   - Default `bpm_lock=False` (a curation workflow leaves
#     beatgrids unlocked).  Pass `bpm_lock=None` to
#     omit the BPMLOCK entry entirely; pass `True` to explicitly lock.
#   - Markers_ V1 blob is NOT generated ("V1 nein").
#     If a future workflow needs V1 for app-back-compat, add it as a
#     separate encoder.
#   - audio-write policy: the MP4 writer is for NEW-or-pre-cleared files
#     only (e.g. a curation workflow's 10-track copies).  The
#     confirm-gate is the only structural protection; caller discipline
#     handles the per-file scope.

def encode_cue_entry(idx: int, pos_ms: int,
                       color_rgb: Tuple[int, int, int],
                       label: str = "") -> bytes:
    """Encode a single Markers2 CUE entry (entry-name cstring + u32 BE
    payload-length + payload).

    Payload layout (mirrors `parse_serato_markers2_full` CUE-branch):

        byte 0      NUL
        byte 1      idx (u8, 0..7)
        bytes 2-5   pos_ms (u32 BE)
        byte 6      NUL
        bytes 7-9   R, G, B (u8 each)
        bytes 10-11 NUL NUL
        bytes 12..  label cstring (UTF-8, NUL-terminated)

    `color_rgb` is a 3-tuple; values are clamped to 0..255.  `label`
    is encoded UTF-8 and NUL-terminated.

    Pure function — no IO.
    """
    r, g, b = (int(c) & 0xFF for c in color_rgb)
    label_bytes = (label or "").encode("utf-8")
    payload = (
        b"\x00"
        + bytes([int(idx) & 0xFF])
        + struct.pack(">I", int(pos_ms) & 0xFFFFFFFF)
        + b"\x00"
        + bytes([r, g, b])
        + b"\x00\x00"
        + label_bytes
        + b"\x00"
    )
    return b"CUE\x00" + struct.pack(">I", len(payload)) + payload


def encode_loop_entry(idx: int, start_ms: int, end_ms: int,
                       color_rgb: Tuple[int, int, int],
                       label: str = "",
                       locked: bool = False) -> bytes:
    """Encode a single Markers2 LOOP entry.

    Payload layout — the community-reverse-engineered Serato Markers2
    format (Holzhaus `serato-tags`, cross-checked against Mixxx
    `SeratoMarkers2LoopEntry`), which is what real Serato and conformant
    third-party tools read/write:

        byte 0       NUL
        byte 1       idx (u8, 0..7)
        bytes 2-5    start_ms (u32 BE)
        bytes 6-9    end_ms   (u32 BE)
        bytes 10-13  0xFFFFFFFF (constant)
        bytes 14-17  color, 4-byte "ARGB" (00 R G B — the A byte is 00)
        byte 18      NUL
        byte 19      locked (u8 bool)
        bytes 20..   label cstring (UTF-8, NUL-terminated)

    Pure function — no IO.
    """
    r, g, b = (int(c) & 0xFF for c in color_rgb)
    label_bytes = (label or "").encode("utf-8")
    payload = (
        b"\x00"
        + bytes([int(idx) & 0xFF])
        + struct.pack(">I", int(start_ms) & 0xFFFFFFFF)
        + struct.pack(">I", int(end_ms) & 0xFFFFFFFF)
        + b"\xff\xff\xff\xff"
        + bytes([0x00, r, g, b])
        + b"\x00"
        + bytes([1 if locked else 0])
        + label_bytes
        + b"\x00"
    )
    return b"LOOP\x00" + struct.pack(">I", len(payload)) + payload


def encode_color_entry(color_rgb: Tuple[int, int, int]) -> bytes:
    """Encode a Markers2 COLOR (track-tile color) entry.

    Payload: NUL + R + G + B (4 bytes total).
    """
    r, g, b = (int(c) & 0xFF for c in color_rgb)
    payload = bytes([0, r, g, b])
    return b"COLOR\x00" + struct.pack(">I", len(payload)) + payload


def encode_bpmlock_entry(locked: bool) -> bytes:
    """Encode a Markers2 BPMLOCK entry.

    Payload: single u8, 0x01 if locked else 0x00.
    """
    payload = b"\x01" if locked else b"\x00"
    return b"BPMLOCK\x00" + struct.pack(">I", len(payload)) + payload


def encode_flip_entry(slot_index: int, name: str, loop: bool,
                       actions: List[dict],
                       *, flag: int = 1) -> bytes:
    """Encode a Markers2 FLIP entry (entry-name cstring + u32 BE
    payload-length + payload).

    Payload layout, verified against real files (the published
    Holzhaus spec disagreed with this entry; this is the corrected,
    verified layout):

        byte 0       NUL prefix
        byte 1       slot_index (u8, 0..5)
        byte 2       flag (u8, 0=old format / 1=new — default 1)
        cstring      name (UTF-8, NUL-terminated, may be empty)
        byte         loop (u8, 0=one-shot / 1=loop FLIP)
        u32 BE       action_count
        action_count × action

    Each action is one of:

        JUMP    {"type":"jump",   "from_s": float, "to_s": float}
                → u8 id=0 + u32 BE 16 + f64 BE from + f64 BE to

        CENSOR  {"type":"censor", "trigger_s": float,
                  "reverse_target_s": float, "speed": float (=-1.0)}
                → u8 id=1 + u32 BE 24 + 3 × f64 BE

    Delegates action-byte construction to
    `sidecaramel.flip_writer.build_jump` / `build_censor` (SPOT).
    Delegates FLIP-entry-body construction to
    `sidecaramel.flip_writer.build_flip_entry`.

    Pure function — no IO.
    """
    from sidecaramel.flip_writer import (build_flip_entry as _bfe,
                                              build_jump as _bj,
                                              build_censor as _bc)

    action_bytes: List[bytes] = []
    for a in (actions or []):
        atype = (a.get("type") or "").lower()
        if atype == "jump":
            action_bytes.append(_bj(float(a["from_s"]),
                                     float(a["to_s"])))
        elif atype == "censor":
            action_bytes.append(_bc(
                float(a["trigger_s"]),
                float(a["reverse_target_s"]),
                float(a.get("speed", -1.0))))
        else:
            raise ValueError(
                f"unknown FLIP action type {atype!r}; "
                "expected 'jump' or 'censor'")

    m2_entry = _bfe(slot=int(slot_index), name=str(name or ""),
                     loop=bool(loop), actions=action_bytes,
                     flag=int(flag))
    body = m2_entry.body
    return b"FLIP\x00" + struct.pack(">I", len(body)) + body


def build_markers2_inner(
        cues: Optional[List[dict]] = None,
        loops: Optional[List[dict]] = None,
        *,
        color_rgb: Optional[Tuple[int, int, int]] = None,
        bpm_lock: Optional[bool] = False,
        flips: Optional[List[dict]] = None,
        ) -> bytes:
    """Build a complete Markers2 INNER stream (pre-base64-wrap).

    Layout (the order real Serato DJ Pro writes):
        b"\\x01\\x01"                              # inner version
        + [COLOR entry, if color_rgb given]
        + [CUE entries, sorted by idx]
        + [LOOP entries, sorted by idx]
        + [BPMLOCK entry, if bpm_lock is not None]
        + [FLIP entries, sorted by slot]
        + b"\\x00"                                  # end-of-stream

    `cues` is a list of dicts with keys: idx, pos_ms, color, label.
        - `color` is either a 3-tuple (r,g,b) or a `#rrggbb` string.
    `loops` is a list of dicts with keys: idx, start_ms, end_ms, color,
                                            label, locked.

    `color_rgb` (track-tile color, optional) — a 3-tuple; None omits.
    `bpm_lock` (default False per ):
        - True  → BPMLOCK=0x01
        - False → BPMLOCK=0x00 (writes the entry, value=unlocked)
        - None  → no BPMLOCK entry written

    Caller is responsible for not exceeding Serato's 8-slot limit per
    type — this function doesn't enforce it (would silently encode 9+
    entries, which Serato would ignore beyond slot 7).
    """
    def _rgb(c):
        # Accept (r,g,b) tuple/list OR "#rrggbb" hex string.
        if c is None:
            return (0, 0, 0)
        if isinstance(c, str):
            s = c.strip().lstrip("#")
            if len(s) >= 6:
                return (int(s[0:2], 16), int(s[2:4], 16),
                        int(s[4:6], 16))
            return (0, 0, 0)
        if isinstance(c, (tuple, list)) and len(c) >= 3:
            return (int(c[0]), int(c[1]), int(c[2]))
        return (0, 0, 0)

    out = bytearray(b"\x01\x01")
    if color_rgb is not None:
        out += encode_color_entry(_rgb(color_rgb))

    for cue in sorted(cues or [], key=lambda d: int(d.get("idx", 0))):
        out += encode_cue_entry(
            idx=int(cue.get("idx", 0)),
            pos_ms=int(cue.get("pos_ms", 0)),
            color_rgb=_rgb(cue.get("color")),
            label=str(cue.get("label", "") or ""))

    for loop in sorted(loops or [], key=lambda d: int(d.get("idx", 0))):
        out += encode_loop_entry(
            idx=int(loop.get("idx", 0)),
            start_ms=int(loop.get("start_ms", 0)),
            end_ms=int(loop.get("end_ms", 0)),
            color_rgb=_rgb(loop.get("color")),
            label=str(loop.get("label", "") or ""),
            locked=bool(loop.get("locked", False)))

    # BPMLOCK sits AFTER the cues/loops and before FLIP — the order real
    # Serato DJ Pro writes (verified against genuine MP3/WAV/MP4 blobs:
    # COLOR, CUE*, LOOP*, BPMLOCK, FLIP*).
    if bpm_lock is not None:
        out += encode_bpmlock_entry(bool(bpm_lock))

    # FLIP entries.  Sorted by slot_index so output is deterministic;
    # caller can interleave by re-ordering the list if needed.  The
    # decoded flip dict uses key `slot` (from parse_serato_markers2_full);
    # `slot_index` is the encoder-side name — accept either.
    for flip in sorted(flips or [],
                        key=lambda d: int(d.get("slot_index",
                                               d.get("slot", 0)))):
        out += encode_flip_entry(
            slot_index=int(flip.get("slot_index", flip.get("slot", 0))),
            name=str(flip.get("name", "") or ""),
            loop=bool(flip.get("loop", False)),
            actions=list(flip.get("actions") or []),
            flag=int(flip.get("flag", 1)))

    out += b"\x00"  # end-of-stream sentinel
    return bytes(out)


def write_serato_markers2_mp4(path: str, inner: bytes,
                                target_size: int = 0,
                                line_width: int = 72,
                                *, confirm: bool = False) -> bool:
    """Write a Markers2 inner stream to an MP4 file's freeform atom
    `----:com.serato.dj:markersv2`.

    Replaces ONLY this single atom — every other tag on the file is
    preserved (iTunes metadata, other `----:com.serato.dj:*` atoms,
    cover art, chapters, non-standard freeform atoms, all of it).
    "achte darauf dass auch keine anderen
    informationen aus dem mutagen verloren gehen, auch wenn sie nicht
    standard sind".  The write goes through mutagen's `mp4.save()`
    on the existing `mp4.tags` object — no `.clear()`, no wholesale
    re-assignment.

    `inner` is the Markers2 inner stream (pre-base64-wrap).  Build
    one with `build_markers2_inner()`.

    `target_size` and `line_width` are forwarded to
    `pack_serato_markers2()` which produces the outer blob.  Default
    `line_width=72` matches the on-disk convention found in real-world files.

    Returns True on success.  Silently returns False for non-MP4
    containers or mutagen errors.

    Confirm-flag check: caller MUST pass `confirm=True`.

    audio-write policy reminder: only call this on NEW files or on files for
    which this has given pro-operation OK.  This function does not
    enforce a path-whitelist — caller discipline.
    """
    _require_confirm(confirm, "write_serato_markers2_mp4", path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".mp4", ".m4a", ".m4v"):
        return False
    if not os.path.isfile(path):
        return False
    try:
        from mutagen.mp4 import MP4, MP4FreeForm
        try:
            # AtomDataType lives at mutagen.mp4.AtomDataType in
            # 1.45+; fall back to the magic constant if a younger
            # mutagen ever lacks it.
            from mutagen.mp4 import AtomDataType
            utf8_format = AtomDataType.UTF8
        except (ImportError, AttributeError):
            utf8_format = 1  # AtomDataType.UTF8
    except ImportError:
        return False

    outer = pack_serato_markers2(inner, target_size=target_size,
                                   line_width=line_width)
    envelope = (b"application/octet-stream\x00\x00"
                + b"Serato Markers2\x00" + outer)
    b64 = base64.b64encode(envelope)

    try:
        mp4 = MP4(path)
    except (MP4StreamInfoError, MutagenError, Exception):
        return False
    if mp4 is None:
        return False
    try:
        if mp4.tags is None:
            # File has no tag block yet — create one.  This does NOT
            # affect any other on-disk data because there's nothing
            # to lose in the ilst container.
            mp4.add_tags()
        # Replace ONLY the markersv2 atom.  Every other key in
        # mp4.tags survives this assignment + save() unchanged.
        mp4.tags["----:com.serato.dj:markersv2"] = [
            MP4FreeForm(b64, dataformat=utf8_format)]
        mp4.save()
        return True
    except Exception:
        return False


def write_serato_markers2_any(path: str, inner: bytes,
                                target_size: int = 0,
                                line_width: int = 72,
                                *, confirm: bool = False) -> bool:
    """Container-aware Markers2 writer.  Dispatches by file extension:

      * MP3 / WAV / AIFF → `write_serato_markers2` (ID3 GEOB)
      * MP4 / M4A / M4V  → `write_serato_markers2_mp4` (freeform atom)
      * FLAC / OGG / Opus → `write_serato_vorbis_blob`
        (`SERATO_MARKERS_V2` Vorbis comment)

    Returns False on extensions outside that set.

    Caller MUST pass `confirm=True`; without it `RuntimeError` is
    raised before any write.
    """
    _require_confirm(confirm, "write_serato_markers2_any", path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp3", ".wav", ".aiff", ".aif"):
        return write_serato_markers2(path, inner,
                                      target_size=target_size,
                                      line_width=line_width,
                                      confirm=True)
    if ext in (".mp4", ".m4a", ".m4v"):
        return write_serato_markers2_mp4(path, inner,
                                          target_size=target_size,
                                          line_width=line_width,
                                          confirm=True)
    if ext in (".flac", ".ogg", ".oga", ".opus"):
        # FLAC stores Markers2 in a SERATO_MARKERS_V2 Vorbis comment.
        # Use the same outer pack as the other containers so caller
        # supplies just the inner stream — pack adds the outer
        # u8/u8 + base64 + NUL padding.
        outer = pack_serato_markers2(inner,
                                       target_size=target_size,
                                       line_width=line_width)
        return write_serato_vorbis_blob(path, "Serato Markers2",
                                          outer, confirm=True)
    return False


# =====================================================================
# Serato Markers_ V1 (Scratch-Live-legacy) — encode + parse + writer
# =====================================================================
# Empirically (Serato sanity-test
# on a curation workflow's files), writing ONLY Markers2 makes
# modern Serato Pro silently drop most cue/loop entries.  Reason:
# Serato writes V1 + V2 in parallel — V2 is the modern data carrier
# (8 cues, 8 loops, labels, locked-flag), V1 is the Scratch-Live-era
# legacy block (5 cues + 9 loops, no labels, no locked-flag) which
# Serato Pro evidently STILL CONSULTS as a presence-signal.
#
# Without V1, Serato Pro treats the track as "not-fully-serato-tagged"
# and falls back to a partial-recovery heuristic.  Adding V1 alongside
# V2 → full conformity → all cues/loops show up.
#
# V1 byte-layout (276 bytes, exact match to this MP4 library):
#   6 B    header: u16 BE version (0x0205) + u32 BE count (14)
#   95 B   5 cue-slots × 19 B each
#   171 B  9 loop-slots × 19 B each
#   4 B    trailer: NUL + 3-byte track-tile-color RGB
#
# Each 19-byte row (decoded from this a sample MP4 file dump):
#   [0]      0x00 (data) or 0xFF (empty)
#   [1..3]   3-byte BE pos_ms / start_ms (24-bit, max ~4.6 h)
#   [4..7]   u32 BE end_ms (0xFFFFFFFF for cues + empty slots)
#   [8]      NUL
#   [9..12]  0xFFFFFFFF (reserved — locked-flag area but always 0xFF
#            in a real Serato library; V1 has no locked-state, just the slot)
#   [13]     NUL
#   [14..16] 3-byte RGB color
#   [17]     type-byte: 0x01 = cue-slot, 0x03 = loop-slot
#   [18]     NUL
#
# Slot 0-4 are cue-slots (type=0x01).  Slot 5-13 are loop-slots
# (type=0x03).  type-byte persists even on empty slots — Serato uses
# it to determine "which kind of slot is this" without re-deriving
# from index.
#
# Atom-name on MP4: `----:com.serato.dj:markers` (no underscore,
# verified via this mp4.tags.keys() dump ).
# On MP3/AIFF: ID3 GEOB "Serato Markers_" (different code path).

_V1_VERSION = 0x0205
_V1_ROW_LEN = 19            # unmasked (MP4 atom) row width
_V1_MASKED_ROW_LEN = 22     # 7-bit-safe (MP3/WAV/AIFF ID3) row width
_V1_N_CUE_SLOTS = 5
_V1_N_LOOP_SLOTS = 9
_V1_N_SLOTS = _V1_N_CUE_SLOTS + _V1_N_LOOP_SLOTS  # 14
_V1_HEADER_LEN = 6
_V1_TRAILER_LEN = 4
_V1_BLOB_LEN = (_V1_HEADER_LEN
                 + _V1_N_SLOTS * _V1_ROW_LEN
                 + _V1_TRAILER_LEN)  # 276


def _v1_pack_cue_row(pos_ms: int,
                       color_rgb: Tuple[int, int, int]) -> bytes:
    """Pack one populated V1 cue row (19 bytes)."""
    r, g, b = (int(c) & 0xFF for c in color_rgb)
    pos_be = (int(pos_ms) & 0xFFFFFF).to_bytes(3, "big")
    return (
        b"\x00"
        + pos_be
        + b"\xff\xff\xff\xff"   # end_ms = unused for cue
        + b"\x00"
        + b"\xff\xff\xff\xff"
        + b"\x00"
        + bytes([r, g, b])
        + b"\x01"               # type = cue
        + b"\x00"
    )


def _v1_pack_loop_row(start_ms: int, end_ms: int,
                       color_rgb: Tuple[int, int, int],
                       locked: bool = False) -> bytes:
    """Pack one populated V1 loop row (19 bytes).

    `locked`: writes byte [18] = 0x01 if True, else 0x00 — the raw
    (MP4) analogue of the masked row's byte [21]. Verified against a
    real Serato MP4 V1 blob whose loop rows carry 0x01 at [18] exactly
    where the loop is locked (cue rows are always 0x00 there)."""
    r, g, b = (int(c) & 0xFF for c in color_rgb)
    start_be = (int(start_ms) & 0xFFFFFF).to_bytes(3, "big")
    end_be = struct.pack(">I", int(end_ms) & 0xFFFFFFFF)
    return (
        b"\x00"
        + start_be
        + end_be
        + b"\x00"
        + b"\xff\xff\xff\xff"
        + b"\x00"
        + bytes([r, g, b])
        + b"\x03"                              # type = loop
        + (b"\x01" if locked else b"\x00")     # [18] locked flag
    )


def _v1_pack_empty_row(*, is_cue: bool) -> bytes:
    """Pack one empty V1 row (19 bytes).  Type-byte persists so the
    slot is still typed as cue-slot vs loop-slot."""
    type_byte = b"\x01" if is_cue else b"\x03"
    return (
        b"\xff"
        + b"\xff\xff\xff"
        + b"\xff\xff\xff\xff"
        + b"\x00"
        + b"\xff\xff\xff\xff"
        + b"\x00"
        + b"\x00\x00\x00"
        + type_byte
        + b"\x00"
    )


def encode_markers_v1_inner(
        cues: Optional[List[dict]] = None,
        loops: Optional[List[dict]] = None,
        *,
        track_color_rgb: Optional[Tuple[int, int, int]] = None,
        ) -> bytes:
    """Encode a complete Serato Markers_ V1 blob (276 bytes).

    Returns the raw blob bytes — for MP4 they go through the
    `application/octet-stream\\x00\\x00Serato Markers_\\x00<blob>`
    envelope + base64, handled by `write_serato_markers_v1_mp4`.

    `cues` / `loops` are lists of dicts (same shape as
    `parse_serato_markers2_full`).  Only the first 5 cues sorted by
    idx are encoded into V1 cue-slots; only the first 9 loops are
    encoded into V1 loop-slots.  Entries beyond those counts have NO
    V1 representation (they fit only in V2) — Serato Pro accepts
    that, since V2 carries the authoritative modern 8-slot data.

    `track_color_rgb` is the V1 trailer's 3-byte RGB tile-color.
    Default (0,0,0) → trailer = `\\x00\\x00\\x00\\x00` ("no tile
    color set"; equivalent to Serato's default for freshly-tagged
    tracks).

    `locked` on input loops IS honoured — the raw (MP4) loop row carries
    the locked flag in its trailing byte [18] (0x01 = locked), the
    analogue of the masked row's byte [21]. Verified against a real
    Serato MP4 V1 blob.

    Pure function — no IO.
    """
    cues_sorted = sorted(cues or [], key=lambda d: int(d.get("idx", 0)))
    loops_sorted = sorted(loops or [], key=lambda d: int(d.get("idx", 0)))

    def _rgb(c):
        if c is None:
            return (0, 0, 0)
        if isinstance(c, str):
            s = c.strip().lstrip("#")
            if len(s) >= 6:
                return (int(s[0:2], 16), int(s[2:4], 16),
                        int(s[4:6], 16))
            return (0, 0, 0)
        if isinstance(c, (tuple, list)) and len(c) >= 3:
            return (int(c[0]), int(c[1]), int(c[2]))
        return (0, 0, 0)

    out = bytearray()
    out += struct.pack(">H", _V1_VERSION)
    out += struct.pack(">I", _V1_N_SLOTS)

    # Cue slots (0..4)
    for i in range(_V1_N_CUE_SLOTS):
        if i < len(cues_sorted):
            cue = cues_sorted[i]
            out += _v1_pack_cue_row(
                pos_ms=int(cue.get("pos_ms", 0)),
                color_rgb=_rgb(cue.get("color")))
        else:
            out += _v1_pack_empty_row(is_cue=True)

    # Loop slots (5..13)
    for i in range(_V1_N_LOOP_SLOTS):
        if i < len(loops_sorted):
            loop = loops_sorted[i]
            out += _v1_pack_loop_row(
                start_ms=int(loop.get("start_ms", 0)),
                end_ms=int(loop.get("end_ms", 0)),
                color_rgb=_rgb(loop.get("color")),
                locked=bool(loop.get("locked", False)))
        else:
            out += _v1_pack_empty_row(is_cue=False)

    # Trailer: NUL + 3-byte RGB (track tile color)
    if track_color_rgb is None:
        out += b"\x00\x00\x00\x00"
    else:
        tr, tg, tb = (int(c) & 0xFF for c in track_color_rgb)
        out += bytes([0, tr, tg, tb])

    assert len(out) == _V1_BLOB_LEN, \
        f"V1 blob size mismatch: {len(out)} != {_V1_BLOB_LEN}"
    return bytes(out)


def parse_serato_markers_v1_full(data: bytes) -> dict:
    """Parse a Serato Markers_ V1 blob slot-by-slot.

    Extension of the legacy `parse_serato_markers_v1` (which only
    returned header + raw entry bytes).  This version decodes each
    slot into the same dict-shape `parse_serato_markers2_full` uses,
    so callers can field-compare V1 vs V2.

    Returns:
        {
          "version":      int,    # raw u16 (0x0205)
          "count":        int,    # slot count from header
          "cues":         [{idx, pos_ms, pos_sec, color}, ...],
          "loops":        [{idx, start_ms, end_ms, start_sec,
                              end_sec, color}, ...],
          "track_color":  (r, g, b) | None,
        }

    Loop `idx` values are normalized to 0..(N-1) of the LOOP slots
    (i.e. V1 slot 5 → idx 0, slot 13 → idx 8), so they line up with
    the V2 schema's loop idx 0..7 (V1 supports 9, V2 supports 8;
    overlap is 0..7).  Cue `idx` matches V1 slot 0..4 == V2 cue
    idx 0..4 directly.
    """
    out = {"version": None, "count": None,
           "cues": [], "loops": [], "track_color": None}
    if not data or len(data) < _V1_HEADER_LEN:
        return out
    try:
        version = struct.unpack(">H", data[:2])[0]
        count = struct.unpack(">I", data[2:6])[0]
    except Exception:
        return out
    out["version"] = int(version)
    out["count"] = int(count)
    # MP3/WAV/AIFF store the V1 blob 7-bit-safe (every byte <= 0x7F, u32
    # fields packed as 5-or-4 bytes of 7 payload bits); MP4 stores it raw.
    masked = all(b <= 0x7F for b in data)
    row_len = _V1_MASKED_ROW_LEN if masked else _V1_ROW_LEN
    body_end = _V1_HEADER_LEN + count * row_len
    if len(data) < body_end:
        return out
    for slot in range(count):
        row = data[_V1_HEADER_LEN + slot * row_len:
                    _V1_HEADER_LEN + (slot + 1) * row_len]
        if len(row) < row_len:
            break
        if masked:
            # 7-bit-safe row (22 B): [0:5] pos / loop-start, [5:10] loop-end,
            # [10] sep, [11:16] reserved, [16:20] colour (0x00RRGGBB),
            # [20] type (0x01 cue / 0x03 loop), [21] trailing.  An empty
            # slot encodes its position as 7f7f7f7f7f (== 0xFFFFFFFF).
            if row[0] == 0x7F:
                continue
            type_byte = row[20]
            col = _decode_4byte_7bit_safe(row[16:20])
            color = ((col >> 16) & 0xFF, (col >> 8) & 0xFF, col & 0xFF)
            start_ms = _decode_5byte_7bit_safe(row[0:5]) & 0xFFFFFFFF
            end_ms = _decode_5byte_7bit_safe(row[5:10]) & 0xFFFFFFFF
        else:
            # Raw row (19 B): [1:4] 3-byte pos, [4:8] loop-end u32,
            # [14:17] colour, [17] type, [18] loop locked flag.
            if row[0] == 0xFF:
                continue
            type_byte = row[17]
            color = (row[14], row[15], row[16])
            start_ms = (row[1] << 16) | (row[2] << 8) | row[3]
            try:
                end_ms = struct.unpack(">I", row[4:8])[0]
            except Exception:
                end_ms = 0
        if type_byte == 0x01:
            out["cues"].append({
                "idx": int(slot),
                "pos_ms": int(start_ms),
                "pos_sec": start_ms / 1000.0,
                "color": color,
            })
        elif type_byte == 0x03:
            loop = {
                # Normalize loop idx to 0..(N_LOOP-1) so it matches
                # V2's loop idx convention.
                "idx": int(slot) - _V1_N_CUE_SLOTS,
                "start_ms": int(start_ms),
                "end_ms": int(end_ms),
                "start_sec": start_ms / 1000.0,
                "end_sec": end_ms / 1000.0,
                "color": color,
            }
            # Both V1 variants carry the per-loop locked flag in the
            # row's trailing byte — masked (MP3/WAV/AIFF) at [21], raw
            # (MP4) at [18] — so a decoded V1 loop re-encodes
            # byte-for-byte via the matching packer.
            loop["locked"] = bool(row[21] if masked else row[18])
            out["loops"].append(loop)
    # Trailer: NUL + 3-byte RGB (a 7-bit-safe 4-byte colour when masked).
    trailer = data[body_end:body_end + _V1_TRAILER_LEN]
    if len(trailer) >= 4:
        if masked:
            tc = _decode_4byte_7bit_safe(trailer[:4])
            out["track_color"] = ((tc >> 16) & 0xFF,
                                  (tc >> 8) & 0xFF, tc & 0xFF)
        else:
            out["track_color"] = (trailer[1], trailer[2], trailer[3])
    return out


def write_serato_markers_v1_mp4(path: str, v1_blob: bytes,
                                  *, confirm: bool = False) -> bool:
    """Write a V1 Markers_ blob to an MP4 freeform atom
    `----:com.serato.dj:markers`.  Companion to
    `write_serato_markers2_mp4` — together they make a Serato-full
    tag set.  See `write_serato_markers_full_mp4` for the single-save
    combined writer.

    Preserves every other tag on the file (a real Serato library's
    preservation-requirement ).

    Confirm-flag check.
    """
    _require_confirm(confirm, "write_serato_markers_v1_mp4", path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".mp4", ".m4a", ".m4v"):
        return False
    if not os.path.isfile(path):
        return False
    try:
        from mutagen.mp4 import MP4, MP4FreeForm
        try:
            from mutagen.mp4 import AtomDataType
            utf8_format = AtomDataType.UTF8
        except (ImportError, AttributeError):
            utf8_format = 1
    except ImportError:
        return False
    envelope = (b"application/octet-stream\x00\x00"
                + b"Serato Markers_\x00" + v1_blob)
    b64 = base64.b64encode(envelope)
    try:
        mp4 = MP4(path)
    except (MP4StreamInfoError, MutagenError, Exception):
        return False
    if mp4 is None:
        return False
    try:
        if mp4.tags is None:
            mp4.add_tags()
        mp4.tags["----:com.serato.dj:markers"] = [
            MP4FreeForm(b64, dataformat=utf8_format)]
        mp4.save()
        return True
    except Exception:
        return False


def _resolve_markers_for_write(path, *, cues, loops, color_rgb,
                                bpm_lock, flips, preserve):
    """Resolve the effective cue/loop set and the V2 inner stream for a
    full-marker write.

    With ``preserve=True`` (default) and an existing Serato Markers2
    blob on ``path``, the V2 stream is produced by
    ``flip_writer.merge_markers2_inner`` — AUTOGAIN, unknown/future
    entries, FLIPs and BPMLOCK survive unless the caller explicitly
    overrides them (a non-None arg replaces that type; None leaves it
    untouched).  V1 is rebuilt from the EFFECTIVE cue/loop set
    (caller's where given, else the existing ones) so V1 and V2 stay
    consistent.

    With ``preserve=False``, or when no existing Markers2 data is
    present, this is the green-field path: build V2 from scratch and
    use the caller's cues/loops verbatim for V1.

    Returns ``(eff_cues, eff_loops, v2_inner)``.
    """
    existing_outer = None
    if preserve:
        try:
            for desc, payload, _src in harvest_serato_blobs(path):
                if desc == "Serato Markers2":
                    existing_outer = payload
                    break
        except Exception:
            existing_outer = None

    if preserve and existing_outer:
        from sidecaramel.flip_writer import decode_outer, merge_markers2_inner
        try:
            existing_inner, _ = decode_outer(existing_outer)
        except Exception:
            existing_inner = None
        if existing_inner is not None:
            try:
                parsed = parse_serato_markers2_full(existing_outer)
            except Exception:
                parsed = {}
            eff_cues = cues if cues is not None else parsed.get("cues")
            eff_loops = loops if loops is not None else parsed.get("loops")
            v2_inner = merge_markers2_inner(
                existing_inner, cues=cues, loops=loops,
                color_rgb=color_rgb, bpm_lock=bpm_lock, flips=flips)
            return eff_cues, eff_loops, v2_inner

    v2_inner = build_markers2_inner(
        cues=cues, loops=loops, color_rgb=color_rgb,
        bpm_lock=bpm_lock, flips=flips)
    return cues, loops, v2_inner


def write_serato_markers_full_mp4(
        path: str, *,
        cues: Optional[List[dict]] = None,
        loops: Optional[List[dict]] = None,
        color_rgb: Optional[Tuple[int, int, int]] = None,
        bpm_lock: Optional[bool] = None,
        flips: Optional[List[dict]] = None,
        track_color_rgb: Optional[Tuple[int, int, int]] = None,
        target_size: int = 0,
        line_width: int = 72,
        preserve: bool = True,
        confirm: bool = False) -> bool:
    """Write a Serato-complete V1+V2 marker pair to an MP4 file in
    a SINGLE mutagen save — atomically updating both
    `----:com.serato.dj:markers` and
    `----:com.serato.dj:markersv2`.

    This is what Serato Pro writes itself.  Modern Serato Pro
    requires BOTH atoms to be present — V2 alone causes silent
    partial-recovery (cue points 1-5 and most loops go
    missing was the empirical signal).

    Preserves every other tag on the file (a real Serato library's
    preservation-requirement ): iTunes-standard tags,
    other Serato atoms (beatgrid, autotags, overview, …), and
    arbitrary non-Serato freeform atoms all survive untouched.

    `cues` / `loops` are dict-lists (same shape as
    `parse_serato_markers2_full` returns) — feed them in directly
    from a read-side roundtrip or from downstream-consumer plan JSON.

    `color_rgb`     — V2 COLOR entry (track-card color, optional).
    `bpm_lock`      — V2 BPMLOCK entry; default False.  None omits
                       the entry.
    `track_color_rgb` — V1 trailer's 3-byte RGB tile-color.
                          Default (0,0,0) = no tile color.

    Confirm-flag check: caller MUST pass `confirm=True`.

    audio-write policy reminder: only call on NEW files or pre-cleared paths.
    """
    _require_confirm(confirm, "write_serato_markers_full_mp4", path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".mp4", ".m4a", ".m4v"):
        return False
    if not os.path.isfile(path):
        return False

    try:
        from mutagen.mp4 import MP4, MP4FreeForm
        try:
            from mutagen.mp4 import AtomDataType
            utf8_format = AtomDataType.UTF8
        except (ImportError, AttributeError):
            utf8_format = 1
    except ImportError:
        return False

    # Resolve V2 (preservation-merged against any existing Markers2)
    # and the effective cue/loop set for V1.
    eff_cues, eff_loops, v2_inner = _resolve_markers_for_write(
        path, cues=cues, loops=loops, color_rgb=color_rgb,
        bpm_lock=bpm_lock, flips=flips, preserve=preserve)

    # Build V1 blob and wrap envelope + base64.
    v1_blob = encode_markers_v1_inner(
        cues=eff_cues, loops=eff_loops, track_color_rgb=track_color_rgb)
    v1_envelope = (b"application/octet-stream\x00\x00"
                   + b"Serato Markers_\x00" + v1_blob)
    v1_b64 = base64.b64encode(v1_envelope)

    # V2 outer + envelope + base64.
    v2_outer = pack_serato_markers2(v2_inner, target_size=target_size,
                                      line_width=line_width)
    v2_envelope = (b"application/octet-stream\x00\x00"
                   + b"Serato Markers2\x00" + v2_outer)
    v2_b64 = base64.b64encode(v2_envelope)

    try:
        mp4 = MP4(path)
    except (MP4StreamInfoError, MutagenError, Exception):
        return False
    if mp4 is None:
        return False
    try:
        if mp4.tags is None:
            mp4.add_tags()
        mp4.tags["----:com.serato.dj:markers"] = [
            MP4FreeForm(v1_b64, dataformat=utf8_format)]
        mp4.tags["----:com.serato.dj:markersv2"] = [
            MP4FreeForm(v2_b64, dataformat=utf8_format)]
        mp4.save()
        return True
    except Exception:
        return False


# =====================================================================
# Serato Markers_ V1 — MP3 / WAV / AIFF variant (Scratch-Live-era)
# =====================================================================
#
# MP3 / WAV / AIFF V1 differs from the MP4 V1 format (see earlier
# section).  Confirmed by byte-equal decoding of a reference .mp3
# V1 blob (318 B) against its V2 ground-truth cue positions:
#
#   V2 cue 0 pos_ms = 35       V1 5-byte field = 00 00 00 00 23   ✓
#   V2 cue 1 pos_ms = 99,042   V1 5-byte field = 00 00 06 05 62   ✓
#   V2 cue 2 pos_ms = 104,869  V1 5-byte field = 00 00 06 33 25   ✓
#   V2 cue 3 pos_ms = 107,630  V1 5-byte field = 00 00 06 48 6e   ✓
#   V2 loop 0 start = 54       V1 5-byte field = 00 00 00 00 36   ✓
#   V2 loop 0 end   = 3,680    V1 5-byte field = 00 00 00 1c 60   ✓
#
# Layout (318 B total for the default 14-slot table):
#   6 B    header: u16 BE version (0x0205) + u32 BE count (14)
#   110 B  5 cue-slots × 22 B each
#   198 B  9 loop-slots × 22 B each
#   4 B    trailer: NUL + 3-byte track-tile-color RGB
#                                                 6+110+198+4 = 318
#
# Per 22-byte row:
#   [0..4]    5-byte 7-bit-safe pos_ms (cue) / start_ms (loop)
#   [5..9]    5-byte 7-bit-safe end_ms — for cues all 0x7F (unused);
#             for loops the real end position
#   [10]      0x00 separator (NUL)
#   [11..15]  5 bytes 0x7F-padded reserved (locked-flag area?
#             observed 0x7F-padded across a real Serato library; V1 has no
#             explicit locked-state per-row)
#   [16]      cue-with-data: 0x00     ; loop-with-data: 0x01
#             empty-cue-slot: 0x00    ; empty-loop-slot: 0x00
#   [17..19]  3-byte RGB color (0x00..02 for empty)
#   [20]      type byte: 0x01 = cue-slot, 0x03 = loop-slot
#             empty-cue-slot: 0x00 (atypical — different from filled)
#             empty-loop-slot: 0x03
#   [21]      cue: 0x00 ; loop-with-data: 0x01 ; empty-loop: 0x00
#
# 5-byte 7-bit-safe encoding (per 5-byte field):
#   Each byte holds 7 payload bits (high bit always 0).  5 bytes
#   together encode a 35-bit integer, MSB-first:
#     decoded = (b0 << 28) | (b1 << 21) | (b2 << 14) | (b3 << 7) | b4
#   For typical millisecond positions (< 2^28 ms = ~3 days) the high
#   bytes are 0x00.
#
# Why this format exists: MP3 ID3 tags can mis-interpret bytes with
# the high bit set (sync-word ambiguity, mid-stream).  Serato's
# legacy Scratch-Live writer kept ALL payload bytes ≤ 0x7F for
# safety — same reason Markers2's outer wrap is base64-encoded.
# MP4 atoms don't have this constraint, hence MP4 V1 uses the
# cleaner 19-byte rows with 4-byte u32 BE positions.
#
# Historical context: the 5 cue / 9 loop split
# is the direct Hardware-Map of the Vestax VCI-300 (Scratch-Live-era
# controller with 5 cue buttons + 9 loop buttons per deck).

_V1_MP3_ROW_LEN = 22
_V1_MP3_BLOB_LEN = (_V1_HEADER_LEN
                     + _V1_N_SLOTS * _V1_MP3_ROW_LEN
                     + _V1_TRAILER_LEN)  # 318


def _encode_5byte_7bit_safe(n: int) -> bytes:
    """Encode an integer as 5 bytes, each holding 7 payload bits
    (high bit cleared, MSB-first).  Up to 2^35 - 1 representable.

    >>> _encode_5byte_7bit_safe(35).hex(" ")
    '00 00 00 00 23'
    >>> _encode_5byte_7bit_safe(99042).hex(" ")
    '00 00 06 05 62'
    """
    n &= (1 << 35) - 1
    return bytes([
        (n >> 28) & 0x7F,
        (n >> 21) & 0x7F,
        (n >> 14) & 0x7F,
        (n >>  7) & 0x7F,
        n & 0x7F,
    ])


def _decode_5byte_7bit_safe(b: bytes) -> int:
    """Inverse of `_encode_5byte_7bit_safe`."""
    if len(b) != 5:
        raise ValueError(f"expected 5 bytes, got {len(b)}")
    return ((b[0] << 28) | (b[1] << 21) | (b[2] << 14)
             | (b[3] << 7) | b[4])


def _encode_4byte_7bit_safe(n: int) -> bytes:
    """Encode a 28-bit-or-less integer as 4 bytes, each holding 7
    payload bits MSB-first.  Used for V1 RGB color (24-bit value)
    and the V1 trailer's track-tile color (24-bit RGB + 4-bit prefix).

    Verified against a V1 slot-1 colour: bytes `06 32 10 00` decode
    to 0xCC8800, i.e. the V2 colour (204, 136, 0).
    """
    n &= (1 << 28) - 1
    return bytes([
        (n >> 21) & 0x7F,
        (n >> 14) & 0x7F,
        (n >>  7) & 0x7F,
        n & 0x7F,
    ])


def _decode_4byte_7bit_safe(b: bytes) -> int:
    """Inverse of `_encode_4byte_7bit_safe`: reassemble a 28-bit integer
    from 4 bytes of 7 payload bits each, MSB-first.  Used to read the V1
    masked RGB colour (`06 30 00 00` -> 0x00CC0000 -> (204, 0, 0))."""
    if len(b) < 4:
        return 0
    return (((b[0] & 0x7F) << 21) | ((b[1] & 0x7F) << 14)
            | ((b[2] & 0x7F) << 7) | (b[3] & 0x7F))


def _v1_mp3_pack_cue_row(pos_ms: int,
                          color_rgb: Tuple[int, int, int]) -> bytes:
    """Pack one populated MP3-V1 cue row (22 bytes).

    Layout verified against a reference file's V1 slots 0-3.
    """
    r, g, b = (int(c) & 0xFF for c in color_rgb)
    rgb24 = (r << 16) | (g << 8) | b
    return (
        _encode_5byte_7bit_safe(int(pos_ms))   # [0..4] pos
        + b"\x7f\x7f\x7f\x7f\x7f"               # [5..9] end_ms unused
        + b"\x00"                                # [10] NUL
        + b"\x7f\x7f\x7f\x7f\x7f"               # [11..15] reserved pad
        + _encode_4byte_7bit_safe(rgb24)         # [16..19] RGB
        + b"\x01"                                # [20] cue-type
        + b"\x00"                                # [21] terminator
    )


def _v1_mp3_pack_loop_row(start_ms: int, end_ms: int,
                            color_rgb: Tuple[int, int, int],
                            locked: bool = False) -> bytes:
    """Pack one populated MP3-V1 loop row (22 bytes).

    `locked`: writes byte [21] = 0x01 if True (this Kingdom-Come
    pattern, both loops were locked); 0x00 if False — V1 has
    limited per-row locked-state, treat as best-effort.
    """
    r, g, b = (int(c) & 0xFF for c in color_rgb)
    rgb24 = (r << 16) | (g << 8) | b
    return (
        _encode_5byte_7bit_safe(int(start_ms))   # [0..4] start
        + _encode_5byte_7bit_safe(int(end_ms))   # [5..9] end
        + b"\x00"                                 # [10] NUL
        + b"\x7f\x7f\x7f\x7f\x7f"                # [11..15] reserved
        + _encode_4byte_7bit_safe(rgb24)          # [16..19] RGB
        + b"\x03"                                 # [20] loop-type
        + (b"\x01" if locked else b"\x00")        # [21]
    )


# Empty-slot patterns are byte-for-byte copies of the empty slots
# observed in a reference file (slot 4 = empty cue; 7-13 = loops).
# Hard-coded as constants since the byte layout differs subtly from
# the filled-slot template — Serato Pro is byte-sensitive here.
_V1_MP3_EMPTY_CUE_ROW = bytes.fromhex(
    "00 7f 7f 7f 7f 7f 7f 7f 7f 7f 7f 00 7f 7f 7f 7f 7f 00 00 00 00 00"
    .replace(" ", ""))
_V1_MP3_EMPTY_LOOP_ROW = bytes.fromhex(
    "7f 7f 7f 7f 7f 7f 7f 7f 7f 7f 00 7f 7f 7f 7f 7f 00 00 00 00 03 00"
    .replace(" ", ""))


def _v1_mp3_pack_empty_cue_row() -> bytes:
    return _V1_MP3_EMPTY_CUE_ROW


def _v1_mp3_pack_empty_loop_row() -> bytes:
    return _V1_MP3_EMPTY_LOOP_ROW


def encode_markers_v1_mp3_inner(
        cues: Optional[List[dict]] = None,
        loops: Optional[List[dict]] = None,
        *,
        track_color_rgb: Optional[Tuple[int, int, int]] = None,
        ) -> bytes:
    """Encode a complete Serato Markers_ V1 blob in the MP3 / WAV /
    AIFF variant (318 bytes, 22-byte rows with 5-byte-7-bit-safe
    pos_ms encoding).

    Use this in ID3-GEOB writes ("Serato Markers_" descriptor) for
    MP3/WAV/AIFF.  For MP4/M4A/M4V V1, use `encode_markers_v1_inner`
    (the 19-byte-row variant).

    `cues` / `loops` are the same dict-shape as
    `parse_serato_markers2_full` returns.  Only the first 5 cues
    and first 9 loops get V1 slots (Scratch-Live legacy limit).

    `track_color_rgb` is the V1 4-byte trailer's tile color.

    `locked` on loops is best-effort (V1 has no explicit per-row
    locked-state — see byte 21 of `_v1_mp3_pack_loop_row`).
    """
    cues_sorted = sorted(cues or [], key=lambda d: int(d.get("idx", 0)))
    loops_sorted = sorted(loops or [], key=lambda d: int(d.get("idx", 0)))

    def _rgb(c):
        if c is None: return (0, 0, 0)
        if isinstance(c, str):
            s = c.strip().lstrip("#")
            if len(s) >= 6:
                return (int(s[0:2], 16), int(s[2:4], 16),
                        int(s[4:6], 16))
            return (0, 0, 0)
        if isinstance(c, (tuple, list)) and len(c) >= 3:
            return (int(c[0]), int(c[1]), int(c[2]))
        return (0, 0, 0)

    out = bytearray()
    out += struct.pack(">H", _V1_VERSION)
    out += struct.pack(">I", _V1_N_SLOTS)

    # Cue slots (0..4)
    for i in range(_V1_N_CUE_SLOTS):
        if i < len(cues_sorted):
            c = cues_sorted[i]
            out += _v1_mp3_pack_cue_row(
                pos_ms=int(c.get("pos_ms", 0)),
                color_rgb=_rgb(c.get("color")))
        else:
            out += _v1_mp3_pack_empty_cue_row()

    # Loop slots (5..13)
    for i in range(_V1_N_LOOP_SLOTS):
        if i < len(loops_sorted):
            l = loops_sorted[i]
            out += _v1_mp3_pack_loop_row(
                start_ms=int(l.get("start_ms", 0)),
                end_ms=int(l.get("end_ms", 0)),
                color_rgb=_rgb(l.get("color")),
                locked=bool(l.get("locked", False)))
        else:
            out += _v1_mp3_pack_empty_loop_row()

    # Trailer: 4-byte 7-bit-safe encoded RGB (same scheme as the
    # color field in each row).  Verified against a reference
    # trailer, `05 6e 77 3b`, which decodes to 0xBBBBBB.
    if track_color_rgb is None:
        out += b"\x00\x00\x00\x00"
    else:
        tr, tg, tb = (int(c) & 0xFF for c in track_color_rgb)
        out += _encode_4byte_7bit_safe((tr << 16) | (tg << 8) | tb)

    assert len(out) == _V1_MP3_BLOB_LEN, \
        f"V1-MP3 blob size: {len(out)} != {_V1_MP3_BLOB_LEN}"
    return bytes(out)


def write_serato_markers_v1_mp3(path: str, v1_blob: bytes,
                                  *, confirm: bool = False) -> bool:
    """Write a V1 Markers_ blob to an MP3 / WAV / AIFF file's ID3
    GEOB frame "Serato Markers_".  Companion to
    `write_serato_markers2` (V2-only).  See
    `write_serato_markers_full_mp3` for the combined V1+V2 writer.

    Confirm-flag check.  Preserves every other ID3 frame.
    """
    _require_confirm(confirm, "write_serato_markers_v1_mp3", path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".mp3", ".wav", ".aiff", ".aif"):
        return False
    return write_serato_geob(path, "Serato Markers_", v1_blob,
                              confirm=True)


def write_serato_markers_full_mp3(
        path: str, *,
        cues: Optional[List[dict]] = None,
        loops: Optional[List[dict]] = None,
        color_rgb: Optional[Tuple[int, int, int]] = None,
        bpm_lock: Optional[bool] = None,
        flips: Optional[List[dict]] = None,
        track_color_rgb: Optional[Tuple[int, int, int]] = None,
        target_size: int = 0,
        line_width: int = 72,
        preserve: bool = True,
        confirm: bool = False) -> bool:
    """Write the complete V1 + V2 Markers pair to an MP3 / WAV / AIFF
    file — ID3 counterpart of `write_serato_markers_full_mp4`.

    Modern Serato Pro on MP3 has the same expectation as on MP4:
    V1 must EXIST for V2 to be fully honoured.  This writer adds BOTH
    GEOB frames to the ID3 tag and persists them in a SINGLE
    `mutagen.save()`, so a failure cannot leave V1 written without V2.
    (The underlying audio write is still in-place — there is no
    tempfile+`os.replace` for audio tags yet; see the README
    "Known limitations".)

    Preserves every other ID3 frame.

    Confirm-flag check.
    """
    _require_confirm(confirm, "write_serato_markers_full_mp3", path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".mp3", ".wav", ".aiff", ".aif"):
        return False
    if not os.path.isfile(path):
        return False

    # Resolve V2 (preservation-merged against any existing Markers2)
    # and the effective cue/loop set for V1.
    eff_cues, eff_loops, v2_inner = _resolve_markers_for_write(
        path, cues=cues, loops=loops, color_rgb=color_rgb,
        bpm_lock=bpm_lock, flips=flips, preserve=preserve)
    v1_blob = encode_markers_v1_mp3_inner(
        cues=eff_cues, loops=eff_loops, track_color_rgb=track_color_rgb)
    v2_outer = pack_serato_markers2(v2_inner, target_size=target_size,
                                      line_width=line_width)

    # Single-save atomicity: add BOTH GEOB frames to the ID3 tag,
    # then ONE save().  (Two separate write_serato_geob() calls would
    # save V1-then-V2, and a failure between the two saves would leave
    # a V1-only file — the exact inconsistency this writer prevents.)
    try:
        from mutagen.id3 import GEOB
    except ImportError:
        return False
    try:
        c = _open_id3_container(path)
    except Exception:
        return False
    tags = c.tags
    try:
        for descriptor, payload in (("Serato Markers_", v1_blob),
                                     ("Serato Markers2", v2_outer)):
            # Wipe any existing GEOB with this exact descriptor first.
            for key in list(tags.keys()):
                ks = str(key)
                if ks.startswith("GEOB:") and descriptor in ks:
                    tags.delall(key)
            tags.add(GEOB(encoding=0, mime="application/octet-stream",
                           filename="", desc=descriptor, data=payload))
        c.save()
        return True
    except Exception:
        return False


# =====================================================================
# Container-agnostic Serato-blob writers — generic encoders + dispatch
# =====================================================================
# an external batch job needs to write Serato blobs into
# brand-new FLAC files (fresh downloads, explicit exception
# for new files).  Existing writers were MP3/MP4-only — this section
# adds:
#   - FLAC dispatch (Vorbis-comment with envelope+base64+line-wrap)
#   - BeatGrid encoder + container-aware writer
#   - Autotags encoder + writer
#   - Analysis encoder + writer
#
# All FLAC values use the same wrapper Serato writes itself:
#   serato_<name> = base64(
#     b"application/octet-stream\0\0Serato <Descriptor>\0<blob>")
#   line-wrapped every 72 chars.  Vorbis comment values are
#   case-insensitive; we emit lowercase.
#
# Serato Pro's "length ≡ 1 mod 4" base64 quirk applies on read; on
# write we emit standard 0-mod-4 padding (Serato Pro's loader accepts
# both forms).

# -----------------------------------------------------------------
# FLAC Vorbis-comment helpers (SPOT)
# -----------------------------------------------------------------
# Mapping from canonical descriptor → Vorbis-comment key.  Lowercase
# per  Mp3tag-screenshot.  Descriptor name does NOT
# always match the key tail (Autotags → autogain, VidAssoc →
# video_assoc, RelVolAd → relvol).

_DESCRIPTOR_TO_VORBIS_KEY = {
    "Serato Analysis":   "serato_analysis",
    "Serato Autotags":   "serato_autogain",
    "Serato BeatGrid":   "serato_beatgrid",
    "Serato Markers_":   "serato_markers",
    "Serato Markers2":   "serato_markers_v2",
    "Serato Overview":   "serato_overview",
    "Serato Playcount":  "serato_playcount",
    "Serato RelVolAd":   "serato_relvol",
    "Serato VidAssoc":   "serato_video_assoc",
    "Serato VideoAssoc": "serato_video_assoc",
}


def _descriptor_to_vorbis_key(descriptor: str) -> str:
    """Map a canonical Serato descriptor (e.g. 'Serato Markers2') to
    its Vorbis-comment key (e.g. 'serato_markers_v2')."""
    return _DESCRIPTOR_TO_VORBIS_KEY.get(
        descriptor,
        "serato_" + descriptor.replace("Serato ", "").lower())


def _wrap_vorbis_value(descriptor: str, blob: bytes,
                         *, line_width: int = 72) -> str:
    """Wrap a raw Serato blob into a Vorbis-comment value string,
    matching what Serato writes itself.

    Output:  base64( application/octet-stream\\0\\0 Serato <Name>\\0
                      <blob> )  line-wrapped every `line_width` chars.

    Pure function — no IO.
    """
    envelope = (b"application/octet-stream\x00\x00"
                + descriptor.encode("ascii") + b"\x00" + blob)
    b64 = base64.b64encode(envelope)
    if line_width > 0:
        chunks = [b64[i:i + line_width]
                  for i in range(0, len(b64), line_width)]
        b64 = b"\n".join(chunks)
    return b64.decode("ascii")


def _open_vorbis_container(path: str):
    """Open a Vorbis-comment-bearing container and return the mutagen
    object.  Routes by extension:

        .flac          → mutagen.flac.FLAC
        .ogg / .oga    → mutagen.oggvorbis.OggVorbis
        .opus          → mutagen.oggopus.OggOpus
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".flac":
        return FLAC(path)
    if ext == ".opus":
        from mutagen.oggopus import OggOpus
        return OggOpus(path)
    if ext in (".ogg", ".oga"):
        from mutagen.oggvorbis import OggVorbis
        return OggVorbis(path)
    raise ValueError(f"unsupported Vorbis-container extension: {ext}")


def write_serato_vorbis_blob(path: str, descriptor: str, blob: bytes,
                              *, confirm: bool = False) -> bool:
    """Write a Serato blob into a Vorbis-comment-bearing container's
    tags.  Supports FLAC, OGG (Vorbis), and Opus.

    Targeted single-key assignment — every other Vorbis tag on the
    file is preserved (artist, album, custom keys, etc.).

    `descriptor` is the canonical name (e.g. 'Serato Markers2').
    `blob` is the raw binary payload (e.g. from `pack_serato_markers2`
    or `build_beatgrid`).

    Returns True on success.  Returns False for unsupported
    extensions or mutagen errors.  Caller MUST pass `confirm=True`;
    without it `RuntimeError` is raised before any write.
    """
    _require_confirm(confirm, "write_serato_vorbis_blob", path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".flac", ".ogg", ".oga", ".opus"):
        return False
    if not os.path.isfile(path):
        return False
    try:
        container = _open_vorbis_container(path)
    except Exception:
        return False
    try:
        key = _descriptor_to_vorbis_key(descriptor)
        container[key] = _wrap_vorbis_value(descriptor, blob)
        container.save()
        return True
    except Exception:
        return False


# -----------------------------------------------------------------
# BeatGrid encoder + writer
# -----------------------------------------------------------------
# Per the docstring "Serato binary formats — verified":
#
#   u8  major (1), u8 minor (0)
#   u32 BE num_markers (includes terminal)
#   (N-1) × non-terminal: f32 BE position_s, u32 BE beats_till_next
#   terminal: f32 BE position_s, f32 BE bpm
#   u8 footer
#
# A sample FLAC dump confirms 15-byte minimal form
# (1 marker, terminal only).

def encode_beatgrid_marker(position_s: float,
                            beats_till_next: int) -> bytes:
    """Encode a single non-terminal BeatGrid marker (8 bytes).

    `position_s`: float seconds, f32 BE.
    `beats_till_next`: u32 BE count of beats until the next marker.
    """
    return struct.pack(">fI", float(position_s),
                        int(beats_till_next) & 0xFFFFFFFF)


def encode_beatgrid_terminal(position_s: float,
                              bpm: float) -> bytes:
    """Encode the terminal BeatGrid marker (8 bytes).

    `position_s`: float seconds, f32 BE.
    `bpm`: the canonical full-precision BPM (per the format docs the
    BeatGrid terminal BPM is authoritative; Autotags BPM is rounded).
    """
    return struct.pack(">ff", float(position_s), float(bpm))


def build_beatgrid(markers: Optional[List[dict]] = None,
                    final_bpm: float = 120.0,
                    *,
                    terminal_position_s: Optional[float] = None,
                    footer: int = 0,
                    version: Tuple[int, int] = (1, 0)) -> bytes:
    """Build a complete Serato BeatGrid blob.

    `markers` — list of non-terminal markers (do NOT include the
    terminal here):
        {"position_s": float, "beats_till_next": int}

    `final_bpm` — the terminal marker's BPM (Serato's authoritative
    BPM per the format docs).

    `terminal_position_s` — explicit terminal-marker position; if None
    the last marker's `position_s` is used, or 0.0 if no markers.
    Real tracks often have a non-zero terminal_position (the downbeat
    of the last analysed beat-block); e.g. a sample FLAC has 0.0167.

    Layout:
        u8 major, u8 minor
        u32 BE num_markers   (= N non-terminal + 1 terminal)
        N × non-terminal (8 B each: f32 pos, u32 beats_till_next)
        terminal (8 B: f32 pos, f32 bpm)
        u8 footer

    Returns the raw blob bytes ready for `write_serato_beatgrid_any`.
    """
    markers = list(markers or [])
    n_non_terminal = len(markers)
    if terminal_position_s is None:
        terminal_pos = (markers[-1]["position_s"]
                         if markers else 0.0)
    else:
        terminal_pos = float(terminal_position_s)
    n_total = n_non_terminal + 1  # +1 for terminal

    out = bytearray()
    out += bytes([version[0] & 0xFF, version[1] & 0xFF])
    out += struct.pack(">I", n_total)
    for m in markers:
        out += encode_beatgrid_marker(
            m["position_s"], int(m.get("beats_till_next", 4)))
    out += encode_beatgrid_terminal(terminal_pos, final_bpm)
    out += bytes([int(footer) & 0xFF])
    return bytes(out)


def write_serato_beatgrid_any(path: str, blob: bytes,
                                *, confirm: bool = False) -> bool:
    """Write a Serato BeatGrid blob to any supported container.

    Dispatches by extension:
        MP3/WAV/AIFF → ID3 GEOB "Serato BeatGrid"
        MP4/M4A/M4V  → freeform atom ----:com.serato.dj:beatgrid
        FLAC/OGG     → Vorbis comment serato_beatgrid

    Confirm-flag check.  Preserves all other tags.
    """
    _require_confirm(confirm, "write_serato_beatgrid_any", path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp3", ".wav", ".aiff", ".aif"):
        return write_serato_geob(path, "Serato BeatGrid", blob,
                                  confirm=True)
    if ext in (".mp4", ".m4a", ".m4v"):
        return _write_serato_mp4_atom(
            path, "Serato BeatGrid", "beatgrid", blob,
            confirm=True)
    if ext in (".flac", ".ogg", ".oga", ".opus"):
        return write_serato_vorbis_blob(path, "Serato BeatGrid",
                                          blob, confirm=True)
    return False


# -----------------------------------------------------------------
# Autotags encoder + writer
# -----------------------------------------------------------------
# Per the format docs + a sample FLAC dump:
# Format: `\x01\x01 + cstring bpm + cstring autoGain + cstring gainDb`
# plus a TRAILING extra NUL byte observed in real Serato Pro files.
# All numeric values are ASCII decimal (bpm 2dp, gain 3dp).

def build_autotags(bpm: float, auto_gain: float, gain_db: float,
                    *, version: Tuple[int, int] = (1, 1),
                    pad_to: int = 21) -> bytes:
    """Build a Serato Autotags blob.

    `bpm`         — ASCII decimal, 2 decimal places (e.g. 94.00)
    `auto_gain`   — ASCII decimal, 3 decimal places (e.g. 0.589)
    `gain_db`     — ASCII decimal, 3 decimal places (e.g. 0.000)
    `pad_to`      — pad output to this length with trailing NULs.
                     Default 21 matches BOTH real-Serato variants
                     observed in a real Serato library:
                       a sample FLAC file (94.00 / 0.589 / 0.000)
                         → 20 content + 1 pad NUL = 21 B
                       a sample MP4 file (81.00 / -0.707 / 0.000)
                         → 21 content + 0 pad NUL = 21 B
                     Set pad_to=0 to skip padding.

    Layout (verified byte-equal against both files ):
        u8 major, u8 minor
        cstring f"{bpm:.2f}"        NUL-terminated
        cstring f"{auto_gain:.3f}"  NUL-terminated
        cstring f"{gain_db:.3f}"    NUL-terminated
        NUL * (pad_to - len(content))   if positive
    """
    body = (
        bytes([version[0] & 0xFF, version[1] & 0xFF])
        + f"{bpm:.2f}".encode("ascii") + b"\x00"
        + f"{auto_gain:.3f}".encode("ascii") + b"\x00"
        + f"{gain_db:.3f}".encode("ascii") + b"\x00"
    )
    if pad_to > len(body):
        body += b"\x00" * (pad_to - len(body))
    return body


def write_serato_autotags_any(path: str, blob: bytes,
                                *, confirm: bool = False) -> bool:
    """Write a Serato Autotags blob to any supported container.
    See `write_serato_beatgrid_any` for dispatch details.

    Autogain is stored TWICE and the two copies must agree: the
    `AutoGain` field of this blob, and the `AUTOGAIN` entry inside
    Serato Markers2. Writing one without the other leaves the track
    inconsistent — which of the two Serato honours depends on version
    and code path, so a mismatch surfaces as "the gain changed back".
    `parse_serato_markers2_full()[\"autogain\"]` reads the Markers2
    side; update both or neither."""
    _require_confirm(confirm, "write_serato_autotags_any", path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp3", ".wav", ".aiff", ".aif"):
        return write_serato_geob(path, "Serato Autotags", blob,
                                  confirm=True)
    if ext in (".mp4", ".m4a", ".m4v"):
        return _write_serato_mp4_atom(
            path, "Serato Autotags", "autgain", blob,
            confirm=True)
    if ext in (".flac", ".ogg", ".oga", ".opus"):
        return write_serato_vorbis_blob(path, "Serato Autotags",
                                          blob, confirm=True)
    return False


# -----------------------------------------------------------------
# Analysis encoder + writer (minimal "version-stamp" blob)
# -----------------------------------------------------------------
# Per the format docs + a sample dump, Analysis is just a
# 2- or 3-byte version stamp.  Common values:
#   b"\x02\x01"     = analyser v2.1 (MP3/MP4 in a real Serato library)
#   b"\x00\x01\x00" = analyser v0.1 + trailing NUL (a sample FLAC)
# The blob serves as a "this file was analysed" presence-signal —
# Serato refuses to honour other blobs (Markers2, BeatGrid,
# Autotags) on a file that lacks an Analysis stamp.

SERATO_ANALYSIS_DEFAULT_BLOB = b"\x02\x01"


def build_analysis_blob(version: Tuple[int, ...] = (2, 1),
                          trailing_nul: bool = False) -> bytes:
    """Build a minimal Serato Analysis blob.

    `version` — (major, minor) reproduces the 2-byte ID3 form (e.g.
    `(2, 1)` -> `02 01`, real MP3/WAV); (major, minor, patch) reproduces
    the 3-byte MP4/FLAC form (e.g. `(0, 1, 0)` -> `00 01 00`).  This
    round-trips `parse_serato_analysis`: feed it `(major, minor)` when
    that parser reports `patch is None`, else `(major, minor, patch)`.
    `trailing_nul` still appends one NUL (the legacy FLAC spelling of the
    3-byte form; prefer the explicit 3-tuple).
    """
    out = bytes([version[0] & 0xFF, version[1] & 0xFF])
    if len(version) >= 3:
        out += bytes([version[2] & 0xFF])
    if trailing_nul:
        out += b"\x00"
    return out


def write_serato_analysis_any(path: str, blob: bytes,
                                *, confirm: bool = False) -> bool:
    """Write a Serato Analysis blob to any supported container.
    See `write_serato_beatgrid_any` for dispatch details."""
    _require_confirm(confirm, "write_serato_analysis_any", path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp3", ".wav", ".aiff", ".aif"):
        return write_serato_geob(path, "Serato Analysis", blob,
                                  confirm=True)
    if ext in (".mp4", ".m4a", ".m4v"):
        return _write_serato_mp4_atom(
            path, "Serato Analysis", "analysis", blob,
            confirm=True)
    if ext in (".flac", ".ogg", ".oga", ".opus"):
        return write_serato_vorbis_blob(path, "Serato Analysis",
                                          blob, confirm=True)
    return False


def write_serato_overview_any(path: str, blob: bytes,
                                *, confirm: bool = False) -> bool:
    """Write a Serato Overview blob to any supported container.

    Overview blobs are 3842 bytes fixed (`u8 1, u8 5` header + 240
    chunks × 16 bytes).  Generate via
    `sidecaramel.overview_encode.build_overview_blob_for_path()`.

    See `write_serato_beatgrid_any` for dispatch details.  B5
    confirm-gate.  Audio-file write — audio-safety policy requires the caller
    to confirm explicitly per operation.
    """
    _require_confirm(confirm, "write_serato_overview_any", path)
    if not blob or len(blob) != 3842:
        raise ValueError(
            f"Overview blob must be exactly 3842 bytes, "
            f"got {len(blob)}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp3", ".wav", ".aiff", ".aif"):
        return write_serato_geob(path, "Serato Overview", blob,
                                  confirm=True)
    if ext in (".mp4", ".m4a", ".m4v"):
        return _write_serato_mp4_atom(
            path, "Serato Overview", "overview", blob,
            confirm=True)
    if ext in (".flac", ".ogg", ".oga", ".opus"):
        return write_serato_vorbis_blob(path, "Serato Overview",
                                          blob, confirm=True)
    return False


# -----------------------------------------------------------------
# Shared MP4 atom writer (extracted SPOT — single-point-of-truth policy)
# -----------------------------------------------------------------
# Previously the MP4 envelope+atom write was inlined in
# `write_serato_markers2_mp4` and `write_serato_markers_v1_mp4`.
# Now Beatgrid / Autotags / Analysis writers also need it.

def _write_serato_mp4_atom(path: str, descriptor: str,
                              atom_short_name: str, blob: bytes,
                              *, confirm: bool = False) -> bool:
    """Write a Serato blob to an MP4 freeform atom.

    `descriptor`       — canonical name (e.g. 'Serato BeatGrid').
    `atom_short_name`  — the short form used inside the freeform key
                          (e.g. 'beatgrid', 'autgain', 'markersv2').
    `blob`             — raw payload bytes (no envelope yet).

    Preserves all other MP4 tags (targeted assignment, no
    `.clear()`).  Confirm-flag check.
    """
    _require_confirm(confirm, "_write_serato_mp4_atom", path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".mp4", ".m4a", ".m4v"):
        return False
    if not os.path.isfile(path):
        return False
    try:
        from mutagen.mp4 import MP4, MP4FreeForm
        try:
            from mutagen.mp4 import AtomDataType
            utf8_format = AtomDataType.UTF8
        except (ImportError, AttributeError):
            utf8_format = 1
    except ImportError:
        return False

    envelope = (b"application/octet-stream\x00\x00"
                + descriptor.encode("ascii") + b"\x00" + blob)
    b64 = base64.b64encode(envelope)

    try:
        mp4 = MP4(path)
    except (MP4StreamInfoError, MutagenError, Exception):
        return False
    if mp4 is None:
        return False
    try:
        if mp4.tags is None:
            mp4.add_tags()
        atom_key = f"----:com.serato.dj:{atom_short_name}"
        mp4.tags[atom_key] = [
            MP4FreeForm(b64, dataformat=utf8_format)]
        mp4.save()
        return True
    except Exception:
        return False


# =====================================================================
# B6.5 — Serato Cue-Color Palette (Pro 3.x, from this MIDI config)
# =====================================================================
# 20 named cue-pad colors typically assigned via a hardware controller.
# Hex from a representative MIDI mapping XML.  The `midi_pad`
# value is the velocity Serato sends to the pad LED for that colour
# (used for hardware-feedback work; not part of the RGB lookup).
#
# applies to CUE entries in Markers2 specifically.
# TRACK-row colours (the ulbl uint32 / Markers2 COLOR entry, e.g.
# this peach #ffbb99) use a different, looser palette and are NOT
# covered here.

SERATO_CUE_COLOR_PALETTE: List[Tuple[str, Tuple[int, int, int], int]] = [
    # (display_name,            (R,    G,    B),    midi_pad_velocity)
    ("WEISS",                  (0xFF, 0xFF, 0xFF), 0x7F),
    ("ROT",                    (0xCC, 0x00, 0x00), 0x0F),
    ("ROTORANGE",              (0xCC, 0x44, 0x00), 0x1F),
    ("ORANGE",                 (0xCC, 0x88, 0x00), 0x2F),
    ("GELB",                   (0xCC, 0xCC, 0x00), 0x5F),
    ("HELLGRÜN",              (0x88, 0xCC, 0x00), 0x7D),
    ("DUNKELGRÜN",            (0x44, 0xCC, 0x00), 0x7F),
    ("GIFTGRÜN",              (0x00, 0xCC, 0x00), 0x7A),
    ("GELBGRÜN",              (0x00, 0xCC, 0x44), 0x6F),
    ("TÜRKISGRÜN",            (0x00, 0xCC, 0x88), 0x7C),
    ("TÜRKISBLAU",            (0x00, 0xCC, 0xCC), 0x7F),
    ("HIMMELBLAU",            (0x00, 0x88, 0xCC), 0x7F),
    ("DUNKELBLAU",            (0x00, 0x00, 0xCC), 0x7F),
    ("KÖNIGSBLAU",            (0x00, 0x44, 0xCC), 0x7F),
    ("VIOLETT",                (0x44, 0x00, 0xCC), 0x7F),
    ("LILA",                   (0x88, 0x00, 0xCC), 0x7F),
    ("MAGENTA",                (0xCC, 0x00, 0xCC), 0x7F),
    ("DUNKELROSA",            (0xCC, 0x00, 0x44), 0x0C),
    ("DUNKELROT",              (0xCC, 0x00, 0x88), 0x0F),
    ("SCHWARZ",                (0x00, 0x00, 0x00), 0x00),
]


def classify_cue_color(rgb: Tuple[int, int, int],
                         palette: Optional[List[Tuple[str,
                                                       Tuple[int, int, int],
                                                       int]]] = None
                         ) -> dict:
    """Classify an RGB triple against the Serato cue-color palette.

    Returns:
        {
          "name":       str            (palette name or "UNKNOWN"),
          "rgb":        (r, g, b),     (original input)
          "match":      "exact" | "nearest",
          "distance":   float,         (Euclidean distance in 0..441)
          "midi_pad":   int | None,    (Serato controller velocity)
        }

    `palette` overrides the default `SERATO_CUE_COLOR_PALETTE` if you
    want to classify against a different colour set (e.g. track-row
    palette once we encode that one).
    """
    pal = palette if palette is not None else SERATO_CUE_COLOR_PALETTE
    if not isinstance(rgb, (tuple, list)) or len(rgb) != 3:
        return {"name": "UNKNOWN", "rgb": rgb,
                "match": "nearest", "distance": float("inf"),
                "midi_pad": None}
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])

    best_name = "UNKNOWN"
    best_dist = float("inf")
    best_midi = None
    for name, (pr, pg, pb), midi in pal:
        d2 = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
        if d2 < best_dist:
            best_dist = d2
            best_name = name
            best_midi = midi
        if d2 == 0:
            return {"name": name, "rgb": (r, g, b),
                    "match": "exact", "distance": 0.0,
                    "midi_pad": midi}
    import math
    return {"name": best_name, "rgb": (r, g, b),
            "match": "nearest",
            "distance": math.sqrt(best_dist),
            "midi_pad": best_midi}


def cue_color_hex(rgb: Tuple[int, int, int]) -> str:
    """Format an (r, g, b) tuple as `#RRGGBB`."""
    if not isinstance(rgb, (tuple, list)) or len(rgb) != 3:
        return "#000000"
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"



# =====================================================================
# Stems readers + sidecar cleanup — thin re-exports of sidecaramel.stems
# so SPOT-consumers can depend on `sidecaramel.tags` alone.
# =====================================================================
# sidecaramel.stems covers the heavy lifting (Serato stems format:
# srtshead header, XOR-decoded MP3 chunks, sidecar discovery with NFC
# normalisation, stem-WAV output paths, cleanup of build artefacts).
# Lazy-imported so this module doesn't pull stems' optional ffmpeg
# dependency at load time.

def _kstems():
    """Lazy import to avoid pulling sidecaramel.stems at module load."""
    from sidecaramel import stems as ks
    return ks


def parse_serato_stems_header(data: bytes) -> Optional[dict]:
    """Decode a `.serato-stems` sidecar's `srtshead` header.

    Returns {header_length, version, n_stems, total_samples,
    sample_rate, duration_sec} or None on malformed input.

    The body decoding (chunk iteration + XOR + MP3-extraction) lives
    in `sidecaramel.stems` — use `iter_serato_stem_chunks` for that.
    """
    return _kstems().parse_serato_stems_header(data)


def iter_serato_stem_chunks(data: bytes, start_offset: int):
    """Walk a `.serato-stems` file body chunk-by-chunk, yielding
    (stem_type:int, encoded_payload:bytes) tuples.  Payload is still
    XOR-encoded — decode with `decode_xor_payload`."""
    return _kstems().iter_serato_stem_chunks(data, start_offset)


def find_serato_stems_sidecar(audio_path: str) -> Optional[str]:
    """Locate the `.serato-stems` sidecar for an audio file.

    Searches the audio's parent + well-known Serato-managed folders
    (Imported, Auto Import, Recording, Latest Import).  Uses NFC
    normalisation so HFS+/APFS path encoding differences match.

    Returns the absolute path as a string, or None.
    """
    ks = _kstems()
    p = ks.find_sidecar(audio_path)
    return str(p) if p else None


def read_serato_stems_header(audio_path: str) -> Optional[dict]:
    """Convenience: locate the sidecar for `audio_path` and parse
    its `srtshead` header.  Returns the same dict shape as
    `parse_serato_stems_header()`, or None when no sidecar exists.
    """
    sidecar = find_serato_stems_sidecar(audio_path)
    if not sidecar:
        return None
    try:
        with open(sidecar, "rb") as f:
            head = f.read(64)
    except OSError:
        return None
    return parse_serato_stems_header(head)


def get_cached_stems(audio_path: str) -> dict:
    """Return {stem_name: absolute_wav_path_str} for every Serato
    stem WAV that's already been extracted next to `audio_path`.

    Empty dict if no stems have been pre-extracted.  Used by
    pipelines that want to know "is this track stem-ready already?"
    without triggering an extraction run.
    """
    raw = _kstems().get_cached_stems(audio_path)
    return {k: str(v) for k, v in raw.items()}


def has_stems(audio_path: str,
                check_db: bool = True,
                check_sidecar: bool = True) -> bool:
    """Multi-source check: does this track have stems available?

    Two independent sources:
      • Serato DB `bstm` flag (set when this ran Serato's stem
        analyser on the track).
      • `.serato-stems` sidecar file existence on disk.

    Returns True if EITHER source confirms stems.  When the DB says
    yes but the sidecar is missing (or vice-versa), still True —
    callers should treat this as "track is stem-flagged, look for
    the sidecar at extraction time".
    """
    if check_db:
        try:
            rec = find_track_in_db(audio_path)
            if rec and rec.get("has_stems"):
                return True
        except Exception:
            pass
    if check_sidecar:
        try:
            if find_serato_stems_sidecar(audio_path):
                return True
        except Exception:
            pass
    return False


# ---- Sidecar cleanup helpers ----------------------------------------
# "sidecar exports always temporary for e2e task,
# clean after every finished task".  Every pipeline / .command job
# that exports WAV/temp files next to an audio track is expected to
# call cleanup_sidecar_artifacts() on its way out.
#
# These do NOT need a confirm-gate — they are the INTENDED side-
# effect of the cleanup convention, not destructive writes to user
# audio files.  Whitelisted suffix matching prevents accidental
# deletion of anything that isn't a recognised temp pattern.

def cleanup_sidecar_artifacts(audio_path: str,
                                include_serato_stems: bool = True,
                                include_external: bool = True,
                                log=print) -> List[str]:
    """Delete known stem / temp-WAV sidecar artefacts next to
    `audio_path`.

    Removes (suffix-whitelisted):
      • `<base>.serato-{vocals,harmony,bass,drums}.wav`
      • any external stem-cache suffixes registered via
        `sidecaramel.stems.add_external_stem_suffix()`

    Preserves (never touched):
      • `.serato-stems`  (source-of-truth Serato sidecar)
      • the audio file itself

    Returns the list of absolute paths actually deleted.
    """
    deleted = _kstems().cleanup_stem_artifacts(
        audio_path,
        include_serato=include_serato_stems,
        include_external=include_external,
        log=log,
    )
    return [str(p) for p in deleted]


def temporary_sidecar_dir(audio_path: str,
                            name: str = ".tmp_sidecars"):
    """Context manager that creates a temp folder next to
    `audio_path` and removes it on exit.

    Use for pipelines that need scratch space (demucs cache, intermediate
    WAV exports, etc.) without leaving artefacts behind:

        with temporary_sidecar_dir(audio_path) as scratch:
            # work in `scratch` — guaranteed wiped on normal exit
            # AND on exception.
            ...

    The folder is created with exist_ok=True; on exit it's removed
    via shutil.rmtree(ignore_errors=True) so a leftover open file
    handle won't blow up the cleanup.
    """
    import shutil
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        dirpath = os.path.join(os.path.dirname(
            os.path.abspath(audio_path)), name)
        os.makedirs(dirpath, exist_ok=True)
        try:
            yield dirpath
        finally:
            shutil.rmtree(dirpath, ignore_errors=True)

    return _ctx()



# =====================================================================
# B3 — Serato Database V2 + Crate readers (re-exported from
#      `sidecaramel.db` so consumers need only one import)
# =====================================================================

def _kdb():
    """Lazy import of `sidecaramel.db`, to avoid cyclic-import risk
    at module load time.  Deliberately without a fallback: if this
    import fails, the failure is real and must be visible.
    """
    from sidecaramel import db as kdb
    return kdb


def parse_serato_database_v2(db_path: Optional[str] = None) -> list:
    """Full track-record list from the Serato DB V2 file.  Each entry
    is a TrackRecord dict — see `sidecaramel.db.SERATO_DB_FIELDS`
    for the mapped field names."""
    kdb = _kdb()
    return kdb.parse_database_v2(db_path or kdb.DEFAULT_DB_PATH)


def write_serato_database_v2(records: list,
                              db_path: str,
                              *,
                              confirm: bool = False,
                              serato_known_closed: bool = False,
                              allow_serato_running: bool = False,
                              log=None) -> bool:
    """Re-export of :func:`sidecaramel.db.write_database_v2`.

    Writes a Serato ``database V2`` file from a list of TrackRecord
    dicts (same shape :func:`parse_serato_database_v2` returns).
    Round-trip-safe — unknown tags survive via the per-record
    ``_raw`` sub-dict.  Gated: requires ``confirm=True`` and refuses
    to write while Serato is running (override with
    ``serato_known_closed=True``).

    ``allow_serato_running`` is a **deprecated** alias for
    ``serato_known_closed`` — see :func:`sidecaramel.db.write_database_v2`.

    ``db_path`` is required: this writer will not default to the
    user's live Serato library.
    """
    kdb = _kdb()
    return kdb.write_database_v2(
        records, db_path,
        confirm=confirm,
        serato_known_closed=serato_known_closed,
        allow_serato_running=allow_serato_running,
        log=log,
    )


def find_track_in_db(audio_path: str,
                       db_path: Optional[str] = None) -> Optional[dict]:
    """Look up a track record in the Serato DB by file path.
    Returns the TrackRecord dict or None."""
    kdb = _kdb()
    return kdb.find_track_record(audio_path,
                                   db_path or kdb.DEFAULT_DB_PATH)


def read_playcount_from_db(audio_path: str,
                             db_path: Optional[str] = None
                             ) -> Optional[int]:
    """Container-agnostic play count via the Serato DB.

    Works for MP3 / WAV / FLAC / AIFF (all containers where Serato
    does NOT write the per-track playcount atom — that's MP4-only
    via `read_serato_playcount()`).  Returns the lifetime play
    count from `utpc` (modern Serato), with `tplc` (legacy) as
    fallback.  None when the track isn't in the DB.
    """
    kdb = _kdb()
    return kdb.db_play_count_for(audio_path,
                                   db_path or kdb.DEFAULT_DB_PATH)


def read_analyzed_from_db(audio_path: str,
                            db_path: Optional[str] = None
                            ) -> Optional[bool]:
    """Has Serato run its track-analysis pass (BPM / key / waveform
    overview) on this audio file?

    Source: DB `bovc` flag.  Verified 99.95% of the
    library has bovc=True; the False outliers are non-audio files
    (.qtz / .mov / .flv) and freshly-imported untouched tracks.

    Returns True / False, or None when the track isn't in the DB.
    """
    rec = find_track_in_db(audio_path, db_path)
    if rec is None:
        return None
    return bool(rec.get("analyzed", False))


def read_beat_grid_locked_from_db(audio_path: str,
                                     db_path: Optional[str] = None
                                     ) -> Optional[bool]:
    """Has the user manually locked the BPM / beat-grid on this
    track?  DB-side mirror of the Markers2 BPMLOCK entry (see
    `read_bpm_lock()` for the audio-tag side).

    Returns True / False, or None when the track isn't in the DB.

    Domain note: beat_grid_locked=True is
    always user-set and signals a fluctuating / odd-downbeat track
    where Serato's auto-grid was wrong.
    """
    rec = find_track_in_db(audio_path, db_path)
    if rec is None:
        return None
    return bool(rec.get("beat_grid_locked", False))


def read_crate(crate_path: str) -> Optional[dict]:
    """Parse a Serato `.crate` file.  See
    `sidecaramel.db.read_crate` for the return shape."""
    return _kdb().read_crate(crate_path)


def read_smart_crate(scrate_path: str) -> Optional[dict]:
    """Parse a Serato `.scrate` Smart Crate file."""
    return _kdb().read_smart_crate(scrate_path)


def list_serato_crates(serato_root: Optional[str] = None,
                         include_smart: bool = True) -> list:
    """List every crate / smart-crate under the Serato root.
    Returns [(display_name, absolute_path), ...]."""
    kdb = _kdb()
    return kdb.list_crates(serato_root or kdb.DEFAULT_SERATO_ROOT,
                             include_smart=include_smart)


def write_crate(crate_path: str,
                  track_paths: list,
                  *,
                  sort_column: Optional[str] = None,
                  sort_reverse: bool = False,
                  columns: Optional[list] = None,
                  confirm: bool = False,
                  serato_known_closed: bool = False,
                  allow_serato_running: bool = False,
                  log=None) -> bool:
    """Write a Serato ``.crate`` file.

    Safety stack:
      * ``confirm=True`` required (audio-write convention)
      * refuses if Serato is currently running (use
        ``serato_known_closed=True`` to assert you have already
        confirmed Serato is closed)
      * NEVER writes ``database V2`` — path basename check refuses
        anything that doesn't end in ``.crate`` / ``.scrate``
      * atomic write (temp file + ``os.replace``) so a crash mid-
        write leaves any existing crate intact

    ``allow_serato_running`` is a **deprecated** alias for
    ``serato_known_closed`` — kept for one minor, removed in 0.2.0.

    Returns ``True`` on success.  Raises
    :class:`sidecaramel.db.CrateWriteError` on safety / build failures
    and :class:`sidecaramel.check.SeratoRunningError` when Serato is
    alive and the override flag isn't set.
    """
    return _kdb().write_crate(
        crate_path,
        track_paths,
        sort_column=sort_column,
        sort_reverse=sort_reverse,
        columns=columns,
        confirm=confirm,
        serato_known_closed=serato_known_closed,
        allow_serato_running=allow_serato_running,
        log=log,
    )



# =====================================================================
# B1 — Serato Markers2 entry-type expansion
# =====================================================================
# `_parse_serato_markers2` (legacy) returns CUE-only list, kept for
# backward-compat.  `parse_serato_markers2_full` returns ALL entry
# types: CUE, LOOP, COLOR, BPMLOCK, AUTOGAIN, FLIP, plus unknown raw.

def _decode_markers2_inner(data: bytes) -> Optional[bytes]:
    """Strip the Markers2 outer wrapper (u16 version + base64 +
    optional NUL padding) and return the decoded inner stream bytes,
    or None on failure.

    A sample FLAC exposed that
    real Serato Pro files write base64 with the "length ≡ 1 mod 4"
    quirk inside the Markers2 wrapper too (not just in the MP4 atom
    envelope).  Naive `base64.b64decode` raised binascii.Error and
    this function silently returned None — read_serato_metadata then
    showed 0 cues / 0 loops on every quirk-affected file.

    Now uses `_robust_b64_decode` (trim-{0,1,2,3} fallback) which
    is the single point of truth for Serato base64 tolerance (it
    lives in `blobs` and is re-exported here).
    """
    if not data or len(data) < 2:
        return None
    try:
        b64 = bytearray()
        for byte in data[2:]:
            if byte == 0:
                break
            if byte == 0x0A:
                continue
            b64.append(byte)
        return _robust_b64_decode(bytes(b64))
    except Exception:
        return None


def _iter_markers2_entries(inner: bytes):
    """Yield (entry_type:str, payload:bytes) tuples from the decoded
    Markers2 inner stream.  Caller responsible for type-specific
    decoding."""
    if not inner or len(inner) < 4:
        return
    p = 2  # skip 2-byte inner version
    while p + 4 < len(inner):
        te = inner.find(0, p)
        if te < 0 or te == p:
            break
        try:
            entry_type = inner[p:te].decode("latin-1", errors="ignore")
        except Exception:
            break
        p = te + 1
        if p + 4 > len(inner):
            break
        entry_len = struct.unpack(">I", inner[p:p + 4])[0]
        p += 4
        if entry_len <= 0 or p + entry_len > len(inner):
            break
        payload = inner[p:p + entry_len]
        p += entry_len
        yield (entry_type, payload)


def parse_serato_markers2_full(data: bytes) -> dict:
    """Parse a full Markers2 GEOB / atom blob, returning every entry
    type Serato writes.

    `data` MUST be the raw outer blob bytes (`u8 1, u8 1, base64(inner),
    NUL...`).  To get it from a file:

        from sidecaramel.blobs import harvest
        for desc, payload, _src in harvest(audio_path):
            if desc == "Serato Markers2":
                blob = payload
                break
        out = parse_serato_markers2_full(blob)

    Returns:
        {
          "cues":     [{idx, pos_ms, pos_sec, color, label}, ...],
          "loops":    [{idx, start_ms, end_ms, start_sec, end_sec,
                         color, label, locked}, ...],
          "color":    "#RRGGBB" | None,    # track-level COLOR
          "bpm_lock": bool | None,
          "autogain": float | None,
          "flips":    [{slot, name, loop, action_count}, ...],
          "raw":      {entry_type: [payload_bytes, ...]}  # everything
        }

    Raises:
        TypeError  — `data` is a path / str / not bytes.
        ValueError — `data` is bytes but cannot be decoded as a
                     Markers2 outer wrapper (likely already-decoded
                     inner bytes, or an unrelated blob).
    """
    # Loud-fail guard: returning an empty result for a path string
    # is indistinguishable from "no markers" and sends readers
    # chasing ghost FLIP bugs.
    if isinstance(data, (str, os.PathLike)):
        raise TypeError(
            "parse_serato_markers2_full() takes the raw Markers2 OUTER "
            "blob (bytes), not a path.  Get the blob first via "
            "`sidecaramel.blobs.harvest(path)`; see the docstring "
            "for the canonical 3-line snippet.")
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(
            "parse_serato_markers2_full() expects bytes, got "
            f"{type(data).__name__}")

    out: dict = {
        "cues": [], "loops": [], "color": None,
        "bpm_lock": None, "autogain": None, "flips": [],
        "raw": {},
    }
    inner = _decode_markers2_inner(data)
    if inner is None:
        raise ValueError(
            "could not decode Markers2 outer blob — input is not a "
            "valid Serato Markers2 wrapper.  Common causes: passing "
            "already-decoded inner bytes (use _iter_markers2_entries "
            "directly for that), or passing an unrelated GEOB payload "
            "(e.g. Serato Autotags) by mistake.")

    for entry_type, payload in _iter_markers2_entries(inner):
        out["raw"].setdefault(entry_type, []).append(payload)

        if entry_type == "CUE" and len(payload) >= 13:
            idx = payload[1]
            pos_ms = struct.unpack(">I", payload[2:6])[0]
            r, g, b = payload[7], payload[8], payload[9]
            label_bytes = payload[12:]
            nul = label_bytes.find(b"\x00")
            if nul >= 0:
                label_bytes = label_bytes[:nul]
            try:
                label = label_bytes.decode("utf-8", errors="ignore")
            except Exception:
                label = ""
            out["cues"].append({
                "idx": int(idx),
                "pos_ms": int(pos_ms),
                "pos_sec": pos_ms / 1000.0,
                "color": (r, g, b),
                "label": label,
            })

        elif entry_type == "LOOP" and len(payload) >= 14:
            # Body layout — Serato Markers2 community spec (Holzhaus
            # serato-tags / Mixxx SeratoMarkers2LoopEntry):
            #   u8  reserved (00)
            #   u8  idx
            #   u32 BE start_ms
            #   u32 BE end_ms
            #   u32 BE 0xFFFFFFFF (constant)
            #   4 bytes color, "ARGB" (00 R G B)
            #   u8  reserved (00)
            #   u8  locked (bool)
            #   bytes label (NUL-terminated, UTF-8)
            idx = payload[1]
            start_ms = struct.unpack(">I", payload[2:6])[0]
            end_ms = struct.unpack(">I", payload[6:10])[0]
            color = (0, 0, 0)
            label = ""
            locked = False
            if len(payload) >= 20:
                # color is 00 R G B at 14..17 → RGB at 15..17
                color = (payload[15], payload[16], payload[17])
                locked = bool(payload[19])
                label_bytes = payload[20:]
                nul = label_bytes.find(b"\x00")
                if nul >= 0:
                    label_bytes = label_bytes[:nul]
                try:
                    label = label_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    label = ""
            out["loops"].append({
                "idx": int(idx),
                "start_ms": int(start_ms),
                "end_ms": int(end_ms),
                "start_sec": start_ms / 1000.0,
                "end_sec": end_ms / 1000.0,
                "color": color,
                "label": label,
                "locked": locked,
            })

        elif entry_type == "COLOR" and len(payload) >= 4:
            # 4 bytes: reserved + 3-byte RGB
            r, g, b = payload[1], payload[2], payload[3]
            out["color"] = f"#{r:02x}{g:02x}{b:02x}"

        elif entry_type == "BPMLOCK" and len(payload) >= 1:
            out["bpm_lock"] = bool(payload[0])

        elif entry_type == "AUTOGAIN" and len(payload) >= 4:
            # Serato sometimes writes a 4-byte f32 BE here; sometimes
            # an ASCII float string.  Try both, record the failure
            # of BOTH paths so callers can detect a malformed blob.
            try:
                out["autogain"] = struct.unpack(">f", payload[:4])[0]
            except Exception:
                try:
                    out["autogain"] = float(payload.decode(
                        "latin-1", errors="ignore").strip("\x00"))
                except Exception as e:
                    out.setdefault("_warnings", []).append(
                        {"entry": "AUTOGAIN", "error": repr(e),
                         "payload_len": len(payload)})

        elif entry_type == "FLIP" and len(payload) >= 6:
            # Body (observed): NUL prefix + slot(1) + flag(1) + name
            # cstring + loop(1) + action_count(u32 BE) + actions...
            try:
                slot = payload[1]
                flag = payload[2]
                name_end = payload.find(b"\x00", 3)
                if name_end < 0:
                    raise ValueError(
                        "FLIP body missing NUL after name")
                name = payload[3:name_end].decode("utf-8",
                                                    errors="ignore")
                cursor = name_end + 1
                loop_flag = bool(payload[cursor]) if cursor < len(
                    payload) else False
                cursor += 1
                action_count = 0
                if cursor + 4 <= len(payload):
                    action_count = struct.unpack(
                        ">I", payload[cursor:cursor + 4])[0]
                    cursor += 4
                # Decode each action: u8 id + u32 BE size + payload
                # id 0 = JUMP    16B  : f64 BE from_s, f64 BE to_s
                # id 1 = CENSOR  24B  : f64 BE trigger_s,
                #                        f64 BE reverse_target_s,
                #                        f64 BE speed
                actions = []
                for _ in range(action_count):
                    if cursor + 5 > len(payload):
                        break
                    act_id = payload[cursor]
                    act_size = struct.unpack(
                        ">I", payload[cursor+1:cursor+5])[0]
                    cursor += 5
                    if cursor + act_size > len(payload):
                        break
                    act_payload = payload[cursor:cursor + act_size]
                    cursor += act_size
                    if act_id == 0 and act_size >= 16:
                        from_s, to_s = struct.unpack(
                            ">dd", act_payload[:16])
                        actions.append({
                            "type": "jump",
                            "from_s": float(from_s),
                            "to_s":   float(to_s),
                        })
                    elif act_id == 1 and act_size >= 24:
                        trigger, reverse_target, speed = struct.unpack(
                            ">ddd", act_payload[:24])
                        actions.append({
                            "type": "censor",
                            "trigger_s":        float(trigger),
                            "reverse_target_s": float(reverse_target),
                            "speed":            float(speed),
                        })
                    else:
                        actions.append({
                            "type": "unknown",
                            "id":   act_id,
                            "raw":  act_payload,
                        })
                out["flips"].append({
                    "slot":         int(slot),
                    "flag":         int(flag),
                    "name":         name,
                    "loop":         loop_flag,
                    "action_count": int(action_count),
                    "actions":      actions,
                })
            except Exception as e:
                # Deliberate tolerate-and-record (Quirk C, ):
                # one malformed FLIP entry must not abort a
                # whole-library scan, but it must not vanish silently
                # either.  Record it on the result dict so callers can
                # detect partial parse without re-reading the blob.
                out.setdefault("_warnings", []).append(
                    {"entry": "FLIP", "error": repr(e),
                     "payload_len": len(payload)})

    out["cues"].sort(key=lambda c: c["idx"])
    out["loops"].sort(key=lambda l: l["idx"])
    return out


# =====================================================================
# B1 — Serato Markers_ V1 legacy parser
# =====================================================================

def parse_serato_markers_v1(data: bytes) -> Optional[dict]:
    """Parse the legacy "Serato Markers_" blob (V1 format).

    Layout (observed across this MP4 library):
        bytes 0..1   u16 BE  version (typically 0x0205)
        bytes 2..5   u32 BE  count   (typically 14: 5 cues + 9 loops)
        rest         fixed-size entry rows, mostly 0xFF in unused slots

    Returns:
        {
          "version": int,        # u16 raw (e.g. 0x0205)
          "count": int,
          "entries_raw": bytes,  # rest of payload — caller decides
        }

    Notes:
      * Most of this V1 blobs contain only 0xFF padding (no real
        data) — Serato writes V1 alongside V2 for app-back-compat.
        Real cues/loops live in Markers2 (`parse_serato_markers2_full`).
      * Row size varies between Serato versions; we expose the raw
        entry bytes so callers can do deep decoding if needed without
        guessing here.
    """
    if not data or len(data) < 6:
        return None
    try:
        version = struct.unpack(">H", data[:2])[0]
        count = struct.unpack(">I", data[2:6])[0]
        return {
            "version": int(version),
            "count": int(count),
            "entries_raw": bytes(data[6:]),
        }
    except Exception:
        return None


# =====================================================================
# B1 — Serato Analysis / RelVol / Overview header / playcount parsers
# =====================================================================

def parse_serato_analysis(data: bytes) -> Optional[dict]:
    """Parse the "Serato Analysis" blob — a short version marker.

    Two on-disk shapes occur in real libraries:
      * 2 bytes  u8 major, u8 minor            — ID3 GEOB on MP3/WAV/AIFF
                                                 (e.g. `02 01` = 2.1)
      * 3 bytes  u8 major, u8 minor, u8 patch   — MP4 atom / FLAC
                                                 (e.g. `00 01 00` = 0.1.0)

    Returns {"version": str, "major": int, "minor": int,
    "patch": int | None}.  `patch` is None for the 2-byte form — which is
    also what tells `build_analysis_blob` to reproduce 2 bytes, not 3.
    The 2-byte form used to return None here (the parser required >= 3
    bytes), so MP3/WAV Analysis silently read as "no data".
    """
    if not data or len(data) < 2:
        return None
    try:
        major, minor = int(data[0]), int(data[1])
        if len(data) >= 3:
            patch = int(data[2])
            return {"version": f"{major}.{minor}.{patch}",
                    "major": major, "minor": minor, "patch": patch}
        return {"version": f"{major}.{minor}",
                "major": major, "minor": minor, "patch": None}
    except Exception:
        return None


def parse_serato_relvol(data: bytes) -> Optional[float]:
    """Parse the "Serato RelVol" blob → gain in dB.

    Serato writes this as a 4-byte f32 BE (best observed); when that
    fails, fall back to an ASCII float in the leading bytes.  Returns
    None on parse failure.
    """
    if not data:
        return None
    try:
        if len(data) >= 4:
            v = struct.unpack(">f", data[:4])[0]
            if -60.0 <= v <= 60.0:
                return float(v)
    except Exception:
        pass
    try:
        s = data.decode("latin-1", errors="ignore").strip("\x00").strip()
        return float(s)
    except Exception:
        return None


def parse_serato_overview_header(data: bytes) -> Optional[dict]:
    """Quick metadata-only peek at a "Serato Overview" waveform blob
    (we don't decode the full waveform here — it's typically 5KB+).

    Returns {"size": int, "version_byte": int} or None.
    """
    if not data or len(data) < 2:
        return None
    return {
        "size": int(len(data)),
        "version_byte": int(data[0]),
    }


# ---------------------------------------------------------------------
# Serato Overview blob → image renderer
# ---------------------------------------------------------------------
#
# Empirically derived blob layout (build 0076, reverse-engineering
# session):
#
#   • Total blob = 2-byte header + 240 × 16-byte chunks (+ optional
#     trailer).  3842 bytes observed for MP3 GEOB payloads.
#   • Each 16-byte chunk = ONE time slice of the waveform.
#   • Within a chunk:
#       offset 0-2  : silence padding (always byte 1)
#       offset 3-7  : top half amplitudes (5 bytes)
#       offset 8    : center axis — frequently holds byte 223
#                     (= "peak / silence-edge sentinel")
#       offset 9-12 : bottom half amplitudes — visually mirror the top
#       offset 13-15: silence padding (always byte 1)
#   • The 240 chunks render stacked VERTICALLY (chunks-untereinander),
#     producing a 16-wide × 240-tall waveform image.  Serato's UI
#     rotates this 90° for the horizontal display in the deck overview.
#   • Top-mirror render: offset 8-12 are forced to mirror offsets 3-7
#     for a clean symmetric column.
#
# Color encoding: empirical byte → hue mapping never matched any single
# clean theory (4-bit nibble, 7-bit hue, 2-bit quadrant, log-spectrum,
# direct LUT — all close but visibly off in spots).  Until we have a
# definitive palette, the renderer ships with a clean grayscale output
# that preserves the SHAPE.  Adding a calibrated color palette is a
# follow-up B-task.

OVERVIEW_BLOB_HEADER = 2          # bytes
OVERVIEW_CHUNK_SIZE = 16          # bytes per time slice
OVERVIEW_NUM_CHUNKS = 240         # time slices per overview
OVERVIEW_WIDTH = OVERVIEW_CHUNK_SIZE
OVERVIEW_HEIGHT = OVERVIEW_NUM_CHUNKS
OVERVIEW_SILENCE = 1              # byte value used as silence/padding
OVERVIEW_PEAK_SENTINEL = 223      # center-axis peak marker
OVERVIEW_PADDING_TOP = 3          # chunk offsets 0..2 are padding
OVERVIEW_PADDING_BOTTOM = 3       # chunk offsets 13..15 are padding


def _overview_apply_top_mirror(chunk: bytes) -> bytes:
    """Return a 16-byte chunk with offsets 8-12 forced to mirror 3-7.

    Padding (offsets 0-2 and 13-15) is set to OVERVIEW_SILENCE.  The
    center axis (offset 8) mirrors offset 7 (i.e. the innermost top
    byte); top-half values are preserved as-is.
    """
    if len(chunk) != OVERVIEW_CHUNK_SIZE:
        return chunk
    out = bytearray(OVERVIEW_CHUNK_SIZE)
    for o in range(OVERVIEW_CHUNK_SIZE):
        if o < OVERVIEW_PADDING_TOP or o >= OVERVIEW_CHUNK_SIZE - OVERVIEW_PADDING_BOTTOM:
            out[o] = OVERVIEW_SILENCE
            continue
        src = o if o <= 7 else (OVERVIEW_CHUNK_SIZE - 1 - o)
        out[o] = chunk[src]
    return bytes(out)


def overview_blob_to_pixel_grid(blob: bytes,
                                  top_mirror: bool = True
                                  ) -> Optional[List[List[int]]]:
    """Convert a raw Serato Overview blob into a 16-wide × 240-tall
    byte grid suitable for paletted-BMP rendering.

    Returns a 240-row list, each row a 16-int list of byte values
    (0..255).  Returns None if the blob is malformed.

    Args:
      blob:        raw GEOB payload (post-envelope-unwrap).
      top_mirror:  if True (default), each chunk's bottom half is
                   replaced with a mirror of its top half — produces
                   a clean symmetric waveform image.  If False, raw
                   chunk bytes pass through unchanged.
    """
    if not blob or len(blob) < OVERVIEW_BLOB_HEADER + OVERVIEW_CHUNK_SIZE:
        return None
    body = blob[OVERVIEW_BLOB_HEADER:
                OVERVIEW_BLOB_HEADER + OVERVIEW_NUM_CHUNKS * OVERVIEW_CHUNK_SIZE]
    if len(body) < OVERVIEW_NUM_CHUNKS * OVERVIEW_CHUNK_SIZE:
        return None
    grid: List[List[int]] = []
    for c in range(OVERVIEW_NUM_CHUNKS):
        chunk = body[c * OVERVIEW_CHUNK_SIZE:(c + 1) * OVERVIEW_CHUNK_SIZE]
        if top_mirror:
            chunk = _overview_apply_top_mirror(chunk)
        grid.append([int(b) for b in chunk])
    return grid


def overview_grayscale_palette() -> List[Tuple[int, int, int]]:
    """256-color grayscale palette for the Overview waveform.

    byte 0/1 → near-white (silence background)
    byte 2..255 → linearly darkening gray (peaks render as near-black)
    byte 223 → kept on the same grayscale ramp; renders mid-gray.

    Used as the default until a calibrated color palette is settled.
    """
    pal: List[Tuple[int, int, int]] = []
    for b in range(256):
        if b <= 1:
            pal.append((245, 245, 245))
        else:
            v = 240 - int(220 * (b / 255))
            v = max(0, min(255, v))
            pal.append((v, v, v))
    return pal


def render_serato_overview_image(blob: bytes,
                                  out_path: str,
                                  *,
                                  scale: int = 4,
                                  palette: Optional[List[Tuple[int, int, int]]] = None,
                                  top_mirror: bool = True) -> bool:
    """Render a Serato Overview blob as an 8-bit indexed BMP.

    SPOT for blob → BMP conversion.  Layout = chunks-untereinander,
    top-mirror geometry.  Default palette = grayscale (color palette
    pending further reverse-engineering of byte→hue mapping).

    Args:
      blob:        raw GEOB payload bytes.
      out_path:    target file path (.bmp recommended).
      scale:       nearest-neighbor upscale factor (default 4 →
                   64-wide × 960-tall image).  Pass 1 for raw size.
      palette:     optional 256-entry RGB list.  Defaults to
                   overview_grayscale_palette().
      top_mirror:  apply chunk top-mirror (default True).

    Returns True on success, False if blob is malformed or PIL is
    unavailable.
    """
    try:
        from PIL import Image
    except ImportError:
        return False
    grid = overview_blob_to_pixel_grid(blob, top_mirror=top_mirror)
    if grid is None:
        return False
    if palette is None:
        palette = overview_grayscale_palette()
    flat_pal: List[int] = []
    for r, g, b in palette:
        flat_pal.extend([r, g, b])
    while len(flat_pal) < 256 * 3:
        flat_pal.extend([0, 0, 0])
    img = Image.new("P", (OVERVIEW_WIDTH, OVERVIEW_HEIGHT))
    flat_data = bytearray()
    for row in grid:
        flat_data.extend(row)
    img.putdata(bytes(flat_data))
    img.putpalette(flat_pal)
    if scale != 1:
        img = img.resize((OVERVIEW_WIDTH * scale, OVERVIEW_HEIGHT * scale),
                          resample=Image.NEAREST)
    try:
        img.save(out_path, "BMP")
        return True
    except Exception:
        return False


def render_serato_overview_for_path(audio_path: str,
                                     out_path: str,
                                     **kwargs) -> bool:
    """Convenience: harvest the Overview blob from `audio_path` and
    render it to `out_path`.  Returns True on success.
    """
    blobs = harvest_serato_blobs(audio_path)
    for desc, payload, _src in blobs:
        if desc == "Serato Overview" and payload:
            return render_serato_overview_image(payload, out_path,
                                                 **kwargs)
    return False


def parse_serato_playcount(data: bytes) -> Optional[int]:
    """Parse the MP4-only `----:com.serato.dj:playcount` atom value.

    Serato stores playcount as base64(utf-8 ASCII decimal string),
    e.g. b"Mw==" → "3" → 3 plays.  Returns int or None.
    """
    if not data:
        return None
    raw = bytes(data)
    # Try base64-decode first (Serato format)
    try:
        decoded = base64.b64decode(raw, validate=False)
        s = decoded.decode("utf-8", errors="replace").strip()
        if s.isdigit():
            return int(s)
    except Exception:
        pass
    # Fall back: raw ASCII int
    try:
        s = raw.decode("utf-8", errors="replace").strip()
        return int(s) if s.isdigit() else None
    except Exception:
        return None



# =====================================================================
# B1 — High-level helpers built on the new parsers
# =====================================================================

def _markers2_blob_for_path(audio_path: str) -> Optional[bytes]:
    """Return the raw Markers2 blob bytes (POST-envelope unwrap for
    MP4, POST-GEOB-data extract for ID3), or None."""
    blobs = harvest_serato_blobs(audio_path)
    for desc, payload, _src in blobs:
        if desc == "Serato Markers2":
            return payload
    return None


def read_track_color(audio_path: str) -> Optional[str]:
    """Return the Serato-assigned track color as "#RRGGBB", or None.

    Color lives in the Markers2 COLOR entry.  Works for any container
    where `blobs.harvest()` can locate Markers2.
    """
    blob = _markers2_blob_for_path(audio_path)
    if not blob:
        return None
    parsed = parse_serato_markers2_full(blob)
    return parsed.get("color")


def read_loops(audio_path: str) -> List[dict]:
    """Return the list of saved Serato loops, [] if none.

    Each loop dict: idx, start_ms, end_ms, start_sec, end_sec,
    color (RGB tuple), label, locked.
    """
    blob = _markers2_blob_for_path(audio_path)
    if not blob:
        return []
    return parse_serato_markers2_full(blob).get("loops", [])


def read_bpm_lock(audio_path: str) -> Optional[bool]:
    """Return the Serato BPMLOCK flag (True if BPM is locked /
    manually fixed), or None if absent."""
    blob = _markers2_blob_for_path(audio_path)
    if not blob:
        return None
    return parse_serato_markers2_full(blob).get("bpm_lock")


def read_serato_playcount(audio_path: str) -> Optional[int]:
    """Read Serato's per-track playcount atom.  MP4-only — Serato
    writes this only into `----:com.serato.dj:playcount` for MP4/M4A
    files; MP3 playcount lives in the Serato database, not the tag.
    """
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in (".mp4", ".m4a", ".m4v", ".mov"):
        return None
    f = open_audio(audio_path)
    if f is None or not isinstance(f, MP4):
        return None
    tags = getattr(f, "tags", None)
    if tags is None:
        return None
    key = "----:com.serato.dj:playcount"
    if key not in tags:
        return None
    try:
        v = tags[key]
        if isinstance(v, list) and v:
            return parse_serato_playcount(bytes(v[0]))
    except Exception:
        return None
    return None


