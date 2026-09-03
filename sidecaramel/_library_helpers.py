"""sidecaramel._library_helpers — minimal Serato DB / crate / playcount
parsers for the consolidate, diff, and verify CLI tools.

The four functions actually needed by those tools
(`enumerate_db_track_paths`, `enumerate_crate_track_paths`,
`crate_volume_root`, `abs_path_from_crate_rel`, `mutagen_playcount`)
are implemented here from scratch as small, self-contained
otrk/pfil/ptrk parsers plus a mutagen-based playcount reader.  This is
original code written for sidecaramel — no third-party source is
vendored (see PROVENANCE.md).

Format spec (Serato `database V2` and `*.crate` files share the same
on-disk wire-format):
    record  := tag(4 ASCII) length(4 BE) payload(length bytes)
    otrk    := track-record (in DB and crates)
      ↳ pfil → UTF-16-BE absolute audio path (DB)
      ↳ ptrk → UTF-16-BE volume-relative audio path (crate)

License: GPL-2.0-or-later, same as the rest of the package (see LICENSE).
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

__all__ = [
    "enumerate_db_track_paths",
    "enumerate_crate_track_paths",
    "crate_volume_root",
    "abs_path_from_crate_rel",
    "mutagen_playcount",
]


# ============================================================
# DATABASE V2 PARSER
# ============================================================

def enumerate_db_track_paths(database_path: str) -> List[str]:
    """Parse a Serato `database V2` file, return absolute audio paths.

    The DB is a flat sequence of `otrk` records, each containing a
    `pfil` sub-record whose payload is a UTF-16-BE path.  Returns
    paths verbatim except for prepending "/" when missing
    (Serato sometimes drops the leading slash on the boot volume).
    """
    paths: List[str] = []
    try:
        with open(database_path, "rb") as f:
            data = f.read()
    except OSError:
        return paths

    i = 0
    n = len(data)
    while i + 8 <= n:
        tag = data[i:i + 4]
        try:
            length = int.from_bytes(data[i + 4:i + 8], "big")
        except Exception:
            break
        body_start = i + 8
        body_end = body_start + length
        if body_end > n:
            break
        if tag == b"otrk":
            path = _extract_pfil_path(data[body_start:body_end])
            if path:
                if not path.startswith("/"):
                    path = "/" + path
                paths.append(path)
        i = body_end

    return paths


def _extract_pfil_path(otrk_payload: bytes) -> Optional[str]:
    """Walk an otrk payload, find pfil, decode UTF-16-BE."""
    i = 0
    n = len(otrk_payload)
    while i + 8 <= n:
        tag = otrk_payload[i:i + 4]
        try:
            length = int.from_bytes(otrk_payload[i + 4:i + 8], "big")
        except Exception:
            return None
        body_start = i + 8
        body_end = body_start + length
        if body_end > n:
            return None
        if tag == b"pfil":
            try:
                return otrk_payload[body_start:body_end].decode(
                    "utf-16-be")
            except Exception:
                return None
        i = body_end
    return None


# ============================================================
# CRATE FILE PARSER (volume-relative ptrk)
# ============================================================

def crate_volume_root(crate_path: str) -> str:
    """Resolve the volume mount-point that a crate's ptrk paths are
    relative to.

    Crates store track paths VOLUME-relative.  For a crate at
    `/Volumes/DJDrive/_Serato_/Subcrates/Foo.crate`, an inner ptrk
    payload `Music/foo.mp3` resolves to
    `/Volumes/DJDrive/Music/foo.mp3`, NOT `/Music/foo.mp3`.

    Walk the crate path: if `/Volumes/<X>/...` appears, return
    `/Volumes/<X>`.  Otherwise return `/` (boot drive)."""
    p = os.path.abspath(crate_path)
    parts = p.split(os.sep)
    if len(parts) >= 3 and parts[1] == "Volumes":
        return "/" + parts[1] + "/" + parts[2]
    return "/"


def abs_path_from_crate_rel(volume_root: str, rel: str) -> str:
    """Convert a volume-relative crate ptrk path to absolute."""
    if rel.startswith("/"):
        return rel
    if volume_root == "/":
        return "/" + rel
    return volume_root + "/" + rel


def enumerate_crate_track_paths(crate_path: str) -> List[str]:
    """Read a `.crate` file, return volume-relative audio paths.

    Caller resolves to absolute via
    `abs_path_from_crate_rel(crate_volume_root(crate_path), rel)`.
    """
    paths: List[str] = []
    try:
        with open(crate_path, "rb") as f:
            data = f.read()
    except OSError:
        return paths

    i = 0
    n = len(data)
    while i + 8 <= n:
        tag = data[i:i + 4]
        try:
            length = int.from_bytes(data[i + 4:i + 8], "big")
        except Exception:
            break
        body_start = i + 8
        body_end = body_start + length
        if body_end > n:
            break
        if tag == b"otrk":
            j = body_start
            while j + 8 <= body_end:
                sub_tag = data[j:j + 4]
                try:
                    sub_len = int.from_bytes(data[j + 4:j + 8], "big")
                except Exception:
                    break
                sub_start = j + 8
                sub_end = sub_start + sub_len
                if sub_end > body_end:
                    break
                if sub_tag == b"ptrk":
                    try:
                        path = data[sub_start:sub_end].decode(
                            "utf-16-be",
                            errors="replace").rstrip("\x00")
                    except Exception:
                        path = ""
                    if path:
                        paths.append(path)
                    break  # one ptrk per otrk
                j = sub_end
        i = body_end

    return paths


# ============================================================
# MUTAGEN PLAYCOUNT (ID3 + MP4)
# ============================================================

_TXXX_PLAYCOUNT_RE = re.compile(
    r"^TXXX:.*(?:SERATO[_\s-]*PLAYCOUNT|"
    r"SeratoPlaycount|Serato\s+Playcount|PCNT|Play\s*Count)",
    re.IGNORECASE)


def mutagen_playcount(audio_path: str) -> Optional[int]:
    """Read Serato-written playcount from the audio file's tags.

    Returns None if mutagen unavailable, file unreadable, or no
    playcount tag found.  Returns 0 explicitly when a playcount tag
    was found and its value parses to 0.

    Recognized:
      - ID3 TXXX:* frames matching the regex above (MP3 / WAV / AIFF)
      - MP4 ----:com.serato.dj.* atoms containing "playcount"
    """
    try:
        from mutagen import File as _MFile
    except ImportError:
        return None
    try:
        f = _MFile(audio_path)
    except Exception:
        return None
    if f is None or not hasattr(f, "tags") or f.tags is None:
        return None
    tags = f.tags

    # ID3 path (MP3, WAV, AIFF)
    try:
        for key in list(tags.keys()):
            if _TXXX_PLAYCOUNT_RE.match(str(key)):
                frame = tags.get(key)
                if frame is None:
                    continue
                val = None
                if hasattr(frame, "text") and frame.text:
                    val = str(frame.text[0]).strip()
                elif isinstance(frame, str):
                    val = frame.strip()
                if val is None:
                    continue
                m = re.match(r"^(\d+)", val.lstrip("0") or "0")
                if m:
                    try:
                        return int(m.group(1))
                    except ValueError:
                        continue
    except Exception:
        pass

    # MP4 path
    try:
        for key in list(tags.keys()):
            k = str(key)
            if ("serato" in k.lower()
                    and "playcount" in k.lower()):
                v = tags.get(key)
                if isinstance(v, list) and v:
                    raw = v[0]
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", "ignore")
                    m = re.match(r"^(\d+)", str(raw).strip())
                    if m:
                        try:
                            return int(m.group(1))
                        except ValueError:
                            continue
    except Exception:
        pass

    return None
