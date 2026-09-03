"""sidecaramel_session — pure-function parser for Serato's
`~/Music/_Serato_/History/Sessions/*.session` binary log.

.  Replaces the DJ.INFO log-tail approach with a
binary parse of Serato's own history files.

Format
======
Top-level: a sequence of length-prefixed chunks::

    [tag:4][size:u32be][payload:size]...

The first chunk is `vrsn` carrying a UTF-16BE version string
("1.0/Serato Scratch LIVE Review" in tested files).  Every
subsequent chunk is an `oent` (object entry) — one per track
load.

Each `oent` wraps a single `adat` chunk holding typed fields::

    [field_id:u32be][value_len:u32be][value:value_len]...

Field IDs (reverse-engineered against tested data — IDs that we
care about for cascade dispatch, the rest are bonus metadata):

  1   u32   track_id (DB row index)
  2   utf16 absolute path
  6   utf16 title
  7   utf16 artist
  8   utf16 album
  9   utf16 genre
  17  utf16 key (camelot, e.g. "4A")
  21  utf16 label
  23  utf16 year
  28  u32   start_time (unix epoch seconds)
  31  u32   deck index (1-based: 1 = deck-0/A, 2 = deck-1/B)
  48  u32   session_id (== filename stem, e.g. 125043)
  50  u8    boolean ?
  51  utf16 key signature (e.g. "Fm")
  52  u8    boolean ?
  53  u32   end_time (unix epoch seconds; equals start_time
              for a freshly-loaded track that hasn't been
              superseded yet, and updates to the
              actual-ended timestamp on next load on the
              same deck)
  68/69/72/78  u32  flags (all zero in tested data — likely
                            played/skipped/eot markers)
  81  u8    boolean ?

Active-load detection
---------------------
this rule (verified against Serato History UI):

    The LAST oent for each fid31 value is the track currently
    loaded on that deck.

The end_time field is always written (initialised to start_time
on load, updated to the next-event timestamp on supersede).  The
UI's empty-end-time column is a presentation quirk over the same
underlying data — what makes a track "live" is being the latest
oent for its deck.

Session rotation
----------------
Serato creates a new `.session` file per app launch, named
`<int>.session` (the int = session_id, monotonically increasing).
The CURRENT session file is the one with the highest mtime in
the directory; it gets rewritten on every track event.

API
===
The two functions external callers use:

    latest_session_path(sessions_dir=None) -> Optional[str]
    parse_session(path)                    -> list[dict]

A convenience for the watcher:

    active_loads_by_deck(oents)            -> dict[int, dict]

All field decoding is best-effort: unknown fields fall back to
hex strings, malformed adat payloads return empty dicts.
"""
from __future__ import annotations

import os
import struct
from typing import Optional


DEFAULT_SESSIONS_DIR = os.path.expanduser(
    "~/Music/_Serato_/History/Sessions")


# --- Field-id constants ---------------------------------------------
FID_PATH        = 2
FID_TITLE       = 6
FID_ARTIST      = 7
FID_ALBUM       = 8
FID_GENRE       = 9
FID_KEY_CAMELOT = 17
FID_LABEL       = 21
FID_YEAR        = 23
FID_START_TIME  = 28
FID_DECK        = 31
FID_SESSION_ID  = 48
FID_KEY_MUSICAL = 51
FID_END_TIME    = 53


def latest_session_path(
    sessions_dir: Optional[str] = None,
) -> Optional[str]:
    """Return the abs path of the .session file with the highest
    mtime in `sessions_dir` (default: `~/Music/_Serato_/History/
    Sessions`).  Returns None if the directory is missing or has
    no .session files.
    """
    d = sessions_dir or DEFAULT_SESSIONS_DIR
    if not os.path.isdir(d):
        return None
    best_p: Optional[str] = None
    best_mt: float = -1.0
    try:
        for entry in os.scandir(d):
            if not entry.name.endswith(".session"):
                continue
            try:
                st = entry.stat()
            except OSError:
                continue
            if st.st_mtime > best_mt:
                best_mt = st.st_mtime
                best_p = entry.path
    except OSError:
        return None
    return best_p


def _iter_chunks(blob: bytes):
    """Yield (tag, payload) pairs from a top-level Serato-format
    blob.  Stops on the first non-ASCII tag or truncated payload."""
    i = 0
    while i + 8 <= len(blob):
        tag = blob[i:i + 4]
        if not all(32 <= b < 127 for b in tag):
            break
        size = struct.unpack(">I", blob[i + 4:i + 8])[0]
        if size > len(blob) - i - 8:
            break
        yield (tag.decode("ascii"), blob[i + 8:i + 8 + size])
        i += 8 + size


def _decode_adat(adat_blob: bytes) -> dict:
    """Decode adat-formatted typed key/value pairs.  Returns a
    dict keyed by field-id (int).  Values are decoded by length:
      • len==4  → big-endian u32 int
      • len==1  → u8 int
      • even len ≥ 2 → UTF-16BE string with trailing nulls stripped
      • anything else → raw bytes
    Malformed encodings degrade gracefully (raw bytes / advance 1).
    """
    out: dict = {}
    i = 0
    n = len(adat_blob)
    while i + 8 <= n:
        fid = struct.unpack(">I", adat_blob[i:i + 4])[0]
        flen = struct.unpack(">I", adat_blob[i + 4:i + 8])[0]
        if flen > n - i - 8 or flen > 0x7FFFFFFF:
            # Mis-aligned or insane length — slide forward.
            i += 1
            continue
        val = adat_blob[i + 8:i + 8 + flen]
        if flen == 4:
            out[fid] = struct.unpack(">I", val)[0]
        elif flen == 1:
            out[fid] = val[0]
        elif flen >= 2 and flen % 2 == 0:
            try:
                s = val.decode("utf-16-be").rstrip("\x00")
                out[fid] = s
            except UnicodeDecodeError:
                out[fid] = val
        else:
            out[fid] = val
        i += 8 + flen
    return out


# Friendly aliases for the most-used fields.  Returned in dicts
# alongside the raw fid:value pairs so callers can pick whichever
# they prefer.
_FRIENDLY = {
    FID_PATH:        "path",
    FID_TITLE:       "title",
    FID_ARTIST:      "artist",
    FID_ALBUM:       "album",
    FID_GENRE:       "genre",
    FID_KEY_CAMELOT: "key_camelot",
    FID_LABEL:       "label",
    FID_YEAR:        "year",
    FID_START_TIME:  "start_time",
    FID_DECK:        "deck_1based",
    FID_SESSION_ID:  "session_id",
    FID_KEY_MUSICAL: "key_musical",
    FID_END_TIME:    "end_time",
}


def parse_session(path: str) -> list[dict]:
    """Parse a Serato .session file at `path`, returning a list
    of dicts (one per oent).  Each dict has friendly aliases for
    common fields plus the raw fid→value mapping under "_raw".

    Per dict (typical keys):
        path:            absolute audio path (str)
        title, artist:   strings
        album, genre:    strings
        key_camelot:     "4A" etc.
        start_time:      unix epoch int (seconds)
        end_time:        unix epoch int (seconds, equals
                          start_time for the active load)
        deck_1based:     1 or 2  (raw Serato encoding)
        deck:            0 or 1  (normalised; 0/1)
        session_id:      int
        _raw:            full fid→value dict

    Returns [] on read/parse failure.
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return []
    out: list[dict] = []
    for tag, payload in _iter_chunks(data):
        if tag != "oent":
            continue
        adat_blob: Optional[bytes] = None
        for sub_tag, sub_payload in _iter_chunks(payload):
            if sub_tag == "adat":
                adat_blob = sub_payload
                break
        if adat_blob is None:
            continue
        raw = _decode_adat(adat_blob)
        if not raw:
            continue
        entry: dict = {"_raw": raw}
        for fid, name in _FRIENDLY.items():
            if fid in raw:
                entry[name] = raw[fid]
        # Normalised 0-based deck index.
        d1 = raw.get(FID_DECK)
        if isinstance(d1, int) and d1 in (1, 2):
            entry["deck"] = d1 - 1
        out.append(entry)
    return out


def active_loads_by_deck(oents: list[dict]) -> dict[int, dict]:
    """From a parsed session, return the LAST oent for each
    `deck` value (0 or 1).  Per this rule, the last oent per
    deck represents the currently-loaded track on that deck.

    Returns a dict keyed by 0-based deck index.  Empty if no
    oent had a recognisable deck field.
    """
    out: dict[int, dict] = {}
    for o in oents:
        d = o.get("deck")
        if d not in (0, 1):
            continue
        out[d] = o  # later iterations overwrite — final = "last"
    return out


# ------------------------------------------------------------------
# CLI: `python sidecaramel_session.py` — pretty-print the
# current state.  Handy for quick verification against the Serato
# History UI.
# ------------------------------------------------------------------
def _main() -> int:
    p = latest_session_path()
    if p is None:
        print("[no .session file found]")
        return 1
    print(f"latest session: {p}")
    print(f"size: {os.path.getsize(p)} bytes")
    oents = parse_session(p)
    print(f"oents: {len(oents)}")
    active = active_loads_by_deck(oents)
    print(f"\n=== currently loaded (last oent per deck) ===")
    for d in (0, 1):
        a = active.get(d)
        if a is None:
            print(f"  deck {d}: <empty>")
            continue
        print(f"  deck {d}:")
        print(f"    artist : {a.get('artist', '?')!r}")
        print(f"    title  : {a.get('title', '?')!r}")
        print(f"    album  : {a.get('album', '?')!r}")
        print(f"    key    : {a.get('key_camelot', '?')!r}"
              f"  ({a.get('key_musical', '?')})")
        print(f"    start  : {a.get('start_time', '?')}")
        delta = (a.get('end_time', 0) or 0) - (a.get('start_time', 0) or 0)
        print(f"    end    : {a.get('end_time', '?')}  (Δ={delta}s)")
        print(f"    path   : {a.get('path', '?')!r}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
