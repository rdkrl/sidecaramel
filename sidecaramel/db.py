"""sidecaramel.db — Serato DJ `database V2` and `.crate` / `.scrate`
parsers and writers.

Both reading and writing are supported.  Writers are gated:

  * `write_crate` and `write_database_v2` refuse without
    `confirm=True` and refuse while a Serato process is running
    (override via `allow_serato_running=True`).
  * Atomic write via tempfile + `os.replace`.
  * Unknown tags survive a parse → write round-trip via each
    record's `_raw` sub-dict.

Format — database V2
--------------------
Reverse-engineered from Serato DJ Pro 2.x and 3.x.

  Top-level structure:
    [vrsn]<version-payload>
    [otrk]<track-record>
    [otrk]<track-record>
    ...

  4-byte ASCII tag + 4-byte big-endian length + payload.
  Inside an `otrk` payload, a sequence of nested tagged fields.

Field-tag prefix convention (Serato's Pascal heritage):
  'p' → UTF-16 BE path string   (e.g. pfil, ptrk, pvid)
  't' → UTF-16 BE text string   (e.g. tart, talb, tkey, tbpm)
  'u' → 4-byte BE uint          (e.g. uadd, utkn, utpc, utme, ulbl)
  'b' → 1-byte bool             (e.g. bstm, bply, biro)
  's' → 2-byte BE short         (e.g. sbav)

Per-tag semantics:
  ttyp  file type string ('mp3','wave','quicktime')
  pfil  file path (volume-relative)
  tsng  title          tart  artist        talb  album
  ttyr  year           tgen  genre         tlbl  imprint label
  tkey  musical key    tcom  comment       tgrp  grouping
  trmx  remix          tcmp  compilation flag string
  tbpm  BPM            tlen  length ms     tbit  bitrate
  tsiz  file size      tsmp  sample rate
  tadd  date added (string)   uadd  date added (uint epoch)
  utme  modified time (uint)
  utpc  PLAY COUNT (uint32 BE)  — canonical "plays" column
  utkn  load count (uint32 BE)  — deck-load events; NOT plays
  tplc  legacy play count (UTF-16 BE string) — older Serato builds
  ulbl  track color (uint, 0x00RRGGBB)
  bstm  has stems    biro  imported remote   bply  loaded-on-deck
  bovc/bbgl/bhrt/bmis/bcrt/bwlb/bwll/bitu/blop/buns/bkrk — additional
        boolean flags decoded by the parser (see SERATO_DB_FIELDS)
  pvid  video file path   tvfx  video effects
  vrsn  format version (header only)

Format — .crate / .scrate
-------------------------
Same Tag-Length-Value envelope.  Top-level fields:
  vrsn  header version
  osrt  sort spec       ovct  column display spec
  otrk  track entry → contains ptrk (track path)
  rart/rlut/rurt  smart-crate rules (.scrate only)
"""

from __future__ import annotations

import os
import re
import struct
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

DEFAULT_DB_PATH = os.path.expanduser("~/Music/_Serato_/database V2")
DEFAULT_SERATO_ROOT = os.path.expanduser("~/Music/_Serato_")

# Field tags we care about. There are many more in real files; we
# silently skip any tag we don't recognise.
TAG_OTRK = b"otrk"
TAG_PFIL = b"pfil"
TAG_PTRK = b"ptrk"
TAG_TPLC = b"tplc"   # legacy play count (UTF-16 BE string)
TAG_UTPC = b"utpc"   # current play count (uint32 BE) — verified
                      # against the Serato app's "plays" column on
                      # a real Serato library; per-track play counts
                      # match the Serato GUI exactly.
TAG_UTKN = b"utkn"   # NOT play count — appears in 4x more tracks
                      # than utpc and includes never-played items.
                      # Best guess: load-into-deck counter (every
                      # time the track is loaded onto a deck, even
                      # without subsequent play).  Distribution:
                      # range 1..2191, median 6 in a real Serato library.


# Mapping: 4-byte tag → (output field name, type code) for the broad
# track-record reader.  Type codes:
#   's' utf-16 BE string,  'u' uint32 BE,  'b' 1-byte bool,
#   'h' uint16 BE.
SERATO_DB_FIELDS: Dict[bytes, Tuple[str, str]] = {
    b"ttyp": ("file_type", "s"),
    b"pfil": ("file_path", "s"),
    b"tsng": ("title", "s"),
    b"tart": ("artist", "s"),
    b"talb": ("album", "s"),
    b"ttyr": ("year", "s"),
    b"tgen": ("genre", "s"),
    b"tlbl": ("label", "s"),
    b"tkey": ("key", "s"),
    b"tcom": ("comment", "s"),
    b"tgrp": ("grouping", "s"),
    b"trmx": ("remix", "s"),
    b"tcmp": ("compilation", "s"),
    b"tbpm": ("bpm", "s"),
    b"tlen": ("length_ms", "s"),
    b"tbit": ("bitrate", "s"),
    b"tsiz": ("file_size", "s"),
    b"tsmp": ("sample_rate", "s"),
    b"tadd": ("date_added_str", "s"),
    b"uadd": ("date_added_uint", "u"),
    b"utme": ("modified_uint", "u"),
    b"utpc": ("play_count", "u"),       # ← verified canonical
    b"utkn": ("load_count", "u"),       # ← deck-load counter (not plays)
    b"ulbl": ("color_uint", "u"),
    b"ufsb": ("file_size_bytes", "u"),  # binary file size (utme-style uint)
    b"udsc": ("disc_number", "u"),      # disc number 1..N
    # Decoded b-flags (verified against a real Serato library):
    b"bovc": ("analyzed", "b"),         # Overview/analysis computed
                                          # True on ≈ 99.95% of tracks
    b"bstm": ("has_stems", "b"),        # Serato stems analyser ran
    b"bbgl": ("beat_grid_locked", "b"), # User-locked BPM (mirrors
                                          # Markers2 BPMLOCK)
    b"bmis": ("file_missing", "b"),     # Serato can't find the file
    b"bcrt": ("corrupted", "b"),        # File flagged corrupt
    b"biro": ("imported_remote", "b"),
    b"bhrt": ("tags_readable", "b"),    # linguistic
                                          # match — bHasReadTags. True
                                          # = Serato could open file
                                          # and read its tags at last
                                          # scan. Goes False when the
                                          # file becomes unreachable
                                          # (correlates with bmis).
                                          # DB tag fields (artist/
                                          # title/...) are last-good-
                                          # read; potentially stale
                                          # when this is False.
    b"bwlb": ("whitelabel_low_bandwidth", "b"),
                                          # empirically
                                          # verified — all 56 bwlb=True
                                          # tracks are 32kbps MP3s from
                                          # `Whitelabel.net/` folders.
                                          # whitelabel.net distributed
                                          # 32kbps preview MP3s of
                                          # unreleased tracks for DJ
                                          # promotion before licensing.
    b"bitu": ("from_itunes", "b"),      # never set across a real Serato
                                          # library (no iTunes-Match tracks)
    b"bwll": ("whitelabel_licensed", "b"),
                                          # never set across a real Serato library —
                                          # likely the bwlb counter-flag
                                          # ("full licensed whitelabel
                                          # version available"), but
                                          # not empirically verifiable
                                          # without a True example.
    b"bkrk": ("has_cdg", "b"),          # set at
                                          # Serato import time when an
                                          # `.cdg` CD+G karaoke-
                                          # graphics sidecar lived next
                                          # to the audio. Persistent
                                          # — stays True even when
                                          # Serato's Auto-Import path-
                                          # mangling later separates
                                          # the .cdg from the audio.
    b"bply": ("loaded_on_deck", "b"),   # matches
                                          # the tracks loaded on the
                                          # decks at the last Serato
                                          # save.
                                          # Ephemeral runtime state —
                                          # Serato persists deck-load
                                          # state into the DB on exit /
                                          # auto-save.  Treat as
                                          # transient, NOT as a
                                          # property of the track
                                          # itself.
    # Undecoded — kept raw under flag_* names so they're available
    # for future reverse-engineering.  Only 1 / 0 True samples in
    # a real Serato library — not enough to confidently rename:
    b"blop": ("flag_blop", "b"),
    b"buns": ("flag_buns", "b"),
    b"sbav": ("sbav", "h"),
    b"pvid": ("video_path", "s"),
    b"tvfx": ("video_effects", "s"),
}

# Cache invalidated by mtime so the user's plays during a session
# are eventually picked up. 30s is short enough that strategy
# decisions feel current without re-parsing on every lookup.
_CACHE_TTL_SEC = 30.0
_cache: Dict[str, object] = {
    "path": None,
    "mtime": None,
    "loaded_at": 0.0,
    "by_path": {},   # absolute path → play_count
}


def _decode_utf16_be(buf: bytes) -> Optional[str]:
    """Decode a Serato UTF-16-BE byte string. Trailing NULs and odd
    lengths are tolerated. Returns None on decode failure."""
    if not buf:
        return ""
    if len(buf) % 2 == 1:
        # Odd length is invalid UTF-16. Drop the trailing byte.
        buf = buf[:-1]
    try:
        s = buf.decode("utf-16-be")
    except UnicodeDecodeError:
        return None
    # Strip trailing NULs the format sometimes pads with
    return s.rstrip("\x00")


def _iter_tagged(buf: bytes,
                 start: int = 0,
                 end: Optional[int] = None,
                 ) -> Iterator[Tuple[bytes, int, int]]:
    """Walk a byte region as a sequence of [4-byte-tag][4-byte-be-len]
    [payload-of-len-bytes] records. Yields (tag, payload_start,
    payload_end) tuples — caller can slice payload as needed.

    Tolerates truncation: stops cleanly when there isn't a full
    8-byte header left or the length runs past the region end.
    """
    if end is None:
        end = len(buf)
    pos = start
    while pos + 8 <= end:
        tag = buf[pos:pos+4]
        length = struct.unpack(">I", buf[pos+4:pos+8])[0]
        payload_start = pos + 8
        payload_end = payload_start + length
        if payload_end > end:
            return
        yield tag, payload_start, payload_end
        pos = payload_end


def _parse_track_record(buf: bytes,
                        start: int,
                        end: int,
                        ) -> Optional[Tuple[str, Optional[int]]]:
    """Parse one `otrk` payload region. Returns (path, play_count)
    or None if no path was found.

    Current Serato builds (Pro 3.x) store the play count as
    `utpc` (uint32 BE), not `tplc` (legacy
    UTF-16 BE string).  Verified against Serato app's "plays"
    column: per-track utpc values match the
    GUI exactly.  utkn is a DIFFERENT counter (deck-load-events,
    appears on never-played tracks too).  Try utpc first, then
    tplc fallback for older Serato exports.
    """
    path: Optional[str] = None
    play_count: Optional[int] = None
    for tag, p_start, p_end in _iter_tagged(buf, start, end):
        if tag == TAG_PFIL:
            path = _decode_utf16_be(buf[p_start:p_end])
        elif tag == TAG_UTPC and play_count is None:
            # Current Serato format: 4-byte BE uint
            payload = buf[p_start:p_end]
            if len(payload) == 4:
                try:
                    play_count = struct.unpack(">I", payload)[0]
                except struct.error:
                    pass
        elif tag == TAG_TPLC and play_count is None:
            # Legacy Serato format: UTF-16 BE ASCII decimal string
            raw = _decode_utf16_be(buf[p_start:p_end])
            if raw is not None:
                cleaned = raw.strip().lstrip("0") or "0"
                m = re.match(r"^(\d+)", cleaned)
                if m:
                    try:
                        play_count = int(m.group(1))
                    except ValueError:
                        pass
    if path is None:
        return None
    return path, play_count


def _decode_field(payload: bytes, type_code: str) -> Any:
    """Decode a Serato DB field payload by its type-code prefix."""
    if type_code == "s":
        return _decode_utf16_be(payload)
    if type_code == "u":
        if len(payload) == 4:
            try:
                return struct.unpack(">I", payload)[0]
            except struct.error:
                return None
        return None
    if type_code == "b":
        return bool(payload[0]) if payload else None
    if type_code == "h":
        if len(payload) == 2:
            try:
                return struct.unpack(">H", payload)[0]
            except struct.error:
                return None
        return None
    return None


def _parse_track_record_full(buf: bytes,
                               start: int,
                               end: int,
                               db_path: str = DEFAULT_DB_PATH,
                               ) -> Optional[dict]:
    """Parse one `otrk` payload into a TrackRecord dict with mapped
    field names.  Unknown tags are preserved under `_raw`.

    Returns None if no `pfil` field was found (a record with no path
    is unusable for downstream consumers).
    """
    rec: dict = {"_raw": {}}
    for tag, p_start, p_end in _iter_tagged(buf, start, end):
        payload = buf[p_start:p_end]
        if tag in SERATO_DB_FIELDS:
            field_name, type_code = SERATO_DB_FIELDS[tag]
            val = _decode_field(payload, type_code)
            if val is not None and val != "":
                rec[field_name] = val
        else:
            rec["_raw"][tag.decode("ascii", errors="replace")] = (
                bytes(payload))
    file_path = rec.get("file_path")
    if not file_path:
        return None
    # Normalise the file_path through the volume-root helper so
    # consumers get an absolute path matching realpath() conventions.
    rec["file_path"] = _normalize_serato_path(file_path, db_path)
    # Derived: color as hex string (when color_uint present)
    if "color_uint" in rec:
        c = rec["color_uint"] & 0x00FFFFFF
        rec["color"] = f"#{c:06x}"
    # Backward-compat alias: keep legacy `play_count` semantic but
    # also expose play_count_alt under same shape.
    return rec


def _db_volume_root(db_path: str) -> str:
    """Derive the volume mount-point that DB tracks are stored
    relative to. Serato stores ptrk paths volume-relative — for a
    DB at /Volumes/DJDrive/_Serato_/database V2, the path
    'Music/foo.mp3' resolves to '/Volumes/DJDrive/Music/foo.mp3'.
    Mirrors the crate_volume_root helper.  Without it, a library on
    an external volume has every DB-iter track reported missing,
    because the paths get resolved against the system root.
    """
    abs_p = os.path.abspath(db_path or DEFAULT_DB_PATH)
    parts = abs_p.split(os.sep)
    if len(parts) >= 3 and parts[1] == "Volumes":
        return "/" + parts[1] + "/" + parts[2]
    return "/"


def _normalize_serato_path(p: str, db_path: str = DEFAULT_DB_PATH) -> str:
    """Serato stores paths VOLUME-relative on macOS. Re-add the
    correct volume-root for absolute-path comparison.

    System-drive DB at ~/Music/_Serato_/database V2 → volume root /
    External-drive DB at /Volumes/X/_Serato_/database V2 → /Volumes/X

    Windows-style paths (C:\\…) are left as-is.
    """
    if not p:
        return p
    if p.startswith("/") or (len(p) >= 2 and p[1] == ":"):
        return p
    volume_root = _db_volume_root(db_path)
    if volume_root == "/":
        return "/" + p
    return volume_root + "/" + p


def _load_db(path: str = DEFAULT_DB_PATH) -> Dict[str, int]:
    """Parse the entire database into a {abs_path: play_count} dict.

    Memory note: a typical DJ library has 10-50k tracks; the dict
    is on the order of a few MB. We parse once and cache by mtime.

    Returns {} on missing file, parse error, or unreadable bytes —
    the caller should treat these as "play count unknown".

    Records without a UTPC/TPLC tag are kept, with a play count of
    0.  Serato only writes the play-count tag once a track has been
    loaded onto a deck, so never-played tracks carry no tag — and
    they are the majority of a typical library.  Skipping them here
    would silently hollow out this dict and, through it,
    `iter_tracks`.  `play_count_for()` returns 0 for them either
    way, so the distinction is invisible to that caller.
    """
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return {}
    out: Dict[str, int] = {}
    for tag, p_start, p_end in _iter_tagged(data):
        if tag != TAG_OTRK:
            continue
        parsed = _parse_track_record(data, p_start, p_end)
        if parsed is None:
            continue
        track_path, play_count = parsed
        # never-played tracks have play_count=None
        # — keep them in the dict at 0 so iter_tracks yields the
        # full DB.
        if play_count is None:
            play_count = 0
        norm = _normalize_serato_path(track_path, db_path=path)
        # Last-write-wins: if a path appears twice in the DB
        # (rare — happens after library re-scans), trust the later
        # record. The DB is append-mostly, so later == newer.
        out[norm] = play_count
    return out


def _ensure_loaded(path: str = DEFAULT_DB_PATH) -> Dict[str, int]:
    """Return the cached path→playcount dict, refreshing if the DB
    file's mtime has changed or the cache is older than TTL."""
    now = time.monotonic()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    cache_path = _cache.get("path")
    cache_mtime = _cache.get("mtime")
    cache_loaded_at = _cache.get("loaded_at", 0.0)
    if (cache_path == path
            and cache_mtime == mtime
            and (now - float(cache_loaded_at)) < _CACHE_TTL_SEC
            and isinstance(_cache.get("by_path"), dict)):
        return _cache["by_path"]  # type: ignore
    # Refresh
    by_path = _load_db(path)
    _cache["path"] = path
    _cache["mtime"] = mtime
    _cache["loaded_at"] = now
    _cache["by_path"] = by_path
    return by_path


def play_count_for(audio_path: str,
                   db_path: str = DEFAULT_DB_PATH,
                   ) -> Optional[int]:
    """Return the Serato play count for the audio file at
    audio_path, or None if not in the DB / DB unreadable / tplc
    field absent.

    Path matching is byte-exact against the absolute path. Symlinks
    are resolved to their real path before lookup, since Serato
    records the resolved path in pfil (per observation, not spec).

    Treat None as "unknown / 0" for downstream strategy logic.
    """
    try:
        abs_path = os.path.realpath(os.path.expanduser(audio_path))
    except OSError:
        return None
    by_path = _ensure_loaded(db_path)
    return by_path.get(abs_path)


def reset_cache() -> None:
    """Force the next play_count_for call to re-parse the DB.
    Useful for tests and for the GUI's manual 'rescan' action."""
    _cache["path"] = None
    _cache["mtime"] = None
    _cache["loaded_at"] = 0.0
    _cache["by_path"] = {}


def iter_tracks(db_path: str = DEFAULT_DB_PATH):
    """Yield every absolute audio_path in the Serato V2 database,
    in DB order. Used by the path-crawler to enumerate the library
    for sidecar-LRC processing.

    Each yielded path has been normalised through
    _normalize_serato_path → realpath → absolute path. Files that
    no longer exist on disk are still yielded (the caller decides
    what to do with stale entries — typically skip them).

    Yields ALL DB records, including tracks that have never been
    played (no UTPC/TPLC tag).  Filtering on the play-count tag here
    would silently drop the never-played majority of a typical
    library — `_load_db` deliberately keeps them.

    For the lighter-weight play-count map, use play_count_for() —
    never-played tracks now show 0 there too.  For full
    TrackRecord access (artist/title/key/colour/etc.) use
    parse_database_v2() instead.

    Returns an empty iterator on missing DB / parse error.
    """
    by_path = _ensure_loaded(db_path)
    for path in by_path.keys():
        yield path


# =====================================================================
# B3 — Full track-record reader + DB-side fallback lookups
# =====================================================================
# Second-tier cache keyed by db_path → {abs_path: track_record_dict}.
# Separate from the play-count-only cache so existing callers of
# play_count_for() pay nothing extra unless they ask for the full
# record API.

_records_cache: Dict[str, object] = {
    "path": None,
    "mtime": None,
    "loaded_at": 0.0,
    "by_path": {},   # abs_path → TrackRecord dict
    "records": [],   # list[TrackRecord] in DB order
}


def _load_records(path: str = DEFAULT_DB_PATH
                   ) -> Tuple[Dict[str, dict], List[dict]]:
    """Parse the whole DB into TrackRecord dicts.  Returns
    ({abs_path: rec}, [rec, ...] in DB order)."""
    if not os.path.isfile(path):
        return {}, []
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return {}, []
    by_path: Dict[str, dict] = {}
    records: List[dict] = []
    for tag, p_start, p_end in _iter_tagged(data):
        if tag != TAG_OTRK:
            continue
        rec = _parse_track_record_full(data, p_start, p_end, db_path=path)
        if rec is None:
            continue
        records.append(rec)
        # Last-write-wins on duplicates (DB is append-mostly)
        by_path[rec["file_path"]] = rec
    return by_path, records


def _ensure_records_loaded(path: str = DEFAULT_DB_PATH
                            ) -> Tuple[Dict[str, dict], List[dict]]:
    """mtime+TTL-cached parser for the full record reader."""
    now = time.monotonic()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    if (_records_cache.get("path") == path
            and _records_cache.get("mtime") == mtime
            and (now - float(_records_cache.get("loaded_at", 0.0)))
                < _CACHE_TTL_SEC):
        return (_records_cache["by_path"],   # type: ignore
                _records_cache["records"])    # type: ignore
    by_path, records = _load_records(path)
    _records_cache["path"] = path
    _records_cache["mtime"] = mtime
    _records_cache["loaded_at"] = now
    _records_cache["by_path"] = by_path
    _records_cache["records"] = records
    return by_path, records


def parse_database_v2(db_path: str = DEFAULT_DB_PATH) -> List[dict]:
    """Return the full list of TrackRecord dicts from the Serato V2
    DB at `db_path`.  Each record has mapped field names (artist,
    title, key, color, play_count, has_stems, ...) plus a `_raw`
    sub-dict with any unrecognised tags as raw bytes (for future
    reverse-engineering)."""
    _by_path, records = _ensure_records_loaded(db_path)
    return list(records)


def find_loaded_on_deck(db_path: str = DEFAULT_DB_PATH,
                          ) -> List[dict]:
    """Return every DB record whose `bply` (loaded_on_deck) flag is
    True at the last Serato DB save.

    Returns a list of 0–2 TrackRecord dicts (Serato can have at most
    two decks loaded — though the field is technically a per-record
    flag, so theoretically more could be set if Serato exits with
    additional decks in play; we don't filter the count).

    Use case: answering "what is loaded right now?" from the DB
    alone, without a live connection to Serato.  Cross-match the
    returned record's artist/title against whatever your player
    reports, and deck → audio_path is resolved deterministically.

    Caveat: `bply` is persisted on Serato exit / auto-save.  Between
    a deck-load and the next save, the DB lags.  In practice Serato
    auto-saves frequently enough that this is sub-second for the
    common workflow, but a live track-change event from the player
    will always be the lower-latency signal.
    """
    by_path, records = _ensure_records_loaded(db_path)
    return [r for r in records if r.get("loaded_on_deck")]


def _mutagen_artist_title(audio_path: str) -> Optional[Tuple[str, str]]:
    """Read `(artist, title)` strictly from the FILE'S CURRENT tags
    via mutagen.  Returns (artist, title) or None on failure.

    This is the ground-truth source for match validation: players
    read the file's own tags, so reading them here produces the same
    strings they display.  Going to the file rather than the Serato
    DB cache protects against the DB carrying stale metadata that has
    since diverged from the file.
    """
    if not audio_path or not os.path.isfile(audio_path):
        return None
    try:
        from mutagen import File as _MFile
    except ImportError:
        return None
    try:
        mf = _MFile(audio_path)
    except Exception:
        return None
    if mf is None or mf.tags is None:
        return None

    def _first(keys):
        for k in keys:
            try:
                v = mf.tags.get(k)
            except Exception:
                v = None
            if v is None:
                continue
            if isinstance(v, list) and v:
                v = v[0]
            if hasattr(v, "text") and v.text:
                v = v.text[0] if isinstance(v.text, list) else v.text
            if isinstance(v, bytes):
                try:
                    v = v.decode("utf-8", "replace")
                except Exception:
                    v = ""
            s = str(v).strip()
            if s:
                return s
        return ""

    # ID3 frames, MP4 atoms, Vorbis comments — try each in order.
    artist = _first(["TPE1", "\xa9ART", "ARTIST", "artist"])
    title = _first(["TIT2", "\xa9nam", "TITLE", "title"])
    return (artist, title)


def find_track_by_artist_title(artist: str, title: str,
                                 db_path: str = DEFAULT_DB_PATH,
                                 only_loaded: bool = False,
                                 ) -> Optional[dict]:
    """Search the DB for a record matching `(artist, title)` with
    ZERO TOLERANCE — strict, mutagen-validated.

    Deliberately strict rather than fuzzy.  Comma-subset and prefix
    matching produce false positives against tracks with unusual
    featuring-artist credit ordering — a query for "X" matches DB
    rows for "X, Y, Z" that do not represent the SAME track.

    New strict contract:
      1. DB tells us WHICH file paths are candidates (filtered by
         `bply` when `only_loaded=True`).
      2. For each candidate, open the file with mutagen and read
         the CURRENT artist+title.  This matches what a player
         displays, since players read the file's tags directly.
      3. Accept only if mutagen's `(artist, title)` equals the
         query exactly (whitespace-trimmed; case-sensitive for
         titles since tags are typically authoritative-cased).

    Returns the DB record dict (with `file_path` filled in) for
    the matched track, or None.
    """
    if not artist and not title:
        return None
    q_a = (artist or "").strip()
    q_t = (title  or "").strip()
    by_path, records = _ensure_records_loaded(db_path)
    pool = (records if not only_loaded
            else [r for r in records if r.get("loaded_on_deck")])
    for r in pool:
        fp = r.get("file_path")
        if not fp:
            continue
        m = _mutagen_artist_title(fp)
        if m is None:
            continue
        m_a, m_t = m[0].strip(), m[1].strip()
        # Strict equality on title — that's the authoritative
        # field.  Empty-title query never matches.
        if not q_t or m_t != q_t:
            continue
        # Strict equality on artist when query side is non-empty.
        # Empty query artist still accepts (overlay sometimes
        # omits artist for re-fetch).
        if q_a and m_a != q_a:
            continue
        return r
    return None


def find_track_record(audio_path: str,
                       db_path: str = DEFAULT_DB_PATH
                       ) -> Optional[dict]:
    """Look up a TrackRecord by absolute file path.  Returns None
    if the DB doesn't list this path.  Symlinks are resolved through
    realpath() before lookup (Serato writes resolved paths in
    pfil)."""
    try:
        abs_path = os.path.realpath(os.path.expanduser(audio_path))
    except OSError:
        return None
    by_path, _records = _ensure_records_loaded(db_path)
    return by_path.get(abs_path)


def db_field_for(audio_path: str, field: str,
                  db_path: str = DEFAULT_DB_PATH) -> Any:
    """Generic fallback lookup: return the DB-side value of `field`
    (one of the keys in SERATO_DB_FIELDS' output names) for the
    given audio file, or None if the file isn't in the DB or the
    field isn't set.

    Useful when the audio tag is empty but the DB has it
    (e.g. playcount on MP3 — never written to tag, only to DB)."""
    rec = find_track_record(audio_path, db_path)
    if rec is None:
        return None
    return rec.get(field)


def db_play_count_for(audio_path: str,
                       db_path: str = DEFAULT_DB_PATH) -> Optional[int]:
    """Lifetime play count from the DB.  Works for all containers
    (unlike the MP4-only `read_serato_playcount` audio-tag reader).
    Returns None if not present in DB."""
    return db_field_for(audio_path, "play_count", db_path)


# =====================================================================
# B3 — Crate / SmartCrate readers
# =====================================================================
# Crates use the same Tag-Length-Value envelope as the DB.
# Crate fields:
#   vrsn  header version
#   osrt  sort spec      (contains tvcn = column-name, brev = reverse)
#   ovct  column spec    (contains tvcn = column-name, tvcw = width)
#   otrk  track entry    (contains ptrk = path)
# SmartCrate extras: rart / rlut / rurt rule entries.

_CRATE_TAG_VRSN = b"vrsn"
_CRATE_TAG_OSRT = b"osrt"   # sort spec
_CRATE_TAG_OVCT = b"ovct"   # column spec
_CRATE_TAG_OTRK = b"otrk"
_CRATE_TAG_TVCN = b"tvcn"   # column name
_CRATE_TAG_TVCW = b"tvcw"   # column width
_CRATE_TAG_BREV = b"brev"   # reverse-sort bool
_CRATE_TAG_PTRK = b"ptrk"   # track path
# Smart-crate rule tags
_CRATE_TAG_RART = b"rart"
_CRATE_TAG_RLUT = b"rlut"
_CRATE_TAG_RURT = b"rurt"


def _parse_crate_inner_container(buf: bytes,
                                   start: int,
                                   end: int) -> dict:
    """Parse the nested fields inside an osrt / ovct / otrk container.

    Returns a small dict with whatever recognised keys we find.
    Unrecognised tags are dropped (these containers are small and
    well-known).
    """
    out: dict = {}
    for tag, p_start, p_end in _iter_tagged(buf, start, end):
        payload = buf[p_start:p_end]
        if tag in (_CRATE_TAG_TVCN, _CRATE_TAG_TVCW, _CRATE_TAG_PTRK):
            out[tag.decode("ascii")] = _decode_utf16_be(payload)
        elif tag == _CRATE_TAG_BREV:
            out["brev"] = bool(payload[0]) if payload else False
    return out


def _crate_name_from_filename(crate_path: str) -> str:
    """Serato encodes nested crate hierarchies in the filename via
    `%%` separators.  e.g. `Subcrates/Hip Hop%%2020.crate` → display
    name "Hip Hop ▸ 2020".  Returns a readable name."""
    fn = os.path.splitext(os.path.basename(crate_path))[0]
    parts = fn.split("%%")
    return " ▸ ".join(parts)


def read_crate(crate_path: str) -> Optional[dict]:
    """Parse a Serato `.crate` file.

    Returns:
        {
          "name": str,                   # readable hierarchical name
          "path": str,                   # absolute file path
          "version": str | None,         # vrsn header text
          "sort": {"column": str,        # osrt content (nested)
                    "reverse": bool} | None,
          "sort_column": str | None,     # flat alias matching the
          "sort_reverse": bool,          # write_crate(sort_column=,
                                          # sort_reverse=) parameter
                                          # names.
          "columns": [{"name": str,      # ovct entries in display order
                        "width": str}, ...],
          "track_paths": [str, ...]      # otrk → ptrk, absolute paths
        }

    Returns None if the file isn't readable.  Path normalisation
    uses the volume-root inferred from the crate's containing
    `_Serato_` parent (mirrors the DB normalisation).

    Note: the flat `sort_column` / `sort_reverse` keys
    are exposed as aliases of the nested `sort` dict, so the obvious
    caller guess (matching the write_crate parameter names) works.
    """
    if not os.path.isfile(crate_path):
        return None
    try:
        with open(crate_path, "rb") as f:
            data = f.read()
    except OSError:
        return None

    # Treat the crate's enclosing _Serato_ folder like a DB root so
    # volume-relative ptrk paths resolve correctly on external drives.
    volume_path = _serato_volume_path_for(crate_path)

    out: dict = {
        "name": _crate_name_from_filename(crate_path),
        "path": os.path.abspath(crate_path),
        "version": None,
        "sort": None,
        "sort_column": None,
        "sort_reverse": False,
        "columns": [],
        "track_paths": [],
    }

    for tag, p_start, p_end in _iter_tagged(data):
        payload = data[p_start:p_end]
        if tag == _CRATE_TAG_VRSN:
            out["version"] = _decode_utf16_be(payload)
        elif tag == _CRATE_TAG_OSRT:
            inner = _parse_crate_inner_container(data, p_start, p_end)
            col = inner.get("tvcn")
            rev = inner.get("brev", False)
            out["sort"] = {"column": col, "reverse": rev}
            out["sort_column"] = col
            out["sort_reverse"] = bool(rev)
        elif tag == _CRATE_TAG_OVCT:
            inner = _parse_crate_inner_container(data, p_start, p_end)
            out["columns"].append({
                "name": inner.get("tvcn"),
                "width": inner.get("tvcw"),
            })
        elif tag == _CRATE_TAG_OTRK:
            inner = _parse_crate_inner_container(data, p_start, p_end)
            ptrk = inner.get("ptrk")
            if ptrk:
                out["track_paths"].append(
                    _normalize_path_to_volume(ptrk, volume_path))
    return out


def read_smart_crate(scrate_path: str) -> Optional[dict]:
    """Parse a Serato `.scrate` (Smart Crate) file.

    Same shape as `read_crate()` plus a `rules` list containing
    raw `rart` / `rlut` / `rurt` payloads.  We don't decode the
    rule semantics here — that's a separate RE problem.  Callers
    typically want the resolved track list (use the regular DB to
    apply the rules client-side, or just show the rule labels).
    """
    if not os.path.isfile(scrate_path):
        return None
    base = read_crate(scrate_path) or {}
    base["rules"] = []
    try:
        with open(scrate_path, "rb") as f:
            data = f.read()
    except OSError:
        return base
    for tag, p_start, p_end in _iter_tagged(data):
        if tag in (_CRATE_TAG_RART, _CRATE_TAG_RLUT, _CRATE_TAG_RURT):
            base["rules"].append({
                "kind": tag.decode("ascii"),
                "raw": bytes(data[p_start:p_end]),
            })
    return base


def list_crates(serato_root: str = DEFAULT_SERATO_ROOT,
                 include_smart: bool = True) -> List[Tuple[str, str]]:
    """List every crate file found under the Serato root.

    Returns [(display_name, absolute_path), ...] sorted by display
    name.  Smart crates are included by default.
    """
    out: List[Tuple[str, str]] = []
    for subdir, ext in (("Subcrates", ".crate"),
                          ("SmartCrates", ".scrate")):
        if subdir == "SmartCrates" and not include_smart:
            continue
        folder = os.path.join(serato_root, subdir)
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(ext):
                continue
            full = os.path.join(folder, fn)
            out.append((_crate_name_from_filename(full), full))
    return out


# =====================================================================
# B6 — Crate WRITER (gated, Serato-running aware, never DB V2)
# =====================================================================
# spec rules
#   1. confirm=True required (audio-write convention)
#   2. refuse if Serato is currently running (file lock + race risk)
#   3. NEVER write database V2 — path basename check refuses it
#   4. atomic write (temp file + os.replace) so a crash mid-write
#      can't corrupt an existing crate
#
# No write_smart_crate yet — smart-crate rules need their own decoding
# pass first.

class CrateWriteError(RuntimeError):
    """Raised when a crate write is refused for safety reasons
    (confirm-gate, Serato running, forbidden path, build failure)."""


def _require_confirm_crate(confirm: bool, path: str) -> None:
    if not confirm:
        raise CrateWriteError(
            f"crate write requires confirm=True; path={path!r}. "
            f"Pass confirm=True to acknowledge that this will modify "
            f"a Serato crate file.")


def _refuse_database_v2_write(path: str) -> None:
    """Hard-guard for the CRATE writer: accepts only `.crate` paths,
    rejects `.scrate` Smart Crates outright (there is no smart-crate
    writer — static-crate bytes would clobber the rule tags), and
    special-cases the DB-V2 filename so a caller can't accidentally
    route a DB write through `write_crate`.

    Note: this does NOT block writing the DB itself — that's what
    `write_database_v2` is for.  This guard exists so the two
    surfaces stay separate."""
    base = os.path.basename(path)
    base_lower = base.lower()
    if base == "database V2" or base_lower == "database v2":
        raise CrateWriteError(
            f"refusing to route a DB-V2 write through the crate "
            f"writer: {path!r}.  Use sidecaramel.db.write_database_v2 "
            f"instead.")
    if base_lower.endswith(".scrate"):
        raise CrateWriteError(
            f"refusing to write a Smart Crate (.scrate): {path!r}.  "
            f"write_crate only serialises STATIC crates; writing static-"
            f"crate bytes to a .scrate would destroy its smart-crate "
            f"rules (rart/rlut/rurt).  Smart-crate writing is not "
            f"implemented — see read_smart_crate().")
    if not base_lower.endswith(".crate"):
        raise CrateWriteError(
            f"crate writer only accepts paths ending in .crate, "
            f"got {path!r}")


# =====================================================================
# Database V2 — writer
# =====================================================================
# Same gating shape as the crate writer (`confirm=True` + refuses
# while Serato is running) — DB-V2 is persistent state, NOT something
# Serato rebuilds on load.  Writing it is supported; you just have to
# do it when Serato isn't holding the file open.


class DatabaseV2WriteError(Exception):
    """Raised by `write_database_v2` on missing confirmation, Serato-
    running collisions, or encoder errors."""


# Inverse of SERATO_DB_FIELDS: field_name → (tag, type_code)
_FIELD_TO_TAG: Dict[str, Tuple[bytes, str]] = {
    name: (tag, type_code)
    for tag, (name, type_code) in SERATO_DB_FIELDS.items()
}


def _encode_field(type_code: str, value) -> bytes:
    """Encode a single field payload by Serato type code."""
    if type_code == "s":
        # UTF-16 BE string.  Tolerate non-str by str()'ing.
        if value is None:
            return b""
        return str(value).encode("utf-16-be")
    if type_code == "u":
        # uint32 BE
        v = int(value) & 0xFFFFFFFF
        return struct.pack(">I", v)
    if type_code == "h":
        # uint16 BE
        v = int(value) & 0xFFFF
        return struct.pack(">H", v)
    if type_code == "b":
        # 1-byte bool
        return b"\x01" if value else b"\x00"
    raise DatabaseV2WriteError(
        f"unknown DB field type code: {type_code!r}")


def _encode_tlv(tag: bytes, payload: bytes) -> bytes:
    """One [4-byte tag][4-byte BE length][payload] envelope."""
    if len(tag) != 4:
        raise DatabaseV2WriteError(f"tag must be 4 bytes: {tag!r}")
    return tag + struct.pack(">I", len(payload)) + payload


def _denormalize_serato_path(absolute_path: str,
                                  db_path: str) -> str:
    """Inverse of `_normalize_serato_path`.  Serato stores the `pfil`
    tag as a volume-relative path (no leading `/` for system-drive
    libraries; volume-stripped for external mounts).  Round-trip
    requires we put the path back in the same form before writing."""
    if not absolute_path:
        return absolute_path
    root = _db_volume_root(db_path).rstrip("/")
    if root and absolute_path.startswith(root + "/"):
        rel = absolute_path[len(root) + 1:]
        return rel
    # System drive case — drop leading slash.
    return absolute_path.lstrip("/")


def _encode_track_record(rec: dict, db_path: str) -> bytes:
    """Encode one TrackRecord dict back to an `otrk` payload.

    Field ordering: pfil first (Serato convention), then the known
    fields in SERATO_DB_FIELDS order, then any preserved `_raw` raw
    bytes verbatim.  Unknown keys on the record (no matching tag)
    are silently dropped — pass them via `_raw` if you need them
    preserved.
    """
    if not rec.get("file_path"):
        raise DatabaseV2WriteError(
            "TrackRecord missing 'file_path' — every otrk needs pfil")

    parts: list = []

    # 1) pfil first — Serato writes this before anything else.
    pfil_value = _denormalize_serato_path(rec["file_path"], db_path)
    parts.append(_encode_tlv(TAG_PFIL,
                                _encode_field("s", pfil_value)))

    # 2) all other known fields in stable order.
    for tag, (name, type_code) in SERATO_DB_FIELDS.items():
        if tag == TAG_PFIL:
            continue
        if name not in rec:
            continue
        # 'color' is a derived view of 'color_uint' — don't double-emit.
        if name == "color":
            continue
        parts.append(_encode_tlv(tag,
                                    _encode_field(type_code, rec[name])))

    # 3) preserved raw bytes for tags we don't recognise.
    for tag_str, raw_payload in (rec.get("_raw") or {}).items():
        try:
            tag_bytes = tag_str.encode("ascii")
        except UnicodeEncodeError:
            continue
        if len(tag_bytes) != 4 or not isinstance(raw_payload, (bytes, bytearray)):
            continue
        parts.append(_encode_tlv(tag_bytes, bytes(raw_payload)))

    return b"".join(parts)


def write_database_v2(records: List[dict],
                       db_path: str,
                       *,
                       vrsn_payload: Optional[bytes] = None,
                       confirm: bool = False,
                       serato_known_closed: bool = False,
                       allow_serato_running: bool = False,
                       log=None) -> bool:
    """Write a Serato ``database V2`` file from a list of TrackRecord
    dicts (same shape :func:`parse_database_v2` returns).

    Args:
        records: list of TrackRecord dicts.  Each must have
            ``file_path`` (absolute).  All other fields are optional.
            Unknown tags from a previous parse are preserved via the
            ``_raw`` sub-dict.
        db_path: target DB file path.  Defaults to the system Serato
            DB at ``~/Music/_Serato_/database V2``.
        vrsn_payload: optional bytes for the leading ``vrsn`` header
            payload.  When ``None`` and ``db_path`` already exists,
            the existing ``vrsn`` is preserved verbatim.  When
            ``None`` and the DB file is new, a minimal hard-coded
            ``vrsn`` is used.
        confirm: confirm-flag gate.  Without ``True``, raises
            :class:`DatabaseV2WriteError`.
        serato_known_closed: explicit user acknowledgement that
            Serato is **not** currently running.  Bypasses the
            ``check.is_running`` probe — useful on hosts without a
            supported probe (Linux), or when the caller has already
            asked the user.  Default ``False``: probe-then-refuse.
        allow_serato_running: **DEPRECATED** legacy alias for
            ``serato_known_closed``.  Kept for one minor; will be
            removed in 0.2.0.  Same effect.

    Returns ``True`` on success.  Round-trip-safe when called with
    the list of records returned by :func:`parse_database_v2` —
    unknown tags are preserved through the ``_raw`` sub-dict.

    Important: DB-V2 is the persistent track-metadata cache for the
    Serato library.  It is NOT rebuilt from ``.crate`` files on
    launch.  But Serato also doesn't merge on-disk changes mid-
    session — it holds the DB in memory while running and writes
    its own version back on quit/save.  Therefore: edit the DB only
    while Serato is closed, otherwise your changes get clobbered.
    """
    if not confirm:
        raise DatabaseV2WriteError(
            "DB-V2 write requires confirm=True.  This will overwrite "
            "your Serato library's track-metadata cache; pass "
            "confirm=True to acknowledge.")
    # Treat the two kwargs as a single OR — deprecation warning on
    # the legacy spelling.
    if allow_serato_running and not serato_known_closed:
        import warnings as _warnings
        _warnings.warn(
            "allow_serato_running is deprecated and will be removed "
            "in 0.2.0; use serato_known_closed=True with the same "
            "effect (= 'I have confirmed Serato is closed', which is "
            "what the caller actually means).",
            DeprecationWarning,
            stacklevel=2,
        )
        serato_known_closed = True
    if not serato_known_closed:
        try:
            from sidecaramel.check import assert_not_running
            assert_not_running(log=log)
        except ImportError:
            pass  # the check module is optional

    # Preserve existing vrsn header if caller didn't supply one.
    if vrsn_payload is None:
        vrsn_payload = b""
        if os.path.isfile(db_path):
            try:
                with open(db_path, "rb") as f:
                    head = f.read(64 * 1024)
                for tag, p_start, p_end in _iter_tagged(head):
                    if tag == b"vrsn":
                        vrsn_payload = head[p_start:p_end]
                        break
            except OSError:
                vrsn_payload = b""
        if not vrsn_payload:
            # Minimal default — matches a fresh Serato install.
            vrsn_payload = "@2.0/Serato Scratch LIVE Database".encode(
                "utf-16-be")

    blob = _encode_tlv(b"vrsn", vrsn_payload)
    for rec in records:
        body = _encode_track_record(rec, db_path)
        blob += _encode_tlv(TAG_OTRK, body)

    # Atomic write via temp file in the same directory.
    target_dir = os.path.dirname(db_path) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp = None, None
    try:
        import tempfile
        fd, tmp = tempfile.mkstemp(dir=target_dir, prefix=".database_V2.")
        with os.fdopen(fd, "wb") as f:
            fd = None
            f.write(blob)
        os.replace(tmp, db_path)
        tmp = None
    finally:
        if fd is not None:
            os.close(fd)
        if tmp is not None and os.path.isfile(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass

    if log is not None:
        try:
            log(f"[db.v2] wrote {len(records)} records "
                 f"({len(blob)} bytes) to {db_path}")
        except Exception:
            pass
    return True


def _build_crate_blob(*,
                       sort_column: Optional[str] = None,
                       sort_reverse: bool = False,
                       columns: Optional[List[Tuple[str, str]]] = None,
                       track_paths: Optional[List[str]] = None,
                       volume_root: str = "/",
                       version_str: str = ("1.0/Serato "
                                              "ScratchLive Crate")
                       ) -> bytes:
    """Serialise crate fields into the binary `.crate` format.

    Pure function — no I/O.  See read_crate() for the inverse.
    """
    def _tlv(tag: bytes, payload: bytes) -> bytes:
        return tag + struct.pack(">I", len(payload)) + payload

    def _utf16be(s: str) -> bytes:
        return s.encode("utf-16-be")

    out = bytearray()
    out += _tlv(_CRATE_TAG_VRSN, _utf16be(version_str))

    if sort_column:
        inner = (_tlv(_CRATE_TAG_TVCN, _utf16be(sort_column))
                  + _tlv(_CRATE_TAG_BREV,
                          b"\x01" if sort_reverse else b"\x00"))
        out += _tlv(_CRATE_TAG_OSRT, bytes(inner))

    for col_name, col_width in (columns or []):
        inner = (_tlv(_CRATE_TAG_TVCN, _utf16be(col_name))
                  + _tlv(_CRATE_TAG_TVCW, _utf16be(str(col_width))))
        out += _tlv(_CRATE_TAG_OVCT, bytes(inner))

    for tp in (track_paths or []):
        # Serato stores VOLUME-relative paths.  Strip the crate's own
        # volume mount root ("/Volumes/X" for external drives, or the
        # leading "/" for the system volume) — inverse of
        # _normalize_path_to_volume() on the read side.  A bare
        # lstrip("/") here duplicated the mount root on read for
        # external-volume crates (volume-relative paths).
        rel = _denormalize_crate_path(tp, volume_root) if tp else ""
        inner = _tlv(_CRATE_TAG_PTRK, _utf16be(rel))
        out += _tlv(_CRATE_TAG_OTRK, bytes(inner))

    return bytes(out)


def write_crate(crate_path: str,
                  track_paths: List[str],
                  *,
                  sort_column: Optional[str] = None,
                  sort_reverse: bool = False,
                  columns: Optional[List[Tuple[str, str]]] = None,
                  version_str: str = ("1.0/Serato "
                                         "ScratchLive Crate"),
                  confirm: bool = False,
                  serato_known_closed: bool = False,
                  allow_serato_running: bool = False,
                  log=None) -> bool:
    """Write a Serato ``.crate`` file.

    Args:
        crate_path: target .crate file path (absolute).
        track_paths: list of absolute audio paths to include.  These
            are stored volume-relative inside the crate (Serato
            convention).
        sort_column: optional column name to sort by (e.g. ``"bpm"``).
        sort_reverse: True for descending sort.
        columns: optional list of ``(column_name, width_string)``
            tuples for the crate's column-display spec.  When
            ``None``, Serato falls back to the user's default
            crate-column layout.
        version_str: crate header version string.  Default mirrors
            what current Serato Pro writes.
        confirm: confirm-flag check.  Must be ``True`` or
            :class:`CrateWriteError` is raised.
        serato_known_closed: explicit user acknowledgement that
            Serato is **not** currently running.  Bypasses the
            ``check.is_running`` probe — useful on hosts without a
            supported probe (Linux), or when the caller has already
            asked the user.  Default ``False``: probe-then-refuse.
        allow_serato_running: **DEPRECATED** legacy alias for
            ``serato_known_closed``.  Kept for one minor; will be
            removed in 0.2.0.  Same effect.
        log: optional logger callable (receives one-line strings).

    Atomic write: builds the full blob in memory, writes to a temp
    file in the same directory, then ``os.replace()``s into the
    target.  A crash mid-write leaves the existing crate untouched.

    Returns ``True`` on success.  Raises:
        :class:`CrateWriteError` — confirm missing / forbidden path /
                                   Serato running / build failure.
        :class:`sidecaramel.check.SeratoRunningError` — same as
                                   ``CrateWriteError`` for the
                                   Serato-running case.
    """
    _require_confirm_crate(confirm, crate_path)
    _refuse_database_v2_write(crate_path)

    # Deprecation-aware OR of the two kwargs.
    if allow_serato_running and not serato_known_closed:
        import warnings as _warnings
        _warnings.warn(
            "allow_serato_running is deprecated and will be removed "
            "in 0.2.0; use serato_known_closed=True with the same "
            "effect.",
            DeprecationWarning,
            stacklevel=2,
        )
        serato_known_closed = True

    if not serato_known_closed:
        try:
            # Must resolve the IN-PACKAGE module.  Importing a
            # top-level `sidecaramel_check` here would fail in a
            # bundled install, fall through to
            # `assert_not_running = None`, and silently skip the
            # Serato-running guard — a gate that fails open.
            from sidecaramel.check import assert_not_running
        except ImportError:
            # Tolerate platforms where the check itself is unavailable
            # (e.g. sandbox-only without psutil).  The guard then
            # degrades to "no protection"; caller's confirm-gate plus
            # the explicit `allow_serato_running` default still apply.
            assert_not_running = None
        if assert_not_running is not None:
            # Raises sidecaramel.check.SeratoRunningError if Serato is
            # alive.  Caller decides whether to catch that.
            assert_not_running(log=log)

    blob = _build_crate_blob(
        sort_column=sort_column,
        sort_reverse=sort_reverse,
        columns=columns,
        track_paths=track_paths,
        volume_root=_serato_volume_path_for(crate_path),
        version_str=version_str,
    )

    # Atomic write — temp file in same directory + os.replace
    import tempfile
    parent = os.path.dirname(os.path.abspath(crate_path)) or "."
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError as e:
        raise CrateWriteError(
            f"cannot create parent directory for {crate_path!r}: {e}")

    fd, tmp = tempfile.mkstemp(
        dir=parent, suffix=".crate.tmp",
        prefix=os.path.basename(crate_path) + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        os.replace(tmp, crate_path)
    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        raise CrateWriteError(
            f"crate write failed for {crate_path!r}: {e}")
    if log:
        try:
            log(f"[crate-write] wrote {crate_path} "
                f"({len(blob)} bytes, {len(track_paths)} tracks)")
        except Exception:
            pass
    return True


def _serato_volume_path_for(crate_path: str) -> str:
    """For a crate inside `/Volumes/X/_Serato_/Subcrates/Foo.crate`
    return `/Volumes/X` so ptrk paths can be made absolute.  For
    system-volume crates return `/`."""
    abs_p = os.path.abspath(crate_path)
    parts = abs_p.split(os.sep)
    if len(parts) >= 3 and parts[1] == "Volumes":
        return "/" + parts[1] + "/" + parts[2]
    return "/"


def _normalize_path_to_volume(p: str, volume_path: str) -> str:
    """Re-add the volume root to a volume-relative ptrk path."""
    if not p:
        return p
    if p.startswith("/") or (len(p) >= 2 and p[1] == ":"):
        return p
    if volume_path == "/":
        return "/" + p
    return volume_path + "/" + p


def _denormalize_crate_path(absolute_path: str, volume_root: str) -> str:
    """Inverse of `_normalize_path_to_volume`: turn an absolute track
    path into the volume-relative form Serato stores in a `ptrk` tag.

    Strips the crate's volume mount root (`/Volumes/X`) for external
    drives, or the leading `/` for the system volume.  Mirrors
    `_denormalize_serato_path` on the DB-V2 side.  An already-relative
    path (or a Windows drive path) is returned unchanged."""
    if not absolute_path:
        return absolute_path
    if not absolute_path.startswith("/") and not (
            len(absolute_path) >= 2 and absolute_path[1] == ":"):
        return absolute_path
    root = (volume_root or "/").rstrip("/")
    if root and absolute_path.startswith(root + "/"):
        return absolute_path[len(root) + 1:]
    # System-volume case (or path outside the crate's volume) — drop
    # the leading slash, matching Serato's system-drive convention.
    return absolute_path.lstrip("/")


# =====================================================================
# B7 — History/Sessions reader
# =====================================================================
# `.session` files share the tag-len-payload envelope
# with database V2 and `.crate`, but each track-entry uses NUMERIC
# 4-byte tag IDs (0x00000001 … 0x0000004x) instead of ASCII codes.
# Empirically identified fields (verified against
# `/Volumes/_Serato_/History/Sessions/1.session`, 2014-vintage):
#
#   0x01: row index (u16 BE)
#   0x02: full path (utf-16-be)
#   0x03: directory  (utf-16-be)
#   0x04: filename   (utf-16-be)
#   0x06: title      (utf-16-be)
#   0x07: artist     (utf-16-be)
#   0x08: album      (utf-16-be)
#   0x09: genre      (utf-16-be)
#   0x0A: duration   (utf-16-be "MM:SS.xx")
#   0x0B: file size  (utf-16-be "83.1MB")
#   0x0D: bitrate    (utf-16-be "256kbps")
#   0x0E: samplerate (utf-16-be "44.1k")
#   0x0F: bpm × 100  (u32 BE)
#   0x12: file type 4-char tag (e.g. b'engn' for engine-rendered video)
#   0x17: year  (utf-16-be "1992")
#   0x1C: play start time (u32 BE Unix)
#   0x1D: play end   time (u32 BE Unix)
#   0x1F: deck number     (u16 BE; 1 or 2)
#   0x21: 1-byte flag
#   0x27: "played" flag    (1 byte; 1 = actually played, 0 = loaded only)
#   0x2D: numeric counter  (u32 BE)
#   0x30: 2-byte flag
#   0x32: 1-byte flag
#   0x34: 1-byte flag
#   0x35: u32 timestamp variant
#   0x46: 1-byte flag
#
# Top-level structure: `vrsn` (UTF-16-BE version string) + N × `oent`
# (one entry per track row).  Each `oent` wraps a single `adat`
# container with the numeric-tag track fields.

_HIST_TAG_FULL_PATH  = b"\x00\x00\x00\x02"
_HIST_TAG_DIR        = b"\x00\x00\x00\x03"
_HIST_TAG_FILENAME   = b"\x00\x00\x00\x04"
_HIST_TAG_TITLE      = b"\x00\x00\x00\x06"
_HIST_TAG_ARTIST     = b"\x00\x00\x00\x07"
_HIST_TAG_ALBUM      = b"\x00\x00\x00\x08"
_HIST_TAG_GENRE      = b"\x00\x00\x00\x09"
_HIST_TAG_DURATION   = b"\x00\x00\x00\x0a"
_HIST_TAG_BITRATE    = b"\x00\x00\x00\x0d"
_HIST_TAG_BPM        = b"\x00\x00\x00\x0f"
_HIST_TAG_YEAR       = b"\x00\x00\x00\x17"
_HIST_TAG_PLAY_START = b"\x00\x00\x00\x1c"
_HIST_TAG_PLAY_END   = b"\x00\x00\x00\x1d"
_HIST_TAG_DECK_NUM   = b"\x00\x00\x00\x1f"
_HIST_TAG_PLAYED     = b"\x00\x00\x00\x27"

_HIST_STR_TAGS = {
    _HIST_TAG_FULL_PATH, _HIST_TAG_DIR, _HIST_TAG_FILENAME,
    _HIST_TAG_TITLE, _HIST_TAG_ARTIST, _HIST_TAG_ALBUM,
    _HIST_TAG_GENRE, _HIST_TAG_DURATION, _HIST_TAG_BITRATE,
    _HIST_TAG_YEAR,
}
_HIST_U32_TAGS = {
    _HIST_TAG_PLAY_START, _HIST_TAG_PLAY_END, _HIST_TAG_BPM,
}
_HIST_U16_TAGS = {
    _HIST_TAG_DECK_NUM,
}


def _parse_session_entry(buf: bytes, start: int, end: int) -> Optional[dict]:
    """Parse one `oent` payload region into a dict of named fields."""
    out: Dict[str, Any] = {}
    # oent wraps a single adat container
    inner = list(_iter_tagged(buf, start, end))
    if not inner:
        return None
    # Walk all sub-tags of the wrapping adat (or the oent itself if
    # adat is missing — defensive)
    for outer_tag, ops, ope in inner:
        if outer_tag == b"adat":
            walk_start, walk_end = ops, ope
        else:
            walk_start, walk_end = start, end
        for stag, sps, spe in _iter_tagged(buf, walk_start, walk_end):
            payload = buf[sps:spe]
            if stag in _HIST_STR_TAGS:
                s = _decode_utf16_be(payload)
                if s is None:
                    continue
                if stag == _HIST_TAG_FULL_PATH:
                    out["full_path"] = s
                elif stag == _HIST_TAG_DIR:
                    out["directory"] = s
                elif stag == _HIST_TAG_FILENAME:
                    out["filename"] = s
                elif stag == _HIST_TAG_TITLE:
                    out["title"] = s
                elif stag == _HIST_TAG_ARTIST:
                    out["artist"] = s
                elif stag == _HIST_TAG_ALBUM:
                    out["album"] = s
                elif stag == _HIST_TAG_GENRE:
                    out["genre"] = s
                elif stag == _HIST_TAG_DURATION:
                    out["duration_str"] = s
                elif stag == _HIST_TAG_BITRATE:
                    out["bitrate"] = s
                elif stag == _HIST_TAG_YEAR:
                    out["year"] = s
            elif stag in _HIST_U32_TAGS and len(payload) == 4:
                v = struct.unpack(">I", payload)[0]
                if stag == _HIST_TAG_PLAY_START:
                    out["play_start_ts"] = v
                elif stag == _HIST_TAG_PLAY_END:
                    out["play_end_ts"] = v
                elif stag == _HIST_TAG_BPM:
                    out["bpm"] = v if v else None
            elif stag in _HIST_U16_TAGS and len(payload) == 4:
                v = struct.unpack(">I", payload)[0]
                if stag == _HIST_TAG_DECK_NUM:
                    out["deck"] = v
            elif stag == _HIST_TAG_PLAYED and len(payload) == 1:
                out["played"] = bool(payload[0])
        break  # only the first adat is the actual record
    return out or None


def read_session(session_path: str) -> Optional[dict]:
    """Parse a single `.session` file from
    `~/Music/_Serato_/History/Sessions/`.

    Returns:
      {
        "path": str,
        "version": Optional[str],     # vrsn payload (UTF-16 BE)
        "entries": [ {fields...}, ... ]
      }
    or None on failure.
    """
    try:
        buf = open(session_path, "rb").read()
    except (OSError, IOError):
        return None
    version: Optional[str] = None
    entries: List[dict] = []
    for tag, ps, pe in _iter_tagged(buf):
        if tag == b"vrsn":
            version = _decode_utf16_be(buf[ps:pe])
        elif tag == b"oent":
            rec = _parse_session_entry(buf, ps, pe)
            if rec:
                entries.append(rec)
    return {
        "path": session_path,
        "version": version,
        "entries": entries,
    }


def list_serato_sessions(serato_root: str = DEFAULT_SERATO_ROOT
                          ) -> List[str]:
    """Return absolute paths of every `.session` file under
    `<serato_root>/History/Sessions/`, sorted by mtime (newest first).
    """
    folder = os.path.join(serato_root, "History", "Sessions")
    if not os.path.isdir(folder):
        return []
    paths = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(".session"):
            paths.append(os.path.join(folder, fn))
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths


def read_history_sessions(serato_root: str = DEFAULT_SERATO_ROOT,
                            limit: Optional[int] = None) -> List[dict]:
    """Read all sessions under `<serato_root>/History/Sessions/`.

    Returns list of session dicts (most recent first).  Pass `limit`
    to cap the number of sessions parsed (newest N).
    """
    paths = list_serato_sessions(serato_root)
    if limit is not None:
        paths = paths[:limit]
    out: List[dict] = []
    for p in paths:
        sess = read_session(p)
        if sess is not None:
            out.append(sess)
    return out


# =====================================================================
# B7 — Recursive Subcrates walker
# =====================================================================
# Serato encodes crate hierarchy via `%%` separators
# in the .crate FILENAME (e.g. "0 META FL%%hip hop 2012.crate" =
# "0 META FL" → "hip hop 2012") AND can also drop crate files into
# nested directories.  Existing `list_crates()` only scans the
# immediate `Subcrates/` directory.  This SPOT walker:
#   • recurses through `Subcrates/`, `SmartCrates/`, and any sibling
#     directories matching `Subcrates*` (e.g. "Subcrates JINXED",
#     "Subcrates___").
#   • returns the same (display_name, absolute_path) shape as
#     list_crates() for drop-in compatibility.

def _crate_display_with_source(crate_path: str, serato_root: str,
                                 source_top: str) -> str:
    """Build a display name for a crate file, prefixed with the
    source-folder tag if the crate is NOT in the canonical
    Subcrates/ (or SmartCrates/) top-level directory.

    Examples:
      Subcrates/Foo.crate            → "Foo"
      Subcrates JINXED/Foo.crate     → "[JINXED] Foo"
      Subcrates___/Videos/Foo.crate  → "[___ ▸ Videos] Foo"
    """
    name = _crate_name_from_filename(crate_path)
    rel_dir = os.path.relpath(os.path.dirname(crate_path),
                                os.path.join(serato_root, source_top))
    parts: List[str] = []
    if source_top not in ("Subcrates", "SmartCrates"):
        suffix = source_top[len("Subcrates"):].strip()
        if suffix:
            parts.append(suffix)
    if rel_dir not in (".", ""):
        for seg in rel_dir.split(os.sep):
            if seg and seg != ".":
                parts.append(seg)
    if parts:
        return f"[{' ▸ '.join(parts)}] {name}"
    return name


def list_crates_recursive(serato_root: str = DEFAULT_SERATO_ROOT,
                            include_smart: bool = True,
                            include_aux_dirs: bool = True
                            ) -> List[Tuple[str, str]]:
    """Recursive variant of list_crates() — walks the full crate tree.

    Disambiguates duplicate filenames found in sibling `Subcrates*`
    directories by tagging the display name with the source folder
    (e.g. "[JINXED] 0 META").

    Args:
      serato_root:      base `_Serato_` directory.
      include_smart:    include `.scrate` files from SmartCrates/.
      include_aux_dirs: also scan sibling `Subcrates*` directories
                        (e.g. "Subcrates JINXED", "Subcrates___").
                        Set False to limit to the canonical
                        `Subcrates/` + `SmartCrates/` pair.
    """
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(serato_root):
        return out
    # Decide which top-level dirs to walk
    top_dirs: List[Tuple[str, str, str]] = []   # (full_path, ext, top_name)
    for entry in sorted(os.listdir(serato_root)):
        full = os.path.join(serato_root, entry)
        if not os.path.isdir(full):
            continue
        if entry == "Subcrates":
            top_dirs.append((full, ".crate", entry))
        elif entry == "SmartCrates" and include_smart:
            top_dirs.append((full, ".scrate", entry))
        elif include_aux_dirs and entry.startswith("Subcrates") \
                and entry != "Subcrates":
            top_dirs.append((full, ".crate", entry))
    for root, ext, top_name in top_dirs:
        for dirpath, _dirs, files in os.walk(root):
            for fn in sorted(files):
                if not fn.lower().endswith(ext):
                    continue
                full = os.path.join(dirpath, fn)
                display = _crate_display_with_source(full, serato_root,
                                                       top_name)
                out.append((display, full))
    out.sort(key=lambda x: x[0])
    return out


# =====================================================================
# B7 — _Serato_ folder survey (informational SPOT)
# =====================================================================

# catalog of folders / file kinds found under
# `~/Music/_Serato_/`.  Used by `list_serato_artifacts()` to inventory
# the full Serato folder for diagnostics and to inform future readers.

SERATO_FOLDER_ROLES = {
    "Subcrates":      ("crates",       "User-defined crates (.crate)"),
    "SmartCrates":    ("crates",       "Smart crates (.scrate) — rule-based"),
    "History":        ("history",      "Played-track sessions + history.database"),
    "Recording":      ("recording",    "Audio recordings of DJ sets"),
    "Recording temp": ("recording",    "In-progress recording temp files"),
    "Auto Import":    ("import",       "Tracks queued for auto-import"),
    "Imported":       ("import",       "Tracks imported into the Serato library"),
    "Lexicon":        ("third_party",  "Lexicon DJ cross-app exports"),
    "Logs":           ("diagnostic",   "Serato application logs"),
    "MIDI":           ("control",      "MIDI mapping files"),
    "Metadata":       ("internal",     "Internal metadata cache"),
    "Reports":        ("reports",      "Playback / session reports"),
    "ReportsLite":    ("reports",      "Reports (Lite)"),
    "ReportsVideo":   ("reports",      "Reports (Video edition)"),
    "SeratoVideo":    ("video",        "Video-DJ assets"),
    "Effects":        ("control",      "Effects presets"),
    "Pulselocker":    ("third_party",  "Pulselocker (deprecated streaming)"),
    "Export Backups": ("internal",     "Export backups"),
    "History Export": ("history",      "Exported history"),
}

SERATO_FILE_ROLES = {
    ".pref":          "Preferences (binary plist or proprietary)",
    ".gai":           "Global app index (DJ.gai / DJLite.gai)",
    ".log":           "Diagnostic log (DropoutCount, MicManager, …)",
    ".database":      "Database file (database V2 / history.database / Remotes.database)",
    ".crate":         "Crate (under Subcrates/, see read_crate)",
    ".scrate":        "Smart crate (under SmartCrates/, see read_smart_crate)",
    ".session":       "Played-session (under History/Sessions/, see read_session)",
}


def list_serato_artifacts(serato_root: str = DEFAULT_SERATO_ROOT
                            ) -> List[dict]:
    """Inventory every readable artifact under the Serato root.

    Returns a list of dicts: {path, kind, size, role} where kind is
    the file extension (or "dir") and role is a human-readable tag
    from SERATO_FOLDER_ROLES / SERATO_FILE_ROLES.

    Useful for diagnostics ("what's in my Serato folder?") and as a
    starting point for new SPOT readers.
    """
    out: List[dict] = []
    if not os.path.isdir(serato_root):
        return out
    for entry in sorted(os.listdir(serato_root)):
        full = os.path.join(serato_root, entry)
        try:
            sz = os.path.getsize(full)
        except OSError:
            sz = 0
        if os.path.isdir(full):
            role = SERATO_FOLDER_ROLES.get(entry,
                                            ("unknown", "Unrecognized folder"))[1]
            out.append({
                "path": full,
                "kind": "dir",
                "size": sz,
                "role": role,
            })
        else:
            _, ext = os.path.splitext(entry.lower())
            role = SERATO_FILE_ROLES.get(ext, "Unrecognized file type")
            out.append({
                "path": full,
                "kind": ext or "(no-ext)",
                "size": sz,
                "role": role,
            })
    return out

def detect_library_format(serato_folder: str) -> dict:
    """Classify a Serato library folder by generation.

    Serato DJ 4.0 (2025) moved the library store from the binary
    `database V2` + `Subcrates/*.crate` files to an SQLite database
    and marks the old files [LEGACY]. The exact SQLite schema is not
    publicly documented, so detection is deliberately content-based:
    any file whose first 16 bytes are the SQLite magic counts,
    whatever Serato names it in a given build.

    Returns a dict with `legacy_database_v2` (path or None),
    `sqlite_files` (paths), `crates` (count of .crate files) and a
    `generation` verdict: "pre-4.0", "4.0", "4.0-with-legacy" or
    "empty". Everything this module writes targets the LEGACY store;
    on a "4.0"-classified library those writes are not authoritative.
    """
    import glob as _glob
    root = os.path.abspath(os.path.expanduser(serato_folder))
    legacy = os.path.join(root, "database V2")
    legacy = legacy if os.path.isfile(legacy) else None
    sqlite_files = []
    for path in _glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if not os.path.isfile(path) or os.path.getsize(path) < 16:
            continue
        try:
            with open(path, "rb") as fh:
                if fh.read(16) == b"SQLite format 3\x00":
                    sqlite_files.append(path)
        except OSError:
            continue
    crates = len(_glob.glob(os.path.join(root, "Subcrates", "*.crate")))
    if sqlite_files and legacy:
        gen = "4.0-with-legacy"
    elif sqlite_files:
        gen = "4.0"
    elif legacy:
        gen = "pre-4.0"
    else:
        gen = "empty"
    return {"folder": root, "generation": gen,
            "legacy_database_v2": legacy,
            "sqlite_files": sqlite_files, "crates": crates}
