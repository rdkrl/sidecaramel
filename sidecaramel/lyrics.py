"""sidecaramel.lyrics — embedded lyric I/O across containers.

Reads + writes timed (LRC / SYLT) and untimed (USLT / ©lyr / Vorbis
LYRICS) lyric fields across MP3, WAV, AIFF, MP4, M4A, M4V, FLAC, OGG.
Container-agnostic dispatchers, mutagen-backed.

The public API is re-exported from `sidecaramel.__init__` for
convenience.  All write functions are gated by `confirm=True` to
prevent accidental modification of user library files (same B5
convention as Serato writers).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from mutagen import File as MutagenFile
from mutagen import MutagenError
from mutagen.id3 import ID3, SYLT, USLT, ID3NoHeaderError
from mutagen.mp4 import MP4, MP4StreamInfoError

from sidecaramel.confirm_gate import _require_confirm


# ---------------------------------------------------------------------
# Untimed lyric readers
# ---------------------------------------------------------------------

def read_embedded_lyrics(audio_path: str) -> Optional[str]:
    """Read the embedded lyric text from any supported container.

    Reads (in container-priority order):
      MP3/WAV/AIFF:  SYLT then USLT (timed beats untimed)
      MP4/M4A/M4V:   ©lyr atom, then `----:com.apple.iTunes:LYRICS`
                       freeform atom
      FLAC/OGG:      LYRICS Vorbis comment

    Returns the lyric text as a string, or None if none present.
    """
    import os
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in (".mp3", ".wav", ".aiff", ".aif"):
        return read_id3_lyrics(audio_path)
    if ext in (".mp4", ".m4a", ".m4v", ".mov"):
        try:
            mp4 = MP4(audio_path)
        except (MP4StreamInfoError, MutagenError, Exception):
            return None
        if mp4 is None or mp4.tags is None:
            return None
        if "\xa9lyr" in mp4.tags:
            v = mp4.tags["\xa9lyr"]
            if isinstance(v, list) and v:
                return str(v[0]) if v[0] else None
        # iTunes-style freeform atom
        for key in mp4.tags.keys():
            ks = str(key)
            if ks == "----:com.apple.iTunes:LYRICS":
                v = mp4.tags[key]
                if isinstance(v, list) and v:
                    return v[0].decode("utf-8", errors="replace")
        return None
    if ext in (".flac", ".ogg", ".oga", ".opus"):
        try:
            f = MutagenFile(audio_path)
        except (MutagenError, Exception):
            return None
        if f is None or f.tags is None:
            return None
        for key in ("LYRICS", "lyrics", "Lyrics", "unsyncedlyrics"):
            if key in f.tags:
                v = f.tags[key]
                if isinstance(v, list) and v:
                    return str(v[0])
                if isinstance(v, str):
                    return v
        return None
    return None


def _open_id3_container(path: str):
    """Open MP3 / WAV / AIFF and return a container with `.tags`
    (ID3-like) and `.save()` that persists to the correct envelope:
    bare ID3v2 for MP3, `id3 ` chunk inside RIFF for WAV, `ID3 `
    chunk inside FORM for AIFF.

    Raises whatever mutagen raises if the file is malformed.
    """
    import os
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
    try:
        id3 = ID3(path)
    except ID3NoHeaderError:
        id3 = ID3()

    class _MP3IDContainer:
        def __init__(self, tags, path):
            self.tags = tags
            self._path = path

        def save(self):
            self.tags.save(self._path)
    return _MP3IDContainer(id3, path)


def _open_vorbis_container(path: str):
    """Open FLAC / OGG / Opus and return a mutagen object with
    `.tags` (VComment, dict-like) and `.save()`.

    Routes by extension:
        .flac          → mutagen.flac.FLAC
        .ogg / .oga    → mutagen.oggvorbis.OggVorbis
        .opus          → mutagen.oggopus.OggOpus
    """
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext == ".flac":
        from mutagen.flac import FLAC
        return FLAC(path)
    if ext == ".opus":
        from mutagen.oggopus import OggOpus
        return OggOpus(path)
    if ext in (".ogg", ".oga"):
        from mutagen.oggvorbis import OggVorbis
        return OggVorbis(path)
    raise ValueError(f"unsupported Vorbis-container extension: {ext}")


def read_id3_lyrics(audio_path: str) -> Optional[str]:
    """Read ID3 lyrics from MP3/WAV/AIFF.  SYLT > USLT priority.

    SYLT (Synchronized Lyrics) is timed line/word data; we collapse
    it to plain text for the untimed reader.  USLT is plain text.
    """
    try:
        c = _open_id3_container(audio_path)
        id3 = c.tags
    except Exception:
        return None
    if id3 is None:
        return None
    # SYLT first (timed); flatten to text
    sylt_frames = id3.getall("SYLT")
    if sylt_frames:
        for frame in sylt_frames:
            entries = getattr(frame, "text", None)
            if entries:
                # entries = [(text, time_ms), ...]
                parts = []
                for item in entries:
                    if isinstance(item, (tuple, list)) and len(item) >= 1:
                        parts.append(str(item[0]).rstrip("\n"))
                if parts:
                    return "\n".join(parts)
    # USLT (untimed)
    uslt_frames = id3.getall("USLT")
    for frame in uslt_frames:
        text = getattr(frame, "text", None)
        if text:
            return text
    return None


def read_embedded_lyrics_with_format(audio_path: str
                                       ) -> Optional[Tuple[str, str]]:
    """Read embedded lyrics and report their FORMAT.

    Returns (text, format) where format is one of:
        "lrc"      — LRC-style timestamped lines `[mm:ss.xx]Text`
        "sylt"     — SYLT frame (timed at line/word level)
        "plain"    — untimed plain text
        None       — no lyrics
    """
    text = read_embedded_lyrics(audio_path)
    if not text:
        return None
    if is_lrc_text(text):
        return (text, "lrc")
    return (text, "plain")


_LRC_TIMESTAMP_RE = re.compile(r"\[\d+:\d+(?:\.\d+)?\]")


def is_lrc_text(text: Optional[str], min_hits: int = 3) -> bool:
    """Heuristic: text counts as LRC if it has at least `min_hits`
    `[mm:ss.xx]` timestamps."""
    if not text:
        return False
    return len(_LRC_TIMESTAMP_RE.findall(text)) >= min_hits


# ---------------------------------------------------------------------
# Timed (synced) lyric reader — LRC output from SYLT
# ---------------------------------------------------------------------

def _ms_to_lrc_timestamp(ms: int) -> str:
    """`53456` → `'[00:53.45]'` (mm:ss.cs where cs = centiseconds)."""
    total_cs = (ms + 5) // 10
    mm, rem = divmod(total_cs, 6000)
    ss, cs = divmod(rem, 100)
    return f"[{mm:02d}:{ss:02d}.{cs:02d}]"


_LRC_TS_RE = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]")


def _parse_lrc_text_to_pairs(lrc_text: str
                                ) -> List[Tuple[int, str]]:
    """Parse LRC text into (time_ms, line_text) tuples, sorted."""
    out: List[Tuple[int, str]] = []
    if not lrc_text:
        return out
    for raw in lrc_text.split("\n"):
        line = raw.rstrip()
        if not line:
            continue
        ts_matches = list(_LRC_TS_RE.finditer(line))
        if not ts_matches:
            continue
        # Skip header tags like [ar:...], [ti:...]
        text_start = ts_matches[-1].end()
        text = line[text_start:]
        for m in ts_matches:
            mm = int(m.group(1)); ss = int(m.group(2))
            frac = m.group(3) or "0"
            if len(frac) == 2:
                ms = int(frac) * 10
            elif len(frac) == 3:
                ms = int(frac)
            else:
                ms = int(frac.ljust(3, "0")[:3])
            t = mm * 60000 + ss * 1000 + ms
            out.append((t, text))
    out.sort(key=lambda x: x[0])
    return out


def read_synced_lyrics(audio_path: str
                          ) -> Optional[List[Tuple[int, str]]]:
    """Read time-aligned lyrics as [(time_ms, text), ...].

    Sources (in priority):
      MP3/WAV/AIFF SYLT frame  →  native timed data
      else fall back to: untimed lyrics text that happens to be LRC

    Returns None if no timed data available.
    """
    import os
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in (".mp3", ".wav", ".aiff", ".aif"):
        try:
            c = _open_id3_container(audio_path)
            id3 = c.tags
        except Exception:
            id3 = None
        if id3:
            sylt_frames = id3.getall("SYLT")
            for frame in sylt_frames:
                entries = getattr(frame, "text", None)
                if not entries:
                    continue
                out: List[Tuple[int, str]] = []
                for item in entries:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        text = str(item[0]).rstrip("\n")
                        t_ms = int(item[1])
                        out.append((t_ms, text))
                if out:
                    out.sort(key=lambda x: x[0])
                    return out
    # Fallback: untimed lyrics text that happens to be LRC
    plain = read_embedded_lyrics(audio_path)
    if plain and is_lrc_text(plain):
        return _parse_lrc_text_to_pairs(plain)
    return None


def read_synced_lyrics_as_lrc(audio_path: str) -> Optional[str]:
    """Convenience: timed lyrics serialized as LRC text."""
    pairs = read_synced_lyrics(audio_path)
    if not pairs:
        return None
    return "\n".join(f"{_ms_to_lrc_timestamp(t)}{text}"
                       for t, text in pairs)


# ---------------------------------------------------------------------
# Writers (untimed text; Confirm-flag check)
# ---------------------------------------------------------------------

def write_lyrics(audio_path: str, lyrics_text: str,
                   lang: str = "eng", desc: str = "",
                   *, confirm: bool = False) -> bool:
    """Write embedded **untimed** lyrics to any supported container.

    Despite the historical ``write_uslt`` name (kept as a deprecated
    alias below), this writer is container-aware: it picks the right
    underlying frame per format.

    Container-side conventions:
        MP3 / WAV / AIFF (ID3): USLT frame, replaces any existing.
        MP4 / M4A / M4V:        ``©lyr`` atom.
        FLAC / OGG / OPUS:      ``LYRICS`` Vorbis comment.

    Args:
        audio_path: target file.
        lyrics_text: plain UTF-8 text.  Returned ``False`` if empty.
        lang: 3-letter ISO-639-2 code, used only for the ID3 USLT
            frame.  Ignored on MP4/Vorbis.
        desc: USLT descriptor, used only on ID3.  Ignored elsewhere.
        confirm: caller MUST pass ``confirm=True``.  Without it,
            :class:`RuntimeError` is raised — see
            :func:`sidecaramel.confirm_gate._require_confirm`.

    Returns:
        ``True`` on success, ``False`` on any failure (silent —
        scheduled for the typed-exception refactor in 0.2.0).

    Timed (SYLT) lyrics are **not** writable in 0.1.0; only readers
    exist.  See ``README`` Known Limitations.
    """
    import os
    _require_confirm(confirm, "write_lyrics", audio_path)
    if not audio_path or not os.path.isfile(audio_path):
        return False
    ext = os.path.splitext(audio_path)[1].lower()
    text = (lyrics_text or "").strip()
    if not text:
        return False

    if ext in (".mp3", ".wav", ".aiff", ".aif"):
        try:
            c = _open_id3_container(audio_path)
        except Exception:
            return False
        try:
            c.tags.delall("USLT")
            c.tags.add(USLT(encoding=3, lang=lang, desc=desc,
                                text=text))
            c.save()
            return True
        except Exception:
            return False

    if ext in (".mp4", ".m4a", ".m4v"):
        try:
            mp4 = MP4(audio_path)
        except (MP4StreamInfoError, MutagenError, Exception):
            return False
        if mp4 is None:
            return False
        try:
            if mp4.tags is None:
                mp4.add_tags()
            mp4.tags["\xa9lyr"] = [text]
            mp4.save()
            return True
        except Exception:
            return False

    if ext in (".flac", ".ogg", ".oga", ".opus"):
        try:
            container = _open_vorbis_container(audio_path)
        except Exception:
            return False
        try:
            container["LYRICS"] = text
            container.save()
            return True
        except Exception:
            return False

    return False


def write_uslt(audio_path: str, lyrics_text: str,
                lang: str = "eng", desc: str = "",
                *, confirm: bool = False) -> bool:
    """DEPRECATED alias for :func:`write_lyrics`.

    The name ``write_uslt`` encodes the ID3 USLT frame implementation
    detail, but this writer also handles MP4 ``©lyr`` and Vorbis
    ``LYRICS`` — different frames per container.  Use
    :func:`write_lyrics` instead.  This alias will be removed in
    0.2.0.
    """
    import warnings as _warnings
    _warnings.warn(
        "write_uslt is deprecated and will be removed in 0.2.0; "
        "use write_lyrics() — same signature and behaviour, just "
        "an honest name (the writer handles ©lyr / LYRICS too, not "
        "only USLT).",
        DeprecationWarning,
        stacklevel=2,
    )
    return write_lyrics(audio_path, lyrics_text,
                          lang=lang, desc=desc, confirm=confirm)


# ---------------------------------------------------------------------
# Timed (synced) lyric writer — ID3 SYLT
# ---------------------------------------------------------------------

def write_sylt(audio_path: str, pairs: List[Tuple[int, str]],
                 *, lang: str = "eng", desc: str = "",
                 confirm: bool = False) -> bool:
    """Write embedded **timed** lyrics as an ID3 SYLT frame.

    ID3-only (MP3 / WAV / AIFF) — MP4 and the Vorbis containers have
    no directly analogous timed-lyrics tag, so this returns ``False``
    there; use :func:`write_lrc_sidecar` for a container-agnostic
    timed-lyrics companion file instead.

    Args:
        audio_path: target file.
        pairs: ``[(time_ms, text), ...]`` — same shape
            :func:`read_synced_lyrics` returns.  Sorted by time
            before writing.  Returns ``False`` if empty.
        lang: 3-letter ISO-639-2 code.
        desc: SYLT descriptor (distinguishes multiple SYLT frames on
            one file); content type is fixed to "lyrics" (type=1).
        confirm: caller MUST pass ``confirm=True``.  Without it,
            :class:`RuntimeError` is raised — see
            :func:`sidecaramel.confirm_gate._require_confirm`.

    Returns:
        ``True`` on success, ``False`` on any failure or unsupported
        container (same silent-failure convention as
        :func:`write_lyrics`; see README Known Limitations).
    """
    import os
    _require_confirm(confirm, "write_sylt", audio_path)
    if not audio_path or not os.path.isfile(audio_path):
        return False
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in (".mp3", ".wav", ".aiff", ".aif"):
        return False
    ordered = sorted(pairs or [], key=lambda p: p[0])
    if not ordered:
        return False
    sylt_entries = [(str(text), int(t_ms)) for t_ms, text in ordered]

    try:
        c = _open_id3_container(audio_path)
    except Exception:
        return False
    try:
        c.tags.delall("SYLT")
        c.tags.add(SYLT(encoding=3, lang=lang, format=2, type=1,
                          desc=desc, text=sylt_entries))
        c.save()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------
# `.lrc` sidecar file I/O — container-agnostic timed lyrics companion
# ---------------------------------------------------------------------

def lrc_sidecar_path(audio_path: str) -> str:
    """Path of the `.lrc` sidecar file for `audio_path` (same base
    name, `.lrc` extension) — the convention
    `consolidate.SIDECAR_SUFFIXES` already moves alongside the audio
    file when consolidating a library.
    """
    import os
    return os.path.splitext(audio_path)[0] + ".lrc"


def read_lrc_sidecar(audio_path: str
                        ) -> Optional[List[Tuple[int, str]]]:
    """Read the `.lrc` sidecar file next to `audio_path`, if present.

    Returns ``[(time_ms, text), ...]`` sorted by time, or ``None`` if
    no sidecar file exists or it has no timestamped lines.
    """
    import os
    path = lrc_sidecar_path(audio_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None
    pairs = _parse_lrc_text_to_pairs(text)
    return pairs or None


def write_lrc_sidecar(audio_path: str, lyrics,
                        *, confirm: bool = False) -> bool:
    """Write an `.lrc` sidecar file next to `audio_path`.

    `lyrics` is either pre-formatted LRC text (``str``) or a
    ``[(time_ms, text), ...]`` pairs list (same shape
    :func:`read_synced_lyrics` returns) — pairs are formatted to LRC
    via the same timestamp formatting :func:`read_synced_lyrics_as_lrc`
    uses.

    Gated by ``confirm=True`` like every other writer in this module
    (see the module docstring): the audio file itself is untouched,
    but the sidecar write can silently overwrite a pre-existing
    `.lrc` file next to the track.

    Returns:
        ``True`` on success, ``False`` on empty input or any I/O
        failure (same silent-failure convention as the rest of this
        module).
    """
    import os
    _require_confirm(confirm, "write_lrc_sidecar", audio_path)
    if not audio_path or not os.path.isfile(audio_path):
        return False
    if isinstance(lyrics, str):
        text = lyrics.strip()
    else:
        pairs = sorted(lyrics or [], key=lambda p: p[0])
        if not pairs:
            return False
        text = "\n".join(f"{_ms_to_lrc_timestamp(t)}{line}"
                           for t, line in pairs)
    if not text:
        return False
    path = lrc_sidecar_path(audio_path)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        return True
    except OSError:
        return False
