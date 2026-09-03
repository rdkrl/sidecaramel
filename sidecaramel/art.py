"""sidecaramel.art — embedded cover-art read + write across containers.

Supports:
    MP3 / WAV / AIFF (ID3 APIC):  read all APIC frames, write/replace
                                    the front-cover APIC.
    MP4 / M4A / M4V (covr atom):  read first cover, write/replace covr.
    FLAC (Picture block):         read all Picture blocks, write/clear
                                    + add a single FRONT-cover picture.
    OGG/OPUS:                      read METADATA_BLOCK_PICTURE Vorbis
                                    comment (base64-encoded FLAC
                                    Picture block); write same.

All writers are gated by `confirm=True` (confirm-flag convention).

Picture MIME inferred from bytes (sniff first 16 bytes):
    JPEG: starts with `\\xff\\xd8\\xff`
    PNG:  starts with `\\x89PNG\\r\\n\\x1a\\n`
    GIF:  starts with `GIF87a` / `GIF89a`
    WEBP: starts with `RIFF`...`WEBP`
Otherwise falls back to `application/octet-stream`.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from mutagen import MutagenError
from mutagen.id3 import ID3, APIC, ID3NoHeaderError
from mutagen.mp4 import MP4, MP4Cover, MP4StreamInfoError
from mutagen.flac import FLAC, Picture

from sidecaramel.confirm_gate import _require_confirm


# APIC type 3 = "Cover (front)" per ID3 spec / FLAC PICTURE block-type.
PICTURE_TYPE_FRONT_COVER = 3


def _sniff_image_mime(data: bytes) -> str:
    """Infer image MIME type from first bytes.  Falls back to
    application/octet-stream if unrecognised."""
    if not data or len(data) < 8:
        return "application/octet-stream"
    head = data[:16]
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


# ---------------------------------------------------------------------
# READ
# ---------------------------------------------------------------------

def read_cover_art(audio_path: str) -> Optional[Tuple[str, bytes]]:
    """Return (mime, data) of the first front-cover picture, or None.

    For MP3/WAV/AIFF (ID3): the FIRST APIC frame of type 3 (front-cover),
    or the first APIC frame of any type if no front-cover.
    For MP4/M4A/M4V: the first `covr` cover.
    For FLAC: the FIRST PICTURE block of type 3, else first of any.
    For OGG/OPUS: the first METADATA_BLOCK_PICTURE Vorbis comment.
    """
    if not audio_path or not os.path.isfile(audio_path):
        return None
    ext = os.path.splitext(audio_path)[1].lower()

    if ext in (".mp3", ".wav", ".aiff", ".aif"):
        try:
            id3 = ID3(audio_path)
        except (ID3NoHeaderError, MutagenError, Exception):
            return None
        front_apic = None
        any_apic = None
        for frame in id3.getall("APIC"):
            data = bytes(getattr(frame, "data", b"") or b"")
            mime = getattr(frame, "mime", "") or _sniff_image_mime(data)
            ptype = getattr(frame, "type", 0)
            if any_apic is None and data:
                any_apic = (mime, data)
            if ptype == PICTURE_TYPE_FRONT_COVER and data:
                front_apic = (mime, data)
                break
        return front_apic or any_apic

    if ext in (".mp4", ".m4a", ".m4v"):
        try:
            mp4 = MP4(audio_path)
        except (MP4StreamInfoError, MutagenError, Exception):
            return None
        if mp4 is None or mp4.tags is None or "covr" not in mp4.tags:
            return None
        covers = mp4.tags["covr"]
        if not covers:
            return None
        cover = covers[0]
        data = bytes(cover)
        # MP4Cover imageformat: MP4Cover.FORMAT_JPEG=13 / PNG=14
        fmt = getattr(cover, "imageformat", None)
        if fmt == MP4Cover.FORMAT_JPEG:
            mime = "image/jpeg"
        elif fmt == MP4Cover.FORMAT_PNG:
            mime = "image/png"
        else:
            mime = _sniff_image_mime(data)
        return (mime, data)

    if ext in (".flac",):
        try:
            flac = FLAC(audio_path)
        except (MutagenError, Exception):
            return None
        if flac is None:
            return None
        pics = list(flac.pictures or [])
        if not pics:
            return None
        front = next((p for p in pics if p.type == PICTURE_TYPE_FRONT_COVER),
                       None) or pics[0]
        return (front.mime or _sniff_image_mime(front.data),
                bytes(front.data))

    if ext in (".ogg", ".oga", ".opus"):
        import base64
        try:
            from mutagen import File as _MF
            f = _MF(audio_path)
        except Exception:
            return None
        if f is None or f.tags is None:
            return None
        for key in ("METADATA_BLOCK_PICTURE", "metadata_block_picture"):
            if key in f.tags:
                v = f.tags[key]
                if isinstance(v, list) and v:
                    try:
                        pic = Picture(base64.b64decode(v[0]))
                        return (pic.mime
                                  or _sniff_image_mime(bytes(pic.data)),
                                bytes(pic.data))
                    except Exception:
                        return None
        return None

    return None


def read_all_cover_art(audio_path: str) -> List[Tuple[int, str, bytes]]:
    """Return every embedded cover as [(picture_type, mime, data), ...].

    Picture types follow the ID3/FLAC convention:
        0  = other
        3  = front cover
        4  = back cover
        5  = leaflet
        6  = media (label)
        ...
    For MP4 (which has no per-cover type), every cover is reported as
    type 3 (front cover).
    """
    if not audio_path or not os.path.isfile(audio_path):
        return []
    ext = os.path.splitext(audio_path)[1].lower()
    out: List[Tuple[int, str, bytes]] = []
    if ext in (".mp3", ".wav", ".aiff", ".aif"):
        try:
            id3 = ID3(audio_path)
        except (ID3NoHeaderError, MutagenError, Exception):
            return []
        for frame in id3.getall("APIC"):
            data = bytes(getattr(frame, "data", b"") or b"")
            if not data:
                continue
            out.append((int(getattr(frame, "type", 0)),
                          getattr(frame, "mime", "")
                          or _sniff_image_mime(data),
                          data))
        return out
    if ext in (".mp4", ".m4a", ".m4v"):
        try:
            mp4 = MP4(audio_path)
        except (MP4StreamInfoError, MutagenError, Exception):
            return []
        if mp4 is None or mp4.tags is None or "covr" not in mp4.tags:
            return []
        for cover in mp4.tags["covr"]:
            data = bytes(cover)
            fmt = getattr(cover, "imageformat", None)
            mime = ("image/jpeg" if fmt == MP4Cover.FORMAT_JPEG
                     else "image/png" if fmt == MP4Cover.FORMAT_PNG
                     else _sniff_image_mime(data))
            out.append((PICTURE_TYPE_FRONT_COVER, mime, data))
        return out
    if ext == ".flac":
        try:
            flac = FLAC(audio_path)
        except (MutagenError, Exception):
            return []
        for pic in flac.pictures or []:
            out.append((int(pic.type),
                          pic.mime or _sniff_image_mime(bytes(pic.data)),
                          bytes(pic.data)))
        return out
    return out


# ---------------------------------------------------------------------
# WRITE
# ---------------------------------------------------------------------

def write_cover_art(audio_path: str, image_data: bytes,
                       *, mime: Optional[str] = None,
                       picture_type: int = PICTURE_TYPE_FRONT_COVER,
                       desc: str = "",
                       replace: bool = True,
                       confirm: bool = False) -> bool:
    """Embed a cover-art image in any supported container.

    `image_data` raw image bytes (JPEG, PNG, GIF, WEBP).
    `mime` inferred from bytes if not provided.
    `picture_type` per ID3/FLAC convention (default 3 = front cover).
    `desc` description string (ID3 + FLAC only).
    `replace` (default True): replace ALL existing covers.  If False,
              append a new one (ID3/FLAC only; MP4 always replaces).

    Returns True on success.  Confirm-flag check: caller MUST pass
    `confirm=True`.
    """
    _require_confirm(confirm, "write_cover_art", audio_path)
    if not audio_path or not os.path.isfile(audio_path):
        return False
    if not image_data:
        return False
    if mime is None:
        mime = _sniff_image_mime(image_data)
    ext = os.path.splitext(audio_path)[1].lower()

    if ext in (".mp3", ".wav", ".aiff", ".aif"):
        try:
            id3 = ID3(audio_path)
        except ID3NoHeaderError:
            try:
                id3 = ID3()
            except Exception:
                return False
        except (MutagenError, Exception):
            return False
        try:
            if replace:
                id3.delall("APIC")
            id3.add(APIC(encoding=3, mime=mime, type=picture_type,
                            desc=desc, data=image_data))
            id3.save(audio_path)
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
            fmt = (MP4Cover.FORMAT_JPEG if mime == "image/jpeg"
                   else MP4Cover.FORMAT_PNG if mime == "image/png"
                   else MP4Cover.FORMAT_JPEG)
            mp4.tags["covr"] = [MP4Cover(image_data, imageformat=fmt)]
            mp4.save()
            return True
        except Exception:
            return False

    if ext == ".flac":
        try:
            flac = FLAC(audio_path)
        except (MutagenError, Exception):
            return False
        try:
            if replace:
                flac.clear_pictures()
            pic = Picture()
            pic.type = int(picture_type)
            pic.mime = mime
            pic.desc = desc
            pic.data = image_data
            flac.add_picture(pic)
            flac.save()
            return True
        except Exception:
            return False

    if ext in (".ogg", ".oga", ".opus"):
        import base64
        try:
            from mutagen import File as _MF
            f = _MF(audio_path)
        except Exception:
            return False
        if f is None:
            return False
        try:
            if f.tags is None:
                return False
            pic = Picture()
            pic.type = int(picture_type)
            pic.mime = mime
            pic.desc = desc
            pic.data = image_data
            encoded = base64.b64encode(pic.write()).decode("ascii")
            if replace:
                f.tags["METADATA_BLOCK_PICTURE"] = [encoded]
            else:
                existing = list(f.tags.get(
                    "METADATA_BLOCK_PICTURE", []))
                existing.append(encoded)
                f.tags["METADATA_BLOCK_PICTURE"] = existing
            f.save()
            return True
        except Exception:
            return False

    return False


def clear_cover_art(audio_path: str, *, confirm: bool = False) -> bool:
    """Remove ALL embedded cover-art from the file.  Confirm-flag check."""
    _require_confirm(confirm, "clear_cover_art", audio_path)
    if not audio_path or not os.path.isfile(audio_path):
        return False
    ext = os.path.splitext(audio_path)[1].lower()
    try:
        if ext in (".mp3", ".wav", ".aiff", ".aif"):
            try:
                id3 = ID3(audio_path)
            except ID3NoHeaderError:
                return True
            id3.delall("APIC")
            id3.save(audio_path)
            return True
        if ext in (".mp4", ".m4a", ".m4v"):
            mp4 = MP4(audio_path)
            if mp4.tags is None:
                return True
            if "covr" in mp4.tags:
                del mp4.tags["covr"]
                mp4.save()
            return True
        if ext == ".flac":
            flac = FLAC(audio_path)
            flac.clear_pictures()
            flac.save()
            return True
        if ext in (".ogg", ".oga", ".opus"):
            from mutagen import File as _MF
            f = _MF(audio_path)
            if f and f.tags and "METADATA_BLOCK_PICTURE" in f.tags:
                del f.tags["METADATA_BLOCK_PICTURE"]
                f.save()
            return True
    except Exception:
        return False
    return False
