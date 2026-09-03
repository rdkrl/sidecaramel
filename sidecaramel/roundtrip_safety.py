"""sidecaramel.roundtrip_safety — ID3/atom data-loss safety net for the
Serato writers.

Design spec: prove that applying a Serato
writer and then removing the edit returns the file byte-for-byte to its
starting state — i.e. a Serato write never silently drops a pre-existing
tag (title / artist / album / cover-art / lyrics / private frames / …).

The flow:

    target              the starting file (a TEMP COPY of a real file
                        or a synthetic rich file — NEVER an original)
    target  -> A        copy
    A       -> edit     apply a Serato writer (markers/cues/loops)
    A       -> B        copy (the "copy copy")
    B       -> unedit   strip the Serato tags back out
    B  ==?  target      MUST be byte-identical

When byte-identity does NOT hold, `roundtrip_check()` distinguishes:

  * REAL DATA LOSS   — a tag present in `target` is gone from `B`
                       (the serious finding)
  * COSMETIC DIFF    — every tag survives; only byte ordering, padding,
                       or mutagen re-serialisation differs

The decision is made by comparing a normalised *tag inventory* of
`target` vs `B` (frame ids + values for ID3, atom keys + values for
MP4, comment keys for Vorbis).  Inventory equal + bytes unequal =
cosmetic.  Inventory differs = data loss.

Write policy: this module only ever writes to caller-supplied temp
paths.  It does NOT open or write any original library file itself.
The CLI's `--from-copy` mode copies a real file into a tempdir FIRST
and treats the copy as the target; the original is opened read-only by
shutil.copy2 only.

Run:
    python -m sidecaramel.roundtrip_safety                # synthetic mp3+mp4
    python -m sidecaramel.roundtrip_safety --from-copy /path/to/song.mp3
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mutagen import File as MFile


# =====================================================================
# Tag inventory — normalised, container-agnostic snapshot of the
# NON-Serato user metadata, used to tell data-loss from cosmetic.
# =====================================================================

# Frame-id prefixes / atom keys that ARE the Serato edit and are thus
# expected to appear after edit and vanish after strip — excluded from
# the inventory so they don't muddy the data-loss check.
def _is_serato_id3_key(key: str) -> bool:
    ks = str(key)
    return ks.startswith("GEOB:") and "Serato" in ks


def _is_serato_mp4_key(key: str) -> bool:
    return "com.serato.dj" in str(key)


def _is_serato_vorbis_key(key: str) -> bool:
    return str(key).lower().startswith("serato_")


def tag_inventory(path: str) -> Dict[str, str]:
    """Return a normalised {key: repr-of-value} dict of all NON-Serato
    tags on the file, across containers.  Used to compare what user
    metadata survived a write/strip cycle.

    Returns {} for an untagged or unreadable file.
    """
    f = MFile(path)
    if f is None or f.tags is None:
        return {}
    inv: Dict[str, str] = {}
    for key in f.tags.keys():
        ks = str(key)
        if (_is_serato_id3_key(ks) or _is_serato_mp4_key(ks)
                or _is_serato_vorbis_key(ks)):
            continue
        val = f.tags[key]
        # Normalise to a stable string.  For ID3 frames we use the
        # frame's own repr of its data; for MP4/Vorbis it's the list
        # value.  Binary payloads (APIC/covr/pictures) are reduced to
        # (len, head-bytes) so a content change is still detected but
        # huge blobs don't bloat the report.
        inv[ks] = _normalise_value(val)
    # FLAC pictures live outside .tags
    pics = getattr(f, "pictures", None)
    if pics:
        for i, pic in enumerate(pics):
            inv[f"PICTURE[{i}]"] = (
                f"type={getattr(pic, 'type', '?')} "
                f"mime={getattr(pic, 'mime', '?')} "
                f"len={len(getattr(pic, 'data', b''))} "
                f"head={bytes(getattr(pic, 'data', b''))[:8].hex()}")
    return inv


def _normalise_value(val) -> str:
    try:
        from mutagen.id3 import APIC
        if isinstance(val, APIC):
            return (f"APIC type={val.type} mime={val.mime} "
                    f"desc={val.desc!r} len={len(val.data)} "
                    f"head={bytes(val.data)[:8].hex()}")
    except Exception:
        pass
    # MP4 cover list
    try:
        from mutagen.mp4 import MP4Cover
        if isinstance(val, list) and val and isinstance(val[0], MP4Cover):
            return "; ".join(
                f"COVR fmt={c.imageformat} len={len(bytes(c))} "
                f"head={bytes(c)[:8].hex()}" for c in val)
    except Exception:
        pass
    # Generic: bytes-ish payloads → (len, head); everything else repr.
    try:
        b = bytes(val)
        if len(b) > 64:
            return f"<bytes len={len(b)} head={b[:8].hex()}>"
    except Exception:
        pass
    return repr(val)


# =====================================================================
# Serato-tag inventory — the COMPLEMENT of tag_inventory().  Catalogs
# every Serato blob (GEOB Serato*, MP4 com.serato.dj:* atom, Vorbis
# serato_* comment) with a content hash, so the disappearance OR
# mutation of a PRE-EXISTING Serato blob is detectable.  Closes the
# blind spot where a write/strip cycle could nuke unrelated Serato data
# and still be scored "cosmetic" (the user-tag inventory excludes
# Serato tags by design).
# =====================================================================

def _serato_value(val) -> str:
    """Content signature for a Serato tag value: length + sha256 so a
    mutation is caught, not just a disappearance."""
    data = getattr(val, "data", None)        # ID3 GEOB frame
    if data is not None:
        b = bytes(data)
        return f"len={len(b)} sha256={hashlib.sha256(b).hexdigest()[:12]}"
    if isinstance(val, list):                # MP4 freeform / Vorbis list
        parts = []
        for v in val:
            try:
                b = bytes(v)
                parts.append(
                    f"len={len(b)} sha256={hashlib.sha256(b).hexdigest()[:12]}")
            except Exception:
                parts.append(repr(v))
        return "; ".join(parts)
    try:
        b = bytes(val)
        return f"len={len(b)} sha256={hashlib.sha256(b).hexdigest()[:12]}"
    except Exception:
        return repr(val)


def serato_inventory(path: str) -> Dict[str, str]:
    """Return {key: content-signature} for every Serato tag on the file.

    Complement of `tag_inventory` (which excludes Serato tags).  Used to
    verify that pre-existing Serato blobs the edit did NOT target survive
    a write/strip cycle byte-for-byte.
    """
    f = MFile(path)
    if f is None or f.tags is None:
        return {}
    inv: Dict[str, str] = {}
    for key in f.tags.keys():
        ks = str(key)
        if (_is_serato_id3_key(ks) or _is_serato_mp4_key(ks)
                or _is_serato_vorbis_key(ks)):
            inv[ks] = _serato_value(f.tags[key])
    return inv


def _written_inventory_keys(written: List[str], ext: str) -> set:
    """Map the descriptors/atoms `apply_serato_edit` returned to the
    inventory-key form they take in `serato_inventory`, so the edit's
    OWN tags can be excluded from the pre-existing-blob preservation
    check."""
    ext = ext.lower()
    if ext in (".mp3", ".wav", ".aiff", ".aif"):
        return {"GEOB:" + w for w in written}
    if ext in (".mp4", ".m4a", ".m4v"):
        return set(written)
    if ext in (".flac", ".ogg", ".oga", ".opus"):
        try:
            from .tags import _descriptor_to_vorbis_key
            return {_descriptor_to_vorbis_key(w).lower() for w in written}
        except Exception:
            return set()
    return set()


# =====================================================================
# Synthetic rich fixtures — deterministic, no pre-committed binaries.
# =====================================================================

def _silent_mpeg1_frame() -> bytes:
    # Mirror of tests/conftest.py: 417-byte CBR MPEG-1 L3 silent frame.
    return bytes([0xFF, 0xFB, 0x90, 0x00]) + b"\x00" * 413


def build_rich_mp3(path: str) -> str:
    """Write a synthetic MP3 carrying a *rich* ID3v2.4 tag — title,
    artist, album, embedded cover (APIC), unsynchronised lyrics
    (USLT), a comment (COMM) and a private frame (PRIV).  This is the
    high-value "can the Serato writer drop any of these?" target."""
    from mutagen.id3 import (
        ID3, ID3NoHeaderError, TIT2, TPE1, TALB, APIC, USLT, COMM, PRIV)
    with open(path, "wb") as fh:
        fh.write(_silent_mpeg1_frame() * 32)  # ~1 s audio
    try:
        tag = ID3(path)
    except ID3NoHeaderError:
        tag = ID3()
    tag.add(TIT2(encoding=3, text=["Edo G Test Title"]))
    tag.add(TPE1(encoding=3, text=["Edo G & Da Bulldogs"]))
    tag.add(TALB(encoding=3, text=["Roundtrip Safety LP"]))
    # A small but non-trivial fake PNG payload for the cover.
    png = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64 + b"IEND\xaeB`\x82")
    tag.add(APIC(encoding=3, mime="image/png", type=3,
                 desc="cover", data=png))
    tag.add(USLT(encoding=3, lang="eng", desc="",
                 text="line one\nline two\nline three"))
    tag.add(COMM(encoding=3, lang="eng", desc="",
                 text="sidecaramel roundtrip fixture"))
    tag.add(PRIV(owner="sidecaramel.test", data=b"\xde\xad\xbe\xef"))
    tag.save(path)
    return path


def build_rich_mp4(path: str) -> str:
    """Synthesize an M4A via ffmpeg, then attach iTunes-style metadata
    (title/artist/album), a cover atom (covr) and a lyrics atom
    (\\xa9lyr) plus a custom freeform atom — the MP4 analogue of the
    rich MP3 target.  Returns the path, or raises if ffmpeg missing."""
    import subprocess
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not on PATH")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=0.5", "-ac", "1",
         "-ar", "16000", "-c:a", "aac", path],
        check=True)
    from mutagen.mp4 import MP4, MP4Cover, MP4FreeForm
    try:
        from mutagen.mp4 import AtomDataType
        utf8 = AtomDataType.UTF8
    except (ImportError, AttributeError):
        utf8 = 1
    mp4 = MP4(path)
    if mp4.tags is None:
        mp4.add_tags()
    mp4.tags["\xa9nam"] = ["Edo G Test Title"]
    mp4.tags["\xa9ART"] = ["Edo G & Da Bulldogs"]
    mp4.tags["\xa9alb"] = ["Roundtrip Safety LP"]
    mp4.tags["\xa9lyr"] = ["line one\nline two"]
    png = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64 + b"IEND\xaeB`\x82")
    mp4.tags["covr"] = [MP4Cover(png, imageformat=MP4Cover.FORMAT_PNG)]
    # A NON-Serato custom freeform atom — even non-standard atoms
    # must survive the Serato write too.
    mp4.tags["----:com.sidecaramel.test:flag"] = [
        MP4FreeForm(b"keepme", dataformat=utf8)]
    mp4.save(path)
    return path


# =====================================================================
# Edit + strip primitives — go through the real sidecaramel writers.
# =====================================================================

# A representative Serato edit: a couple of cues + one loop, written as
# the full V1+V2 marker pair (what Serato Pro itself writes).  This
# exercises the heaviest writer path.  Container is auto-detected.
# Dict shape matches build_markers2_inner / encode_markers_v1_*_inner:
# cues  -> idx, pos_ms, color, label
# loops -> idx, start_ms, end_ms, color, label, locked
_SAMPLE_CUES = [
    {"idx": 0, "pos_ms": 35, "color": (204, 0, 0), "label": ""},
    {"idx": 1, "pos_ms": 12000, "color": (0, 204, 0), "label": "drop"},
]
_SAMPLE_LOOPS = [
    {"idx": 0, "start_ms": 54, "end_ms": 3680,
     "color": (0, 0, 204), "label": "lp", "locked": False},
]


def apply_serato_edit(path: str) -> List[str]:
    """Apply a representative Serato marker edit to `path` (in place).
    Returns the list of descriptors/atoms that were written, so the
    strip step knows what to remove.  Routes by extension to the
    matching full-marker writer.
    """
    from . import tags as t
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp3", ".wav", ".aiff", ".aif"):
        ok = t.write_serato_markers_full_mp3(
            path, cues=_SAMPLE_CUES, loops=_SAMPLE_LOOPS, confirm=True)
        if not ok:
            raise RuntimeError("write_serato_markers_full_mp3 failed")
        return ["Serato Markers_", "Serato Markers2"]
    if ext in (".mp4", ".m4a", ".m4v"):
        ok = t.write_serato_markers_full_mp4(
            path, cues=_SAMPLE_CUES, loops=_SAMPLE_LOOPS, confirm=True)
        if not ok:
            raise RuntimeError("write_serato_markers_full_mp4 failed")
        return ["----:com.serato.dj:markers",
                "----:com.serato.dj:markersv2"]
    if ext in (".flac", ".ogg", ".oga", ".opus"):
        inner = t.build_markers2_inner(
            cues=_SAMPLE_CUES, loops=_SAMPLE_LOOPS)
        ok = t.write_serato_markers2_any(path, inner, confirm=True)
        if not ok:
            raise RuntimeError("write_serato_markers2_any failed")
        return ["Serato Markers2"]
    raise RuntimeError(f"unsupported container for edit: {ext}")


def strip_serato_edit(path: str,
                       written: Optional[List[str]] = None) -> int:
    """Remove Serato tags from `path` (in place), reversing
    `apply_serato_edit`.  Returns the count of removed keys.

    When `written` (the descriptor/atom list `apply_serato_edit`
    returned) is supplied, ONLY those tags are removed — pre-existing
    Serato blobs the edit never touched (BeatGrid, Autotags, a private
    future entry, …) survive untouched.  This is the correct inverse:
    stripping every Serato tag indiscriminately (the fallback mode,
    used when `written is None`) destroys unrelated Serato data and
    makes the roundtrip mis-score that loss as cosmetic.
    """
    from . import tags as t
    ext = os.path.splitext(path)[1].lower()

    if ext in (".mp3", ".wav", ".aiff", ".aif"):
        if written is None:
            # SPOT: sidecaramel already ships a GEOB stripper (legacy:
            # removes ALL Serato GEOBs).
            return t.wipe_serato_geobs(path, confirm=True)
        from mutagen.id3 import ID3
        try:
            tag = ID3(path)
        except Exception:
            return 0
        targets = {"GEOB:" + w for w in written}
        removed = 0
        for key in [k for k in list(tag.keys()) if str(k) in targets]:
            tag.delall(key)
            removed += 1
        if removed:
            tag.save(path)
        return removed

    if ext in (".mp4", ".m4a", ".m4v"):
        from mutagen.mp4 import MP4
        mp4 = MP4(path)
        if mp4.tags is None:
            return 0
        if written is None:
            keys = [k for k in mp4.tags.keys() if _is_serato_mp4_key(k)]
        else:
            wset = set(written)
            keys = [k for k in mp4.tags.keys() if k in wset]
        removed = 0
        for key in keys:
            del mp4.tags[key]
            removed += 1
        if removed:
            mp4.save()
        return removed

    if ext in (".flac", ".ogg", ".oga", ".opus"):
        from .tags import _open_vorbis_container
        c = _open_vorbis_container(path)
        if written is None:
            keys = [k for k in list(c.keys()) if _is_serato_vorbis_key(k)]
        else:
            wkeys = _written_inventory_keys(written, ext)
            keys = [k for k in list(c.keys()) if k.lower() in wkeys]
        removed = 0
        for key in keys:
            del c[key]
            removed += 1
        if removed:
            c.save()
        return removed

    raise RuntimeError(f"unsupported container for strip: {ext}")


# =====================================================================
# The roundtrip check itself.
# =====================================================================

@dataclass
class RoundtripResult:
    target: str
    container: str
    byte_identical: bool
    written: List[str] = field(default_factory=list)
    stripped_count: int = 0
    verdict: str = ""            # "byte-identical" / "cosmetic" / "DATA LOSS"
    lost_keys: List[str] = field(default_factory=list)
    changed_keys: List[str] = field(default_factory=list)
    extra_keys: List[str] = field(default_factory=list)
    serato_lost_keys: List[str] = field(default_factory=list)
    serato_changed_keys: List[str] = field(default_factory=list)
    target_size: int = 0
    b_size: int = 0
    notes: str = ""

    def ok(self) -> bool:
        """True when no user data was lost.  Cosmetic byte diffs pass;
        only DATA LOSS fails."""
        return self.verdict in ("byte-identical", "cosmetic")


def roundtrip_check(target: str, workdir: Optional[str] = None
                    ) -> RoundtripResult:
    """Run the edit→copy→strip→compare cycle on `target`.

    `target` MUST already be a safe-to-mutate file (a temp copy of a
    real file, or a synthetic file).  This function copies it to A and
    B inside a tempdir and never touches `target` itself after the
    initial read — but as a belt-and-braces guard it operates only on
    the A/B copies, so `target` is the immutable reference.

    Returns a `RoundtripResult`.  `.ok()` is False only on real data
    loss.
    """
    ext = os.path.splitext(target)[1].lower()
    res = RoundtripResult(target=target, container=ext,
                          byte_identical=False)
    res.target_size = os.path.getsize(target)

    own_tmp = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="sidecaramel_rt_")
    try:
        a = os.path.join(workdir, "A" + ext)
        b = os.path.join(workdir, "B" + ext)

        # target -> A
        shutil.copy2(target, a)
        # edit A
        res.written = apply_serato_edit(a)
        # A -> B  (the "copy copy")
        shutil.copy2(a, b)
        # strip B (targeted: remove only what apply_serato_edit wrote,
        # so pre-existing unrelated Serato blobs survive)
        res.stripped_count = strip_serato_edit(b, written=res.written)

        # compare B vs target
        res.b_size = os.path.getsize(b)
        tgt_bytes = open(target, "rb").read()
        b_bytes = open(b, "rb").read()
        res.byte_identical = (tgt_bytes == b_bytes)

        if res.byte_identical:
            res.verdict = "byte-identical"
            return res

        # Not byte-identical → is it data loss or cosmetic?
        inv_t = tag_inventory(target)
        inv_b = tag_inventory(b)

        res.lost_keys = sorted(k for k in inv_t if k not in inv_b)
        res.extra_keys = sorted(k for k in inv_b if k not in inv_t)
        res.changed_keys = sorted(
            k for k in inv_t
            if k in inv_b and inv_t[k] != inv_b[k])

        # Pre-existing Serato blobs the edit did NOT target MUST survive
        # the write/strip cycle unchanged.  The user-tag inventory above
        # excludes Serato tags, so without this an unrelated Serato blob
        # nuked by the strip would be mis-scored "cosmetic".
        own = _written_inventory_keys(res.written, ext)
        ser_t = {k: v for k, v in serato_inventory(target).items()
                 if k not in own and k.lower() not in own}
        ser_b = serato_inventory(b)
        res.serato_lost_keys = sorted(k for k in ser_t if k not in ser_b)
        res.serato_changed_keys = sorted(
            k for k in ser_t
            if k in ser_b and ser_t[k] != ser_b[k])

        if (res.lost_keys or res.changed_keys
                or res.serato_lost_keys or res.serato_changed_keys):
            res.verdict = "DATA LOSS"
            res.notes = (
                "data lost/changed after strip: "
                f"user_lost={res.lost_keys} "
                f"user_changed={res.changed_keys} "
                f"serato_lost={res.serato_lost_keys} "
                f"serato_changed={res.serato_changed_keys}")
        elif res.extra_keys:
            # Strip left a residual non-Serato key behind — also a
            # correctness problem, flag it loudly.
            res.verdict = "DATA LOSS"
            res.notes = (f"strip introduced/left extra keys: "
                         f"{res.extra_keys}")
        else:
            res.verdict = "cosmetic"
            res.notes = (
                "all user + pre-existing Serato tags intact; byte diff "
                "is padding / frame-order / mutagen re-serialisation "
                f"(target={res.target_size}B, B={res.b_size}B)")
        return res
    finally:
        if own_tmp:
            shutil.rmtree(workdir, ignore_errors=True)


# =====================================================================
# CLI
# =====================================================================

def _run_synthetic() -> List[RoundtripResult]:
    results: List[RoundtripResult] = []
    tmp = tempfile.mkdtemp(prefix="sidecaramel_rt_synth_")
    try:
        mp3 = build_rich_mp3(os.path.join(tmp, "target.mp3"))
        results.append(roundtrip_check(mp3))
        try:
            mp4 = build_rich_mp4(os.path.join(tmp, "target.m4a"))
            results.append(roundtrip_check(mp4))
        except RuntimeError as e:
            print(f"  (skipping MP4: {e})", file=sys.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return results


def _run_from_copy(src: str) -> RoundtripResult:
    """Copy a REAL file into a tempdir and roundtrip the copy.  The
    original `src` is only ever read (shutil.copy2), never written."""
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    tmp = tempfile.mkdtemp(prefix="sidecaramel_rt_copy_")
    ext = os.path.splitext(src)[1].lower()
    target = os.path.join(tmp, "target" + ext)
    shutil.copy2(src, target)            # original opened read-only
    try:
        return roundtrip_check(target)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _print(res: RoundtripResult) -> None:
    mark = "OK " if res.ok() else "FAIL"
    print(f"[{mark}] {res.container}  "
          f"byte-identical={res.byte_identical}  "
          f"verdict={res.verdict}")
    print(f"       wrote={res.written} stripped={res.stripped_count} "
          f"sizes target={res.target_size} B={res.b_size}")
    if res.notes:
        print(f"       {res.notes}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Serato-writer ID3/atom data-loss safety net "
                    "(edit -> copy -> strip -> byte-compare).")
    ap.add_argument(
        "--from-copy", metavar="AUDIO",
        help="Copy this real file into a tempdir and roundtrip the "
             "copy (original is read-only). Default: synthetic mp3+mp4.")
    ap.add_argument(
        "--serato-known-closed", action="store_true",
        help="Skip the Serato-running probe. Means \"I have confirmed "
             "Serato is closed\" — required on hosts where the probe "
             "is unavailable.")
    args = ap.parse_args(argv)

    # Serato must not be running for any write op. Fail CLOSED: an
    # unavailable probe counts as "running" (see check.py), and the
    # only way past either verdict is the explicit flag. A gate that
    # swallows its own failure is worse than no gate — it reports
    # protection it is not providing.
    if not args.serato_known_closed:
        from sidecaramel.check import (is_running,
                                       SeratoCheckUnavailableError)
        try:
            running, processes = is_running()
        except SeratoCheckUnavailableError as e:
            print(f"ABORT: cannot determine whether Serato is running "
                  f"({e}). Close Serato and pass "
                  f"--serato-known-closed to proceed.", file=sys.stderr)
            return 3
        if running:
            print(f"ABORT: Serato is running ({', '.join(processes)}). "
                  f"No writes.", file=sys.stderr)
            return 3

    print("sidecaramel roundtrip safety net")
    print("=" * 60)
    if args.from_copy:
        results = [_run_from_copy(args.from_copy)]
    else:
        results = _run_synthetic()

    for r in results:
        print()
        _print(r)

    print()
    print("=" * 60)
    losses = [r for r in results if not r.ok()]
    if losses:
        print(f"RESULT: {len(losses)} DATA-LOSS finding(s)")
        return 1
    print("RESULT: PASS — no Serato writer lost any user tag")
    return 0


if __name__ == "__main__":
    sys.exit(main())
