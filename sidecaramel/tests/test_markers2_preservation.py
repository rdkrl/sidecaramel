"""Regression tests for the preservation-oriented cue/loop write.

The external review's remaining valid finding: the green-field cue
writers (write_serato_markers_full_mp3/mp4, build_markers2_inner)
rebuilt the whole Markers2 stream, so a cue rewrite dropped AUTOGAIN,
dropped unknown/future entries, and flipped a locked BPMLOCK to
unlocked by default.

This is handled: write_serato_markers_full_mp3/mp4 default preserve=True
and merge against any existing Markers2 via flip_writer.merge_markers2_inner,
keeping AUTOGAIN / unknown / FLIP / BPMLOCK verbatim unless explicitly
overridden.  bpm_lock now defaults to None ("don't touch") instead of
False ("write unlocked").
"""
import pytest

from sidecaramel.flip_writer import (
    M2Entry, parse_inner, serialize_inner, decode_outer,
    merge_markers2_inner,
)
from sidecaramel.tags import (
    build_markers2_inner, write_serato_markers2_any,
    write_serato_markers_full_mp3, harvest_serato_blobs,
    parse_serato_markers2_full,
)


def _seed_existing_markers2(path):
    """Seed `path` with a Markers2 blob containing a locked BPMLOCK,
    one cue, an AUTOGAIN entry, and an unknown FUTUREX entry."""
    base = build_markers2_inner(
        cues=[{"idx": 0, "pos_ms": 1000, "color": (255, 0, 0),
               "label": "old"}],
        bpm_lock=True)
    entries = parse_inner(base)
    entries.append(M2Entry(type="AUTOGAIN", body=b"-3.50"))
    entries.append(M2Entry(type="FUTUREX", body=b"\xde\xad\xbe\xef"))
    inner = serialize_inner(entries)
    assert write_serato_markers2_any(path, inner, confirm=True)


def _read_inner(path):
    outer = None
    for desc, payload, _src in harvest_serato_blobs(path):
        if desc == "Serato Markers2":
            outer = payload
            break
    assert outer is not None
    inner, _ = decode_outer(outer)
    return outer, inner


# ---- the merge primitive (unit) -------------------------------------

def test_merge_primitive_preserves_and_replaces():
    base = build_markers2_inner(
        cues=[{"idx": 0, "pos_ms": 1000, "color": (255, 0, 0),
               "label": "old"}],
        bpm_lock=True)
    entries = parse_inner(base)
    entries.append(M2Entry("AUTOGAIN", b"-3.50"))
    entries.append(M2Entry("FUTUREX", b"\xde\xad\xbe\xef"))
    existing = serialize_inner(entries)

    merged = merge_markers2_inner(
        existing,
        cues=[{"idx": 1, "pos_ms": 2000, "color": (0, 255, 0),
               "label": "new"}])
    by_type = {}
    for e in parse_inner(merged):
        by_type.setdefault(e.type, []).append(e)

    # AUTOGAIN + unknown FUTUREX survive byte-for-byte.
    assert by_type["AUTOGAIN"][0].body == b"-3.50"
    assert by_type["FUTUREX"][0].body == b"\xde\xad\xbe\xef"
    # BPMLOCK left untouched (bpm_lock not passed) → still locked.
    assert by_type["BPMLOCK"][0].body == b"\x01"
    # The cue was replaced, not appended.
    assert len(by_type["CUE"]) == 1


def test_merge_can_clear_with_empty_list():
    base = build_markers2_inner(
        cues=[{"idx": 0, "pos_ms": 1000, "color": (1, 2, 3)}])
    merged = merge_markers2_inner(base, cues=[])
    assert not [e for e in parse_inner(merged) if e.type == "CUE"]


# ---- through the documented MP3 cue writer --------------------------

def test_cue_write_preserves_autogain_unknown_and_lock(
        minimal_mp3_with_id3):
    p = str(minimal_mp3_with_id3)
    _seed_existing_markers2(p)

    # Write NEW cues; default preserve=True, no bpm_lock (None=keep).
    assert write_serato_markers_full_mp3(
        p,
        cues=[{"idx": 0, "pos_ms": 2000, "color": (0, 255, 0),
               "label": "new"}],
        confirm=True)

    outer, inner = _read_inner(p)
    by_type = {}
    for e in parse_inner(inner):
        by_type.setdefault(e.type, []).append(e)

    assert by_type["AUTOGAIN"][0].body == b"-3.50"
    assert by_type["FUTUREX"][0].body == b"\xde\xad\xbe\xef"
    assert by_type["BPMLOCK"][0].body == b"\x01"      # still locked

    full = parse_serato_markers2_full(outer)
    assert any(c["pos_ms"] == 2000 for c in full["cues"])
    assert not any(c["pos_ms"] == 1000 for c in full["cues"])


def test_preserve_false_rebuilds_from_scratch(minimal_mp3_with_id3):
    p = str(minimal_mp3_with_id3)
    _seed_existing_markers2(p)

    assert write_serato_markers_full_mp3(
        p,
        cues=[{"idx": 0, "pos_ms": 2000, "color": (0, 255, 0)}],
        preserve=False, confirm=True)

    _outer, inner = _read_inner(p)
    types = [e.type for e in parse_inner(inner)]
    # preserve=False is the explicit clobber escape hatch.
    assert "AUTOGAIN" not in types
    assert "FUTUREX" not in types
    # bpm_lock default None → no BPMLOCK fabricated on a green-field write.
    assert "BPMLOCK" not in types


def test_full_mp3_writes_both_v1_and_v2(minimal_mp3_with_id3):
    p = str(minimal_mp3_with_id3)
    assert write_serato_markers_full_mp3(
        p,
        cues=[{"idx": 0, "pos_ms": 500, "color": (1, 2, 3), "label": "x"}],
        confirm=True)
    descs = [d for d, _p, _s in harvest_serato_blobs(p)]
    assert "Serato Markers_" in descs       # V1 present
    assert "Serato Markers2" in descs       # V2 present (single save)


# ── The V1 + V2 pair, per container ──────────────────────────────────
# Serato honours cues and loops only when BOTH the V1 `Serato Markers_`
# and the V2 `Serato Markers2` blob exist. A writer that emits one of
# them alone leaves the file inconsistent, so pin it for every
# container the full writers claim to cover.

import pytest


def _both_blobs(path: str) -> tuple:
    from sidecaramel.tags import harvest_serato_blobs
    descs = [d for d, _p, _s in harvest_serato_blobs(path)]
    return ("Serato Markers_" in descs, "Serato Markers2" in descs)


CUES = [{"idx": 0, "pos_ms": 500, "color": (1, 2, 3), "label": "x"}]
LOOPS = [{"idx": 0, "start_ms": 1000, "end_ms": 2000,
          "color": (4, 5, 6), "label": "y", "locked": True}]


@pytest.mark.parametrize("fixture_name", ["ffmpeg_wav", "ffmpeg_aiff"])
def test_full_id3_writer_pairs_v1_and_v2(request, fixture_name):
    from sidecaramel.tags import write_serato_markers_full_mp3
    path = str(request.getfixturevalue(fixture_name))
    assert write_serato_markers_full_mp3(
        path, cues=CUES, loops=LOOPS, confirm=True)
    v1, v2 = _both_blobs(path)
    assert v1, f"{fixture_name}: V1 Markers_ missing — cues/loops unpaired"
    assert v2, f"{fixture_name}: V2 Markers2 missing"


def test_full_mp4_writer_pairs_v1_and_v2(ffmpeg_m4a):
    from sidecaramel.tags import write_serato_markers_full_mp4
    path = str(ffmpeg_m4a)
    assert write_serato_markers_full_mp4(
        path, cues=CUES, loops=LOOPS, confirm=True)
    v1, v2 = _both_blobs(path)
    assert v1, "MP4: V1 Markers_ missing — cues/loops unpaired"
    assert v2, "MP4: V2 Markers2 missing"


# ── Foreign Serato blobs survive a cue/loop write ────────────────────
# A cue write touches Markers_ and Markers2 and nothing else. Every
# other Serato blob — Autotags (autogain), BeatGrid, Overview,
# Playcount, RelVolAd, VideoAssoc, Offsets_, StemSet, Analysis — must
# come back byte-for-byte, or the write silently destroyed analysis
# data the user cannot regenerate without re-analysing the track.

FOREIGN_BLOBS = {
    "Serato Analysis":   b"\x02\x01",
    "Serato BeatGrid":   b"\x01\x00" + b"\xde\xad\xbe\xef" * 4,
    "Serato Overview":   bytes(range(256)) * 3,
    "Serato Playcount":  b"\x01\x00\x00\x00\x07",
    "Serato RelVolAd":   b"\x01\x01" + b"\x00" * 6,
    "Serato VideoAssoc": b"\x01\x00vid",
    "Serato Offsets_":   b"\x01\x00off",
    "Serato StemSet":    b"\x01\x00stem",
}


def test_cue_write_preserves_every_foreign_blob(minimal_mp3_with_id3):
    from sidecaramel.tags import (write_serato_geob, build_autotags,
                                  write_serato_markers_full_mp3,
                                  harvest_serato_blobs)
    path = str(minimal_mp3_with_id3)
    blobs = dict(FOREIGN_BLOBS)
    # Autogain through the package's own builder, not a hand-made blob.
    blobs["Serato Autotags"] = build_autotags(
        bpm=128.0, auto_gain=-3.5, gain_db=0.0)
    for name, payload in blobs.items():
        assert write_serato_geob(path, name, payload, confirm=True), name

    before = {d: raw for d, raw, _s in harvest_serato_blobs(path)}
    assert write_serato_markers_full_mp3(
        path, cues=CUES, loops=LOOPS, confirm=True)
    after = {d: raw for d, raw, _s in harvest_serato_blobs(path)}

    for name in blobs:
        assert name in after, f"{name} was DROPPED by the cue write"
        assert after[name] == before[name], f"{name} was MODIFIED"
    # …and the pair itself was in fact rewritten.
    assert after["Serato Markers_"] != before.get("Serato Markers_")
    assert after["Serato Markers2"] != before.get("Serato Markers2")
