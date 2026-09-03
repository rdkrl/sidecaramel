"""Golden tests against REAL Serato DJ Pro output.

Unlike the rest of the suite (which round-trips sidecaramel's own
writers through its own parsers on synthetic fixtures), these assert
sidecaramel decodes byte-for-byte blobs that Serato DJ Pro itself wrote
— extracted from a track cued/looped/grid/flipped in Serato across three
container envelopes (MP3 + WAV = ID3 GEOB, MP4 = freeform atoms).

The blobs live in `fixtures/serato_golden/` (no audio — just the tag
payloads, a few KB each). See `dev/serato_fixtures/` for how they were
extracted, and `manifest.json` for the decoded ground-truth values.

The load-bearing assertion is V1<->V2 consistency: the `Serato Markers_`
(V1) and `Serato Markers2` (V2) blobs are two independent encodings of
the same cues/loops, so they must agree. That property caught the real
bug this file guards: the MP3/WAV V1 blob is 7-bit-safe masked, and the
V1 parser used to skip the masking and return 0 cues for MP3.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from sidecaramel.tags import (parse_serato_markers2_full,
                              parse_serato_markers_v1_full,
                              build_beatgrid, encode_flip_entry,
                              encode_markers_v1_mp3_inner,
                              encode_markers_v1_inner,
                              build_markers2_inner, pack_serato_markers2,
                              parse_serato_analysis, build_analysis_blob)
from sidecaramel.blobs import decode_beatgrid, _decode_flip
from sidecaramel.flip_writer import decode_outer

FIX = Path(__file__).parent / "fixtures" / "serato_golden"
CONTAINERS = ["mp3", "wav", "mp4"]

pytestmark = pytest.mark.skipif(
    not (FIX.exists() and any(FIX.glob("*.blob"))),
    reason="real-Serato golden blobs not present")


def _blob(container: str, name: str) -> bytes:
    p = FIX / f"Round_Tip_Parity_{container}__Serato_{name}.blob"
    return p.read_bytes()


@pytest.mark.parametrize("container", CONTAINERS)
def test_markers2_shape(container):
    m = parse_serato_markers2_full(_blob(container, "Markers2"))
    assert len(m["cues"]) == 8, "Serato wrote 8 hot cues"
    assert len(m["loops"]) == 8, "Serato wrote 8 saved loops"
    assert len(m["flips"]) == 6, "Serato wrote 6 flips"
    # Serato's default slot colours + default loop colour (0x27aae1).
    assert m["cues"][0]["color"] == (204, 0, 0)
    assert all(l["color"] == (39, 170, 225) for l in m["loops"])


@pytest.mark.parametrize("container", CONTAINERS)
def test_v1_masked_decode_and_v1_v2_consistency(container):
    """The regression guard: V1 must decode (5 cues, not 0) AND agree
    with V2 on the cues/loops they share."""
    v2 = parse_serato_markers2_full(_blob(container, "Markers2"))
    v1 = parse_serato_markers_v1_full(_blob(container, "Markers"))

    # V1 (legacy Markers_) caps at 5 cue slots; all 5 must be read
    # (pre-fix this was 0 for MP3, 2 for WAV).
    assert len(v1["cues"]) == 5, f"{container}: V1 cues lost (masking bug)"
    assert len(v1["loops"]) == 8, f"{container}: V1 loops lost"

    v2cue = {c["idx"]: c for c in v2["cues"]}
    for c in v1["cues"]:
        exp = v2cue[c["idx"]]
        assert c["pos_ms"] == exp["pos_ms"], f"{container} cue {c['idx']} pos"
        assert c["color"] == exp["color"], f"{container} cue {c['idx']} colour"

    v2loop = {l["idx"]: l for l in v2["loops"]}
    for l in v1["loops"]:
        exp = v2loop.get(l["idx"])
        if exp is None:
            continue
        assert l["start_ms"] == exp["start_ms"], f"{container} loop start"
        assert l["end_ms"] == exp["end_ms"], f"{container} loop end"


@pytest.mark.parametrize("container", CONTAINERS)
def test_beatgrid_bpm(container):
    bg = decode_beatgrid(_blob(container, "BeatGrid"))
    bpm = bg["terminal_marker"]["bpm"]
    assert 127.0 <= bpm <= 129.0, f"{container}: unexpected BPM {bpm}"


@pytest.mark.parametrize("container", CONTAINERS)
def test_all_flips_decode(container):
    m = parse_serato_markers2_full(_blob(container, "Markers2"))
    assert len(m["raw"].get("FLIP", [])) == 6
    for i, raw in enumerate(m["raw"]["FLIP"]):
        dec = _decode_flip(raw)          # must not raise
        acts = dec.get("actions") or []
        assert acts, f"{container}: FLIP[{i}] decoded to zero actions"
        for a in acts:
            assert a.get("type") in ("JUMP", "CENSOR")


@pytest.mark.parametrize("container", CONTAINERS)
def test_overview_present(container):
    ov = _blob(container, "Overview")
    assert len(ov) >= 3842   # Serato Overview is a fixed-size waveform blob


def test_mp3_exact_anchor():
    """One concrete, human-checkable anchor so a silent fixture swap
    can't pass unnoticed."""
    v1 = parse_serato_markers_v1_full(_blob("mp3", "Markers"))
    assert v1["cues"][0]["pos_ms"] == 1454
    assert v1["cues"][0]["color"] == (204, 0, 0)      # Serato cue-1 red


def test_named_cues_decode():
    """Real Serato cue labels decode (UTF-8). The fixture track carries
    named cues, so the label field is exercised against real bytes."""
    names = {c["idx"]: c["label"]
             for c in parse_serato_markers2_full(_blob("mp3", "Markers2"))["cues"]
             if c["label"]}
    # Concrete anchor from the mp3 fixture (cue names set in Serato).
    assert names.get(0) == "Un"
    assert names.get(1) == "Dos"
    assert names.get(6) == "Sette"
    # Every container has at least one named cue that decodes to a str.
    for container in CONTAINERS:
        labels = [c["label"] for c
                  in parse_serato_markers2_full(_blob(container, "Markers2"))["cues"]
                  if c["label"]]
        assert labels and all(isinstance(x, str) for x in labels), container


# ---------------------------------------------------------------------------
# Write round-trip: decode Serato's real bytes, re-encode with sidecaramel's
# own builders, and assert the result is byte-for-byte identical to what
# Serato wrote. This is the strongest fidelity check — it proves the writers
# don't just produce *a* valid blob, they reproduce Serato's exact bytes.
#
# Scope note: BeatGrid, the V1 Markers_ blob (masked MP3/WAV *and* raw MP4),
# every FLIP entry, and the whole Markers2 stream round-trip byte-exact here
# (see the cross-container shuffle below). The Markers2 OUTER (base64 wrapped
# at 72 + NUL-padded) also reproduces byte-for-byte when the original blob
# size is preserved — see test_serato_audio_roundtrip.py, which asserts the
# full GEOB/atom payload off real files. What is NOT byte-exact is the
# Overview blob (recomputed from audio, not re-encoded), the Analysis version
# marker, and the MP4-only extra atoms (RelVolAd, VidAssoc, playcount), which
# have no builder — those gaps are deliberately left out.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("container", CONTAINERS)
def test_beatgrid_write_roundtrip(container):
    """decode_beatgrid -> build_beatgrid reproduces Serato's bytes.
    Guards the un-rounding fix: rounding the f32 position/BPM used to
    make the re-encode diverge from Serato's blob."""
    orig = _blob(container, "BeatGrid")
    d = decode_beatgrid(orig)
    major, minor = (int(x) for x in d["version"].split("."))
    re = build_beatgrid(
        markers=[{"position_s": m["position_s"],
                  "beats_till_next": m["beats_till_next"]}
                 for m in d["non_terminal_markers"]],
        final_bpm=d["terminal_marker"]["bpm"],
        terminal_position_s=d["terminal_marker"]["position_s"],
        footer=int(d["footer_byte"], 16),
        version=(major, minor))
    assert re == orig, f"{container}: BeatGrid re-encode diverged"


@pytest.mark.parametrize("container", ["mp3", "wav"])
def test_v1_markers_write_roundtrip(container):
    """parse_serato_markers_v1_full -> encode_markers_v1_mp3_inner
    reproduces the masked (MP3/WAV) V1 blob byte-exact. Exercises the
    per-loop `locked` flag round-trip (byte [21]) and the 7-bit-safe
    position/colour encoders. (MP4's 19-byte raw V1 is a separate path,
    not asserted here.)"""
    orig = _blob(container, "Markers")
    p = parse_serato_markers_v1_full(orig)
    re = encode_markers_v1_mp3_inner(
        cues=p["cues"], loops=p["loops"],
        track_color_rgb=p["track_color"])
    assert re == orig, f"{container}: V1 Markers_ re-encode diverged"


@pytest.mark.parametrize("container", CONTAINERS)
def test_flip_write_roundtrip(container):
    """_decode_flip -> encode_flip_entry reproduces every FLIP entry
    byte-exact. Guards the schema reconcile: the decoder now emits the
    same action vocabulary (trigger_s/reverse_target_s, unrounded f64s)
    the encoder consumes, so a decoded FLIP re-encodes losslessly."""
    m = parse_serato_markers2_full(_blob(container, "Markers2"))
    raws = m["raw"].get("FLIP", [])
    assert len(raws) == 6, f"{container}: expected 6 FLIP entries"
    for i, raw in enumerate(raws):
        dec = _decode_flip(raw)
        orig_entry = b"FLIP\x00" + struct.pack(">I", len(raw)) + raw
        re_entry = encode_flip_entry(
            slot_index=dec.get("slot_index", 0),
            name=dec.get("name", ""),
            loop=bool(dec.get("loop", False)),
            actions=[a for a in dec.get("actions", [])
                     if a.get("type") in ("JUMP", "CENSOR")],
            flag=dec.get("flag_byte", 1))
        assert re_entry == orig_entry, \
            f"{container}: FLIP[{i}] re-encode diverged"


@pytest.mark.parametrize("container", CONTAINERS)
def test_analysis_write_roundtrip(container):
    """parse_serato_analysis -> build_analysis_blob reproduces the version
    marker byte-exact. Guards the 2-byte fix: MP3/WAV write `02 01` (2 bytes,
    patch is None) and MP4 writes `00 01 00` (3 bytes) — the parser used to
    return None for the 2-byte form, so it read as 'no Analysis data'."""
    orig = _blob(container, "Analysis")
    a = parse_serato_analysis(orig)
    assert a is not None, f"{container}: Analysis parsed to None"
    ver = ((a["major"], a["minor"]) if a["patch"] is None
           else (a["major"], a["minor"], a["patch"]))
    assert build_analysis_blob(ver) == orig, \
        f"{container}: Analysis re-encode diverged ({orig.hex(' ')})"


# ---------------------------------------------------------------------------
# Cross-container "shuffle" round-trip.
#
# The strongest round-trip there is: carry one track's Serato metadata
# through EVERY container envelope and back, parsing and RE-SYNTHESISING at
# each hop (never byte-copying), and prove the bytes you started with are the
# bytes you end with. This exercises the real container difference — the V1
# `Serato Markers_` blob is 7-bit-safe *masked* on MP3/WAV (22-byte rows) but
# *raw* on MP4 (19-byte rows) — so a lossless mp3->mp4->wav->mp3 cycle proves
# the masked<->raw transcode (incl. the per-loop `locked` flag, which lives at
# byte [21] masked / [18] raw) is exact in both directions.
#
# The container-neutral waypoint ("plain text") is the decoded dict; Markers2
# (V2) is its authoritative source for cues/loops/flips/colour/bpm-lock (it
# carries all 8 slots and the labels), V1 supplies its own tile-colour trailer,
# and BeatGrid is container-independent. Markers2 is compared at the INNER
# stream (the actual data); the outer base64 blob's trailing NUL padding is a
# per-container size artifact, not metadata.
# ---------------------------------------------------------------------------

_MASKED = {"mp3", "wav"}          # ID3 GEOB, 7-bit-safe V1
                                   # mp4 -> raw (unmasked) V1


def _shuffle_parse(blobs: dict) -> dict:
    """{markers2_inner, v1, beatgrid} bytes -> container-neutral dict."""
    m = parse_serato_markers2_full(
        pack_serato_markers2(blobs["markers2_inner"], target_size=0))
    v1 = parse_serato_markers_v1_full(blobs["v1"])
    bg = decode_beatgrid(blobs["beatgrid"])
    major, minor = (int(x) for x in bg["version"].split("."))
    return {
        "cues": m["cues"], "loops": m["loops"], "flips": m["flips"],
        "color_rgb": m.get("color"), "bpm_lock": m.get("bpm_lock"),
        "v1_track_color": v1["track_color"],
        "beatgrid": {
            "markers": [{"position_s": x["position_s"],
                         "beats_till_next": x["beats_till_next"]}
                        for x in bg["non_terminal_markers"]],
            "terminal_pos": bg["terminal_marker"]["position_s"],
            "bpm": bg["terminal_marker"]["bpm"],
            "footer": int(bg["footer_byte"], 16),
            "version": (major, minor)},
    }


def _shuffle_synth(neutral: dict, container: str) -> dict:
    """container-neutral dict -> {markers2_inner, v1, beatgrid} for `container`."""
    m2_inner = build_markers2_inner(
        cues=neutral["cues"], loops=neutral["loops"],
        color_rgb=neutral["color_rgb"], bpm_lock=neutral["bpm_lock"],
        flips=neutral["flips"])
    v1_encoder = (encode_markers_v1_mp3_inner if container in _MASKED
                  else encode_markers_v1_inner)
    v1 = v1_encoder(cues=neutral["cues"], loops=neutral["loops"],
                    track_color_rgb=neutral["v1_track_color"])
    bg = neutral["beatgrid"]
    beatgrid = build_beatgrid(
        markers=bg["markers"], final_bpm=bg["bpm"],
        terminal_position_s=bg["terminal_pos"], footer=bg["footer"],
        version=bg["version"])
    return {"markers2_inner": m2_inner, "v1": v1, "beatgrid": beatgrid}


# Each cycle starts and ends in `start`, visiting all three envelopes.
_SHUFFLE_CYCLES = [
    ("mp3", ["mp4", "wav", "mp3"]),
    ("wav", ["mp3", "mp4", "wav"]),
    ("mp4", ["wav", "mp3", "mp4"]),
]


@pytest.mark.parametrize("start,hops", _SHUFFLE_CYCLES,
                          ids=[c[0] for c in _SHUFFLE_CYCLES])
def test_cross_container_shuffle_roundtrip(start, hops):
    """Parse the real Serato `start` blobs, then re-synthesise through every
    container envelope and back — anfang == ende, byte-for-byte, and equal to
    what Serato itself wrote for `start`."""
    real = {
        "markers2_inner": decode_outer(_blob(start, "Markers2"))[0],
        "v1": _blob(start, "Markers"),
        "beatgrid": _blob(start, "BeatGrid"),
    }
    neutral = _shuffle_parse(real)

    # "anfang": synth in the start container reproduces Serato's own bytes.
    anfang = _shuffle_synth(neutral, start)
    for k in ("markers2_inner", "v1", "beatgrid"):
        assert anfang[k] == real[k], f"{start}: synth != Serato [{k}]"

    # Shuffle through every container, parsing + re-synthesising each hop.
    for container in hops:
        neutral = _shuffle_parse(_shuffle_synth(neutral, container))

    # "ende": back in the start container, byte-identical to anfang == Serato.
    ende = _shuffle_synth(neutral, start)
    for k in ("markers2_inner", "v1", "beatgrid"):
        assert ende[k] == anfang[k] == real[k], \
            f"{start}: shuffle diverged [{k}] (anfang != ende)"
