"""Real-audio round-trip against Serato DJ Pro-tagged FILES.

Where `test_serato_golden.py` works on pre-extracted `.blob` payloads, this
module drives the whole path through actual audio containers: it opens the
real Serato-tagged `Round-Tip Parity.{mp3,wav,mp4}`, reads their tags with
the real reader, RE-SYNTHESISES each blob from the decoded data, writes it
into a copy of the *blank baseline* file through the real (mutagen) writers,
reads the written file back, and asserts the tag bytes match what Serato
itself wrote — end to end, through real ID3 / MP4-atom I/O.

This is the "copy the blank files, write the tagged data into them, flag
what you don't know" round-trip: the baseline files carry NO Serato tags,
so every asserted byte comes from sidecaramel's own writers landing on disk.

The audio (tagged + blank baseline, a few tens of MB) lives ONLY on the
`gold-dropbox` branch, checked out to `sidecaramel/roundtrip/`. When it is
absent — main, CI, a normal clone — every test here skips.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("mutagen")

from sidecaramel.tags import (                                  # noqa: E402
    harvest_serato_blobs,
    parse_serato_markers2_full, build_markers2_inner,
    parse_serato_markers_v1_full,
    encode_markers_v1_mp3_inner, encode_markers_v1_inner,
    parse_serato_autotags, build_autotags, build_beatgrid,
    parse_serato_analysis, build_analysis_blob,
    write_serato_markers2_any,
    write_serato_markers_v1_mp3, write_serato_markers_v1_mp4,
    write_serato_beatgrid_any, write_serato_autotags_any,
    write_serato_analysis_any)
from sidecaramel.blobs import decode_beatgrid                   # noqa: E402
from sidecaramel.flip_writer import decode_outer                # noqa: E402

RT = Path(__file__).parent.parent / "roundtrip"
TAGGED = RT / "tagged"
BASELINE = RT / "baseline"
NAME = "Round-Tip Parity"

# (extension, is the V1 blob 7-bit-safe *masked*?)  MP3/WAV = masked ID3
# GEOB; MP4 = raw freeform atom.
_CONTAINERS = [("mp3", True), ("wav", True), ("mp4", False)]

# Blobs we re-synthesise from decoded data and expect byte-identical.
_REBUILT = {"Serato Markers_", "Serato Markers2",
            "Serato BeatGrid", "Serato Autotags", "Serato Analysis"}
# Blobs we knowingly do NOT reproduce byte-exact — flagged, not asserted:
#   Overview   -> recomputed from the audio, not re-encoded from the blob
#   MP4 extras -> RelVolAd / VidAssoc / playcount have no builder yet
_FLAGGED = {"Serato Overview",
            "Serato RelVolAd", "Serato VidAssoc", "Serato playcount"}

pytestmark = pytest.mark.skipif(
    not (TAGGED.exists() and BASELINE.exists()
         and (TAGGED / f"{NAME}.mp3").exists()),
    reason="real Serato-tagged audio (gold-dropbox roundtrip/) not present")


def _blobs(path) -> dict:
    return {d: p for d, p, _ in harvest_serato_blobs(str(path))}


def _synth_and_write(src: dict, work: str, masked: bool) -> None:
    """Parse each source blob, re-synthesise it, and write it into `work`
    through the real container writers (each a separate mutagen save that
    preserves the other tags)."""
    m = parse_serato_markers2_full(src["Serato Markers2"])
    inner = build_markers2_inner(
        cues=m["cues"], loops=m["loops"], color_rgb=m.get("color"),
        bpm_lock=m.get("bpm_lock"), flips=m["flips"])
    # Preserve the original GEOB size so the outer blob (base64 wrapped at
    # 72 + NUL padding to that size — Serato's convention) reproduces
    # byte-for-byte, not just the inner stream.
    write_serato_markers2_any(work, inner, target_size=len(src["Serato Markers2"]),
                              line_width=72, confirm=True)

    v1p = parse_serato_markers_v1_full(src["Serato Markers_"])
    if masked:
        v1 = encode_markers_v1_mp3_inner(
            cues=v1p["cues"], loops=v1p["loops"],
            track_color_rgb=v1p["track_color"])
        write_serato_markers_v1_mp3(work, v1, confirm=True)
    else:
        v1 = encode_markers_v1_inner(
            cues=v1p["cues"], loops=v1p["loops"],
            track_color_rgb=v1p["track_color"])
        write_serato_markers_v1_mp4(work, v1, confirm=True)

    bgp = decode_beatgrid(src["Serato BeatGrid"])
    major, minor = (int(x) for x in bgp["version"].split("."))
    bg = build_beatgrid(
        markers=[{"position_s": x["position_s"],
                  "beats_till_next": x["beats_till_next"]}
                 for x in bgp["non_terminal_markers"]],
        final_bpm=bgp["terminal_marker"]["bpm"],
        terminal_position_s=bgp["terminal_marker"]["position_s"],
        footer=int(bgp["footer_byte"], 16), version=(major, minor))
    write_serato_beatgrid_any(work, bg, confirm=True)

    atp = parse_serato_autotags(src["Serato Autotags"])
    write_serato_autotags_any(
        work, build_autotags(atp["bpm"], atp["gain"], atp["gain_db"]),
        confirm=True)

    anp = parse_serato_analysis(src["Serato Analysis"])
    # 2-byte form (MP3/WAV, patch is None) vs 3-byte (MP4, patch set).
    ver = ((anp["major"], anp["minor"]) if anp["patch"] is None
           else (anp["major"], anp["minor"], anp["patch"]))
    write_serato_analysis_any(work, build_analysis_blob(ver), confirm=True)


@pytest.mark.parametrize("ext,masked", _CONTAINERS,
                          ids=[c[0] for c in _CONTAINERS])
def test_real_audio_write_roundtrip(ext, masked, tmp_path):
    """Read Serato's tags from the real tagged file, re-synthesise them into
    a copy of the blank baseline through real mutagen writes, read that file
    back, and assert every rebuilt blob is byte-identical to Serato's."""
    tagged = TAGGED / f"{NAME}.{ext}"
    base = BASELINE / f"{NAME}.{ext}"
    if not (tagged.exists() and base.exists()):
        pytest.skip(f"{ext} audio not present")

    src = _blobs(tagged)
    work = tmp_path / f"work.{ext}"
    shutil.copy2(base, work)
    assert not _blobs(work), \
        f"{ext}: baseline is supposed to be blank but carries Serato tags"

    _synth_and_write(src, str(work), masked)
    got = _blobs(work)

    # Full byte-identity — the whole GEOB/atom payload, not just the data:
    # preserving the original blob size makes the base64-wrapped, NUL-padded
    # Markers2 OUTER reproduce exactly too.
    for name in ("Serato Markers2", "Serato Markers_",
                 "Serato BeatGrid", "Serato Autotags", "Serato Analysis"):
        assert got[name] == src[name], \
            f"{ext}: {name} diverged after real-file write"
    # (Belt and braces: the decoded inner must of course match too.)
    assert decode_outer(got["Serato Markers2"])[0] \
        == decode_outer(src["Serato Markers2"])[0], \
        f"{ext}: Markers2 inner diverged"


def test_stems_sidecar_container_roundtrip():
    """The `.serato-stems` sidecar re-encodes byte-identical. Unlike the
    Markers2 GEOB (base64 + a NUL reserve), the sidecar has NO padding
    field: its size is fully determined by the `srtshead` header's
    `total_samples` (each stem's MP3 stream is trimmed/padded to exactly
    that on write), so header + XOR'd stem chunks rebuild byte-for-byte."""
    from sidecaramel.stems import (parse_serato_stems_header,
                                   iter_serato_stem_chunks,
                                   build_serato_stems)
    sc = TAGGED / f"{NAME}.1.2.serato-stems"
    if not sc.exists():
        pytest.skip("stems sidecar not present")
    data = sc.read_bytes()
    hdr = parse_serato_stems_header(data)
    assert hdr is not None and hdr["version"] == (1, 2)
    # Chunk payloads pass through verbatim (still XOR-masked on disk); we
    # are testing the container framing, not the audio.
    chunks = list(iter_serato_stem_chunks(data, hdr["body_offset"]))
    assert len(chunks) == hdr["n_stems"]
    rebuilt = build_serato_stems(
        {"header_length": hdr["header_length"], "version": hdr["version"],
         "n_stems": hdr["n_stems"], "total_samples": hdr["total_samples"],
         "sample_rate": hdr["sample_rate"]},
        chunks)
    assert rebuilt == data, \
        "stems sidecar container did not round-trip byte-exact"


def _count_mp3_frames(mp3: bytes) -> int:
    """Count MPEG-1 Layer III frames in a bare stream (incl. the Xing frame)."""
    _br = {1: 32, 2: 40, 3: 48, 4: 56, 5: 64, 6: 80, 7: 96, 8: 112,
           9: 128, 10: 160, 11: 192, 12: 224, 13: 256, 14: 320}
    _sr = {0: 44100, 1: 48000, 2: 32000}
    off = n = 0
    while off + 4 <= len(mp3):
        h = mp3[off:off + 4]
        if h[0] != 0xFF or (h[1] & 0xE0) != 0xE0:
            break
        br = _br.get((h[2] >> 4) & 0xF)
        sr = _sr.get((h[2] >> 2) & 0x3)
        if not br or not sr:
            break
        off += (144 * br * 1000) // sr + ((h[2] >> 1) & 1)
        n += 1
    return n


def test_stems_gapless_identity():
    """(A) The gapless identity that proves the +12 chunk-payload boundary
    is right, on the real sidecar: for the first stem,
        audio_frames * 1152 - encoder_delay - end_padding == total_samples
    Reading the payload 4 bytes late would drop LAME's Xing frame and break
    this by exactly 1152 samples."""
    from sidecaramel.stems import (parse_serato_stems_header,
                                   iter_serato_stem_chunks,
                                   decode_xor_payload, read_lame_gapless)
    sc = TAGGED / f"{NAME}.1.2.serato-stems"
    if not sc.exists():
        pytest.skip("stems sidecar not present")
    data = sc.read_bytes()
    hdr = parse_serato_stems_header(data)
    gl = read_lame_gapless(str(sc))
    assert gl is not None, "no LAME Xing frame found (payload boundary wrong?)"
    delay, padding = gl
    first = next(iter(iter_serato_stem_chunks(data, hdr["body_offset"])))
    mp3 = decode_xor_payload(first[1])
    audio_frames = _count_mp3_frames(mp3) - 1        # minus the Xing info frame
    assert audio_frames * 1152 - delay - padding == hdr["total_samples"], (
        f"gapless identity broken: {audio_frames}*1152 - {delay} - {padding} "
        f"!= {hdr['total_samples']}")


@pytest.mark.parametrize("ext", ["mp4"])
def test_marker_write_preserves_other_blobs(ext, tmp_path):
    """(B) A cue/loop edit is non-destructive on a real file: rewriting only
    Serato Markers2 into the tagged MP4 leaves every other Serato blob —
    including the MP4-only RelVolAd / VidAssoc / playcount that sidecaramel
    has no builder for — byte-identical, and preserves all container atoms."""
    tagged = TAGGED / f"{NAME}.{ext}"
    if not tagged.exists():
        pytest.skip(f"{ext} audio not present")
    work = tmp_path / f"work.{ext}"
    shutil.copy2(tagged, work)                       # start from the tagged file
    before = _blobs(tagged)

    m = parse_serato_markers2_full(before["Serato Markers2"])
    inner = build_markers2_inner(
        cues=m["cues"], loops=m["loops"], color_rgb=m.get("color"),
        bpm_lock=m.get("bpm_lock"), flips=m["flips"])
    write_serato_markers2_any(str(work), inner,
                              target_size=len(before["Serato Markers2"]),
                              line_width=72, confirm=True)
    after = _blobs(work)

    # Every OTHER Serato blob survives the write byte-for-byte.
    for name, blob in before.items():
        if name == "Serato Markers2":
            continue
        assert after.get(name) == blob, \
            f"{ext}: {name} was disturbed by a Markers2-only write"

    # And no container atom was dropped.
    from mutagen.mp4 import MP4
    assert len(MP4(str(work)).tags or {}) == len(MP4(str(tagged)).tags or {})


def test_every_source_blob_is_accounted_for():
    """'flag what you don't know': every Serato blob the real files actually
    carry is either rebuilt byte-exact (`_REBUILT`) or a known, documented
    gap (`_FLAGGED`). A Serato blob type we've never seen trips this test so
    it can't slip through silently."""
    seen: set = set()
    for ext, _ in _CONTAINERS:
        tagged = TAGGED / f"{NAME}.{ext}"
        if tagged.exists():
            seen |= set(_blobs(tagged))
    unknown = seen - _REBUILT - _FLAGGED
    assert not unknown, f"unrecognised Serato blob type(s): {sorted(unknown)}"
    assert _REBUILT <= seen, \
        f"expected rebuilt blobs missing from the fixtures: {_REBUILT - seen}"
