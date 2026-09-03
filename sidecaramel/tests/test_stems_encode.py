"""Sidecar payload conditioning — the byte discipline a chunk needs."""
from __future__ import annotations

import shutil
import struct
import subprocess

import pytest

from sidecaramel import stems_encode as SE
from sidecaramel.stems import (decode_xor_payload, iter_serato_stem_chunks,
                               looks_like_mp3, parse_serato_stems_header)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="ffmpeg not on PATH")


def _fake_frame_header(kbps_idx=9, sr_idx=0):
    """A syntactically valid MPEG-1 Layer III header (9 -> 128 kbps)."""
    b1 = 0xFF
    b2 = 0b11111011                       # MPEG-1, Layer III, no CRC
    b3 = (kbps_idx << 4) | (sr_idx << 2)
    return bytes((b1, b2, b3, 0x00))


def test_strip_id3v2_and_junk_and_id3v1():
    frame = _fake_frame_header() + b"\x00" * 64
    id3v2 = b"ID3" + bytes((4, 0, 0)) + struct.pack(">I", 20)[0:4]
    # syncsafe 20 -> bytes 00 00 00 14
    id3v2 = b"ID3\x04\x00\x00\x00\x00\x00\x14" + b"J" * 20
    id3v1 = b"TAG" + b"x" * 125
    data = id3v2 + b"\x01\x02junk" + frame + id3v1
    out = SE.strip_mp3_tags_and_align_frames(data)
    assert out[0] == 0xFF and (out[1] & 0xE0) == 0xE0
    assert b"TAG" + b"x" * 125 not in out
    with pytest.raises(ValueError):
        SE.strip_mp3_tags_and_align_frames(b"ID3\x04\x00\x00\x00\x00\x00\x02xxnothing here")


def test_frame_props_and_compat_decision():
    good = _fake_frame_header()                     # 128k / 44100
    p = SE.mp3_frame_props(good)
    assert p == {"kbps": 128, "sample_rate": 44100, "mpeg1_layer3": True}
    assert SE.is_sidecar_compatible(good)
    assert SE.mp3_frame_props(_fake_frame_header(sr_idx=1)) \
        == {"kbps": 128, "sample_rate": 48000, "mpeg1_layer3": True}
    assert not SE.is_sidecar_compatible(_fake_frame_header(sr_idx=1))
    assert not SE.is_sidecar_compatible(_fake_frame_header(kbps_idx=14))
    assert SE.mp3_frame_props(b"\x00\x00\x00\x00") is None


@needs_ffmpeg
def test_silent_stem_is_bare_compatible_and_exact(tmp_path):
    n = 44100  # 1 s
    payload = SE.make_silent_stem_mp3(n)
    assert payload[:3] != b"ID3"
    assert SE.is_sidecar_compatible(payload)
    f = tmp_path / "s.mp3"
    f.write_bytes(payload)
    r = subprocess.run(["ffmpeg", "-v", "quiet", "-i", str(f), "-f",
                        "s16le", "-ac", "1", "-ar", "44100", "-"],
                       capture_output=True)
    assert len(r.stdout) // 2 == n, "gapless length must be exact"


@needs_ffmpeg
def test_build_sidecar_bytes_fills_silence_and_self_checks(tmp_path):
    tone = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=1.5",
                    "-ar", "44100", "-ac", "2", str(tone)], check=True)
    blob = SE.build_sidecar_bytes({0: str(tone), 2: str(tone)}, 44100)
    hdr = parse_serato_stems_header(blob)
    assert hdr["n_stems"] == 4 and hdr["total_samples"] == 44100
    assert hdr["body_offset"] == 28
    stems = {}
    for st, enc in iter_serato_stem_chunks(blob, hdr["body_offset"]):
        dec = decode_xor_payload(enc)
        assert looks_like_mp3(dec) and SE.is_sidecar_compatible(dec)
        stems[st] = dec
    assert sorted(stems) == [0, 1, 2, 3]
    # silent slots exist and are real full-length streams, not stubs
    assert len(stems[1]) > 10_000 and len(stems[3]) > 10_000


def test_detect_library_format(tmp_path):
    from sidecaramel.db import detect_library_format
    root = tmp_path / "_Serato_"
    (root / "Subcrates").mkdir(parents=True)
    assert detect_library_format(str(root))["generation"] == "empty"
    (root / "database V2").write_bytes(b"vrsn\x00\x00" + b"x" * 32)
    (root / "Subcrates" / "a.crate").write_bytes(b"vrsn")
    r = detect_library_format(str(root))
    assert r["generation"] == "pre-4.0" and r["crates"] == 1
    (root / "Library").mkdir()
    (root / "Library" / "whatever.bin").write_bytes(
        b"SQLite format 3\x00" + b"\x00" * 64)
    r = detect_library_format(str(root))
    assert r["generation"] == "4.0-with-legacy"
    assert len(r["sqlite_files"]) == 1
