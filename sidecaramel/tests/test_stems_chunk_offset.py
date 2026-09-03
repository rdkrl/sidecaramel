"""The stem payload starts at chunk offset +12, not +16.

Two layers, both guarding the payload boundary:

1. Synthetic — pins the offset without a committed binary. A chunk is
   built the way a real sidecar is laid out and the parser must hand
   back the payload starting at the MPEG sync word.
2. Golden — opt-in via `SIDECARAMEL_TEST_SIDECAR=/path/to/x.serato-stems`.
   Asserts against a real Serato file that every chunk starts on a
   frame header and that the LAME gapless values reconcile with the
   sidecar header EXACTLY:

       audio_frames * 1152 - encoder_delay - end_padding == total_samples

   That identity is what proves the payload boundary is right; it does
   not hold if the first frame is truncated.
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from sidecaramel.stems import (
    SERATO_XOR_KEY,
    STEM_WAV_SR,
    build_serato_stems,
    decode_xor_payload,
    expected_stem_frames,
    header_derived_drift_samples,
    iter_serato_stem_chunks,
    looks_like_mp3,
    parse_serato_stems_header,
    read_lame_gapless,
    wav_frame_count,
)

SYNC = bytes([0xFF, 0xFB, 0x90, 0x64])          # MPEG-1 LIII 128k 44.1k


def _sidecar(bodies: dict[int, bytes], sample_rate: int = 44100,
             total_samples: int = 44100) -> bytes:
    """Build a sidecar the way Serato lays one out."""
    out = bytearray(b"srtshead")
    out += struct.pack(">I", 16)
    out += struct.pack(">HH", 1, 2)
    out += struct.pack(">I", len(bodies))
    out += struct.pack(">I", total_samples)
    out += struct.pack(">I", sample_rate)
    for stem_type, body in bodies.items():
        payload = bytes(b ^ SERATO_XOR_KEY for b in body)
        out += b"stem"
        out += struct.pack(">I", 4 + len(payload))   # type counts in
        out += struct.pack(">I", stem_type)
        out += payload
    return bytes(out)


def test_payload_starts_at_plus_twelve():
    body = SYNC + b"\x00" * 413
    data = _sidecar({0: body})
    hdr = parse_serato_stems_header(data)
    chunks = list(iter_serato_stem_chunks(data, hdr["body_offset"]))

    assert len(chunks) == 1
    stem_type, enc = chunks[0]
    assert stem_type == 0
    decoded = decode_xor_payload(enc)
    assert decoded == body, "payload boundary moved"
    assert decoded[:4] == SYNC, "first frame header was cut off"


def test_all_four_types_and_no_trailing_bytes():
    bodies = {t: SYNC + bytes([t]) * 413 for t in (0, 3, 2, 1)}
    data = _sidecar(bodies)
    hdr = parse_serato_stems_header(data)
    got = list(iter_serato_stem_chunks(data, hdr["body_offset"]))

    assert [t for t, _ in got] == [0, 3, 2, 1]
    consumed = 28 + sum(12 + len(b) for b in bodies.values())
    assert consumed == len(data), "chunk walk must end exactly at EOF"
    for stem_type, enc in got:
        assert decode_xor_payload(enc) == bodies[stem_type]


def test_strict_guard_rejects_a_shifted_payload():
    """A guard that scans 512 bytes passes a truncated first frame.
    The strict one must not."""
    body = SYNC + b"\x00" * 409 + SYNC          # sync again further in
    assert looks_like_mp3(body) is True
    assert looks_like_mp3(body[4:]) is False    # 4-byte shift → reject
    assert looks_like_mp3(b"ID3" + b"\x00" * 10) is True
    assert looks_like_mp3(b"") is False
    assert looks_like_mp3(b"\xff") is False


def test_drift_is_zero_without_a_lame_frame(tmp_path: Path):
    """No info frame → nothing to correct. Must be 0, never a guess."""
    p = tmp_path / "x.serato-stems"
    p.write_bytes(_sidecar({0: SYNC + b"\x00" * 413}))
    assert read_lame_gapless(p) is None
    assert header_derived_drift_samples(p, 44100) == 0


# ============================================================
# Golden — against a real Serato sidecar, opt-in
# ============================================================

_REAL = os.environ.get("SIDECARAMEL_TEST_SIDECAR", "")
_skip = pytest.mark.skipif(
    not (_REAL and Path(_REAL).is_file()),
    reason="set SIDECARAMEL_TEST_SIDECAR to a real .serato-stems file",
)


def _frame_len(h: bytes) -> int | None:
    br = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128,
          160, 192, 224, 256, 320, 0]
    sr = [44100, 48000, 32000, 0]
    if len(h) < 4 or h[0] != 0xFF or (h[1] & 0xE0) != 0xE0:
        return None
    ver, lay = (h[1] >> 3) & 3, (h[1] >> 1) & 3
    bri, sri, pad = (h[2] >> 4) & 0xF, (h[2] >> 2) & 3, (h[2] >> 1) & 1
    if ver != 3 or lay != 1 or bri in (0, 15) or sri == 3:
        return None
    return int(144 * br[bri] * 1000 / sr[sri]) + pad


@_skip
def test_real_sidecar_chunks_start_on_a_frame_header():
    data = Path(_REAL).read_bytes()
    hdr = parse_serato_stems_header(data)
    assert hdr is not None
    chunks = list(iter_serato_stem_chunks(data, hdr["body_offset"]))
    assert len(chunks) == hdr["n_stems"]
    for stem_type, enc in chunks:
        decoded = decode_xor_payload(enc)
        assert _frame_len(decoded[:4]) is not None, (
            f"stem {stem_type} does not start on a frame header")
        assert looks_like_mp3(decoded)


@_skip
def test_real_sidecar_gapless_reconciles_with_header():
    data = Path(_REAL).read_bytes()
    hdr = parse_serato_stems_header(data)
    gapless = read_lame_gapless(_REAL)
    assert gapless is not None, "expected a LAME info frame"
    delay, padding = gapless

    for stem_type, enc in iter_serato_stem_chunks(
            data, hdr["body_offset"]):
        body = decode_xor_payload(enc)
        pos = n = 0
        while pos + 4 <= len(body):
            flen = _frame_len(body[pos:pos + 4])
            if flen:
                n += 1
                pos += flen
            else:
                pos += 1
        audio_frames = n - 1                     # minus the Xing frame
        assert audio_frames * 1152 - delay - padding == hdr["total_samples"], (
            f"stem {stem_type}: frame count does not reconcile with the "
            f"sidecar header — payload boundary or gapless values wrong")


@_skip
def test_real_sidecar_drift_is_the_encoder_delay():
    delay, _padding = read_lame_gapless(_REAL)
    src_sr = parse_serato_stems_header(
        Path(_REAL).read_bytes()[:28])["sample_rate"]
    assert header_derived_drift_samples(_REAL, src_sr) == -delay
    assert header_derived_drift_samples(_REAL, 16000) == -round(
        delay * 16000 / src_sr)


def test_roundtrip_is_byte_identical_synthetic():
    """parse → rebuild must reproduce the input exactly."""
    bodies = {t: SYNC + bytes([t]) * 413 for t in (0, 3, 2, 1)}
    data = _sidecar(bodies)
    hdr = parse_serato_stems_header(data)
    chunks = list(iter_serato_stem_chunks(data, hdr["body_offset"]))
    assert build_serato_stems(hdr, chunks) == data


@_skip
def test_roundtrip_is_byte_identical_real_sidecar():
    """The decisive one: a real Serato sidecar, decomposed into header
    plus chunks and reassembled, must come back byte-for-byte. Any
    mis-modelled reserved field or unaccounted byte breaks this."""
    data = Path(_REAL).read_bytes()
    hdr = parse_serato_stems_header(data)
    chunks = list(iter_serato_stem_chunks(data, hdr["body_offset"]))

    rebuilt = build_serato_stems(hdr, chunks)
    assert len(rebuilt) == len(data), (
        f"length differs by {len(rebuilt) - len(data):+d} bytes")
    assert rebuilt == data, "rebuilt sidecar is not byte-identical"


# ============================================================
# End-to-end — needs ffmpeg on PATH
# ============================================================

_have_ffmpeg = shutil.which("ffmpeg") is not None
_skip_ff = pytest.mark.skipif(
    not (_have_ffmpeg and _REAL and Path(_REAL).is_file()),
    reason="needs ffmpeg on PATH and SIDECARAMEL_TEST_SIDECAR",
)


@_skip_ff
def test_extracted_wav_is_gapless_and_sample_exact(tmp_path: Path):
    """ffmpeg honours the LAME `Xing` frame, so an extracted stem lands
    exactly on the header's total_samples — no correction needed and
    none allowed on top."""
    data = Path(_REAL).read_bytes()
    hdr = parse_serato_stems_header(data)
    stem_type, enc = next(iter(iter_serato_stem_chunks(
        data, hdr["body_offset"])))
    mp3 = tmp_path / "stem.mp3"
    mp3.write_bytes(decode_xor_payload(enc))
    wav = tmp_path / "stem.wav"

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                    "-ac", "1", "-ar", str(STEM_WAV_SR), str(wav)],
                   check=True, capture_output=True, timeout=120)

    assert wav_frame_count(wav) == expected_stem_frames(hdr, STEM_WAV_SR)


@_skip_ff
def test_disabling_gapless_puts_the_frame_back(tmp_path: Path):
    """Guards the ffmpeg invocation: with gapless handling switched
    off the stem is one MPEG frame too long. If this ever stops being
    true, the extractor's flag choice needs revisiting."""
    data = Path(_REAL).read_bytes()
    hdr = parse_serato_stems_header(data)
    _t, enc = next(iter(iter_serato_stem_chunks(
        data, hdr["body_offset"])))
    mp3 = tmp_path / "stem.mp3"
    mp3.write_bytes(decode_xor_payload(enc))
    wav = tmp_path / "raw.wav"

    src_sr = hdr["sample_rate"]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-flags2", "+skip_manual", "-i", str(mp3),
                    "-ac", "1", "-ar", str(src_sr), str(wav)],
                   check=True, capture_output=True, timeout=120)

    delay, padding = read_lame_gapless(_REAL)
    assert wav_frame_count(wav) == hdr["total_samples"] + delay + padding
