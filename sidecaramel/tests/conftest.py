"""Pytest config + shared fixtures.

Synthesises a 1-second silent MP3 + M4A on-the-fly via mutagen so tests
need zero pre-committed binary fixtures.  Each fixture writes into a
tmp_path provided by pytest; nothing touches user audio.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest


# ============================================================
# Minimal MP3 fixture — single MPEG-1 Layer III silent frame
# repeated to form a ~1 s file, big enough for mutagen to parse
# the header and accept ID3 tags via mutagen.id3.ID3.
# ============================================================

# 32 ms silent MPEG-1 Layer III frame, 128 kbps, 44.1 kHz, stereo
# (constant content, hand-built; valid MPEG sync 0xFFF + 4-byte
# header, payload all zeros).  417 bytes per frame.
def _silent_mpeg1_frame() -> bytes:
    # Header: 11111111 11111011 10010000 00000000
    #          sync ver layer protect  bitrate  freq  pad chan
    header = bytes([0xFF, 0xFB, 0x90, 0x00])
    # Payload: 413 zero bytes → 417-byte CBR frame
    return header + b"\x00" * 413


@pytest.fixture
def minimal_mp3(tmp_path: Path) -> Path:
    """Write a ~1 s silent MP3 with no ID3 tags."""
    p = tmp_path / "fixture.mp3"
    frame = _silent_mpeg1_frame()
    p.write_bytes(frame * 32)  # ~1 s
    return p


@pytest.fixture
def minimal_mp3_with_id3(tmp_path: Path, minimal_mp3: Path) -> Path:
    """Same fixture, with an empty ID3v2.4 tag pre-attached so
    mutagen.id3.ID3.add(...) round-trips cleanly."""
    from mutagen.id3 import ID3, ID3NoHeaderError
    try:
        tag = ID3(str(minimal_mp3))
    except ID3NoHeaderError:
        tag = ID3()
    tag.save(str(minimal_mp3))
    return minimal_mp3


@pytest.fixture
def serato_crate_path(tmp_path: Path) -> Path:
    """A `.crate` filename to write to (path only, file doesn't exist)."""
    return tmp_path / "TestCrate.crate"


# ============================================================
# Cross-container fixtures — real WAV/AIFF/M4A/FLAC/OGG via ffmpeg
#
# These tests verify the writers don't corrupt non-MP3 containers.
# Skipped automatically if ffmpeg isn't available.
# ============================================================

import shutil
import subprocess


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _ffmpeg_synth(out_path: Path, codec_args: list[str]) -> Path:
    """Synthesize a 0.5 s 440 Hz sine to `out_path` via ffmpeg.
    `codec_args` are appended after the sine generator (e.g.
    `["-c:a", "libvorbis"]` for OGG)."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
        "-ac", "1", "-ar", "16000",
    ] + codec_args + [str(out_path)]
    subprocess.run(cmd, check=True)
    return out_path


@pytest.fixture
def ffmpeg_wav(tmp_path: Path) -> Path:
    if not _have_ffmpeg():
        pytest.skip("ffmpeg not installed")
    return _ffmpeg_synth(tmp_path / "fixture.wav", [])


@pytest.fixture
def ffmpeg_aiff(tmp_path: Path) -> Path:
    if not _have_ffmpeg():
        pytest.skip("ffmpeg not installed")
    return _ffmpeg_synth(tmp_path / "fixture.aiff", [])


@pytest.fixture
def ffmpeg_m4a(tmp_path: Path) -> Path:
    if not _have_ffmpeg():
        pytest.skip("ffmpeg not installed")
    return _ffmpeg_synth(tmp_path / "fixture.m4a",
                            ["-c:a", "aac"])


@pytest.fixture
def ffmpeg_flac(tmp_path: Path) -> Path:
    if not _have_ffmpeg():
        pytest.skip("ffmpeg not installed")
    return _ffmpeg_synth(tmp_path / "fixture.flac",
                            ["-c:a", "flac"])


@pytest.fixture
def ffmpeg_ogg(tmp_path: Path) -> Path:
    if not _have_ffmpeg():
        pytest.skip("ffmpeg not installed")
    return _ffmpeg_synth(tmp_path / "fixture.ogg",
                            ["-c:a", "libvorbis"])
