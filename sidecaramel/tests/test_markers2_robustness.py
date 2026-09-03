"""Quirk A and Quirk C robustness regressions.

Quirk A: parse_serato_markers2_full() raises TypeError / ValueError
         rather than silently returning an empty dict on the wrong
         input shape.

Quirk C: a malformed FLIP entry inside an otherwise-valid Markers2
         is recorded in `_warnings` rather than silently dropped on
         the floor.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest


def test_parse_markers2_rejects_path_string():
    """Quirk A: handing a path string used to return an empty-but-
    valid result dict, which looked indistinguishable from an
    un-analyzed track and burned reviewer time."""
    from sidecaramel.tags import parse_serato_markers2_full
    with pytest.raises(TypeError, match="bytes"):
        parse_serato_markers2_full("/path/to/track.mp3")


def test_parse_markers2_rejects_path_object(tmp_path: Path):
    from sidecaramel.tags import parse_serato_markers2_full
    with pytest.raises(TypeError, match="bytes"):
        parse_serato_markers2_full(tmp_path / "track.mp3")


def test_parse_markers2_rejects_int():
    from sidecaramel.tags import parse_serato_markers2_full
    with pytest.raises(TypeError):
        parse_serato_markers2_full(42)


def test_parse_markers2_rejects_undecodable_blob():
    """Bytes that don't decode as a Markers2 outer wrapper must raise
    ValueError rather than silently return empty.  A 2-byte input is
    smaller than the header (u8 1, u8 1, ...payload) so the inner
    decoder produces None."""
    from sidecaramel.tags import parse_serato_markers2_full
    # Less than 3 bytes → cannot contain any base64 inner stream
    with pytest.raises(ValueError, match="decode"):
        parse_serato_markers2_full(b"\x01")
    with pytest.raises(ValueError, match="decode"):
        parse_serato_markers2_full(b"")


def test_parse_markers2_accepts_valid_blob():
    """Positive-path sanity check: a real built blob still parses
    without exception."""
    from sidecaramel.tags import (build_markers2_inner,
                                       pack_serato_markers2,
                                       parse_serato_markers2_full)
    inner = build_markers2_inner(cues=[], loops=[], flips=[])
    blob = pack_serato_markers2(inner, target_size=0)
    out = parse_serato_markers2_full(blob)
    assert isinstance(out, dict)
    assert out["cues"] == []
    assert out["loops"] == []


def test_malformed_flip_recorded_in_warnings():
    """Quirk C: previously a bare `except: pass` swallowed the
    malformed FLIP silently.  Now it surfaces in `_warnings` so
    callers can detect partial parse without re-reading the blob."""
    from sidecaramel.tags import (pack_serato_markers2,
                                       parse_serato_markers2_full)
    # Build a Markers2 inner stream by hand with one good CUE
    # entry and one broken FLIP entry (truncated body — no NUL
    # after slot/flag, so name parse fails).
    parts = bytearray(b"\x01\x01")
    # CUE entry (good) — minimum 13-byte body
    cue_body = (b"\x00\x00"           # null + idx=0
                  + struct.pack(">I", 1000)   # pos_ms
                  + b"\x00\xcc\x00\x00"        # NUL + RGB
                  + b"\x00\x00" + b"hi\x00")   # padding + label
    parts += b"CUE\x00" + struct.pack(">I", len(cue_body)) + cue_body
    # FLIP entry (truncated — no NUL after slot, so name parse
    # raises ValueError; should land in _warnings, not crash)
    bad_flip = b"\x00\x00\x01ABCDEFG"  # no NUL anywhere → name_end < 0
    parts += b"FLIP\x00" + struct.pack(">I", len(bad_flip)) + bad_flip
    parts += b"\x00"  # EOS sentinel

    blob = pack_serato_markers2(bytes(parts), target_size=0)
    out = parse_serato_markers2_full(blob)

    # The CUE survived
    assert len(out["cues"]) == 1
    assert out["cues"][0]["idx"] == 0
    # The FLIP did NOT silently appear in flips[] …
    assert out["flips"] == []
    # … but it WAS surfaced in _warnings
    assert "_warnings" in out
    assert any(w["entry"] == "FLIP" for w in out["_warnings"])
