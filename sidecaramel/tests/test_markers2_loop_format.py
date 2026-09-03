"""Markers2 LOOP entries must match the community-reverse-engineered
Serato format (Holzhaus `serato-tags`, cross-checked against Mixxx
`SeratoMarkers2LoopEntry`) — NOT merely round-trip through our own
encoder/parser.

These fixtures are built independently from the published spec bytes, so
they fail if the wire layout drifts from Serato even when our own
encode↔parse pair stays internally consistent. Regression guard for the
bug where the LOOP color was read one byte off and the locked flag was
read from the wrong offset (a genuine Serato loop came back with rotated
colors and locked=False).
"""
from __future__ import annotations

import base64
import struct

from sidecaramel.tags import parse_serato_markers2_full, encode_loop_entry
from sidecaramel.blobs import _decode_markers2_entry


# --- spec-conformant byte builders (independent of our encoder) ------

def _spec_loop_body(idx, start_ms, end_ms, rgb, locked, name):
    """A LOOP entry body exactly as the Serato Markers2 spec documents it."""
    r, g, b = rgb
    return (
        b"\x00"                                   # 0x00 padding
        + bytes([idx])                            # 0x01 index
        + struct.pack(">I", start_ms)             # 0x02-05 start_ms
        + struct.pack(">I", end_ms)               # 0x06-09 end_ms
        + b"\xff\xff\xff\xff"                     # 0x0a-0d constant
        + bytes([0x00, r, g, b])                  # 0x0e-11 color 00 R G B
        + b"\x00"                                 # 0x12 padding
        + bytes([1 if locked else 0])             # 0x13 locked
        + name.encode("utf-8") + b"\x00"          # 0x14+ name cstring
    )


def _wrap_outer(*entries):
    """Wrap (type, body) entries into a full Markers2 OUTER blob
    (u8 1, u8 1, base64(inner)), matching what harvest() hands the parser."""
    inner = b"\x01\x01"                           # 2-byte inner version
    for etype, body in entries:
        inner += (etype.encode("ascii") + b"\x00"
                  + struct.pack(">I", len(body)) + body)
    return b"\x01\x01" + base64.b64encode(inner)


# --- the parser reads the spec correctly -----------------------------

def test_parse_reads_spec_loop_color_locked_and_name():
    # Independently built: locked loop, colored #27aae1, named "hot".
    body = _spec_loop_body(3, 1000, 2000, (0x27, 0xAA, 0xE1),
                           locked=True, name="hot")
    parsed = parse_serato_markers2_full(_wrap_outer(("LOOP", body)))

    assert len(parsed["loops"]) == 1
    lp = parsed["loops"][0]
    assert lp["idx"] == 3
    assert lp["start_ms"] == 1000
    assert lp["end_ms"] == 2000
    # Pre-fix these three were wrong: color rotated to (0xaa,0xe1,0x00),
    # locked read as False, and (via the blobs decoder) name "ot".
    assert lp["color"] == (0x27, 0xAA, 0xE1)
    assert lp["locked"] is True
    assert lp["label"] == "hot"


def test_blobs_decoder_reads_spec_loop_name_untruncated():
    body = _spec_loop_body(1, 500, 1500, (0x10, 0x20, 0x30),
                           locked=False, name="verse")
    dec = _decode_markers2_entry("LOOP", body)
    assert dec["rgb"] == "#102030"
    assert dec["locked"] is False
    assert dec["name"] == "verse"        # not "erse" (was name_start off by 1)


# --- our encoder emits spec-conformant bytes -------------------------

def test_encode_loop_entry_is_spec_conformant():
    raw = encode_loop_entry(3, 1000, 2000, (0x27, 0xAA, 0xE1),
                            label="hot", locked=True)
    assert raw[:5] == b"LOOP\x00"
    body_len = struct.unpack(">I", raw[5:9])[0]
    body = raw[9:9 + body_len]

    assert body[0] == 0x00
    assert body[1] == 3
    assert body[2:6] == struct.pack(">I", 1000)
    assert body[6:10] == struct.pack(">I", 2000)
    assert body[10:14] == b"\xff\xff\xff\xff"
    assert body[14:18] == bytes([0x00, 0x27, 0xAA, 0xE1])   # 00 R G B
    assert body[18] == 0x00
    assert body[19] == 0x01                                 # locked
    assert body[20:23] == b"hot"


# --- and encode → parse still round-trips ----------------------------

def test_encode_then_parse_roundtrips():
    raw = encode_loop_entry(5, 32000, 64000, (0x00, 0xCC, 0x00),
                            label="main8", locked=False)
    body = raw[9:]
    lp = parse_serato_markers2_full(_wrap_outer(("LOOP", body)))["loops"][0]
    assert lp["idx"] == 5
    assert lp["start_ms"] == 32000
    assert lp["end_ms"] == 64000
    assert lp["color"] == (0x00, 0xCC, 0x00)
    assert lp["locked"] is False
    assert lp["label"] == "main8"
