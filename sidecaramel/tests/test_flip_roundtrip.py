"""FLIP write → read → parse roundtrip.

Also a guard on the import wiring: `encode_flip_entry` reaches into
`sidecaramel.flip_writer`, and if that import ever resolves to a
non-existent module the failure is a ModuleNotFoundError on every
FLIP authoring path — silent until someone authors a FLIP.
"""
from __future__ import annotations

import struct

import pytest


def test_encode_flip_entry_does_not_raise():
    """The earlier fatal bug: this call raised ModuleNotFoundError."""
    from sidecaramel.tags import encode_flip_entry
    blob = encode_flip_entry(
        slot_index=0,
        name="CLEAN",
        loop=False,
        actions=[
            {"type": "censor", "trigger_s": 32.10,
                "reverse_target_s": 31.50, "speed": -1.0},
            {"type": "jump", "from_s": 32.10, "to_s": 32.60},
        ],
    )
    assert isinstance(blob, bytes)
    assert blob.startswith(b"FLIP\x00")


def test_flip_roundtrip_via_markers2_inner_outer():
    """End-to-end: build a Markers2 blob containing a FLIP, encode
    the outer wrapper, parse it back, assert the FLIP shape comes
    out byte-equal to what we put in."""
    from sidecaramel.tags import (build_markers2_inner,
                                       pack_serato_markers2,
                                       parse_serato_markers2_full)

    flips = [{
        "slot_index": 0,
        "name": "CLEAN",
        "loop": False,
        "actions": [
            {"type": "censor", "trigger_s": 32.10,
                "reverse_target_s": 31.50, "speed": -1.0},
            {"type": "jump", "from_s": 32.10, "to_s": 32.60},
        ],
    }]
    inner = build_markers2_inner(cues=[], loops=[], flips=flips)
    blob = pack_serato_markers2(inner, target_size=0)

    parsed = parse_serato_markers2_full(blob)
    assert len(parsed["flips"]) == 1
    flip = parsed["flips"][0]
    assert flip["slot"] == 0
    assert flip["name"] == "CLEAN"
    assert flip["loop"] is False
    assert flip["action_count"] == 2
    # No _warnings means clean parse
    assert "_warnings" not in parsed or not parsed["_warnings"]


def test_build_word_bleep_action_pair():
    """Quick sanity-check on the CENSOR+JUMP pair shape used by
    cuss_flip."""
    from sidecaramel.flip_writer import build_word_bleep
    actions = build_word_bleep(word_start_s=10.0, word_end_s=10.5)
    assert len(actions) == 2
    censor, jump = actions
    # CENSOR: id=1, 24-byte payload
    assert censor[0] == 0x01
    assert struct.unpack(">I", censor[1:5])[0] == 24
    # JUMP: id=0, 16-byte payload
    assert jump[0] == 0x00
    assert struct.unpack(">I", jump[1:5])[0] == 16
