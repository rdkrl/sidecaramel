"""write_serato_markers_full_mp3 → harvest → read back.
Cues + loops + color round-trip with exact field preservation.
"""
from __future__ import annotations

from pathlib import Path


def test_cue_loop_mp3_roundtrip(minimal_mp3_with_id3: Path):
    from sidecaramel.tags import (write_serato_markers_full_mp3,
                                       harvest_serato_blobs,
                                       parse_serato_markers2_full)

    cues = [
        {"idx": 0, "pos_ms": 1000,
            "color": (0xCC, 0x88, 0x00), "label": "intro"},
        {"idx": 4, "pos_ms": 64000,
            "color": (0x00, 0xCC, 0xCC), "label": "drop"},
    ]
    loops = [
        {"idx": 0, "start_ms": 32000, "end_ms": 64000,
            "color": (0x00, 0xCC, 0x00), "label": "main8",
            "locked": True},
    ]
    ok = write_serato_markers_full_mp3(
        str(minimal_mp3_with_id3),
        cues=cues, loops=loops,
        confirm=True,
    )
    assert ok is True

    # Read back
    blobs = list(harvest_serato_blobs(str(minimal_mp3_with_id3)))
    m2_blobs = [p for d, p, _ in blobs if d == "Serato Markers2"]
    assert m2_blobs, "no Markers2 blob written"
    parsed = parse_serato_markers2_full(m2_blobs[0])
    assert len(parsed["cues"]) == 2
    assert len(parsed["loops"]) == 1
    assert parsed["cues"][0]["pos_ms"] == 1000
    assert parsed["cues"][0]["label"] == "intro"
    assert parsed["cues"][1]["pos_ms"] == 64000
    assert parsed["loops"][0]["start_ms"] == 32000
    assert parsed["loops"][0]["end_ms"] == 64000
    assert parsed["loops"][0]["locked"] is True
