"""Regression: cuss_flip must detect occupied Serato FLIP slots.

The pre-fix bug read ``parse_serato_markers2_full()``'s result under a
non-existent ``"entries"`` key (expecting ``type`` / ``slot_index``
fields), so ``flip_slots_used`` stayed empty, every slot 0..5 looked
free, and the occupied-slot / duplicate-name guards in
``analyze_track`` never fired — the writer could overwrite an occupied
FLIP slot it believed was empty.

These tests build a real Markers2 blob (the same encoder path Serato
authoring uses), parse it, and assert cuss_flip's slot collector sees
the slot as used.
"""
from __future__ import annotations


def _markers2_blob_with_flip(slot_index: int, name: str) -> bytes:
    """Build a Serato Markers2 outer blob carrying one FLIP."""
    from sidecaramel.tags import build_markers2_inner, pack_serato_markers2
    flips = [{
        "slot_index": slot_index,
        "name": name,
        "loop": False,
        "actions": [
            {"type": "censor", "trigger_s": 12.0,
                "reverse_target_s": 11.5, "speed": -1.0},
            {"type": "jump", "from_s": 12.0, "to_s": 12.5},
        ],
    }]
    inner = build_markers2_inner(cues=[], loops=[], flips=flips)
    return pack_serato_markers2(inner, target_size=0)


def test_occupied_flip_slot_is_detected_as_used():
    from sidecaramel.tags import parse_serato_markers2_full
    from sidecaramel.cuss_flip import _flip_slots_from_parsed

    parsed = parse_serato_markers2_full(
        _markers2_blob_with_flip(3, "CLEAN"))
    used, names = _flip_slots_from_parsed(parsed)

    assert used == {3}
    assert "CLEAN" in names
    # The slot must drop out of the "empty" list (analyze_track logic).
    empty = [s for s in range(6) if s not in used]
    assert 3 not in empty


def test_no_flips_means_all_slots_free():
    from sidecaramel.tags import (build_markers2_inner, pack_serato_markers2,
                                       parse_serato_markers2_full)
    from sidecaramel.cuss_flip import _flip_slots_from_parsed

    inner = build_markers2_inner(cues=[], loops=[], flips=[])
    parsed = parse_serato_markers2_full(
        pack_serato_markers2(inner, target_size=0))
    used, names = _flip_slots_from_parsed(parsed)

    assert used == set()
    assert names == []
    assert [s for s in range(6) if s not in used] == [0, 1, 2, 3, 4, 5]


def test_flips_live_under_flips_key_not_entries():
    """Guards the exact regression: the data is under ``flips`` with a
    ``slot`` key — the old ``entries`` / ``type`` / ``slot_index``
    shape does not exist."""
    from sidecaramel.tags import parse_serato_markers2_full

    parsed = parse_serato_markers2_full(
        _markers2_blob_with_flip(0, "CLEAN"))
    assert parsed.get("flips")            # populated
    assert not parsed.get("entries")      # the buggy key is empty/absent
    flip = parsed["flips"][0]
    assert flip.get("slot") == 0          # key is "slot", not "slot_index"
    assert flip.get("slot_index") is None


def test_none_parsed_is_safe():
    """A failed parse (parsed=None) must not raise — it yields no
    used slots."""
    from sidecaramel.cuss_flip import _flip_slots_from_parsed
    used, names = _flip_slots_from_parsed(None)
    assert used == set()
    assert names == []
