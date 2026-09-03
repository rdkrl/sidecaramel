"""Regression tests for the roundtrip-safety Serato-blob inventory.

The external review found that the harness excluded Serato tags from its
inventory AND stripped every Serato tag indiscriminately, so it scored
total loss of a pre-existing private Serato blob ("Serato ExistingFuture")
as verdict=cosmetic / lost_keys=[].

This is handled: serato_inventory() catalogs Serato blobs with a content
hash, strip_serato_edit() removes only what apply_serato_edit wrote this
cycle, and roundtrip_check() flags loss/mutation of any untargeted
pre-existing Serato blob as DATA LOSS.
"""
import pytest

from sidecaramel.roundtrip_safety import (
    build_rich_mp3, roundtrip_check, serato_inventory,
    apply_serato_edit, strip_serato_edit,
)
from sidecaramel.tags import write_serato_geob


def test_serato_inventory_catalogs_blobs_with_hash(tmp_path):
    p = str(tmp_path / "t.mp3")
    build_rich_mp3(p)
    assert serato_inventory(p) == {}          # no Serato tags yet
    assert write_serato_geob(p, "Serato BeatGrid",
                             b"\x01PREEXISTING-GRID\x00", confirm=True)
    inv = serato_inventory(p)
    assert any("Serato BeatGrid" in k for k in inv)
    # signature carries a sha256 so a mutation (not just a disappearance)
    # is detectable.
    assert "sha256=" in next(iter(inv.values()))


def test_targeted_strip_preserves_unrelated_serato_blob(tmp_path):
    p = str(tmp_path / "t.mp3")
    build_rich_mp3(p)
    write_serato_geob(p, "Serato BeatGrid", b"GRID-DATA", confirm=True)
    written = apply_serato_edit(p)            # adds Markers_/Markers2
    strip_serato_edit(p, written=written)     # TARGETED strip
    inv = serato_inventory(p)
    assert any("Serato BeatGrid" in k for k in inv)   # survived
    assert not any("Markers" in k for k in inv)       # edit removed


def test_roundtrip_does_not_silently_nuke_preexisting_serato(tmp_path):
    # The reviewer's exact scenario.
    p = str(tmp_path / "t.mp3")
    build_rich_mp3(p)
    write_serato_geob(p, "Serato ExistingFuture",
                      b"ORIGINAL-SERATO-DATA", confirm=True)
    res = roundtrip_check(p)
    assert res.serato_lost_keys == []
    assert res.serato_changed_keys == []
    assert res.ok()                            # pre-existing blob preserved
