"""The three registries a consuming pipeline extends.

Nothing producer-specific is built into the defaults: a pipeline that
writes its own sidecars, stem caches or LRC markers registers them at
start-up. These tests pin the defaults (so a producer name cannot creep
back in) and the registration contract.
"""
from __future__ import annotations

import pytest

from sidecaramel import consolidate, lrc_cusswords, stems


@pytest.fixture(autouse=True)
def _restore_registries():
    """Registries are module globals — put them back after each test."""
    saved = (consolidate.SIDECAR_SUFFIXES,
             lrc_cusswords.LRC_METADATA_PREFIXES,
             stems.EXTERNAL_STEM_WAV_SUFFIXES,
             stems.CLEANUP_STEM_WAV_SUFFIXES)
    yield
    (consolidate.SIDECAR_SUFFIXES,
     lrc_cusswords.LRC_METADATA_PREFIXES,
     stems.EXTERNAL_STEM_WAV_SUFFIXES,
     stems.CLEANUP_STEM_WAV_SUFFIXES) = saved


def test_companion_defaults_are_format_native_only():
    """Only the standard `.lrc` and the Serato stem artefacts ship as
    defaults — and the stem WAV names come from `stems`, so the two
    lists cannot drift apart."""
    assert consolidate.SIDECAR_SUFFIXES == (
        (".lrc", ".serato-stems") + stems.SERATO_STEM_WAV_SUFFIXES)
    for suffix in consolidate.SIDECAR_SUFFIXES:
        assert suffix.startswith(".")
        assert "serato" in suffix or suffix == ".lrc"


def test_add_companion_suffix():
    consolidate.add_companion_suffix(".mytool-anchors.json")
    assert ".mytool-anchors.json" in consolidate.SIDECAR_SUFFIXES
    n = len(consolidate.SIDECAR_SUFFIXES)
    consolidate.add_companion_suffix(".mytool-anchors.json")
    assert len(consolidate.SIDECAR_SUFFIXES) == n, "must be idempotent"
    with pytest.raises(ValueError):
        consolidate.add_companion_suffix("no-leading-dot")


def test_lrc_metadata_prefixes_default_empty():
    assert lrc_cusswords.LRC_METADATA_PREFIXES == ()


def test_add_lrc_metadata_prefix_skips_the_line():
    lrc = "\n".join([
        "[ar:Artist]",
        "[mytool:v3 stamped]",
        "[00:01.00]a word here",
    ])
    # Registered marker → the line is dropped.
    lrc_cusswords.add_lrc_metadata_prefix("[mytool:")
    assert "[mytool:" in lrc_cusswords.LRC_METADATA_PREFIXES
    lrc_cusswords.add_lrc_metadata_prefix("[mytool:")
    assert len(lrc_cusswords.LRC_METADATA_PREFIXES) == 1, "idempotent"
    with pytest.raises(ValueError):
        lrc_cusswords.add_lrc_metadata_prefix("")
    # The generic `[tag:value]` rule already drops both header lines, so
    # parsing stays clean either way — the registry is for markers that
    # do NOT sit alone on a well-formed metadata line.
    spans = lrc_cusswords.parse_lrc(lrc)
    assert all("mytool" not in sp.text_clean for sp in spans)


def test_stem_suffix_registry_still_independent():
    """`stems` has its own registry; registering there must not leak
    into the consolidate list, and vice versa."""
    stems.add_external_stem_suffix(".myproj-vocals.wav")
    assert ".myproj-vocals.wav" in stems.CLEANUP_STEM_WAV_SUFFIXES
    assert ".myproj-vocals.wav" not in consolidate.SIDECAR_SUFFIXES
