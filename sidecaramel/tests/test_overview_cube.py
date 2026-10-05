"""Overview blob — the 6x6x6 colour-cube reading.

Model (see `sidecaramel.overview_palette`):
    v = 36a + 6b + c, a = bass, b = mid, c = treble, each 0..5
    0 / 1 background, >= 216 flagged (223 = flag + 7)
    240 columns x 16 rows, column-major, row 0 at the top

The fixtures in `fixtures/overview_calibration/` are Serato-written
blobs for generated test tones; see the README there.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sidecaramel.overview_palette import (
    BAND_TIER_DIGIT, CUBE_MAX, blob_payload, byte_to_cube, byte_to_digits,
    cube_to_byte, digits_to_byte, is_background, mix_band_tiers, split_flag)

FIX = Path(__file__).parent / "fixtures" / "overview_calibration"
GOLDEN = (Path(__file__).parent / "fixtures" / "serato_golden"
          / "Round_Tip_Parity_mp3__Serato_Overview.blob")
HEADER = b"\x01\x05"


def _fixture(name: str) -> bytes:
    return (FIX / f"{name}.blob").read_bytes()


def _max_digits(blob: bytes):
    colour = [v for v in blob_payload(blob) if not is_background(v)]
    return tuple(max(byte_to_digits(v)[i] for v in colour) for i in range(3))


ALL_FIXTURES = sorted(p.stem for p in FIX.glob("*.blob"))


# ── the model ────────────────────────────────────────────────────────

def test_digits_roundtrip_over_the_cube():
    for v in range(CUBE_MAX):
        assert digits_to_byte(*byte_to_digits(v)) == v


def test_cube_colour_roundtrip_over_the_cube():
    for v in range(CUBE_MAX):
        assert cube_to_byte(*byte_to_cube(v)) == v


def test_known_values():
    assert byte_to_digits(128) == (3, 3, 2), "128 is a colour, not a marker"
    assert byte_to_cube(128) == (153, 153, 102)
    assert byte_to_digits(43) == (1, 1, 1)
    assert split_flag(223) == (True, 7)
    assert byte_to_cube(223) == byte_to_cube(7) == (0, 51, 51)
    assert split_flag(215) == (False, 215)
    assert is_background(0) and is_background(1) and not is_background(2)


def test_band_tiers_set_one_digit_each_and_never_spill():
    for band, axis in (("bass", 0), ("mid", 1), ("treble", 2)):
        for tier in range(8):
            d = byte_to_digits(mix_band_tiers(**{f"{band}_tier": tier}))
            assert d[axis] == BAND_TIER_DIGIT[band][tier]
            assert sum(d) == d[axis], f"{band} tier {tier} spills: {d}"


def test_band_mix_is_per_digit():
    v = mix_band_tiers(bass_tier=4, treble_tier=1)
    assert byte_to_digits(v) == (BAND_TIER_DIGIT["bass"][4], 0,
                                 BAND_TIER_DIGIT["treble"][1])
    assert mix_band_tiers() is None


def test_band_tables_are_monotone():
    for band, table in BAND_TIER_DIGIT.items():
        assert list(table) == sorted(table), band
        assert all(0 <= d <= 5 for d in table), band


# ── Serato-written fixtures ──────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_fixture_stays_inside_the_cube(name):
    blob = _fixture(name)
    assert blob[:2] == HEADER
    body = blob_payload(blob)
    assert {v for v in body if v >= CUBE_MAX} <= {223}
    assert {i % 16 for i, v in enumerate(body) if v >= CUBE_MAX} <= {8}


@pytest.mark.parametrize("name, axis", [("amp_sweep_60Hz", 0),
                                        ("amp_sweep_1kHz", 1),
                                        ("amp_sweep_8kHz", 2)])
def test_single_tone_lands_on_its_own_digit(name, axis):
    """Bass drives a, mid drives b, treble drives c."""
    mx = _max_digits(_fixture(name))
    assert mx[axis] >= 4, mx
    assert all(d <= 1 for i, d in enumerate(mx) if i != axis), mx


@pytest.mark.parametrize("name, absent", [("mix_60Hz_10kHz", 1),
                                          ("mix_60Hz_1kHz", 2),
                                          ("mix_1kHz_10kHz", 0)])
def test_two_tone_mix_leaves_the_third_digit_empty(name, absent):
    """Mixes combine per digit, with no spill into the absent band."""
    mx = _max_digits(_fixture(name))
    assert mx[absent] == 0, mx
    assert all(d >= 3 for i, d in enumerate(mx) if i != absent), mx


def test_three_tone_mix_lights_every_digit():
    assert all(d >= 2 for d in _max_digits(_fixture("mix_60Hz_1kHz_10kHz")))


def test_background_can_be_zero():
    body = blob_payload(_fixture("log_sweep_background_0"))
    outer = [body[c * 16 + r] for c in range(240) for r in (0, 15)]
    assert set(outer) == {0}


def test_golden_overview_stays_inside_the_cube():
    body = blob_payload(GOLDEN.read_bytes())
    assert {v for v in body if v >= CUBE_MAX} <= {223}


# ── renderer ─────────────────────────────────────────────────────────

def _blob_with(cells: dict, fill: int = 1) -> bytes:
    body = bytearray([fill] * 3840)
    for (col, row), v in cells.items():
        body[col * 16 + row] = v
    return HEADER + bytes(body)


def test_render_orientation_and_colours():
    pytest.importorskip("PIL")
    from sidecaramel.overview import overview_image
    blob = _blob_with({(0, 0): 180, (239, 15): 5, (10, 8): 223})
    img = overview_image(blob)
    assert img.size == (240, 16)
    assert img.getpixel((0, 0)) == (255, 0, 0), "column 0 / row 0 is top-left"
    assert img.getpixel((239, 15)) == (0, 0, 255)
    assert img.getpixel((10, 8)) == (0, 51, 51), "flag draws its base colour"
    assert img.getpixel((5, 5)) == (0, 0, 0), "background 1 is background"
    assert overview_image(_blob_with({}, fill=0)).getpixel((5, 5)) == (0, 0, 0)


def test_render_options():
    pytest.importorskip("PIL")
    from sidecaramel.overview import overview_image
    img = overview_image(_blob_with({(10, 8): 223}),
                         background=(9, 9, 9), flag_colour=(255, 255, 255))
    assert img.getpixel((10, 8)) == (255, 255, 255)
    assert img.getpixel((0, 0)) == (9, 9, 9)


def test_render_is_not_mirrored():
    """Rows are not mirror images in real data; the renderer must keep
    the top and bottom halves as they are."""
    pytest.importorskip("PIL")
    from sidecaramel.overview import overview_image
    img = overview_image(_blob_with({(3, 2): 36, (3, 13): 6}))
    assert img.getpixel((3, 2)) == (51, 0, 0)
    assert img.getpixel((3, 13)) == (0, 51, 0)


def test_render_overview_writes_the_named_format(tmp_path):
    pytest.importorskip("PIL")
    from sidecaramel.overview import render_overview
    blob = _fixture("mix_60Hz_10kHz")
    png, bmp = tmp_path / "o.png", tmp_path / "o.bmp"
    assert render_overview(blob, str(png), scale=2)
    assert render_overview(blob, str(bmp), scale=1)
    assert png.read_bytes()[:4] == b"\x89PNG"
    assert bmp.read_bytes()[:2] == b"BM"
    from PIL import Image
    assert Image.open(png).size == (480, 32)
    assert not render_overview(b"\x01\x05\x01", str(tmp_path / "short.png"))


# ── encoder ──────────────────────────────────────────────────────────

def _sines(freqs, seconds=30.0, sr=22050, amp=0.7):
    # 30 s gives ~125 ms columns. Much shorter audio makes each column
    # so short that 60 Hz leaks into the mid band of the per-column FFT.
    np = pytest.importorskip("numpy")
    t = np.arange(int(seconds * sr)) / sr
    return sum(amp * np.sin(2 * np.pi * f * t) for f in freqs).astype(
        "float32"), sr


@pytest.mark.parametrize("freqs, name", [([60, 10000], "mix_60Hz_10kHz"),
                                         ([60, 1000], "mix_60Hz_1kHz"),
                                         ([1000, 10000], "mix_1kHz_10kHz")])
def test_encoder_mix_matches_serato_digits(freqs, name):
    """Per-digit mixing: the encoder reaches the same peak digits as
    Serato for the same two-tone signal, with the third digit empty."""
    pytest.importorskip("numpy")
    from sidecaramel.overview_encode import build_overview_blob
    samples, sr = _sines(freqs)
    assert _max_digits(build_overview_blob(samples, sr)) == \
        _max_digits(_fixture(name))


def test_encoder_emits_only_cube_bytes_and_223():
    pytest.importorskip("numpy")
    from sidecaramel.overview_encode import build_overview_blob
    for freqs in ([60], [1000], [10000], [60, 1000, 10000]):
        samples, sr = _sines(freqs)
        body = blob_payload(build_overview_blob(samples, sr))
        assert {v for v in body if v >= CUBE_MAX} <= {223}, freqs


def test_encoder_silence_matches_serato_pattern():
    np = pytest.importorskip("numpy")
    from sidecaramel.overview_encode import build_overview_blob
    body = blob_payload(build_overview_blob(np.zeros(22050, "float32"),
                                            22050))
    assert [body[c * 16 + 8] for c in range(4)] == [223, 43, 223, 43]
    assert all(body[c * 16 + 7] == 1 for c in range(240))


# ── inventory decoder ────────────────────────────────────────────────

def test_decode_overview_does_not_eat_a_zero_background():
    from sidecaramel.blobs import decode_overview
    d = decode_overview(_fixture("log_sweep_background_0"))
    assert (d["rows"], d["cols"], d["trailing_bytes"]) == (16, 240, 0)
    assert d["first_column"] == list(blob_payload(
        _fixture("log_sweep_background_0"))[:16])
    assert d["background_0"] > 0 and d["flagged_cells"] > 0
