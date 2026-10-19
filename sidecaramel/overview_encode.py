"""sidecaramel.overview_encode — audio → Overview-blob encoder.

Round-trip companion to `sidecaramel.overview` (decoder/renderer).
Takes a WAV / FLAC / MP3 path and produces a 3842-byte Overview blob
that the audio-tag writers embed in the container's metadata block.

Blob layout:

    [0:2]    header: 0x01 0x05
    [2:..]   240 chunks × 16 bytes each (one chunk = one time-bin
             vertical slice)

    Within a chunk:
        offsets 0, 1, 14, 15  → always byte 1 (silence padding)
        offsets 2..13          → active waveform area (12 rows)
        offsets 7, 8           → centerline pair

Silence (column below the audible threshold): byte 1 everywhere,
offset 8 alternating 223 / 43 across columns, offset 7 = 1.  Low
amplitude: offsets 7/8 alternate between (43, 223) and (1, 43) across
columns, as Serato writes at the start of a fade-in.  Higher
amplitude OVERWRITES the marker with audio bytes.

Byte encoding for audio cells: a 6x6x6 colour cube (see
`sidecaramel.overview_palette`), one digit per band —

    v = 36*a + 6*b + c      a = bass, b = mid, c = treble, each 0..5

Each band is measured as a tier 0..7 and mapped to its own digit; the
three digits are set independently.  Calibration tones show Serato
doing the same: 60 Hz + 10 kHz gives (3, 0, 3), 60 Hz + 1 kHz gives
(3, 3, 0), with no spill into the third digit.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# ---- Layout constants (empirical) ----------------------------------

HEADER = bytes([0x01, 0x05])
N_COLS = 240
ROWS_PER_COL = 16

# Padding rows — ALWAYS byte 1
PADDING_ROWS = (0, 1, 14, 15)
# Active waveform rendering area
ACTIVE_ROWS = tuple(range(2, 14))    # 12 rows: 2..13 inclusive
# Centerline pair — special handling for silence/low-amp
CENTERLINE_UPPER = 7
CENTERLINE_LOWER = 8

# Universal marker bytes (constant across tracks)
SILENCE_BYTE          = 1     # background: padding + below-threshold audio
CENTERLINE_UPPER_BYTE = 43    # cube (1, 1, 1)
CENTERLINE_LOWER_BYTE = 223   # flag + cube (0, 1, 1)

# Threshold below which a column gets only the silence-centerline
# dot at offset 8 alternating across columns (the silence pattern).
# Above this but below the first audio tier the centreline pair
# alternates between (43, 223) and (1, 43).
SILENCE_RMS_THRESHOLD = 1e-5
LOW_AMP_RMS_THRESHOLD = 1e-3


# ---- Band-energy extraction ----------------------------------------

# Band edges in Hz (matches sidecaramel.audio_render).
BASS_RANGE   = (20.0, 200.0)
MID_RANGE    = (200.0, 2000.0)
TREBLE_RANGE = (2000.0, 20000.0)


def _band_energies(samples: np.ndarray, sr: float
                     ) -> Tuple[float, float, float]:
    """Return (bass, mid, treble) PEAK FFT magnitude per band.

    Peak-bin (not sum) gives consistent scale across narrow-band
    pure tones and broadband real music: a pure 60Hz sine produces
    1 huge bin in the bass range; a broadband mix produces several
    smaller bins.  Per-bin max stays comparable.

    Sum-based scoring made real music score ALL bands as ~equal
    shares because broadband energy spreads across many bins.
    """
    if len(samples) < 16:
        return (0.0, 0.0, 0.0)
    win = samples * np.hanning(len(samples))
    spec = np.abs(np.fft.rfft(win))
    # Normalize per-bin magnitude by FFT length so values stay
    # roughly comparable across column-windows of different sizes.
    spec = spec * (2.0 / len(win))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / sr)

    def band_peak(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            return 0.0
        return float(spec[mask].max())

    return (band_peak(*BASS_RANGE),
            band_peak(*MID_RANGE),
            band_peak(*TREBLE_RANGE))


# Anchor: peak-bin magnitude of a 0.7-amp pure sine in its own band
# after the 2/N normalisation above (peak bin = 0.7 × Hann window
# correction).
BAND_FULLSCALE_REF = 0.35


# ---- Band tiers ----------------------------------------------------
# Each band's tier 0..7 becomes that band's cube digit via
# `sidecaramel.overview_palette.BAND_TIER_DIGIT`.
#
# Per-band tier-reference values: each channel uses a different
# amp-anchor so that equal audio amplitude produces lower display
# tiers for higher-frequency bands.  Tier-4 reference per band:
#   bass:   0.34
#   mid:    0.68
#   treble: 1.24
BAND_TIER4_REF = {
    "bass":   0.34,
    "mid":    0.68,
    "treble": 1.24,
}


def _amp_to_tier(linear_intensity: float,
                    band: str = "bass") -> Optional[int]:
    """Map a band's peak-bin magnitude to its discrete tier 0..7.

    Returns None if below noise floor (= channel not present).
    """
    ref = BAND_TIER4_REF.get(band, 0.085)
    # Linear tier mapping: amp=ref → tier 4, amp=1.75×ref → tier 7
    # Sub-noise threshold = ref / 8 = tier 0 boundary
    if linear_intensity < ref / 16:
        return None
    tier = int(round(linear_intensity / ref * 4))
    return max(0, min(7, tier))


# ---- Encoder core --------------------------------------------------

def _column_chunk(samples: np.ndarray, sr: float,
                    col_idx: int) -> bytes:
    """Build the 16-byte chunk for one time-bin column."""
    chunk = bytearray([SILENCE_BYTE] * ROWS_PER_COL)
    if len(samples) == 0:
        # Silence-special: alternate 223/43 at offset 8, offset 7 = 1
        if col_idx % 2 == 0:
            chunk[CENTERLINE_LOWER] = CENTERLINE_LOWER_BYTE
        else:
            chunk[CENTERLINE_LOWER] = CENTERLINE_UPPER_BYTE
        return bytes(chunk)

    # Per-col RMS (overall loudness)
    rms = float(np.sqrt(np.mean(samples * samples)))

    # True silence
    if rms < SILENCE_RMS_THRESHOLD:
        if col_idx % 2 == 0:
            chunk[CENTERLINE_LOWER] = CENTERLINE_LOWER_BYTE
        else:
            chunk[CENTERLINE_LOWER] = CENTERLINE_UPPER_BYTE
        return bytes(chunk)

    # Low-amp: centreline pair alternates (43, 223) / (1, 43) across
    # columns
    if rms < LOW_AMP_RMS_THRESHOLD:
        if col_idx % 2 == 0:
            chunk[CENTERLINE_UPPER] = CENTERLINE_UPPER_BYTE
            chunk[CENTERLINE_LOWER] = CENTERLINE_LOWER_BYTE
        else:
            chunk[CENTERLINE_LOWER] = CENTERLINE_UPPER_BYTE
        return bytes(chunk)

    # Audio rendering — each band → tier via _amp_to_tier → its own
    # cube digit via sidecaramel.overview_palette.mix_band_tiers.
    from sidecaramel.overview_palette import mix_band_tiers

    bass, mid, treble = _band_energies(samples, sr)
    # Per-band tier lookup with per-band reference
    bass_tier   = _amp_to_tier(bass,   "bass")
    mid_tier    = _amp_to_tier(mid,    "mid")
    treble_tier = _amp_to_tier(treble, "treble")

    audio_byte = mix_band_tiers(bass_tier=bass_tier,
                                   mid_tier=mid_tier,
                                   treble_tier=treble_tier)
    if audio_byte is None:
        # Silence-style centerline (alternating)
        if col_idx % 2 == 0:
            chunk[CENTERLINE_LOWER] = CENTERLINE_LOWER_BYTE
        else:
            chunk[CENTERLINE_LOWER] = CENTERLINE_UPPER_BYTE
        return bytes(chunk)

    # Amplitude tier (overall): maps RMS → row-count from centerline.
    import math
    log_amp = math.log10(max(rms, 1e-6))
    amp_tier = int(round(6 * (log_amp + 2.0) / 1.7))
    amp_tier = max(0, min(6, amp_tier))

    # "Edge" byte for col-alternation at centerline-upper: one tier
    # lower per channel.  Cmcal pure-tone calibrated:
    #   1kHz base=18 (0,3,0) edge=12 (0,2,0)
    #   10kHz base=3 (0,0,3) edge=2 (0,0,2)
    edge_byte = mix_band_tiers(
        bass_tier=  bass_tier - 1 if bass_tier and bass_tier > 0 else None,
        mid_tier=   mid_tier - 1 if mid_tier and mid_tier > 0 else None,
        treble_tier=treble_tier - 1
                    if treble_tier and treble_tier > 0 else None)
    if edge_byte is None:
        edge_byte = audio_byte

    # Fill rows symmetrically from centerline outward
    for dist in range(amp_tier + 1):
        ru = CENTERLINE_UPPER - dist
        rl = CENTERLINE_LOWER + dist
        if ru in ACTIVE_ROWS:
            chunk[ru] = audio_byte
        if rl in ACTIVE_ROWS:
            chunk[rl] = audio_byte

    # Row-7 col-alternation (texture pattern)
    if col_idx % 2 == 1 and chunk[CENTERLINE_UPPER] != SILENCE_BYTE:
        chunk[CENTERLINE_UPPER] = edge_byte
    return bytes(chunk)


def build_overview_blob(samples: np.ndarray, sr: float
                          ) -> bytes:
    """Build a 3842-byte Serato Overview blob from raw audio
    samples (mono, float32 in -1..1).

    Length is fixed at 3842 bytes regardless of audio duration —
    Serato resamples to 240 time-bins for the thumbnail.

    Args:
        samples: 1-D numpy array of float audio samples in [-1, 1].
                  Stereo input should be downmixed to mono BEFORE
                  calling (Serato MERGES L+R for the Overview —
                  see the format docs).
        sr:      Sample rate in Hz.

    Returns:
        Exactly 3842 bytes: `0x01 0x05` header + 240 × 16 cell
        bytes.
    """
    if samples.ndim != 1:
        # Downmix stereo to mono
        samples = samples.mean(axis=-1)
    samples = np.asarray(samples, dtype=np.float32)

    n_total = len(samples)
    blob = bytearray()
    blob.extend(HEADER)
    if n_total == 0:
        # Pure silence — emit silence chunks
        for col_idx in range(N_COLS):
            blob.extend(_column_chunk(np.zeros(0, dtype=np.float32),
                                        sr, col_idx))
        return bytes(blob)

    # Slice audio into 240 equal-duration columns
    edges = np.linspace(0, n_total, N_COLS + 1, dtype=int)
    for col_idx in range(N_COLS):
        a, b = edges[col_idx], edges[col_idx + 1]
        col_samples = samples[a:b]
        blob.extend(_column_chunk(col_samples, sr, col_idx))

    assert len(blob) == 3842, (
        f"blob length {len(blob)} != 3842")
    return bytes(blob)


# ---- File-path convenience wrapper --------------------------------

def build_overview_blob_for_path(audio_path: str) -> bytes:
    """Decode an audio file, downmix to mono, build Overview blob.

    Tries soundfile first (handles WAV/FLAC natively), falls back
    to ffmpeg via sidecaramel.audio_render._decode_audio for
    formats that need it.
    """
    try:
        import soundfile as sf
        samples, sr = sf.read(audio_path, always_2d=False)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        samples = np.asarray(samples, dtype=np.float32)
    except Exception:
        # Fall back to the ffmpeg-based decoder
        from sidecaramel.audio_render import _decode_audio
        samples, sr = _decode_audio(audio_path, target_sr=22050,
                                       stereo=False)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
    return build_overview_blob(samples, float(sr))


# ---- Blob comparison utility (encoder iteration) -------------------

def diff_blobs(ours: bytes, ref: bytes) -> dict:
    """Per-byte / per-column / per-channel diff report.

    Returns dict with: bytes_equal (bool), n_diff (int), per_col
    diff-counts, per-byte-value frequency of mismatches, header_eq.
    """
    if len(ours) != len(ref):
        return {"length_mismatch": (len(ours), len(ref))}
    if ours == ref:
        return {"bytes_equal": True, "n_diff": 0}
    n_diff = sum(1 for a, b in zip(ours, ref) if a != b)
    header_eq = ours[:2] == ref[:2]
    # Per-column diff count
    col_diff = []
    for c in range(N_COLS):
        start = 2 + c * ROWS_PER_COL
        end   = start + ROWS_PER_COL
        d = sum(1 for a, b in zip(ours[start:end], ref[start:end])
                  if a != b)
        col_diff.append(d)
    # Most-common (ours, ref) byte mismatches
    from collections import Counter
    mismatch_pairs = Counter(
        (a, b) for a, b in zip(ours, ref) if a != b)
    return {
        "bytes_equal":      False,
        "n_diff":           n_diff,
        "total_bytes":      len(ours),
        "diff_pct":         round(100 * n_diff / len(ours), 2),
        "header_eq":        header_eq,
        "n_cols_with_diff": sum(1 for d in col_diff if d > 0),
        "max_col_diff":     max(col_diff),
        "mean_col_diff":    round(sum(col_diff) / len(col_diff), 2),
        "top_mismatches":   [
            {"ours": a, "ref": b, "count": n}
            for (a, b), n in mismatch_pairs.most_common(10)
        ],
    }
