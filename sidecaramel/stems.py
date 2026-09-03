"""sidecaramel.stems — Serato-Stems sidecar decoder.

Decodes a `.serato-stems` sidecar into all four stem types and
produces individual WAV files.  Typical consumers: offline
pre-bake tools or on-demand pipeline stages that need clean stems
without having to re-run Demucs.

The format is reverse-engineered from Serato 3.x output. Each
`stem` chunk in the file carries one of four stem types:

  Type 0 → VOCALS    (the singing track — used for STT, phoneme CTC)
  Type 1 → BASS      (low-end — used for energy-drop section
                      boundaries)
  Type 2 → DRUMS     (percussion — used for BPM-grid validation
                      via beat onset detection)
  Type 3 → HARMONY   (melodic/chordal content — used for section
                      detection via harmonic onset/offset)

Each payload is XOR-encoded MP3 (key 0x26). We decode → write to
a temp .mp3 → ffmpeg-transcode to mono 16kHz WAV.

WAV outputs are named `<basename>.serato-{type_name}.wav` and live
next to the original audio file. Sidecar-only invariant: NEVER
modify the .serato-stems file or the audio file itself.
"""
from __future__ import annotations

import os
import re
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

# ============================================================
# CONSTANTS
# ============================================================

SERATO_XOR_KEY = 0x26

# Output format of extract_all_stems(). Single point of truth: the
# ffmpeg command AND the length check derive from these, so the two
# can never drift apart.
STEM_WAV_SR = 16000
STEM_WAV_CHANNELS = 1

# Type-to-name mapping. Names map 1:1 to the WAV filename suffix
# (`<base>.serato-{name}.wav`) and to the kwarg key in
# extract_all_stems' return dict.
#
# Mapping verified empirically by extracting all four stems from a
# reference sidecar and identifying each by ear and by spectrum
# against the same track in Serato. The order matters: an intuitive
# guess of 1=harmony, 2=bass, 3=drums is wrong in all three slots —
# slot 1 is the bass, slot 2 the drums, slot 3 the harmony/melody.
STEM_TYPE_NAMES: Dict[int, str] = {
    0: "vocals",
    1: "bass",
    2: "drums",
    3: "harmony",
}
STEM_NAME_TO_TYPE: Dict[str, int] = {v: k for k, v in STEM_TYPE_NAMES.items()}

# Suffixes for build-artifact stem WAVs produced next to the audio
# by extract_all_stems().  These are persistent caches by design
# (avoid re-extracting on every pipeline run); cleanup_stem_artifacts
# removes them when the caller is done with them.
SERATO_STEM_WAV_SUFFIXES = tuple(
    f".serato-{name}.wav" for name in STEM_TYPE_NAMES.values()
)

# Optional extra sidecar suffixes that downstream pipelines may use
# for their own quick-vocals caches.  Callers can append their own
# suffixes via add_external_stem_suffix() if their pipeline writes
# alongside `.serato-*.wav`.
EXTERNAL_STEM_WAV_SUFFIXES: tuple = tuple()

# Whitelist of suffixes we're allowed to WRITE. Used by the
# write-safety check; never modify .serato-stems or bare audio files.
ALLOWED_STEM_WAV_SUFFIXES = SERATO_STEM_WAV_SUFFIXES

# Whitelist of suffixes we're allowed to DELETE during cleanup.
CLEANUP_STEM_WAV_SUFFIXES = (
    SERATO_STEM_WAV_SUFFIXES + EXTERNAL_STEM_WAV_SUFFIXES
)


def add_external_stem_suffix(suffix: str) -> None:
    """Register an external pipeline's stem-cache suffix so it gets
    swept up by cleanup_stem_artifacts.  Suffix must start with `.`
    (e.g. ".myproj-vocals.wav")."""
    global EXTERNAL_STEM_WAV_SUFFIXES, CLEANUP_STEM_WAV_SUFFIXES
    if not suffix.startswith("."):
        raise ValueError("suffix must start with '.'")
    if suffix in EXTERNAL_STEM_WAV_SUFFIXES:
        return
    EXTERNAL_STEM_WAV_SUFFIXES = EXTERNAL_STEM_WAV_SUFFIXES + (suffix,)
    CLEANUP_STEM_WAV_SUFFIXES = (
        SERATO_STEM_WAV_SUFFIXES + EXTERNAL_STEM_WAV_SUFFIXES
    )


# ============================================================
# HEADER + CHUNK PARSING
# ============================================================

def parse_serato_stems_header(data: bytes) -> Optional[dict]:
    """Decode the `srtshead` magic-prefixed header at file start.

    Returns dict with header_length, version, n_stems, total_samples,
    sample_rate, duration_sec. None on malformed input.

    Layout (big-endian, byte offsets):
      [0:8]   magic "srtshead"
      [8:12]  header_length (always 16 in known versions)
      [12:14] version_major
      [14:16] version_minor
      [16:20] n_stems (1-16)
      [20:24] total_samples
      [24:28] sample_rate
    """
    if len(data) < 28 or data[:8] != b"srtshead":
        return None
    try:
        hdr_len = struct.unpack(">I", data[8:12])[0]
        vmaj, vmin = struct.unpack(">HH", data[12:16])
        n_stems = struct.unpack(">I", data[16:20])[0]
        total_samples = struct.unpack(">I", data[20:24])[0]
        sr = struct.unpack(">I", data[24:28])[0]
    except Exception:
        return None
    if hdr_len != 16 or n_stems == 0 or n_stems > 16 or sr == 0:
        return None
    return {
        "header_length": hdr_len,
        # Chunks start at header_length + 12 (magic "srtshead" is 8
        # bytes plus the 4-byte length field are NOT counted in
        # header_length). Exposed so callers stop hand-adding the 12.
        "body_offset": hdr_len + 12,
        "version": (vmaj, vmin),
        "n_stems": n_stems,
        "total_samples": total_samples,
        "sample_rate": sr,
        "duration_sec": total_samples / sr if sr else 0.0,
    }


def iter_serato_stem_chunks(data: bytes,
                             start_offset: int
                             ) -> Iterator[Tuple[int, bytes]]:
    """Walk chunk-by-chunk through the file body.

    Each chunk has a 12-byte header:
      [0:4]   magic "stem"
      [4:8]   length (stem_type + payload, i.e. 4 + len(payload))
      [8:12]  stem_type (0-3) — counts towards `length`

    The XOR-encoded MP3 payload starts at +12, directly after the
    stem_type. There is NO reserved/padding field.

    2026-08-13: verified against a real Serato 3.x sidecar — the four
    bytes at +12 XOR-decode to `ff fb 90 64` (MPEG-1 Layer III sync,
    128 kbps, 44100 Hz) in all four chunks, and that first frame is the
    LAME `Xing` info frame. Reading the payload from +16 instead — i.e.
    treating +12..+16 as a reserved field — cuts that frame's header
    off: a decoder then resyncs on frame 2 and the extracted stem
    silently loses 1152 samples (26.1 ms) at the front.

    Yields (stem_type, encoded_payload). Payload is still
    XOR-encoded; caller must decode via decode_xor_payload.
    """
    offset = start_offset
    while offset + 12 <= len(data):
        if data[offset:offset+4] != b"stem":
            break
        length = struct.unpack(">I", data[offset+4:offset+8])[0]
        stem_type = struct.unpack(">I", data[offset+8:offset+12])[0]
        payload_bytes = length - 4
        payload_start = offset + 12
        payload_end = payload_start + payload_bytes
        if payload_end > len(data):
            break
        yield stem_type, data[payload_start:payload_end]
        offset = payload_end


def build_serato_stems(header: dict,
                       chunks: List[Tuple[int, bytes]]) -> bytes:
    """Serialise a sidecar back to bytes — the inverse of
    `parse_serato_stems_header` + `iter_serato_stem_chunks`.

    `chunks` are (stem_type, encoded_payload) exactly as the iterator
    yields them, i.e. still XOR-encoded.

    Returns bytes and touches no file. The sidecar-only invariant
    holds: nothing in this module writes a `.serato-stems` path. It
    exists so the parser can be proven lossless — parse a real sidecar,
    rebuild it, compare byte-for-byte. If a single reserved field were
    mis-modelled or a byte unaccounted for, that comparison fails.
    """
    out = bytearray(b"srtshead")
    out += struct.pack(">I", header["header_length"])
    out += struct.pack(">HH", *header["version"])
    out += struct.pack(">I", header["n_stems"])
    out += struct.pack(">I", header["total_samples"])
    out += struct.pack(">I", header["sample_rate"])
    for stem_type, payload in chunks:
        out += b"stem"
        out += struct.pack(">I", 4 + len(payload))
        out += struct.pack(">I", stem_type)
        out += payload
    return bytes(out)


def decode_xor_payload(payload: bytes) -> bytes:
    """XOR-decode a stem payload. Returns the underlying MP3 bytes."""
    return bytes(b ^ SERATO_XOR_KEY for b in payload)


def looks_like_mp3(decoded: bytes) -> bool:
    """Sanity-check a decoded payload — true only when the stream
    STARTS with an ID3 prefix or an MPEG sync word.

    Strict on purpose. A stem payload begins exactly at the first frame
    header, so a sync word found only further in means the payload start
    is misaligned — not that the stream is fine. A guard that scans the
    first 512 bytes for any sync word cannot do this job: it returns
    True for a payload whose first frame has been truncated, which is
    precisely the failure this check exists to catch.
    """
    if len(decoded) < 4:
        return False
    if decoded[:3] == b"ID3":
        return True
    return decoded[0] == 0xFF and (decoded[1] & 0xE0) == 0xE0


# ============================================================
# OUTPUT PATH RESOLUTION
# ============================================================

def _legacy_in_place_stem_wav_path(audio_path: str | Path,
                                    stem_name: str) -> Path:
    """The in-place location next to the audio file.

    This is the default write target. It stays a read-only fallback
    even when a consumer redirects writes elsewhere via
    `set_stem_path_resolver`, so stems written before the redirect
    still resolve — see `find_stem_wav`.
    """
    p = Path(audio_path)
    stem = p.stem
    m = re.match(r"^(.+?)(\.\d+(?:\.\d+)*)$", stem)
    if m:
        stem = m.group(1)
    return p.parent / f"{stem}.serato-{stem_name}.wav"


def stem_wav_path(audio_path: str | Path,
                  stem_name: str) -> Path:
    """Canonical WAV output path for a given audio file and stem.

    Default: places the WAV next to the audio file as
    `<base>.serato-<stem>.wav`.

    Consumers that want to redirect stems into a shadow tree (so the
    user's audio folder stays clean) can override this resolver via
    `set_stem_path_resolver()` — see `find_stem_wav()` for the
    dual-lookup pattern that supports both layouts side-by-side.
    """
    resolver = _STEM_PATH_RESOLVER
    if resolver is not None:
        try:
            return Path(resolver(str(audio_path), stem_name))
        except Exception:
            pass
    return _legacy_in_place_stem_wav_path(audio_path, stem_name)


# Optional consumer-provided resolver: (audio_path, stem_name) → path.
# Set via set_stem_path_resolver() to redirect stems off the audio
# folder.  When None, stems land in-place next to the audio.
_STEM_PATH_RESOLVER: Optional[Callable[[str, str], str]] = None


def set_stem_path_resolver(
        resolver: Optional[Callable[[str, str], str]]) -> None:
    """Install a consumer-provided stem-path resolver.  Pass None to
    revert to the default in-place behaviour."""
    global _STEM_PATH_RESOLVER
    _STEM_PATH_RESOLVER = resolver


def _spleeter_in_place_stem_wav_path(audio_path: str | Path,
                                       stem_name: str) -> Path:
    """Alternate provenance: `<base>.spleeter-<stem>.wav` next to
    the audio file.  An external batch job populates this layout
    for cheap stems that are NOT from Serato — they must not be
    named as Serato vocals when they are not Serato vocals.
    """
    p = Path(audio_path)
    stem = p.stem
    m = re.match(r"^(.+?)(\.\d+(?:\.\d+)*)$", stem)
    if m:
        stem = m.group(1)
    return p.parent / f"{stem}.spleeter-{stem_name}.wav"


def find_stem_wav(audio_path: str | Path,
                  stem_name: str) -> Optional[Path]:
    """Locate an existing stem WAV.  Checks (in order):

      1. Consumer-overridden location via stem_wav_path()
      2. Legacy in-place `<base>.serato-<stem>.wav`
      3. Spleeter alternate `<base>.spleeter-<stem>.wav`

    Returns the first existing path, or None.
    """
    primary = stem_wav_path(audio_path, stem_name)
    if primary.is_file():
        return primary
    legacy = _legacy_in_place_stem_wav_path(audio_path, stem_name)
    if legacy.is_file() and legacy != primary:
        return legacy
    spleeter = _spleeter_in_place_stem_wav_path(audio_path, stem_name)
    if (spleeter.is_file() and spleeter != primary
            and spleeter != legacy):
        return spleeter
    return None


def read_lame_gapless(sidecar_path: str | Path
                      ) -> Optional[Tuple[int, int]]:
    """`(encoder_delay, end_padding)` in SOURCE-rate samples, read from
    the LAME `Xing`/`Info` frame of the first stem chunk.

    Serato encodes each stem with LAME and keeps the info frame, so the
    gapless values are carried in the sidecar itself — no guessing, no
    per-track residual. Returns None when the first chunk has no info
    frame (then there is nothing to correct from this source).

    Verified 2026-08-13 against a Serato 3.x sidecar: delay=576,
    padding=576, and
        audio_frames * 1152 - delay - padding == header total_samples
    exactly, for all four stems.
    """
    try:
        with open(sidecar_path, "rb") as fh:
            head = fh.read(28)
            if len(head) < 28 or head[0:8] != b"srtshead":
                return None
            if fh.read(4) != b"stem":
                return None
            fh.read(8)                       # length + stem_type
            body = bytes(b ^ SERATO_XOR_KEY for b in fh.read(4096))
    except OSError:
        return None
    if len(body) < 40 or body[0] != 0xFF or (body[1] & 0xE0) != 0xE0:
        return None
    # MPEG-1 side info: 17 bytes mono, 32 bytes otherwise.
    side = 17 if ((body[3] >> 6) & 3) == 3 else 32
    if body[4 + side:8 + side] not in (b"Xing", b"Info"):
        return None
    pos = 8 + side
    flags = struct.unpack(">I", body[pos:pos + 4])[0]
    pos += 4
    for bit, size in ((1, 4), (2, 4), (4, 100), (8, 4)):
        if flags & bit:
            pos += size
    d = body[pos + 21:pos + 24]              # LAME tag: delay/padding
    if len(d) < 3:
        return None
    delay = (d[0] << 4) | (d[1] >> 4)
    padding = ((d[1] & 0x0F) << 8) | d[2]
    return delay, padding


def header_derived_drift_samples(sidecar_path: str | Path,
                                    target_sr: int = 16000) -> int:
    """Stem-vs-mix sample drift for a `.serato-stems` sidecar.

    Returns a NEGATIVE integer when the stem leads the mix. To align a
    same-SR-loaded stem with the mix:

        drift = header_derived_drift_samples(sidecar)   # e.g. -209
        aligned_stem = stem_array[-drift:]              # drop the delay

    The value is the LAME **encoder delay** (`read_lame_gapless`),
    rescaled to `target_sr`. It is exact, not an estimate.

    IMPORTANT — do not apply this on top of a gapless decode. ffmpeg
    reads the same `Xing`/LAME frame and trims delay and padding by
    itself, so a stem extracted through `extract_all_stems()` is
    ALREADY aligned and needs no correction. This function is for
    callers that decode the payload without gapless support.

    Note the value is NOT one whole MPEG frame. Deriving it from the
    sidecar header alone (1152 samples at source rate) and attributing
    it to a first-frame strip in Serato's writer gets both halves
    wrong: a missing frame there is a symptom of a mis-set payload
    boundary (see `iter_serato_stem_chunks`), and the real delay is
    576 samples on the material observed, carried in the LAME tag.

    Returns 0 when the sidecar is unreadable or carries no info frame.
    """
    try:
        head = Path(sidecar_path).read_bytes()[:28]
    except OSError:
        return 0
    if len(head) < 28 or head[0:8] != b"srtshead":
        return 0
    source_sr = struct.unpack(">I", head[24:28])[0]
    if source_sr <= 0:
        return 0
    gapless = read_lame_gapless(sidecar_path)
    if gapless is None:
        return 0
    delay, _padding = gapless
    return -round(delay * target_sr / source_sr)


def assert_safe_stem_write(out_path: Path,
                           audio_path: Optional[Path] = None) -> None:
    """Defence against ever overwriting a non-sidecar file. Raises
    RuntimeError when the target path doesn't end in one of the
    allowed `serato-{type}.wav` suffixes, or when it equals the
    original audio path."""
    full_lower = str(out_path.resolve()).lower()
    if not any(full_lower.endswith(s) for s in ALLOWED_STEM_WAV_SUFFIXES):
        raise RuntimeError(
            f"REFUSING TO WRITE non-stem-suffix path: {out_path}. "
            f"Allowed suffixes: {ALLOWED_STEM_WAV_SUFFIXES}"
        )
    if full_lower.endswith(".serato-stems"):
        raise RuntimeError(
            f"REFUSING TO WRITE to .serato-stems file: {out_path}"
        )
    if audio_path is not None:
        if out_path.resolve() == Path(audio_path).resolve():
            raise RuntimeError(
                f"REFUSING TO WRITE: target equals audio file: "
                f"{out_path}"
            )


def expected_stem_frames(header: dict, target_sr: int) -> int:
    """Sample count a correctly decoded stem must have at `target_sr`.

    The sidecar header's `total_samples` is the gapless length at the
    source rate — encoder delay and end padding already excluded. A
    decode that honours the LAME `Xing` frame lands on exactly this
    number; measured against a Serato 3.x sidecar at 16 k, 22.05 k,
    44.1 k and 48 kHz, the deviation was 0 samples at every rate.
    """
    src = header.get("sample_rate") or 0
    if src <= 0:
        return 0
    return round(header.get("total_samples", 0) * target_sr / src)


def wav_frame_count(path: str | Path) -> Optional[int]:
    """Frame count of a PCM WAV, or None when it cannot be read."""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes()
    except Exception:
        return None


# ============================================================
# EXTRACTION
# ============================================================

def cleanup_stem_artifacts(audio_path: str | Path,
                            include_serato: bool = True,
                            include_external: bool = True,
                            log: Callable[[str], None] = print,
                            ) -> List[Path]:
    """Delete persistent stem build-artefacts next to `audio_path`.

    Stems are extracted as a CACHE — they speed up repeat consumer
    runs but they're not deliverables.  Once downstream work has
    converged, the cached WAVs are dead weight.  This function
    removes them safely.

    Two families are cleaned:
      include_serato     — <base>.serato-{vocals,harmony,bass,
                           drums}.wav (this module, ~34 MB total)
      include_external   — any suffixes registered via
                           add_external_stem_suffix() (e.g. an
                           upstream pipeline's quick-vocals cache).

    NOT cleaned:
      .serato-stems      — source-of-truth sidecar from Serato
                           DJ; we don't own this format
      audio file itself  — never touched (audio-write policy)

    Safety: every candidate path is filename-checked against
    CLEANUP_STEM_WAV_SUFFIXES before unlink.  A path that doesn't
    match (e.g. someone passed an unrelated audio_path that resolves
    near a foreign file with a similar name) is logged and skipped.

    Returns the list of paths that were actually deleted (post-
    success), so callers can report accurate cleanup metrics.
    """
    audio = Path(audio_path)
    base = audio.with_suffix("")  # strip extension
    # Strip Serato's `.N.M` version suffix from the base too — same
    # logic as stem_wav_path. Otherwise "Track.1.2.mp3" wouldn't
    # find "Track.serato-vocals.wav".
    base_stem = base.name
    m = re.match(r"^(.+?)(\.\d+(?:\.\d+)*)$", base_stem)
    if m:
        base_stem = m.group(1)
    base = base.parent / base_stem

    candidates: List[Path] = []
    if include_serato:
        for suffix in SERATO_STEM_WAV_SUFFIXES:
            candidates.append(Path(str(base) + suffix))
    if include_external:
        for suffix in EXTERNAL_STEM_WAV_SUFFIXES:
            candidates.append(Path(str(base) + suffix))

    deleted: List[Path] = []
    for path in candidates:
        if not path.exists():
            continue
        # Defence: never delete what isn't whitelisted, even if our
        # own filename construction is wrong.
        name_lower = path.name.lower()
        if not any(name_lower.endswith(s)
                   for s in CLEANUP_STEM_WAV_SUFFIXES):
            log(f"  [stems] REFUSING to delete non-stem path: {path}")
            continue
        try:
            path.unlink()
            deleted.append(path)
            log(f"  [stems] cleaned {path.name}")
        except OSError as e:
            log(f"  [stems] could not delete {path.name}: {e}")
    return deleted


def get_cached_stems(audio_path: str | Path) -> Dict[str, Path]:
    """Return dict of stem-name → path for every stem WAV that
    already exists on disk for this audio file. Empty dict if none.

    Used by the pipeline's fast-only Stage 0 visit: don't trigger
    extraction (which is slow on cache miss), just report what's
    already cached. The slow extraction runs later in the same
    pipeline run via a non-fast Stage 0 visit.
    """
    audio_path = Path(audio_path)
    base = audio_path.with_suffix("")  # strip extension
    out: Dict[str, Path] = {}
    for name in STEM_TYPE_NAMES.values():
        candidate = base.parent / f"{base.name}.serato-{name}.wav"
        if candidate.exists():
            out[name] = candidate
    return out


def find_sidecar(audio_path: str | Path) -> Optional[Path]:
    """Locate the .serato-stems file for a given audio file.

    Serato writes the sidecar with a version suffix that may not
    match the audio's exact filename. We look in:
      1. The audio file's own parent directory
      2. A few well-known Serato-managed folders (Imported, Auto
         Import, Latest Import, Stems)
    The sidecar matching uses `<base_stem>*.serato-stems` so a file
    `Track.1.2.serato-stems` matches `Track.flac`.

    2026-05-14:
      • extended search to handle the common case where the audio
        file lives in an arbitrary music folder but Serato stored
        the sidecar under `_Serato_/Imported/Latest Import/`.
      • NFC-normalise both audio + sidecar stems before equality
        check. macOS HFS+ filenames decompose accented characters
        (NFD: `ö`) while APFS preserves precomposed (NFC:
        `ö`).  A Serato sidecar written on an HFS+ era volume
        will not byte-match an NFC audio path on a newer volume
        unless we normalise.
    """
    import unicodedata as _ud
    audio = Path(audio_path)
    if not audio.exists():
        return None
    base_stem = _ud.normalize("NFC", audio.stem)
    # Strip trailing version suffix from audio's stem too, so
    # "Track.1.2.mp3" → "Track" matches "Track.serato-stems"
    m = re.match(r"^(.+?)(\.\d+(?:\.\d+)*)$", base_stem)
    if m:
        base_stem = m.group(1)

    def _scan(folder: Path) -> Optional[Path]:
        if not folder.is_dir():
            return None
        try:
            entries = list(folder.iterdir())
        except OSError:
            return None
        for f in entries:
            name = f.name
            if not name.endswith(".serato-stems"):
                continue
            f_stem = _ud.normalize("NFC",
                                     name[:-len(".serato-stems")])
            m2 = re.match(r"^(.+?)(\.\d+(?:\.\d+)*)?$", f_stem)
            if m2 and m2.group(1) == base_stem:
                return f
        return None

    # ---- Priority 1: parent of the audio file ----
    hit = _scan(audio.parent)
    if hit:
        return hit

    # ---- Priority 2: known Serato-managed folders ----
    # Walk up from the audio file looking for an `_Serato_` folder,
    # then probe its standard sub-locations. Caps at 6 levels to avoid
    # walking $HOME forever.
    candidates: list[Path] = []
    seen_serato_roots: set[Path] = set()
    cur = audio.parent
    for _ in range(6):
        sibling_serato = cur / "_Serato_"
        if sibling_serato.is_dir() and sibling_serato not in seen_serato_roots:
            seen_serato_roots.add(sibling_serato)
            candidates.extend([
                sibling_serato / "Imported" / "Latest Import",
                sibling_serato / "Imported",
                sibling_serato / "Auto Import",
                sibling_serato / "Recording",
                sibling_serato,
            ])
        if cur.parent == cur:
            break
        cur = cur.parent

    # Also try the canonical ~/Music/_Serato_ location.
    home_serato = Path.home() / "Music" / "_Serato_"
    if home_serato.is_dir() and home_serato not in seen_serato_roots:
        seen_serato_roots.add(home_serato)
        candidates.extend([
            home_serato / "Imported" / "Latest Import",
            home_serato / "Imported",
            home_serato / "Auto Import",
            home_serato / "Recording",
            home_serato,
        ])

    for cand in candidates:
        hit = _scan(cand)
        if hit:
            return hit

    return None


def extract_all_stems(audio_path: str | Path,
                       which: Optional[set] = None,
                       force: bool = False,
                       log: Callable[[str], None] = print
                       ) -> Dict[str, Path]:
    """Extract stem WAVs for an audio file from its `.serato-stems`
    sidecar. Returns dict mapping stem-name → output path for each
    successfully extracted stem.

    Args:
        audio_path: Path to the original audio file.
        which: Set of stem names to extract (e.g. {"vocals","drums"}).
            None → extract all 4 (vocals, harmony, bass, drums).
        force: If False (default), skip stems whose WAV already
            exists. Set True to force re-extraction (e.g. after
            updating ffmpeg flags).
        log: Logging callback.

    Empty dict if sidecar not found. Per-stem failure is non-fatal —
    other stems still extract.

    The on-disk layout this writes:
        <audio_dir>/<base>.serato-vocals.wav
        <audio_dir>/<base>.serato-harmony.wav
        <audio_dir>/<base>.serato-bass.wav
        <audio_dir>/<base>.serato-drums.wav
    """
    if which is None:
        which = set(STEM_TYPE_NAMES.values())
    audio_path = Path(audio_path)
    sidecar = find_sidecar(audio_path)
    if sidecar is None:
        log(f"  [stems] no .serato-stems sidecar for {audio_path.name}")
        return {}

    try:
        data = sidecar.read_bytes()
    except OSError as e:
        log(f"  [!] could not read {sidecar.name}: {e}")
        return {}

    header = parse_serato_stems_header(data)
    if header is None:
        log(f"  [!] {sidecar.name} — missing 'srtshead' magic")
        return {}

    out_paths: Dict[str, Path] = {}

    # Walk ALL chunks, keep payloads of types we want
    payloads: Dict[int, bytes] = {}
    for stem_type, payload in iter_serato_stem_chunks(
        data, header["body_offset"]
    ):
        name = STEM_TYPE_NAMES.get(stem_type)
        if name is None or name not in which:
            continue
        if stem_type in payloads:
            # Duplicate type — Serato shouldn't, but defensively keep first
            continue
        payloads[stem_type] = payload

    if not payloads:
        log(f"  [stems] {sidecar.name} — no requested types found "
            f"(wanted {which})")
        return {}

    # Decode + transcode each
    for stem_type, payload in payloads.items():
        name = STEM_TYPE_NAMES[stem_type]
        out_path = stem_wav_path(audio_path, name)
        if out_path.exists() and not force:
            out_paths[name] = out_path
            continue

        try:
            assert_safe_stem_write(out_path, audio_path)
        except RuntimeError as e:
            log(f"  [!] {e}")
            continue

        # 2026-05-28: stem WAV path is now in the shadow
        # tree.  Ensure the parent directory exists before
        # ffmpeg tries to write there.
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log(f"  [!] cannot mkdir {out_path.parent}: {e}")
            continue

        decoded = decode_xor_payload(payload)
        if not looks_like_mp3(decoded):
            log(f"  [!] decoded {name} stem doesn't look like MP3 — "
                f"Serato may have changed XOR key or format")
            continue

        # Write decoded MP3 to tempfile, transcode to WAV
        mp3_tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp3", prefix="sidecaramel-stems-"
        )
        mp3_tmp.write(decoded)
        mp3_tmp.close()
        try:
            # NOTE the absent flags. ffmpeg reads the LAME `Xing` frame
            # and trims encoder delay and end padding by itself, so the
            # WAV lands sample-exact on the sidecar header's
            # total_samples. Do NOT add `-flags2 +skip_manual` (or any
            # other switch that disables gapless handling): measured
            # against a Serato 3.x sidecar, it puts the 1152 trimmed
            # samples back in and every stem is 26 ms long at the front.
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", mp3_tmp.name,
                "-ac", str(STEM_WAV_CHANNELS), "-ar", str(STEM_WAV_SR),
                str(out_path),
            ]
            r = subprocess.run(cmd, capture_output=True, timeout=120)
            if r.returncode != 0:
                log(f"  [!] ffmpeg failed for {name}: "
                    f"{r.stderr.decode('utf-8', errors='ignore')[:200]}")
                continue
            # Fail closed on length: a stem that disagrees with the
            # sidecar header is misaligned, and a misaligned stem in the
            # cache is worse than no stem — every later run would trust
            # it. Delete it and say so, rather than return it.
            expected = expected_stem_frames(header, STEM_WAV_SR)
            actual = wav_frame_count(out_path)
            if actual is None or abs(actual - expected) > 1:
                try:
                    out_path.unlink()
                except OSError:
                    pass
                log(f"  [!] {name}: length mismatch — got "
                    f"{'unreadable' if actual is None else format(actual, ',')}"
                    f" frames, sidecar header says {expected:,}. Stem "
                    f"discarded (gapless decode or payload boundary "
                    f"wrong).")
                continue
            out_paths[name] = out_path
            log(f"  [stems] {name}: {out_path.name} ({actual:,} frames)")
        except Exception as e:
            log(f"  [!] {name} transcode failed: {e}")
        finally:
            try:
                os.unlink(mp3_tmp.name)
            except OSError:
                pass

    return out_paths


# ============================================================
# DEPRECATED COMPAT — for crawler back-compat
# ============================================================

def extract_serato_acapella(sidecar_path: str | Path,
                             out_path: Optional[Path] = None,
                             log: Callable[[str], None] = print
                             ) -> Optional[Path]:
    """Legacy entry point — extract vocals only. New code should
    use extract_all_stems() with which={"vocals"}.

    Kept for backward-compat with vocals-only call sites.
    Prefer extract_all_stems() for new code."""
    sidecar = Path(sidecar_path)
    audio_path = sidecar.with_suffix("")
    # Strip version suffix if present
    m = re.match(r"^(.+?)(\.\d+(?:\.\d+)*)$", str(audio_path))
    if m:
        audio_path = Path(m.group(1))
    # Try common audio extensions to find the real audio file
    for ext in (".mp3", ".m4a", ".wav", ".flac", ".aiff"):
        candidate = audio_path.with_suffix(ext)
        if candidate.exists():
            audio_path = candidate
            break
    out = extract_all_stems(
        audio_path, which={"vocals"}, log=log,
    )
    if out_path is not None and "vocals" in out:
        # Caller wanted a specific path — copy/rename
        try:
            out["vocals"].rename(out_path)
            out["vocals"] = out_path
        except OSError as e:
            log(f"  [!] could not rename to {out_path}: {e}")
    return out.get("vocals")
