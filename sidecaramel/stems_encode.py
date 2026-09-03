"""sidecaramel.stems_encode — condition stem audio for sidecars.

The decoder side of `.serato-stems` lives in `stems.py` and never
writes a sidecar. This module is the encoder side's preparation work:
turning arbitrary audio into payloads that match what Serato itself
puts into sidecar chunks, byte-discipline included.

The rules encoded here come from measuring real Serato sidecars and
from studying sidecars produced by Good Clean Stems
(github.com/Fotsbeats/GOOD-CLEAN-STEMS) — a separate, independently
authored Serato-sidecar-building app whose repository is public but
whose software is licensed proprietary ("for evaluation and testing")
per its own README. No code from it is used or vendored here; what
follows was derived by measuring sidecars it produced that Serato
demonstrably accepts:

  * a chunk payload is a BARE MP3 STREAM — no ID3v2 header, no ID3v1
    tail, nothing before the first frame sync. A payload that starts
    with a tag shifts the first frame away from the chunk start; the
    first frame is LAME's Xing/Info frame and carries the gapless
    fields, so it must sit at offset 0.
  * frames are MPEG-1 Layer III, 44100 Hz, CBR — real sidecars use
    128 kbps. Anything else is transcoded rather than trusted.
  * every stem in one sidecar has the SAME sample count, equal to the
    header's `total_samples`. Length is conformed by trim/pad before
    encoding, never by hoping.
  * a missing stem is not a missing chunk — it is a SILENT stem of
    full length, so `n_stems` and the slot layout stay canonical.

Pure-bytes helpers have no dependencies; the conform/build helpers
shell out to ffmpeg and raise `RuntimeError` when it is absent.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

SIDE_SR = 44100
SIDE_KBPS = 128
XOR_KEY = 0x26

# MPEG-1 Layer III bitrate table (kbps), index = header bits 12..15.
_BITRATES_V1L3 = (None, 32, 40, 48, 56, 64, 80, 96, 112, 128,
                  160, 192, 224, 256, 320, None)
_SAMPLERATES_V1 = (44100, 48000, 32000, None)


def _syncsafe(b4: bytes) -> int:
    return ((b4[0] & 0x7F) << 21) | ((b4[1] & 0x7F) << 14) \
        | ((b4[2] & 0x7F) << 7) | (b4[3] & 0x7F)


def strip_mp3_tags_and_align_frames(data: bytes) -> bytes:
    """Remove ID3v2 (with footer, if flagged), ID3v1, and any junk
    before the first MPEG frame sync. Returns a stream whose byte 0
    is the first frame's 0xFF. Raises ValueError when no frame sync
    exists at all."""
    buf = data
    while buf[:3] == b"ID3" and len(buf) >= 10:
        size = _syncsafe(buf[6:10]) + 10
        if buf[5] & 0x10:                     # footer present flag
            size += 10
        buf = buf[size:]
    if len(buf) >= 128 and buf[-128:-125] == b"TAG":
        buf = buf[:-128]
    for i in range(len(buf) - 1):
        if buf[i] == 0xFF and (buf[i + 1] & 0xE0) == 0xE0:
            return buf[i:]
    raise ValueError("no MPEG frame sync found")


def mp3_frame_props(data: bytes) -> Optional[dict]:
    """Header fields of the frame at offset 0, or None when the four
    bytes are not an MPEG-1 Layer III header."""
    if len(data) < 4 or data[0] != 0xFF or (data[1] & 0xE0) != 0xE0:
        return None
    version = (data[1] >> 3) & 0x03           # 3 = MPEG-1
    layer = (data[1] >> 1) & 0x03             # 1 = Layer III
    if version != 3 or layer != 1:
        return None
    kbps = _BITRATES_V1L3[(data[2] >> 4) & 0x0F]
    sr = _SAMPLERATES_V1[(data[2] >> 2) & 0x03]
    if kbps is None or sr is None:
        return None
    return {"kbps": kbps, "sample_rate": sr, "mpeg1_layer3": True}


def is_sidecar_compatible(data: bytes) -> bool:
    """True when the (already aligned) stream opens with an MPEG-1
    Layer III frame at 44100 Hz and the sidecar bitrate."""
    p = mp3_frame_props(data)
    return bool(p and p["sample_rate"] == SIDE_SR
                and p["kbps"] == SIDE_KBPS)


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH")
    return exe


def _encode(args_in: list, total_samples: int) -> bytes:
    """Common encode tail: exact-length trim/pad, sidecar bitrate,
    no tags, then strip/align defensively and validate."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        out = f.name
    try:
        subprocess.run(
            [_ffmpeg(), "-y", "-loglevel", "error", *args_in,
             "-af", f"apad,atrim=end_sample={total_samples}",
             "-ar", str(SIDE_SR), "-ac", "2",
             "-codec:a", "libmp3lame", "-b:a", f"{SIDE_KBPS}k",
             "-id3v2_version", "0", "-write_id3v1", "0", out],
            check=True)
        payload = strip_mp3_tags_and_align_frames(Path(out).read_bytes())
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass
    if not is_sidecar_compatible(payload):
        raise RuntimeError("encoded stream is not sidecar-compatible "
                           f"({mp3_frame_props(payload)})")
    return payload


def conform_stem_mp3(src_path: str | Path, total_samples: int) -> bytes:
    """Any audio file -> sidecar-ready bare MP3 stream of exactly
    `total_samples` samples at 44100 Hz."""
    return _encode(["-i", str(src_path)], total_samples)


def make_silent_stem_mp3(total_samples: int) -> bytes:
    """A silent, full-length stem — the canonical filler for an empty
    slot, so the sidecar always carries n_stems=4."""
    return _encode(["-f", "lavfi", "-i",
                    f"anullsrc=r={SIDE_SR}:cl=stereo"], total_samples)


def build_sidecar_bytes(stems: Dict[int, Optional[str]],
                        total_samples: int) -> bytes:
    """Serialise a complete `.serato-stems` blob from up to four stem
    audio files. `stems` maps stem type (0..3) to a path or None;
    missing/None slots become silence. The result is re-parsed and
    every payload re-validated before it is returned — a blob this
    function hands back has already survived the package's own
    decoder."""
    from sidecaramel.stems import (build_serato_stems,
                                   decode_xor_payload,
                                   iter_serato_stem_chunks, looks_like_mp3,
                                   parse_serato_stems_header)
    chunks = []
    for st in range(4):
        src = stems.get(st)
        payload = (conform_stem_mp3(src, total_samples) if src
                   else make_silent_stem_mp3(total_samples))
        chunks.append((st, bytes(b ^ XOR_KEY for b in payload)))
    header = {"header_length": 16, "version": (1, 2), "n_stems": 4,
              "total_samples": int(total_samples), "sample_rate": SIDE_SR}
    blob = build_serato_stems(header, chunks)

    hdr = parse_serato_stems_header(blob)
    if not hdr or hdr["total_samples"] != int(total_samples):
        raise RuntimeError("self-check failed: header does not parse back")
    seen = []
    for st, enc in iter_serato_stem_chunks(blob, hdr["body_offset"]):
        dec = decode_xor_payload(enc)
        if not looks_like_mp3(dec) or not is_sidecar_compatible(dec):
            raise RuntimeError(f"self-check failed: stem {st} payload")
        seen.append(st)
    if sorted(seen) != [0, 1, 2, 3]:
        raise RuntimeError(f"self-check failed: stems present {seen}")
    return blob
