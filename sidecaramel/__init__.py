"""sidecaramel — DJ-audio metadata toolkit.

Alpha, not thoroughly tested — use at your own risk.

Read + write Serato DJ Pro tag formats, embedded lyrics (timed
reads + untimed writes), and embedded cover-art across MP3 / WAV /
AIFF / MP4 / M4A / M4V / FLAC / OGG.

Architectural layering:
    container-agnostic top   ←  ``*_any()`` dispatchers, take a
                                file path, route by extension.
    format-correct bottom    ←  per-format encoders/parsers,
                                roundtrip-validated against the
                                maintainer's own Serato library
                                (see README "Known limitations" —
                                this is author-claim, not yet
                                third-party-corroborated).

Public sub-modules:

  tags        — Serato Markers2 + Markers_ V1 + BeatGrid + Autotags
                + Analysis encoders + writers + parsers +
                dispatchers.
  blobs       — Decoder + harvest() (one call returns every Serato
                blob on a file regardless of container).  Renamed
                from ``inspect`` in 0.1.0 to avoid shadowing the
                stdlib ``inspect`` module; an ``inspect`` alias
                still imports but emits DeprecationWarning.
  flip_writer — Serato FLIP / JUMP / CENSOR action authoring.
  lyrics      — Embedded lyrics: USLT / SYLT / ©lyr / Vorbis
                LYRICS read.  Untimed write (``write_lyrics``,
                container-aware).  Timed write: ``write_sylt``
                (ID3 SYLT — MP3/WAV/AIFF) and the container-agnostic
                ``.lrc`` sidecar (``read_lrc_sidecar`` /
                ``write_lrc_sidecar``).
  art         — Embedded cover-art: APIC / covr / FLAC Picture /
                Vorbis METADATA_BLOCK_PICTURE read + write.
  db          — Serato ``database V2`` reader and writer; ``.crate``
                reader and writer.
  session     — History ``*.session`` file reader.
  paths       — Path-resolution helpers using Serato DBs / History.
  check       — Runtime "is Serato running?" guard.  macOS via
                ``pgrep``; Windows via ``tasklist``; other hosts
                raise ``SeratoCheckUnavailableError``.

CLI: ``python -m sidecaramel <subcommand>`` or ``sidecaramel <sub>``.
Subcommands: read, cues, loops, bpm, lyrics, art, blobs (alias:
inspect), overview, overview-encode, overview-roundtrip.

License: GPL-2.0-or-later (matches mutagen, the sole runtime
dependency).  Format support is reverse-engineered.
Not affiliated with, endorsed by, or sponsored by Serato Limited.
"""

__version__ = "0.1.0"


# Curated public API — common entry points.  Sub-modules are
# importable directly for the full surface.
from sidecaramel.tags import (  # noqa: F401
    # High-level reads
    read_serato_metadata,
    read_loops,
    read_track_color,
    read_bpm_lock,
    read_serato_playcount,
    harvest_serato_blobs,
    # Container-aware writers (all confirm-gated)
    write_serato_markers2_any,
    write_serato_beatgrid_any,
    write_serato_autotags_any,
    write_serato_analysis_any,
    write_serato_vorbis_blob,
    write_serato_markers_full_mp3,
    write_serato_markers_full_mp4,
    write_serato_markers_v1_mp3,
    write_serato_markers_v1_mp4,
    write_serato_markers2,
    write_serato_markers2_mp4,
    write_serato_geob,
    # Encoders (pure)
    build_markers2_inner,
    build_beatgrid,
    build_autotags,
    build_analysis_blob,
    encode_cue_entry,
    encode_loop_entry,
    encode_color_entry,
    encode_bpmlock_entry,
    encode_flip_entry,
    encode_markers_v1_inner,
    encode_markers_v1_mp3_inner,
    pack_serato_markers2,
    # Parsers
    parse_serato_markers2_full,
    parse_serato_markers_v1_full,
    parse_serato_autotags,
    # Constants
    SERATO_ANALYSIS_DEFAULT_BLOB,
    SERATO_CUE_COLOR_PALETTE,
)

__all__ = [
    "read_serato_metadata",
    "read_loops",
    "read_track_color",
    "read_bpm_lock",
    "read_serato_playcount",
    "harvest_serato_blobs",
    "write_serato_markers2_any",
    "write_serato_beatgrid_any",
    "write_serato_autotags_any",
    "write_serato_analysis_any",
    "write_serato_vorbis_blob",
    "write_serato_markers_full_mp3",
    "write_serato_markers_full_mp4",
    "write_serato_markers_v1_mp3",
    "write_serato_markers_v1_mp4",
    "write_serato_markers2",
    "write_serato_markers2_mp4",
    "write_serato_geob",
    "build_markers2_inner",
    "build_beatgrid",
    "build_autotags",
    "build_analysis_blob",
    "encode_cue_entry",
    "encode_loop_entry",
    "encode_color_entry",
    "encode_bpmlock_entry",
    "encode_flip_entry",
    "encode_markers_v1_inner",
    "encode_markers_v1_mp3_inner",
    "pack_serato_markers2",
    "parse_serato_markers2_full",
    "parse_serato_markers_v1_full",
    "parse_serato_autotags",
    "SERATO_ANALYSIS_DEFAULT_BLOB",
    "SERATO_CUE_COLOR_PALETTE",
    # Lyrics
    "read_embedded_lyrics",
    "read_embedded_lyrics_with_format",
    "read_id3_lyrics",
    "read_synced_lyrics",
    "read_synced_lyrics_as_lrc",
    "is_lrc_text",
    "write_uslt",
    "write_sylt",
    "lrc_sidecar_path",
    "read_lrc_sidecar",
    "write_lrc_sidecar",
    # Cover-art
    "read_cover_art",
    "read_all_cover_art",
    "write_cover_art",
    "clear_cover_art",
]


# Lyrics + art re-exports for the top-level public API.
from sidecaramel.lyrics import (  # noqa: F401
    read_embedded_lyrics,
    read_embedded_lyrics_with_format,
    read_id3_lyrics,
    read_synced_lyrics,
    read_synced_lyrics_as_lrc,
    is_lrc_text,
    write_uslt,
    write_sylt,
    lrc_sidecar_path,
    read_lrc_sidecar,
    write_lrc_sidecar,
)
from sidecaramel.art import (  # noqa: F401
    read_cover_art,
    read_all_cover_art,
    write_cover_art,
    clear_cover_art,
)
from sidecaramel.overview import (  # noqa: F401
    render_overview,
    render_overview_for_path,
    grayscale_palette,
)
