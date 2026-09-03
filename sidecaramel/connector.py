"""sidecaramel.connector — local MCP server over the package.

Exposes the library to any stdio MCP host as a set of typed tools. Run
it with::

    sidecaramel-mcp            # console script
    python -m sidecaramel.connector

Design rules, in order of precedence:

1. **Writes fail CLOSED.** Every tool that touches Serato-owned data
   calls the same gate as the `roundtrip_safety` CLI: if the
   Serato-running probe is unavailable, that counts as "Serato is
   running" and the write is refused. `serato_known_closed=True` is
   the only way past an unavailable probe, and it is the caller's
   explicit statement, never a default.
2. **No format invariant is bypassable from a tool argument.** Cue
   writes keep V1+V2 pairing, blob writes go through the same
   builders the package tests pin, sidecars are only written next to
   their audio file and never overwrite silently.
3. **Big data stays on disk.** Tools return file paths and compact
   JSON, not audio payloads.

The `mcp` dependency is optional (`pip install sidecaramel[connector]`);
importing this module without it raises with that hint.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations
except ImportError as _exc:                                 # pragma: no cover
    raise ImportError(
        "the connector needs the 'mcp' package (>= 2.0) - install with "
        "pip install sidecaramel[connector]") from _exc

# A tool that raises ToolError returns is_error=True with the message intact
# for the model to read; any other exception is treated as a crash and the
# client sees only "Error executing tool <name>". So every ANTICIPATED
# failure below (write refused, missing file, no blob, ...) goes through
# ToolError, not a bare ValueError/RuntimeError/FileNotFoundError.
try:
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:            # pragma: no cover - older mcp lacks this path
    class ToolError(RuntimeError):  # noqa: N818
        pass

from sidecaramel.check import SeratoCheckUnavailableError, is_running

mcp = MCPServer(
    "sidecaramel_mcp",
    instructions=(
        "Serato library tooling. Read tools are safe anywhere; every "
        "write tool refuses while Serato runs and fails CLOSED when "
        "that cannot be determined."))


class WriteRefused(ToolError):
    pass


def _gate_write(serato_known_closed: bool) -> None:
    """Fail-closed write gate, identical in spirit to roundtrip_safety.

    Raises WriteRefused unless Serato is provably not running, or the
    caller explicitly asserted it is closed (which only helps when the
    probe itself is unavailable — a positive "it IS running" is never
    overridable).
    """
    try:
        running, procs = is_running()
    except SeratoCheckUnavailableError as exc:
        if serato_known_closed:
            return
        raise WriteRefused(
            f"cannot determine whether Serato is running ({exc}); "
            "refusing to write. If you are certain Serato is closed, "
            "retry with serato_known_closed=true.") from exc
    if running:
        raise WriteRefused(
            f"Serato appears to be running ({', '.join(procs) or 'process'}); "
            "close it and retry. This is not overridable.")


def _need_file(path: str) -> Path:
    p = Path(os.path.expanduser(path))
    if not p.is_file():
        raise ToolError(f"no such file: {p}")
    return p


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str, indent=1)


# ── read-only tools ─────────────────────────────────────────────────

@mcp.tool(
    name="serato_track_metadata",
    annotations=ToolAnnotations(title="Read Serato track metadata",
                 readOnlyHint=True, openWorldHint=False))
def serato_track_metadata(audio_path: str) -> str:
    """BPM, beatgrid, cues, loops, flips, autogain, colour and grid
    lock for one audio file, decoded from its Serato tags.

    Returns JSON (`null` when the file has no Serato tags). Raises on
    a nonexistent path so a typo is not mistaken for "no metadata".
    """
    from sidecaramel.tags import read_serato_metadata
    return _j(read_serato_metadata(str(_need_file(audio_path))))


@mcp.tool(
    name="serato_blob_inventory",
    annotations=ToolAnnotations(title="List raw Serato blobs in a file",
                 readOnlyHint=True, openWorldHint=False))
def serato_blob_inventory(audio_path: str) -> str:
    """Names and sizes of every Serato blob (GEOB / atom / Vorbis)
    the file carries — the ground truth of what a write must preserve.
    """
    from sidecaramel.tags import harvest_serato_blobs
    rows = [{"name": d, "bytes": len(raw)}
            for d, raw, _s in harvest_serato_blobs(str(_need_file(audio_path)))]
    return _j(rows)


@mcp.tool(
    name="serato_sidecar_info",
    annotations=ToolAnnotations(title="Inspect a .serato-stems sidecar",
                 readOnlyHint=True, openWorldHint=False))
def serato_sidecar_info(audio_path: str) -> str:
    """Header and per-stem payload sizes of the audio file's
    `.serato-stems` sidecar, or `null` when none exists."""
    from sidecaramel.stems import (STEM_TYPE_NAMES, find_sidecar,
                                   iter_serato_stem_chunks,
                                   parse_serato_stems_header)
    sc = find_sidecar(_need_file(audio_path))
    if sc is None:
        return _j(None)
    data = Path(sc).read_bytes()
    hdr = parse_serato_stems_header(data)
    chunks = [{"stem": STEM_TYPE_NAMES.get(t, str(t)), "bytes": len(p)}
              for t, p in iter_serato_stem_chunks(data, hdr["body_offset"])] \
        if hdr else []
    return _j({"sidecar": str(sc), "header": hdr, "chunks": chunks})


@mcp.tool(
    name="serato_library_tracks",
    annotations=ToolAnnotations(title="List tracks from a Serato database V2",
                 readOnlyHint=True, openWorldHint=False))
def serato_library_tracks(db_path: str, offset: int = 0,
                          limit: int = 50) -> str:
    """Tracks from a `database V2` file, paginated (`offset`/`limit`,
    limit capped at 500). Returns paths plus a `total` count."""
    from sidecaramel.db import iter_tracks
    limit = max(1, min(int(limit), 500))
    all_tracks = list(iter_tracks(str(_need_file(db_path))))
    page = all_tracks[offset:offset + limit]
    return _j({"total": len(all_tracks), "offset": offset,
               "tracks": page})


@mcp.tool(
    name="serato_render_overview",
    annotations=ToolAnnotations(title="Render the Overview blob to an image",
                 readOnlyHint=False, destructiveHint=True,
                 idempotentHint=False, openWorldHint=False))
def serato_render_overview(audio_path: str, out_png: str,
                           mode: str = "grayscale",
                           overwrite: bool = False) -> str:
    """Render the file's Serato Overview blob (240x16) to an image at
    `out_png`. Returns the path, or an error when no blob exists.

    This tool WRITES a file (hence not read-only): it refuses to
    overwrite an existing `out_png` unless `overwrite=true`, and writes
    atomically (temp file + rename) so a failed render never truncates
    the target."""
    import tempfile
    from sidecaramel.overview import render_overview
    from sidecaramel.tags import harvest_serato_blobs
    blobs = {d: raw for d, raw, _s
             in harvest_serato_blobs(str(_need_file(audio_path)))}
    blob = blobs.get("Serato Overview")
    if not blob:
        raise ToolError("file carries no Serato Overview blob")
    out = Path(os.path.expanduser(out_png)).resolve()
    if out.exists() and not overwrite:
        raise WriteRefused(f"{out} exists; pass overwrite=true to replace")
    fd, tmp = tempfile.mkstemp(prefix=".sidecaramel_ov_", dir=str(out.parent))
    os.close(fd)
    try:
        if not render_overview(blob, tmp, mode=mode):
            raise ToolError(
                "render failed (Pillow missing or blob malformed)")
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return _j({"png": str(out), "mode": mode})


@mcp.tool(
    name="serato_extract_stems",
    annotations=ToolAnnotations(title="Extract stem WAVs from the sidecar",
                 readOnlyHint=False, destructiveHint=False,
                 openWorldHint=False))
def serato_extract_stems(audio_path: str, force: bool = False) -> str:
    """Decode the four stems from the file's `.serato-stems` sidecar
    into WAV files next to the audio. Never modifies Serato data; the
    length of every WAV is verified against the sidecar header and a
    mismatching WAV is deleted rather than kept."""
    from sidecaramel.stems import extract_all_stems
    out = extract_all_stems(str(_need_file(audio_path)), force=force,
                            log=lambda *_: None)
    return _j({k: str(v) for k, v in out.items()})


@mcp.tool(
    name="serato_library_format",
    annotations=ToolAnnotations(title="Detect Serato library generation",
                 readOnlyHint=True, openWorldHint=False))
def serato_library_format(serato_folder: str) -> str:
    """Classify a _Serato_ folder: pre-4.0 (binary `database V2` +
    crates), 4.0 (SQLite store), or both during migration. Detection
    is content-based (SQLite magic bytes), not filename-based. All
    write paths in this package target the LEGACY store — on a 4.0
    library they are not authoritative, and this tool is how a client
    finds that out BEFORE writing."""
    from sidecaramel.db import detect_library_format
    p = Path(os.path.expanduser(serato_folder))
    if not p.is_dir():
        raise ToolError(f"no such folder: {p}")
    return _j(detect_library_format(str(p)))


# ── gated write tools ───────────────────────────────────────────────

@mcp.tool(
    name="serato_write_overview",
    annotations=ToolAnnotations(title="Rebuild and write the Overview blob",
                 readOnlyHint=False, destructiveHint=True,
                 idempotentHint=True, openWorldHint=False))
def serato_write_overview(audio_path: str,
                          serato_known_closed: bool = False) -> str:
    """Recompute the Overview blob from the audio and write it into
    the file's Serato tags. All other blobs are preserved
    byte-for-byte. Refused while Serato runs (fail-closed)."""
    _gate_write(serato_known_closed)
    from sidecaramel.overview_encode import build_overview_blob_for_path
    from sidecaramel.tags import write_serato_geob
    p = _need_file(audio_path)
    blob = build_overview_blob_for_path(str(p))
    ok = write_serato_geob(str(p), "Serato Overview", blob, confirm=True)
    if not ok:
        raise ToolError("write_serato_geob returned False")
    return _j({"written": len(blob), "path": str(p)})


@mcp.tool(
    name="serato_build_stems_sidecar",
    annotations=ToolAnnotations(title="Build a .serato-stems sidecar from 4 stems",
                 readOnlyHint=False, destructiveHint=True,
                 idempotentHint=False, openWorldHint=False))
def serato_build_stems_sidecar(audio_path: str,
                               vocals: Optional[str] = None,
                               bass: Optional[str] = None,
                               drums: Optional[str] = None,
                               harmony: Optional[str] = None,
                               overwrite: bool = False,
                               serato_known_closed: bool = False) -> str:
    """Package stem audio files into a `.serato-stems` sidecar next to
    `audio_path`. Stems may come from any separator — or be arbitrary
    tracks. A missing slot becomes a SILENT full-length stem, so the
    sidecar always carries the canonical four chunks. Every payload
    is a bare MPEG-1 Layer III 44.1 kHz 128k stream (tags stripped,
    first frame at byte 0 — the Xing frame must sit at chunk start),
    length-conformed to the shortest provided stem, and the finished
    blob is re-parsed and re-validated before anything is written.

    Refuses to overwrite an existing sidecar unless `overwrite=true`,
    and refuses entirely while Serato runs."""
    _gate_write(serato_known_closed)
    import subprocess

    from sidecaramel.stems_encode import SIDE_SR, build_sidecar_bytes
    p = _need_file(audio_path)
    out = Path(str(p.with_suffix("")) + ".serato-stems")
    if out.exists() and not overwrite:
        raise WriteRefused(f"{out} exists; pass overwrite=true to replace")
    srcs = {0: vocals, 1: bass, 2: drums, 3: harmony}
    provided = {st: str(_need_file(s)) for st, s in srcs.items() if s}
    if not provided:
        raise ToolError("at least one stem file is required")

    def n_samples(path: str) -> int:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return int(float(r.stdout.strip()) * SIDE_SR)
        raise ToolError(f"cannot probe duration of {path}")

    total = min(n_samples(s) for s in provided.values())
    blob = build_sidecar_bytes({st: provided.get(st) for st in range(4)},
                               total)
    out.write_bytes(blob)
    return _j({"sidecar": str(out), "total_samples": total,
               "silent_slots": [st for st in range(4)
                                if st not in provided]})


@mcp.tool(
    name="serato_wipe_blobs",
    annotations=ToolAnnotations(title="Remove ALL Serato blobs from a file",
                 readOnlyHint=False, destructiveHint=True,
                 idempotentHint=True, openWorldHint=False))
def serato_wipe_blobs(audio_path: str, confirm_wipe: bool = False,
                      serato_known_closed: bool = False) -> str:
    """Strip every Serato blob from the file. Destructive and not
    undoable; requires BOTH `confirm_wipe=true` and a closed Serato."""
    if not confirm_wipe:
        raise WriteRefused("destructive: pass confirm_wipe=true to proceed")
    _gate_write(serato_known_closed)
    from sidecaramel.tags import harvest_serato_blobs, wipe_serato_geobs
    p = _need_file(audio_path)
    before = len(harvest_serato_blobs(str(p)))
    wipe_serato_geobs(str(p), confirm=True)
    after = len(harvest_serato_blobs(str(p)))
    return _j({"removed": before - after, "remaining": after})


@mcp.tool(
    name="serato_library_consolidate",
    annotations=ToolAnnotations(title="Consolidate Serato libraries (dry-run default)",
                 readOnlyHint=False, destructiveHint=True,
                 idempotentHint=False, openWorldHint=False))
def serato_library_consolidate(canonical: str, secondaries: list[str],
                       dry_run: bool = True, confirm_move: bool = False,
                       serato_known_closed: bool = False) -> str:
    """Merge secondary _Serato_ folders into a canonical one.
    `dry_run=true` (the default) only reports; a real run additionally
    requires `confirm_move=true` (checked before the write gate, same
    order as `serato_wipe_blobs`'s `confirm_wipe`) and the write gate
    to pass. This moves every unique/orphan audio file it finds —
    thousands of files on a real library — so it gets the same
    explicit double-gate as the other destructive tool rather than
    the CLI's typed-MOVE-confirmation being its only safeguard."""
    if not dry_run:
        if not confirm_move:
            raise WriteRefused(
                "destructive: pass confirm_move=true to proceed with a "
                "live (non-dry-run) consolidate")
        _gate_write(serato_known_closed)
    from sidecaramel.consolidate import consolidate
    rep = consolidate(canonical, list(secondaries), dry_run=dry_run,
                      log_cb=lambda *_: None)
    return _j({k: v for k, v in vars(rep).items()
               if not k.startswith("_")})


def main() -> None:
    """Console entry point — stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main()
