"""sidecaramel CLI entry point.

Run via:
    python -m sidecaramel <subcommand> [args]
or (after pip install):
    sidecaramel <subcommand> [args]

Subcommands:
    read       Dump everything Serato + lyrics + art knows about a file
                as JSON.
    cues       List cue points (idx / pos_ms / color / label).
    loops      List loops.
    bpm        Print BPM + beatgrid-terminal-position.
    lyrics     Print embedded lyrics (auto-detect timed vs untimed).
    art        Extract embedded cover-art to a file (or print info).
    inspect    Hex-preview every Serato blob on the file (debug).

Examples:
    sidecaramel read track.mp3 --json
    sidecaramel cues track.m4a
    sidecaramel lyrics track.flac
    sidecaramel art track.mp3 --out cover.jpg
    sidecaramel bpm track.wav
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _cmd_read(args) -> int:
    import sidecaramel
    path = args.path
    meta = sidecaramel.read_serato_metadata(path) or {}
    out: dict = {
        "path": path,
        "bpm": meta.get("bpm"),
        "first_cue_sec": meta.get("first_cue"),
        "cues": meta.get("cues", []),
        "loops": meta.get("loops", []),
        "color": meta.get("color"),
        "bpm_lock": meta.get("bpm_lock"),
    }
    # Lyrics
    from sidecaramel.lyrics import read_embedded_lyrics_with_format
    lyr = read_embedded_lyrics_with_format(path)
    out["lyrics"] = {"present": lyr is not None,
                       "format": lyr[1] if lyr else None,
                       "char_count": len(lyr[0]) if lyr else 0}
    # Art
    from sidecaramel.art import read_cover_art
    art = read_cover_art(path)
    out["cover_art"] = {"present": art is not None,
                          "mime": art[0] if art else None,
                          "byte_count": len(art[1]) if art else 0}
    if args.json:
        # Cues/loops contain RGB tuples — convert for JSON
        def _serialize(o):
            if isinstance(o, tuple):
                return list(o)
            return str(o)
        print(json.dumps(out, indent=2, default=_serialize))
    else:
        print(f"path:      {path}")
        print(f"bpm:       {out['bpm']}")
        print(f"cues:      {len(out['cues'])}")
        print(f"loops:     {len(out['loops'])}")
        print(f"color:     {out['color']}")
        print(f"bpm_lock:  {out['bpm_lock']}")
        print(f"lyrics:    "
              f"{'yes (' + out['lyrics']['format'] + ', ' + str(out['lyrics']['char_count']) + ' chars)' if out['lyrics']['present'] else 'no'}")
        print(f"cover_art: "
              f"{'yes (' + out['cover_art']['mime'] + ', ' + str(out['cover_art']['byte_count']) + ' B)' if out['cover_art']['present'] else 'no'}")
    return 0


def _cmd_cues(args) -> int:
    import sidecaramel
    meta = sidecaramel.read_serato_metadata(args.path) or {}
    cues = meta.get("cues", [])
    if not cues:
        print(f"no cues in {args.path}", file=sys.stderr)
        return 1
    for c in cues:
        ms = c["pos_ms"]
        sec = ms / 1000.0
        rgb = c["color"]
        label = c.get("label") or ""
        print(f"  cue[{c['idx']}]  {sec:>7.3f}s  ({ms:>7} ms)  "
              f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}  {label}")
    return 0


def _cmd_loops(args) -> int:
    import sidecaramel
    meta = sidecaramel.read_serato_metadata(args.path) or {}
    loops = meta.get("loops", [])
    if not loops:
        print(f"no loops in {args.path}", file=sys.stderr)
        return 1
    for l in loops:
        s = l["start_ms"] / 1000.0
        e = l["end_ms"] / 1000.0
        rgb = l["color"]
        locked = "🔒" if l.get("locked") else ""
        label = l.get("label") or ""
        print(f"  loop[{l['idx']}]  {s:>7.3f} → {e:>7.3f}s  "
              f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}  {locked}  {label}")
    return 0


def _cmd_bpm(args) -> int:
    import sidecaramel
    meta = sidecaramel.read_serato_metadata(args.path) or {}
    bpm = meta.get("bpm")
    if bpm is None:
        print(f"no bpm in {args.path}", file=sys.stderr)
        return 1
    print(f"{bpm:.4f}")
    return 0


def _cmd_lyrics(args) -> int:
    from sidecaramel.lyrics import (read_embedded_lyrics,
                                       read_synced_lyrics_as_lrc)
    if args.lrc:
        text = read_synced_lyrics_as_lrc(args.path)
        if text is None:
            print(f"no synced lyrics in {args.path}", file=sys.stderr)
            return 1
    else:
        text = read_embedded_lyrics(args.path)
        if text is None:
            print(f"no lyrics in {args.path}", file=sys.stderr)
            return 1
    print(text)
    return 0


def _cmd_art(args) -> int:
    from sidecaramel.art import read_cover_art
    art = read_cover_art(args.path)
    if art is None:
        print(f"no cover-art in {args.path}", file=sys.stderr)
        return 1
    mime, data = art
    if args.out:
        with open(args.out, "wb") as f:
            f.write(data)
        print(f"wrote {len(data)} B {mime} → {args.out}")
    else:
        print(f"mime:  {mime}")
        print(f"bytes: {len(data)}")
    return 0


def _cmd_inspect(args) -> int:
    from sidecaramel.blobs import harvest
    blobs = harvest(args.path)
    if not blobs:
        print(f"no Serato blobs in {args.path}", file=sys.stderr)
        return 1
    for desc, payload, src in blobs:
        preview = (payload[:48].hex(" ") if payload else "(empty)")
        if len(payload) > 48:
            preview += " ..."
        print(f"  {desc:<26}  {len(payload):>6} B  {preview}")
    return 0


def _cmd_overview(args) -> int:
    from sidecaramel.overview import (render_overview_for_path,
                                         compare_against_reference)
    from sidecaramel.blobs import harvest

    if args.compare:
        # Comparison mode: render via mode + diff against reference
        blob = None
        for desc, payload, _ in harvest(args.path):
            if desc == "Serato Overview" and payload:
                blob = payload
                break
        if blob is None:
            print(f"no Serato Overview blob in {args.path}",
                  file=sys.stderr)
            return 1
        result = compare_against_reference(
            blob, args.compare, mode=args.mode)
        if result is None:
            print("comparison failed (Pillow missing or blob bad)",
                  file=sys.stderr)
            return 1
        print(f"mode:              {result['mode']}")
        print(f"mean RGB error:    {result['mean_rgb_error']:.2f} "
              f"(0=perfect, 255=opposite)")
        print()
        print("=== by column-spread band ===")
        for band, st in result["by_spread_band"].items():
            print(f"  {band:<20}  n={st['n']:>5}  "
                  f"err={st['mean_rgb_error']:.2f}")
        print()
        print("=== by byte zone ===")
        for zone, st in result["by_byte_zone"].items():
            print(f"  {zone:<15}  n={st['n']:>5}  "
                  f"err={st['mean_rgb_error']:.2f}")
        print()
        print(f"rendered:    {result['rendered_image_path']}")
        print(f"diff heatmap: {result['diff_image_path']}")
        return 0

    # Render mode (no comparison)
    if not args.out:
        print("--out required for render", file=sys.stderr)
        return 2
    ok = render_overview_for_path(args.path, args.out,
                                     mode=args.mode, scale=args.scale)
    if not ok:
        print(f"render failed (no Overview blob or PIL missing)",
              file=sys.stderr)
        return 1
    print(f"wrote {args.mode} BMP → {args.out}")
    return 0


def _cmd_overview_encode(args) -> int:
    """Encode an audio file → 3842-byte Serato Overview blob."""
    from sidecaramel.overview_encode import build_overview_blob_for_path
    blob = build_overview_blob_for_path(args.path)
    if len(blob) != 3842:
        print(f"unexpected blob size: {len(blob)}", file=sys.stderr)
        return 1
    with open(args.out, "wb") as f:
        f.write(blob)
    print(f"wrote {len(blob)}-byte Overview blob → {args.out}")
    return 0


def _cmd_overview_roundtrip(args) -> int:
    """Encode audio + compare against in-file Serato blob.  Optional
    PNG diff panel.  Never writes to the audio file (audio-safety policy safe)."""
    from sidecaramel.overview_encode import (
        build_overview_blob_for_path, diff_blobs)
    from sidecaramel.tags import harvest_serato_blobs

    # Find in-file Serato Overview blob
    ref = None
    for desc, payload, _src in harvest_serato_blobs(args.path):
        if desc == "Serato Overview":
            if hasattr(payload, "data"):
                ref = bytes(payload.data)
            else:
                ref = bytes(payload)
            break

    ours = build_overview_blob_for_path(args.path)
    print(f"OURS:  {len(ours)} B  "
          f"first8={ours[:8].hex()}")

    if ref is None:
        print("REF:   no in-file Serato Overview blob to compare "
              "against (file isn't Serato-analysed yet).  Encode "
              "succeeded; can't diff.")
        return 0

    print(f"REF:   {len(ref)} B  first8={ref[:8].hex()}")
    report = diff_blobs(ours, ref)
    if report.get("bytes_equal"):
        print("→ BYTE-EQUAL  ✓")
    else:
        print(f"→ {report['n_diff']}/{report['total_bytes']} bytes "
              f"differ ({report['diff_pct']}%), "
              f"{report['n_cols_with_diff']}/240 cols affected")
        print(f"  max col diff: {report['max_col_diff']}/16, "
              f"mean: {report['mean_col_diff']}")
        print("  top mismatches (ours → ref):")
        for m in report["top_mismatches"][:5]:
            print(f"    {m['ours']:3d} → {m['ref']:3d}  ×{m['count']}")

    if args.render_diff:
        try:
            from PIL import Image, ImageDraw, ImageFont
            from sidecaramel.overview import render_overview
        except ImportError:
            print("Pillow missing — skipping --render-diff",
                  file=sys.stderr)
            return 0
        import tempfile
        # Vertical block-wise render: time runs top→bottom, each
        # 16-byte chunk = one horizontal row.  Strip 48 wide × 720
        # tall (scale 3 from 16×240 native).
        STRIP_W, STRIP_H = 48, 720
        def _render(blob):
            tmp = tempfile.NamedTemporaryFile(
                suffix=".bmp", delete=False).name
            render_overview(blob, tmp, mode="rgb332", scale=1)
            img = (Image.open(tmp).convert("RGB")
                       .transpose(Image.TRANSPOSE)
                       .resize((STRIP_W, STRIP_H), Image.NEAREST))
            os.remove(tmp)
            return img
        ref_img  = _render(ref)
        ours_img = _render(ours)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except Exception:
            font = ImageFont.load_default()

        PAD       = 12
        INNER_GAP = 8
        TITLE_H   = 36
        SUB_LABEL = 22
        panel_w   = STRIP_W * 2 + INNER_GAP + PAD * 2
        panel_h   = TITLE_H + STRIP_H + SUB_LABEL + PAD
        panel = Image.new("RGB", (panel_w, panel_h), (20, 20, 20))
        d = ImageDraw.Draw(panel)
        d.text((PAD, 8),
                f"{os.path.basename(args.path)}  ·  vertical (block-wise)",
                fill=(255, 255, 255), font=font)
        panel.paste(ref_img,  (PAD, TITLE_H))
        panel.paste(ours_img,
                       (PAD + STRIP_W + INNER_GAP, TITLE_H))
        sub_y = TITLE_H + STRIP_H + 4
        d.text((PAD + (STRIP_W - 24) // 2, sub_y), "REF",
                 fill=(180, 220, 180), font=font)
        d.text((PAD + STRIP_W + INNER_GAP + (STRIP_W - 32) // 2,
                  sub_y), "OURS",
                 fill=(180, 180, 255), font=font)
        panel.save(args.render_diff)
        print(f"  side-by-side panel → {args.render_diff}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sidecaramel",
        description="Read + write Serato DJ Pro tag formats and "
                    "embedded lyrics / cover-art across MP3, WAV, AIFF, "
                    "MP4, M4A, M4V, FLAC, OGG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  sidecaramel read track.mp3 --json\n"
            "  sidecaramel cues track.m4a\n"
            "  sidecaramel loops track.flac\n"
            "  sidecaramel bpm track.wav\n"
            "  sidecaramel lyrics track.mp3\n"
            "  sidecaramel lyrics track.mp3 --lrc\n"
            "  sidecaramel art track.flac --out cover.jpg\n"
            "  sidecaramel inspect track.m4a\n"
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("read", help="Dump full metadata (Serato + "
                                       "lyrics + art) for a file.")
    sp.add_argument("path")
    sp.add_argument("--json", action="store_true",
                     help="Print as machine-readable JSON.")
    sp.set_defaults(func=_cmd_read)

    sp = sub.add_parser("cues", help="List cue points.")
    sp.add_argument("path")
    sp.set_defaults(func=_cmd_cues)

    sp = sub.add_parser("loops", help="List loops.")
    sp.add_argument("path")
    sp.set_defaults(func=_cmd_loops)

    sp = sub.add_parser("bpm", help="Print Serato BPM.")
    sp.add_argument("path")
    sp.set_defaults(func=_cmd_bpm)

    sp = sub.add_parser("lyrics", help="Print embedded lyrics.")
    sp.add_argument("path")
    sp.add_argument("--lrc", action="store_true",
                     help="Output as LRC (timed) if synced lyrics "
                          "available.")
    sp.set_defaults(func=_cmd_lyrics)

    sp = sub.add_parser("art", help="Extract / inspect embedded cover-art.")
    sp.add_argument("path")
    sp.add_argument("--out", help="Write the image bytes to this path.")
    sp.set_defaults(func=_cmd_art)

    # Canonical subcommand name from 0.1.0 onwards: `blobs`.
    # `inspect` registered below as a deprecated alias for one minor
    # — same handler.
    sp = sub.add_parser("blobs",
                          help="Hex-preview every Serato blob on the file.")
    sp.add_argument("path")
    sp.set_defaults(func=_cmd_inspect)

    sp = sub.add_parser("inspect",
                          help="DEPRECATED — use `blobs` instead.  "
                               "Same behaviour; alias for one minor.")
    sp.add_argument("path")
    sp.set_defaults(func=_cmd_inspect)

    sp = sub.add_parser("overview",
                          help="Render the Serato Overview waveform as a "
                               "BMP, or diff it against a reference image.")
    sp.add_argument("path")
    sp.add_argument("--out",
                     help="Output BMP path (required unless --compare).")
    sp.add_argument("--mode",
                     choices=("grayscale", "color", "alpha", "hsl",
                               "rgb332", "serato_hue", "serato_palette",
                               "column_agg", "column_tornado"),
                     default="serato_palette",
                     help="Render mode (default: serato_palette).  "
                          "serato_palette = 40-byte hand-tuned LUT "
                          "with additive R=bass G=mid B=treble "
                          "anchors; serato_hue = interpolated "
                          "spectral gradient; rgb332 = structural bit "
                          "decode; grayscale = darkness ramp; "
                          "color/alpha/hsl = legacy experimental.")
    sp.add_argument("--scale", type=int, default=4,
                     help="Nearest-neighbor upscale factor "
                          "(default: 4).")
    sp.add_argument("--compare",
                     help="Path to a reference image; runs pixel-diff "
                          "+ prints per-band/per-zone error stats.")
    sp.set_defaults(func=_cmd_overview)

    # ---- overview-encode: audio → blob.bin ------------------------
    sp = sub.add_parser(
        "overview-encode",
        help="Encode an audio file to a 3842-byte Serato Overview "
              "blob (pixel-similar v1 encoder).")
    sp.add_argument("path")
    sp.add_argument("--out", required=True,
                     help="Output .bin path (3842 bytes).")
    sp.set_defaults(func=_cmd_overview_encode)

    # ---- overview-roundtrip: encode + diff vs Serato's blob -------
    sp = sub.add_parser(
        "overview-roundtrip",
        help="Encode the audio + diff against the Serato-written "
              "Overview blob already in the file.  Read-only on the "
              "audio file.  read-only.")
    sp.add_argument("path")
    sp.add_argument("--render-diff",
                     help="Optional output PNG path; renders REF + "
                          "OURS side-by-side as a comparison panel.")
    sp.set_defaults(func=_cmd_overview_roundtrip)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not os.path.isfile(args.path):
        print(f"file not found: {args.path}", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
