#!/usr/bin/env python3
"""Extract real Serato-produced blobs from tagged audio (and .serato-stems
sidecars) into committable fixtures — the raw material for golden tests.

Usage:
  # dump every Serato blob in a tagged file, with a decoded summary
  python extract_serato_fixtures.py dump TAGGED.mp3 [more files...] --out DIR

  # diff a raw baseline against its Serato-tagged copy — shows exactly which
  # blobs Serato added/changed (the ground truth we want to pin)
  python extract_serato_fixtures.py diff RAW.mp3 TAGGED.mp3

No audio is written — only the tag blobs (a few KB each) plus a JSON manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

from sidecaramel.tags import harvest_serato_blobs


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def _summary(desc: str, raw: bytes) -> dict:
    """Best-effort decoded summary so a human can eyeball it against what
    they set in Serato. Never raises — a decode failure is itself a finding."""
    out: dict = {}
    try:
        if desc == "Serato Markers2":
            from sidecaramel.tags import parse_serato_markers2_full
            m = parse_serato_markers2_full(raw)
            out["cues"] = [
                {"idx": c["idx"], "pos_ms": c["pos_ms"],
                 "color": c["color"], "label": c["label"]} for c in m["cues"]]
            out["loops"] = [
                {"idx": l["idx"], "start_ms": l["start_ms"], "end_ms": l["end_ms"],
                 "color": l["color"], "label": l["label"], "locked": l["locked"]}
                for l in m["loops"]]
            out["color"] = m["color"]
            out["bpm_lock"] = m["bpm_lock"]
            out["autogain"] = m["autogain"]
            # keep flip counts, drop the (large) per-action lists
            out["flips"] = [{k: f.get(k) for k in
                             ("slot", "name", "loop", "action_count")}
                            for f in m["flips"]]
            out["entry_types"] = sorted(m["raw"].keys())
        elif desc == "Serato BeatGrid":
            from sidecaramel.blobs import decode_beatgrid
            out["beatgrid"] = decode_beatgrid(raw)
        elif desc == "Serato Overview":
            out["overview_len"] = len(raw)
    except Exception as e:  # pragma: no cover - diagnostic path
        out["_decode_error"] = f"{type(e).__name__}: {e}"
    return out


def _collect(path: str) -> dict:
    """{descriptor: {raw, source}} for every Serato blob in `path`."""
    blobs = {}
    for desc, raw, src in harvest_serato_blobs(path):
        blobs[desc] = {"raw": bytes(raw), "source": src}
    return blobs


def cmd_dump(paths, outdir):
    os.makedirs(outdir, exist_ok=True)
    manifest = {"files": []}
    for path in paths:
        # keep the extension in the stem so multi-container dumps of the
        # same track name don't collide (…_mp3 vs …_mp4 vs …_wav).
        stem = _safe(os.path.basename(path))
        blobs = _collect(path)
        rec = {"file": os.path.basename(path), "container":
               os.path.splitext(path)[1].lstrip("."), "blobs": []}
        for desc, info in sorted(blobs.items()):
            raw = info["raw"]
            fn = f"{stem}__{_safe(desc)}.blob"
            with open(os.path.join(outdir, fn), "wb") as fh:
                fh.write(raw)
            rec["blobs"].append({
                "descriptor": desc, "source": info["source"],
                "fixture": fn, "len": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "decoded": _summary(desc, raw),
            })
        manifest["files"].append(rec)
        print(f"{path}: {len(blobs)} Serato blob(s) -> {outdir}")
    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))


def cmd_diff(raw_path, tagged_path):
    a, b = _collect(raw_path), _collect(tagged_path)
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    changed = sorted(d for d in set(a) & set(b) if a[d]["raw"] != b[d]["raw"])
    same = sorted(d for d in set(a) & set(b) if a[d]["raw"] == b[d]["raw"])
    print(f"raw:    {raw_path}  ({len(a)} Serato blobs)")
    print(f"tagged: {tagged_path}  ({len(b)} Serato blobs)")
    print(f"\nADDED by Serato ({len(added)}):")
    for d in added:
        print(f"  + {d:22} {len(b[d]['raw']):6} B  via {b[d]['source']}")
        s = _summary(d, b[d]["raw"])
        if s:
            print("      " + json.dumps(s, ensure_ascii=False, default=str)[:400])
    if changed:
        print(f"\nCHANGED ({len(changed)}): " + ", ".join(changed))
    if removed:
        print(f"REMOVED ({len(removed)}): " + ", ".join(removed))
    if same:
        print(f"UNCHANGED ({len(same)}): " + ", ".join(same))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("paths", nargs="+")
    d.add_argument("--out", default="fixtures")
    df = sub.add_parser("diff")
    df.add_argument("raw")
    df.add_argument("tagged")
    args = ap.parse_args(argv)
    if args.cmd == "dump":
        cmd_dump(args.paths, args.out)
    else:
        cmd_diff(args.raw, args.tagged)


if __name__ == "__main__":
    main()
