"""sidecaramel.consolidate — safely move unique tracks + crates from
secondary _Serato_ folders into one canonical workspace.

After running, the secondary folders contain only Serato metadata
(database V2, recordings, slot configs) which you can safely delete
by hand — every audio file + every non-duplicate crate has been
preserved outside of them.

Audio handling:
  • For each secondary, enumerate the folder recursively for audio
    files (mp3/m4a/mp4/wav/aac/aif/aiff/flac/ogg/opus/wma/mov).
  • Each audio file `F` is classified:
       (a) F is referenced by CANONICAL DB V2 — already accounted
           for. We move F anyway (so the secondary becomes audio-
           free), and emit a warning: the canonical DB will report
           the track as missing on next Serato launch; user fixes
           with right-click → relocate, or by dragging the moved
           folder back into Serato.
       (b) F is in SECONDARY DB V2 only — unique track. Moved.
       (c) F is in neither DB — orphan audio inside the secondary
           folder. Moved (the bytes are real, don't trash them).
  • Companion sidecars (.lrc, .serato-stems, .serato-*.wav; extend
    with add_companion_suffix()
    extracts) move alongside their audio so the work survives.

Crate handling:
  • Every .crate in secondary's Subcrates/ that is NOT also present
    by exact name in canonical's Subcrates/ gets copied to
    canonical/Subcrates/imported_from_<secondary_basename>__<name>.
  • Crates whose only difference vs canonical is the iTunes-style
    " 2" duplicate suffix are detected and skipped (they're noise).

Destination layout:
  ~/Music/sidecaramel_consolidated/
    └─ <secondary_basename>/      e.g. _Serato_backup
        ├─ Auto Import/...        (preserves rel. structure)
        ├─ Imported/...
        └─ <other audio subdirs>

The script is DRY-RUN by default. Pass --apply to actually move
files. After --apply you'll get a concrete list of secondary
folders that should now contain only Serato metadata and are safe
to delete.

2026-05-11.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import shutil
import sys
import time

from sidecaramel.stems import SERATO_STEM_WAV_SUFFIXES
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# Audio extensions to walk.
AUDIO_EXTS = (".mp3", ".m4a", ".mp4", ".wav", ".aac", ".aif",
              ".aiff", ".flac", ".ogg", ".opus", ".wma", ".mov")

# Companion sidecar suffixes moved alongside the audio file (same base
# name + suffix). Only format-native ones are built in: the standard
# `.lrc` lyrics file and the Serato stem artefacts — the stem WAV names
# come from sidecaramel.stems so the two cannot drift apart.
#
# A pipeline that writes its own sidecars registers them with
# add_companion_suffix(); nothing producer-specific is hard-coded here.
SIDECAR_SUFFIXES: tuple = (
    ".lrc",
    ".serato-stems",
) + SERATO_STEM_WAV_SUFFIXES


def add_companion_suffix(suffix: str) -> None:
    """Register an extra sidecar suffix to move with the audio file.

    Suffix must start with `.` (e.g. ".mytool-anchors.json").
    """
    global SIDECAR_SUFFIXES
    if not suffix.startswith("."):
        raise ValueError("suffix must start with '.'")
    if suffix not in SIDECAR_SUFFIXES:
        SIDECAR_SUFFIXES = SIDECAR_SUFFIXES + (suffix,)
# Versioned .serato-stems variants like `track.1.2.serato-stems`.
VERSIONED_STEMS_RE = re.compile(
    r"\.\d+\.\d+\.serato-stems$", re.IGNORECASE)

# iTunes-collision " 2" suffix detector for crates.
ITUNES_DUP_RE = re.compile(r"\s+2\.crate$", re.IGNORECASE)


DEFAULT_CONSOLIDATE_DIR = os.path.expanduser(
    "~/Music/sidecaramel_consolidated")


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ConsolidateReport:
    canonical: str
    secondaries: List[str]
    dry_run: bool
    started_at: float
    finished_at: Optional[float] = None
    n_audio_moved: int = 0
    n_audio_in_canonical_db: int = 0
    n_audio_in_secondary_db: int = 0
    n_audio_orphan: int = 0
    n_audio_skipped_basename_dup: int = 0
    n_audio_skipped_unmatched: int = 0
    n_sidecars_moved: int = 0
    n_crates_copied: int = 0
    n_crates_skipped_dup: int = 0
    n_crates_skipped_itunes_dup: int = 0
    warnings: List[str] = field(default_factory=list)
    basename_dup_paths: List[str] = field(default_factory=list)
    unmatched_paths: List[str] = field(default_factory=list)
    safe_to_delete: List[str] = field(default_factory=list)

    @property
    def elapsed_sec(self) -> float:
        end = self.finished_at or time.monotonic()
        return round(end - self.started_at, 1)


# ============================================================
# DB V2 + SIDECAR HELPERS
# ============================================================

def _db_paths(serato_folder: str) -> List[str]:
    db = os.path.join(serato_folder, "database V2")
    if not os.path.isfile(db):
        return []
    # 2026-06-08: vendored helper (was an upstream module).
    from sidecaramel._library_helpers import (
        enumerate_db_track_paths)
    return enumerate_db_track_paths(db)


def _walk_audio(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root,
                                                  followlinks=False):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            if name.lower().endswith(AUDIO_EXTS):
                out.append(os.path.join(dirpath, name))
    return out


def _sidecars_for(audio_path: str) -> List[str]:
    """Return all sidecar files that exist next to `audio_path`."""
    base = os.path.splitext(audio_path)[0]
    parent = os.path.dirname(audio_path)
    base_name = os.path.basename(base)
    sidecars: List[str] = []
    for suf in SIDECAR_SUFFIXES:
        cand = base + suf
        if os.path.isfile(cand):
            sidecars.append(cand)
    # Versioned .serato-stems variants
    try:
        for name in os.listdir(parent):
            if (name.startswith(base_name + ".")
                    and VERSIONED_STEMS_RE.search(name)):
                full = os.path.join(parent, name)
                if os.path.isfile(full) and full not in sidecars:
                    sidecars.append(full)
    except OSError:
        pass
    return sidecars


def _safe_move(src: str, dst: str, *, dry_run: bool,
                log) -> bool:
    """Move src → dst. If dst exists, append `.dup-N` to avoid
    overwriting. Returns True on success / would-succeed."""
    if not os.path.isfile(src):
        return False
    if os.path.exists(dst):
        # Pick a non-colliding suffix
        base, ext = os.path.splitext(dst)
        n = 1
        while True:
            cand = f"{base}.dup-{n}{ext}"
            if not os.path.exists(cand):
                dst = cand
                break
            n += 1
    if dry_run:
        log(f"[move-dry] {src} → {dst}")
        return True
    try:
        Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)
        log(f"[moved]    {src} → {dst}")
        return True
    except OSError as e:
        log(f"[move-fail] {src}: {e!r}")
        return False


def _safe_copy(src: str, dst: str, *, dry_run: bool,
                log) -> bool:
    if not os.path.isfile(src):
        return False
    if os.path.exists(dst):
        base, ext = os.path.splitext(dst)
        n = 1
        while True:
            cand = f"{base}.dup-{n}{ext}"
            if not os.path.exists(cand):
                dst = cand
                break
            n += 1
    if dry_run:
        log(f"[copy-dry] {src} → {dst}")
        return True
    try:
        Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        log(f"[copied]   {src} → {dst}")
        return True
    except OSError as e:
        log(f"[copy-fail] {src}: {e!r}")
        return False


# ============================================================
# MAIN CONSOLIDATION LOGIC
# ============================================================

def consolidate(canonical: str,
                  secondaries: List[str],
                  *,
                  dry_run: bool = True,
                  consolidate_dir: str = DEFAULT_CONSOLIDATE_DIR,
                  skip_itunes_dup_crates: bool = True,
                  only_canonical_matched: bool = False,
                  log_cb=print,
                  ) -> ConsolidateReport:
    log = log_cb
    canonical = os.path.abspath(os.path.expanduser(canonical))
    rep = ConsolidateReport(canonical=canonical,
                              secondaries=list(secondaries),
                              dry_run=dry_run,
                              started_at=time.monotonic())

    canonical_db_paths = set(_db_paths(canonical))
    canonical_subcrates = os.path.join(canonical, "Subcrates")
    # Build basename → True lookup for fast same-name-elsewhere
    # checks (catches duplicates that live in iTunes etc).
    canonical_basenames: Set[str] = {
        os.path.basename(p) for p in canonical_db_paths}
    log(f"[consolidate] canonical: {canonical} "
        f"({len(canonical_db_paths)} DB tracks, "
        f"{len(canonical_basenames)} unique basenames)")
    if only_canonical_matched:
        log(f"[consolidate] mode: --only-orphans — only moves files "
            f"that are EITHER referenced by canonical DB by exact "
            f"path OR truly unique (no basename match anywhere in "
            f"canonical). Skips iTunes-style ' 2' duplicates and "
            f"same-name siblings in canonical/Imported/.")

    # iTunes-suffix stripper (mirrors check_siblings.py)
    _ITUNES_DUP_RE = re.compile(r"\s+\d+(\.\w+)$",
                                  re.IGNORECASE)

    def _strip_itunes_dup_basename(name: str) -> str:
        m = _ITUNES_DUP_RE.search(name)
        if m:
            return _ITUNES_DUP_RE.sub(m.group(1), name)
        return name

    # Canonical crate names (for dedup of secondary crates)
    canonical_crate_names: Set[str] = set()
    if os.path.isdir(canonical_subcrates):
        for n in os.listdir(canonical_subcrates):
            if n.lower().endswith(".crate"):
                canonical_crate_names.add(n)

    for sec in secondaries:
        sec_abs = os.path.abspath(os.path.expanduser(sec))
        sec_base = os.path.basename(sec_abs.rstrip(os.sep))
        if not os.path.isdir(sec_abs):
            log(f"[consolidate] skip (not a dir): {sec_abs}")
            continue
        if sec_abs == canonical:
            log(f"[consolidate] skip (same as canonical): "
                f"{sec_abs}")
            continue

        log("")
        log("=" * 60)
        log(f"[consolidate] processing secondary: {sec_abs}")
        log("=" * 60)

        # ---- audio walk ----
        all_audio = _walk_audio(sec_abs)
        log(f"[consolidate]   {len(all_audio)} audio files in {sec_base}")

        sec_db_paths = set(_db_paths(sec_abs))
        target_root = os.path.join(consolidate_dir, sec_base)

        for audio_path in all_audio:
            abs_p = os.path.abspath(audio_path)
            rel = os.path.relpath(abs_p, sec_abs)
            dst = os.path.join(target_root, rel)
            bn = os.path.basename(abs_p)

            # ---- Classification ----
            #   canonical_db: exact path in canonical DB → MUST move
            #   basename_match: same filename exists elsewhere in
            #                   canonical (sibling) → safe to leave
            #   basename_stripped: " 2" iTunes-dup of a canonical
            #                      file → safe to leave
            #   orphan: no match anywhere → MUST move to survive
            #           secondary deletion
            classification = "orphan"
            if abs_p in canonical_db_paths:
                classification = "canonical_db"
            elif bn in canonical_basenames:
                classification = "basename_match"
            else:
                stripped = _strip_itunes_dup_basename(bn)
                if (stripped != bn
                        and stripped in canonical_basenames):
                    classification = "basename_stripped"

            # ---- Filter mode ----
            should_move = True
            if only_canonical_matched:
                # Move only canonical_db + orphan. Skip the two
                # basename-dup buckets (they're safe in canonical).
                if classification in ("basename_match",
                                       "basename_stripped"):
                    should_move = False

            # ---- Stats + warnings ----
            if classification == "canonical_db":
                rep.n_audio_in_canonical_db += 1
                rep.warnings.append(
                    f"canonical DB references file inside "
                    f"secondary — after move it will show as "
                    f"missing in Serato (right-click → relocate "
                    f"or re-drag from new path): {abs_p}")
            elif classification == "basename_match":
                rep.n_audio_skipped_basename_dup += 1
                if len(rep.basename_dup_paths) < 100:
                    rep.basename_dup_paths.append(abs_p)
            elif classification == "basename_stripped":
                rep.n_audio_skipped_basename_dup += 1
                if len(rep.basename_dup_paths) < 100:
                    rep.basename_dup_paths.append(abs_p)
            elif abs_p in sec_db_paths:
                # Edge case: only-in-secondary-DB (no canonical
                # reference, no basename match). Treat as orphan
                # but track separately.
                rep.n_audio_in_secondary_db += 1
                rep.n_audio_orphan += 1
            else:
                rep.n_audio_orphan += 1

            if not should_move:
                rep.unmatched_paths.append(abs_p)
                continue

            if _safe_move(abs_p, dst, dry_run=dry_run, log=log):
                rep.n_audio_moved += 1

                # Move sidecars next to it
                for sc in _sidecars_for(abs_p):
                    sc_rel = os.path.relpath(sc, sec_abs)
                    sc_dst = os.path.join(target_root, sc_rel)
                    if _safe_move(sc, sc_dst,
                                    dry_run=dry_run, log=log):
                        rep.n_sidecars_moved += 1

        # ---- crate copy ----
        sec_subcrates = os.path.join(sec_abs, "Subcrates")
        if not os.path.isdir(sec_subcrates):
            log(f"[consolidate]   no Subcrates/ in {sec_base}")
        else:
            for crate_name in sorted(os.listdir(sec_subcrates)):
                if not crate_name.lower().endswith(".crate"):
                    continue
                src = os.path.join(sec_subcrates, crate_name)
                # Skip iTunes-collision " 2.crate" duplicates by
                # default
                if skip_itunes_dup_crates and ITUNES_DUP_RE.search(
                        crate_name):
                    rep.n_crates_skipped_itunes_dup += 1
                    log(f"[crate-skip-itunes-dup] {crate_name}")
                    continue
                if crate_name in canonical_crate_names:
                    rep.n_crates_skipped_dup += 1
                    continue
                # Prefix the imported crate so it's easy to spot
                # in Serato (and avoid future collisions).
                imp_name = (f"imported_from_{sec_base}__"
                             f"{crate_name}")
                dst = os.path.join(canonical_subcrates, imp_name)
                if _safe_copy(src, dst, dry_run=dry_run, log=log):
                    rep.n_crates_copied += 1

        # ---- safe-to-delete suggestion ----
        rep.safe_to_delete.append(sec_abs)

    rep.finished_at = time.monotonic()
    return rep


# ============================================================
# CLI
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Safely consolidate multiple _Serato_-style "
                    "folders into one canonical workspace. Moves "
                    "audio + crates out of the secondaries so you "
                    "can delete the leftover Serato metadata "
                    "folders without losing any tracks.")
    ap.add_argument("canonical",
                     help="Canonical Serato folder to KEEP (e.g. "
                          "~/Music/_Serato_)")
    ap.add_argument("secondaries", nargs="+",
                     help="One or more secondary Serato folders to "
                          "harvest audio + crates from. The folders "
                          "themselves are NOT deleted — after the "
                          "script completes you do that by hand.")
    ap.add_argument("--target-dir", default=DEFAULT_CONSOLIDATE_DIR,
                     help=f"Where moved audio + sidecars land. "
                          f"Subfolder per secondary preserves "
                          f"relative structure. Default: "
                          f"{DEFAULT_CONSOLIDATE_DIR}")
    ap.add_argument("--include-itunes-dup-crates",
                     action="store_true",
                     help="Also copy crates whose name ends with "
                          "' 2.crate' (typically iTunes-collision "
                          "duplicates of an existing crate). "
                          "Default: skipped as noise.")
    ap.add_argument("--only-orphans", action="store_true",
                     help="Smart mode: move only files that are "
                          "(a) referenced by canonical DB by exact "
                          "path, or (b) truly unique (no basename "
                          "match anywhere in canonical's "
                          "database — also strips iTunes ' 2' "
                          "suffix before matching). Skips "
                          "basename-duplicates that already exist "
                          "somewhere in canonical's library — "
                          "those stay in secondary and can be "
                          "deleted manually with the leftover "
                          "folder.")
    ap.add_argument("--apply", action="store_true",
                     help="LIVE — actually move files. Default is "
                          "DRY-RUN (no disk changes).")
    args = ap.parse_args()

    dry_run = not args.apply

    print()
    print("=" * 68)
    print("sidecaramel.consolidate")
    print(f"  canonical:        {args.canonical}")
    print(f"  secondaries:      {len(args.secondaries)}")
    for s in args.secondaries:
        print(f"    - {s}")
    print(f"  target dir:       {args.target_dir}")
    print(f"  mode:             "
          f"{'DRY-RUN (no disk writes)' if dry_run else 'LIVE'}")
    print("=" * 68)
    if not dry_run:
        print()
        print("LIVE mode — this will MOVE audio + sidecars out of")
        print("the secondary folders into the target directory.")
        print("The originals will no longer exist at their old")
        print("paths. Make sure you have a backup of anything")
        print("irreplaceable before proceeding.")
        print()
        confirm = input("Type 'MOVE' to proceed: ").strip()
        if confirm != "MOVE":
            print("aborted.")
            return 1
    print()

    rep = consolidate(
        args.canonical,
        args.secondaries,
        dry_run=dry_run,
        consolidate_dir=args.target_dir,
        skip_itunes_dup_crates=not args.include_itunes_dup_crates,
        only_canonical_matched=args.only_orphans)

    print()
    print("=" * 68)
    print("SUMMARY")
    print(f"  mode:                       "
          f"{'DRY-RUN' if rep.dry_run else 'LIVE'}")
    print(f"  elapsed:                    {rep.elapsed_sec}s")
    print(f"  audio moved:                {rep.n_audio_moved}")
    print(f"    referenced by canonical:  "
          f"{rep.n_audio_in_canonical_db} (re-link in Serato)")
    print(f"    secondary-DB only:        "
          f"{rep.n_audio_in_secondary_db}")
    print(f"    orphan (not in any DB):   "
          f"{rep.n_audio_orphan}")
    print(f"  audio SKIPPED (basename dup): "
          f"{rep.n_audio_skipped_basename_dup} "
          f"(already in canonical under another path)")
    print(f"  sidecars moved alongside:   "
          f"{rep.n_sidecars_moved}")
    print(f"  crates copied to canonical: "
          f"{rep.n_crates_copied}")
    print(f"  crates skipped (dup name):  "
          f"{rep.n_crates_skipped_dup}")
    print(f"  crates skipped (iTunes ' 2'): "
          f"{rep.n_crates_skipped_itunes_dup}")
    if rep.warnings:
        print()
        print(f"  WARNINGS ({len(rep.warnings)} — first 10 shown):")
        for w in rep.warnings[:10]:
            print(f"    {w}")
        if len(rep.warnings) > 10:
            print(f"    … +{len(rep.warnings) - 10} more")
    print()
    print(f"  Secondaries now safe to delete BY HAND "
          f"(only Serato metadata remains):")
    for s in rep.safe_to_delete:
        print(f"    {s}")
    print()
    print("=" * 68)
    print()
    if rep.n_audio_in_canonical_db > 0 and not rep.dry_run:
        print(f"NOTE: {rep.n_audio_in_canonical_db} tracks that moved "
              f"were referenced by your CANONICAL Serato DB.")
        print(f"After launching Serato, those will show as missing.")
        print(f"Fix by dragging the consolidated folder into "
              f"Serato (it'll re-index at the new paths) OR by "
              f"using right-click → 'Relocate File...' per track.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
