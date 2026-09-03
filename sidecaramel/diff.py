"""sidecaramel.diff — compare two Serato config folders.

Reports on three layers:

  1. **Track index (database V2)** — which absolute audio paths
     each folder's DB knows about. Set-diff: only-A, only-B, both.
  2. **Subcrates** — .crate files in each folder's Subcrates/ dir.
     For overlapping crate names: same content / different content.
  3. **Sidecar coverage** — for tracks in either DB:
     - .serato-stems present (analysed for stems)
     - .lrc present (consumer-touched)
     - playcount tag in ID3

Usage:
    python -m sidecaramel.diff \\
        ~/Music/_Serato_  \\
        ~/Music/_Serato_backup

With --verbose: also print first 25 sample paths from each diff
bucket.

2026-05-11.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _enumerate_db_paths(db_path: str) -> List[str]:
    """Wrapper around vendored Serato DB parser."""
    from sidecaramel._library_helpers import (
        enumerate_db_track_paths)
    if not os.path.isfile(db_path):
        return []
    return enumerate_db_track_paths(db_path)


def _enumerate_crates(subcrates_dir: str) -> Dict[str, str]:
    """Return {crate_name: crate_path} for every .crate in dir."""
    out: Dict[str, str] = {}
    if not os.path.isdir(subcrates_dir):
        return out
    for entry in os.listdir(subcrates_dir):
        if not entry.lower().endswith(".crate"):
            continue
        full = os.path.join(subcrates_dir, entry)
        if os.path.isfile(full):
            out[entry] = full
    return out


def _enumerate_crate_paths(crate_path: str) -> List[str]:
    """Wrapper around vendored Serato crate parser."""
    from sidecaramel._library_helpers import (
        enumerate_crate_track_paths,
        crate_volume_root,
        abs_path_from_crate_rel,
    )
    rel = enumerate_crate_track_paths(crate_path)
    vol = crate_volume_root(crate_path)
    return [abs_path_from_crate_rel(vol, r) for r in rel]


def _has_lrc(audio_path: str) -> bool:
    return os.path.isfile(os.path.splitext(audio_path)[0] + ".lrc")


def _has_serato_stems(audio_path: str) -> bool:
    base = os.path.splitext(audio_path)[0]
    if os.path.isfile(base + ".serato-stems"):
        return True
    # Versioned variant
    parent = os.path.dirname(audio_path)
    pref = os.path.basename(base) + "."
    try:
        for name in os.listdir(parent):
            if (name.startswith(pref)
                    and name.endswith(".serato-stems")):
                return True
    except OSError:
        pass
    return False


def _id3_playcount(audio_path: str) -> int:
    """Read Serato-written playcount via vendored mutagen helper."""
    try:
        from sidecaramel._library_helpers import mutagen_playcount
        pc = mutagen_playcount(audio_path)
        return pc if isinstance(pc, int) else 0
    except Exception:
        return 0


def _print_diff_bucket(label: str, paths: List[str],
                         *, verbose: bool, sample: int = 25
                         ) -> None:
    print(f"  {label}: {len(paths)}")
    if verbose and paths:
        for p in paths[:sample]:
            tag_l = "L" if _has_lrc(p) else "·"
            tag_s = "S" if _has_serato_stems(p) else "·"
            pc = _id3_playcount(p)
            tag_p = f"p{pc}" if pc > 0 else "·"
            print(f"    [{tag_l}{tag_s} {tag_p:>3}] {p}")
        if len(paths) > sample:
            print(f"    … +{len(paths) - sample} more")


def diff_folders(folder_a: str, folder_b: str,
                  *, verbose: bool = False) -> None:
    db_a = os.path.join(folder_a, "database V2")
    db_b = os.path.join(folder_b, "database V2")
    sub_a = os.path.join(folder_a, "Subcrates")
    sub_b = os.path.join(folder_b, "Subcrates")

    print("=" * 72)
    print(f"A: {folder_a}")
    print(f"B: {folder_b}")
    print("=" * 72)

    # --- DB-track diff -----------------------------------------------
    print()
    print("[track index]")
    paths_a = _enumerate_db_paths(db_a)
    paths_b = _enumerate_db_paths(db_b)
    set_a = set(paths_a)
    set_b = set(paths_b)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    both = sorted(set_a & set_b)
    print(f"  total A:    {len(paths_a)}")
    print(f"  total B:    {len(paths_b)}")
    print(f"  in both:    {len(both)}")
    _print_diff_bucket("only in A", only_a, verbose=verbose)
    _print_diff_bucket("only in B", only_b, verbose=verbose)

    # --- Sidecar coverage breakdown ----------------------------------
    print()
    print("[sidecar coverage]")

    def _coverage(paths: List[str], label: str) -> None:
        n = len(paths)
        if not n:
            return
        n_lrc = sum(1 for p in paths if _has_lrc(p))
        n_stems = sum(1 for p in paths if _has_serato_stems(p))
        print(f"  {label}: {n} tracks  → "
              f"LRC={n_lrc} ({100*n_lrc//max(n,1)}%)  "
              f"stems={n_stems} ({100*n_stems//max(n,1)}%)")

    _coverage(paths_a, "A")
    _coverage(paths_b, "B")
    _coverage(both, "in both")
    _coverage(only_a, "only A")
    _coverage(only_b, "only B")

    # --- Crate diff --------------------------------------------------
    print()
    print("[crates]")
    crates_a = _enumerate_crates(sub_a)
    crates_b = _enumerate_crates(sub_b)
    only_crate_a = sorted(set(crates_a) - set(crates_b))
    only_crate_b = sorted(set(crates_b) - set(crates_a))
    common = sorted(set(crates_a) & set(crates_b))
    print(f"  crates A:   {len(crates_a)}")
    print(f"  crates B:   {len(crates_b)}")
    print(f"  common:     {len(common)}")
    if only_crate_a:
        print(f"  only in A ({len(only_crate_a)}):")
        for c in only_crate_a[:20]:
            print(f"    {c}")
    if only_crate_b:
        print(f"  only in B ({len(only_crate_b)}):")
        for c in only_crate_b[:20]:
            print(f"    {c}")
    if common and verbose:
        # Content diff per common crate
        print(f"  common-crate track diff:")
        for c in common[:20]:
            try:
                ta = set(_enumerate_crate_paths(crates_a[c]))
                tb = set(_enumerate_crate_paths(crates_b[c]))
            except Exception:
                continue
            if ta == tb:
                print(f"    [=] {c}  ({len(ta)} tracks identical)")
            else:
                print(f"    [≠] {c}  "
                       f"A={len(ta)} B={len(tb)}  "
                       f"only-A={len(ta-tb)} only-B={len(tb-ta)}")

    print()
    print("=" * 72)
    print("Legend for verbose lines: [L=lrc S=stems p<count>]")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare two _Serato_ config folders. Reports "
                    "track-index diff, subcrate diff, and consumer "
                    "sidecar coverage. Useful for deciding which "
                    "library to keep as canonical when you have "
                    "multiple _Serato_-style folders.")
    ap.add_argument("folder_a",
                     help="First Serato folder "
                          "(e.g. ~/Music/_Serato_)")
    ap.add_argument("folder_b",
                     help="Second Serato folder "
                          "(e.g. ~/Music/_Serato_backup)")
    ap.add_argument("--verbose", "-v", action="store_true",
                     help="Show first 25 paths in each diff bucket "
                          "plus per-crate track-set diffs for "
                          "overlapping crate names.")
    args = ap.parse_args()

    # Ensure imports resolve
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)

    fa = os.path.abspath(os.path.expanduser(args.folder_a))
    fb = os.path.abspath(os.path.expanduser(args.folder_b))
    if not os.path.isdir(fa):
        print(f"ERROR: not a directory: {fa}", file=sys.stderr)
        return 1
    if not os.path.isdir(fb):
        print(f"ERROR: not a directory: {fb}", file=sys.stderr)
        return 1

    diff_folders(fa, fb, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
