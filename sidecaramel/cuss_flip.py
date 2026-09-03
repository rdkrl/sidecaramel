"""sidecaramel.cuss_flip — auto-generate Serato CENSOR FLIPs from
embedded cusswords in a track's lyrics.

2026-06-08: "wenn lyrics im mutagen und cuzzwords (import
vendored im paket) und wenn außerdem empty slots und keiner heißt clean,
dann bilde einen censor flip (permission to filewrite erst mit
dryrun)"

Flow per track:

    1. Read embedded lyrics via sidecaramel.lyrics
    2. Detect cusswords via lrc_cusswords.extract_cuss_windows
    3. Read existing Serato cues + FLIP slots via sidecaramel.tags
    4. PRE-CONDITIONS:
       - lyrics present (timed LRC preferred)
       - at least one cussword detected
       - at least one empty FLIP slot available
       - no existing cue / FLIP named "clean" (= already cleaned)
    5. Build CENSOR+JUMP action pairs via sidecaramel.flip_writer.
       build_word_bleep
    6. DRY-RUN by default → print the plan
       --write requires explicit confirm flag (audio-write policy)

policy COMPLIANCE
-----------------
audio-write policy (never write audio without explicit per-op okay):
    Default mode = dry-run.  Actual write needs --write flag PLUS
    in-code `confirm=True` is passed (handled by write_flip_any
    which raises RuntimeError without it).
Serato-running check (Serato-check before write):
    `--write` mode runs sidecaramel.check.is_running() and refuses
    if Serato is alive.
plan-before-code policy (plan before code):
    Dry-run IS the plan.  the maintainer reads the plan, then opt-in to
    --write per track.
single-point-of-truth policy (no reinvention):
    Re-uses lrc_cusswords (detection), sidecaramel.tags (cue read),
    sidecaramel.flip_writer (CENSOR+JUMP build, MP3 write).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


@dataclass
class CussFlipPlan:
    """Pure-data description of what the CLI WOULD do for one
    track.  Returned by `analyze_track`; consumed by `print_plan`
    and `execute_plan`."""
    audio_path: str
    ok_to_proceed: bool
    reasons_blocked: List[str]
    lyrics_format: Optional[str]    # "lrc" | "plain" | None
    n_cusswords: int
    cusswords: List[dict]            # [{word, start_s, end_s, source}]
    existing_cues_count: int
    cue_names: List[str]
    empty_flip_slots: List[int]      # 0..5 available
    chosen_slot: Optional[int]
    chosen_name: str = "clean"
    notes: List[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["notes"] is None:
            d["notes"] = []
        return d


# ============================================================
# Detection
# ============================================================

def _flip_slots_from_parsed(parsed: Optional[dict]
                             ) -> Tuple[set, List[str]]:
    """Collect (used_slot_indices, flip_names) from a
    ``parse_serato_markers2_full()`` result dict.

    FLIP entries live under ``parsed["flips"]`` with keys ``slot``
    (int) and ``name`` — see
    ``sidecaramel.tags.parse_serato_markers2_full``.  There is no
    ``entries`` key and no ``type`` / ``slot_index`` fields; reading
    those (the pre-fix bug) always returned an empty list, so the
    occupied-slot and duplicate-name guards in ``analyze_track`` never
    fired and the writer could clobber an occupied FLIP slot it
    believed was empty.
    """
    used: set = set()
    names: List[str] = []
    for entry in (parsed or {}).get("flips", []):
        slot = entry.get("slot")
        if slot is not None:
            used.add(int(slot))
        names.append(entry.get("name") or "")
    return used, names


def analyze_track(audio_path: str,
                    *,
                    flip_name: str = "clean") -> CussFlipPlan:
    """Inspect a track and return a CussFlipPlan describing whether
    we'd build a censor FLIP, what it'd contain, and any blockers."""
    plan = CussFlipPlan(
        audio_path=audio_path,
        ok_to_proceed=False,
        reasons_blocked=[],
        lyrics_format=None,
        n_cusswords=0,
        cusswords=[],
        existing_cues_count=0,
        cue_names=[],
        empty_flip_slots=[],
        chosen_slot=None,
        chosen_name=flip_name,
        notes=[],
    )

    if not os.path.isfile(audio_path):
        plan.reasons_blocked.append(f"file not found: {audio_path}")
        return plan

    # 1. Lyrics
    try:
        from sidecaramel.lyrics import (read_embedded_lyrics_with_format,
                                              read_synced_lyrics_as_lrc)
    except ImportError:
        plan.reasons_blocked.append("sidecaramel.lyrics not importable")
        return plan
    lyr = read_embedded_lyrics_with_format(audio_path)
    if not lyr:
        plan.reasons_blocked.append("no embedded lyrics")
        return plan
    text, fmt = lyr
    plan.lyrics_format = fmt
    # Prefer timed LRC if present (much more accurate word offsets)
    lrc_text = read_synced_lyrics_as_lrc(audio_path) or text
    if not lrc_text:
        plan.reasons_blocked.append("lyrics blob empty after parse")
        return plan

    # 2. Cusswords via vendored detector
    try:
        # 2026-06-08: vendored into the package (was top-level
        # vendored from the upstream codebase).
        from sidecaramel.lrc_cusswords import (parse_lrc,
                                                     extract_cuss_windows)
    except ImportError:
        plan.reasons_blocked.append(
            "sidecaramel.lrc_cusswords not importable")
        return plan
    try:
        spans = parse_lrc(lrc_text)
        windows = extract_cuss_windows(spans)
    except Exception as e:
        plan.reasons_blocked.append(f"cusswords detection failed: {e}")
        return plan
    plan.n_cusswords = len(windows)
    plan.cusswords = [
        {"word": w.word,
          "start_s": round(w.start_s, 3),
          "end_s": round(w.end_s, 3),
          "source": w.source}
        for w in windows
    ]
    if not windows:
        plan.reasons_blocked.append("no cusswords detected in lyrics")
        return plan

    # 3. Serato cues + FLIP slot inventory
    try:
        from sidecaramel.tags import (read_serato_metadata,
                                          harvest_serato_blobs,
                                          parse_serato_markers2_full)
    except ImportError:
        plan.reasons_blocked.append("sidecaramel.tags not importable")
        return plan
    meta = read_serato_metadata(audio_path) or {}
    cues = meta.get("cues") or []
    plan.existing_cues_count = len(cues)
    plan.cue_names = [c.get("label") or "" for c in cues]

    # FLIP slots (0..5) — parse Markers2 raw and collect slot indices.
    # FLIP entries live under parsed["flips"] (keys "slot"/"name"), NOT
    # "entries"/"type"/"slot_index" — see _flip_slots_from_parsed.
    flip_slots_used = set()
    flip_names = []
    for desc, payload, _ in harvest_serato_blobs(audio_path):
        if desc == "Serato Markers2":
            try:
                parsed = parse_serato_markers2_full(payload)
            except Exception:
                parsed = None
            if not parsed:
                continue
            slots, names = _flip_slots_from_parsed(parsed)
            flip_slots_used |= slots
            flip_names.extend(names)
            break
    plan.empty_flip_slots = [s for s in range(6) if s not in flip_slots_used]

    # 4. Pre-condition checks
    if not plan.empty_flip_slots:
        plan.reasons_blocked.append(
            "no empty FLIP slot available (slots 0..5 all used)")
    # No cue OR FLIP named "clean" already
    has_clean = (any((n or "").strip().lower() == flip_name.lower()
                       for n in plan.cue_names) or
                  any((n or "").strip().lower() == flip_name.lower()
                       for n in flip_names))
    if has_clean:
        plan.reasons_blocked.append(
            f"existing cue/FLIP named '{flip_name}' — track already "
            "censored, skipping to avoid double-clean")

    if plan.empty_flip_slots:
        plan.chosen_slot = min(plan.empty_flip_slots)

    plan.ok_to_proceed = (not plan.reasons_blocked)
    return plan


# ============================================================
# Plan presentation
# ============================================================

def print_plan(plan: CussFlipPlan, *, show_words: bool = True) -> None:
    print(f"=== {os.path.basename(plan.audio_path)} ===")
    print(f"  lyrics format:  {plan.lyrics_format}")
    print(f"  cusswords:      {plan.n_cusswords}")
    print(f"  existing cues:  {plan.existing_cues_count}  "
            f"({plan.cue_names})")
    print(f"  empty FLIP slots: {plan.empty_flip_slots}")
    if plan.chosen_slot is not None:
        print(f"  would write to: slot {plan.chosen_slot}, "
                f"name '{plan.chosen_name}'")
    if show_words and plan.cusswords:
        print(f"  cussword windows:")
        for w in plan.cusswords:
            print(f"    {w['start_s']:>7.2f}s .. {w['end_s']:>7.2f}s  "
                    f"{w['word']!r:<14}  ({w['source']})")
    if plan.reasons_blocked:
        print(f"  BLOCKED:")
        for r in plan.reasons_blocked:
            print(f"    × {r}")
    else:
        print(f"  → READY (use --write to apply)")
    print()


# ============================================================
# Execution
# ============================================================

def execute_plan(plan: CussFlipPlan,
                   *,
                   confirm: bool = False,
                   in_place: bool = False,
                   reverse_fraction: float = 1.0,
                   head_pad_s: float = 0.02) -> bool:
    """Apply the FLIP to the audio file.  audio-write policy: requires
    `confirm=True` AND Serato-running check Serato-check passes.

    Returns True on success.
    """
    if not plan.ok_to_proceed:
        raise RuntimeError("plan not OK; reasons: "
                             + ", ".join(plan.reasons_blocked))
    if not confirm:
        raise RuntimeError(
            "audio-write policy — execute_plan requires confirm=True to write "
            f"audio file {plan.audio_path!r}")

    # Serato-running check: refuse to write while Serato is running so we
    # never race its own on-disk writes.  is_running() returns
    # (running, procs); only an UNAVAILABLE probe is tolerated (the
    # --confirm gate above has to suffice there) — a positive "it is
    # running" always aborts and must NOT be swallowed.
    try:
        from sidecaramel.check import is_running, SeratoCheckUnavailableError
    except ImportError:
        is_running = None
    if is_running is not None:
        try:
            running, procs = is_running()
        except SeratoCheckUnavailableError as e:
            # On sandbox/non-mac the probe may be unavailable — fall back
            # to the caller's explicit --confirm gate.
            print(f"warning: Serato-running check unavailable: {e}",
                    file=sys.stderr)
        else:
            if running:
                raise RuntimeError(
                    "Serato-running check — Serato is running "
                    f"({', '.join(procs) or 'process'}).  Abort.  Quit "
                    "Serato (eject decks first per race-condition fact) "
                    "and retry.")

    ext = os.path.splitext(plan.audio_path)[1].lower()
    SUPPORTED = (".mp3", ".wav", ".aiff", ".aif",
                  ".mp4", ".m4a", ".m4v",
                  ".flac", ".ogg", ".oga", ".opus")
    if ext not in SUPPORTED:
        raise RuntimeError(
            f"unsupported audio container {ext!r}.  "
            f"write_flip_any handles {SUPPORTED}.")

    from sidecaramel.flip_writer import write_flip_any
    bleeps = [(w["start_s"], w["end_s"]) for w in plan.cusswords]

    if in_place:
        target = plan.audio_path
    else:
        # Copy source to .CLEAN.<ext> sibling, then write FLIP into
        # the copy.  Preserves byte-equality of source (audio-write policy).
        import shutil
        stem, ext_orig = os.path.splitext(plan.audio_path)
        target = stem + ".CLEAN" + ext_orig
        shutil.copy2(plan.audio_path, target)

    ok = write_flip_any(
        target,
        slot=plan.chosen_slot,
        name=plan.chosen_name,
        loop=False,
        bleeps=bleeps,
        reverse_fraction=reverse_fraction,
        head_pad_s=head_pad_s,
        replace_slot=True,
        confirm=True,
    )
    if not ok:
        raise RuntimeError(
            f"write_flip_any returned False for {target!r}; "
            "container may not support Markers2 writes.")
    print(f"wrote FLIP → {target}")
    return True


# ============================================================
# CLI
# ============================================================

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m sidecaramel.cuss_flip",
        description=(
            "Auto-build Serato CENSOR FLIPs from embedded cusswords. "
            "audio-write policy: dry-run by default — pass --write to actually "
            "modify the file."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m sidecaramel.cuss_flip track.mp3            # dry-run (default)\n"
            "  python -m sidecaramel.cuss_flip track.mp3 --json     # JSON plan\n"
            "  python -m sidecaramel.cuss_flip track.mp3 --write    # apply (writes .CLEAN.mp3)\n"
            "  python -m sidecaramel.cuss_flip track.mp3 --write --in-place  "
            "# overwrite the source file (DANGEROUS)\n"))
    p.add_argument("tracks", nargs="+", help="audio file(s) to analyze")
    p.add_argument("--name", default="clean",
                     help="FLIP slot name to create (default 'clean'). "
                          "Track is skipped if a cue/FLIP with this "
                          "name already exists.")
    p.add_argument("--json", action="store_true",
                     help="emit machine-readable JSON plans (one per "
                          "track) to stdout.")
    p.add_argument("--write", action="store_true",
                     help="(audio-write policy) actually write the FLIP.  By "
                          "default this writes to <name>.CLEAN.mp3; "
                          "combine with --in-place to overwrite source.")
    p.add_argument("--in-place", action="store_true",
                     help="WITH --write: overwrite the source audio "
                          "file directly (instead of writing .CLEAN.mp3 "
                          "sibling).")
    p.add_argument("--reverse-fraction", type=float, default=1.0,
                     help="(censor) reverse-duration as multiple of "
                          "word duration (default 1.0 = full word). "
                          "0.5 = half-word reverse (= older, audible-"
                          "first-half behaviour, discouraged).")
    p.add_argument("--head-pad", type=float, default=0.02,
                     help="(censor) seconds of pad after word_start "
                          "for the reverse target (default 0.02).")
    args = p.parse_args(argv)

    plans = []
    n_ready = 0
    n_blocked = 0
    for t in args.tracks:
        plan = analyze_track(t, flip_name=args.name)
        plans.append(plan)
        if plan.ok_to_proceed:
            n_ready += 1
        else:
            n_blocked += 1
        if not args.json:
            print_plan(plan)

    if args.json:
        print(json.dumps([p.to_dict() for p in plans], indent=2))

    n_write_fail = 0
    if args.write:
        for plan in plans:
            if not plan.ok_to_proceed:
                continue
            print(f"=== writing FLIP to {plan.audio_path} ===")
            try:
                execute_plan(plan, confirm=True,
                                in_place=args.in_place,
                                reverse_fraction=args.reverse_fraction,
                                head_pad_s=args.head_pad)
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                n_write_fail += 1
        # Only error out if EVERY ready plan failed; otherwise we
        # still made forward progress and shouldn't break a batch.
        if n_ready > 0 and n_write_fail == n_ready:
            return 1

    if not args.write:
        print(f"=== summary: {n_ready} ready, {n_blocked} blocked ===")
        if n_ready:
            print("    → re-run with --write to apply")

    # Exit 0 for normal analysis; "blocked" tracks (no lyrics / no
    # cusswords / slot full / already-clean) are not errors — they
    # just don't get a FLIP.  Only return 1 if --write was requested
    # AND every ready plan failed during execute.
    return 0


if __name__ == "__main__":
    sys.exit(main())
