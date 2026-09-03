"""sidecaramel.lrc_cusswords — pull cuss-word time windows out of an LRC.

Supports both line-level LRCs (timestamps only at the start of each
line) and word-level LRCs that include per-word inline timestamps
like  `[00:31.44]It's <00:33.20>fuck <00:33.55>Poppa`.

The cuss-word list is English+German (Anglicism-aware). Words like
"asshole" or "motherfucker" are matched as multi-token spans.

Output: a list of (start_s, end_s, word) tuples, plus a CLI mode
that prints them in a copy-pastable form for `flip_writer`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# ============================================================
# Cuss-word dictionary
# ============================================================
# Word-boundary matched, case-insensitive. Order matters — longer
# multi-word forms come first so they consume the bytes before
# single-word matchers run.
#
# Patterns are loaded from `sidecaramel/data/cusswords.json` at
# import-time so users can override the bundled list via
# $SIDECARAMEL_CUSSWORDS pointing to a custom JSON file.

# Fallback / built-in patterns — kept here for emergency-fallback
# if the bundled data file is missing or unparseable.
_BUILTIN_CUSS_PATTERNS = [
    r"motherfuck(?:er|ers|ing|in[' ]?)?",
    r"asshole(?:s)?",
    r"bullshit",
    r"fuck(?:in[g']?|ing|ers?|er|ed)?",
    r"shit(?:ty|s|ting|ted)?",
    r"bitch(?:es|in[g']?)?",
    r"prick(?:s)?",
    r"dick(?:s|head|heads)?",
    r"cock(?:s|sucker)?",
    r"cum(?:ming|med|s)?",
    r"piss(?:ed|ing)?",
    r"slut(?:s|ty)?",
    r"whore(?:s)?",
    r"nigga(?:s|h|hs)?",
    r"hoe(?:s)?",
    r"ass",
    r"schei(?:ß|ss)e?(?:r|n)?",
    r"ficke?n?",
    r"ficker(?:s)?",
    r"hure(?:n)?",
    r"fotze(?:n)?",
    r"wichser(?:n)?",
    r"arschloch(?:es)?",
    r"schwanz",
    r"möse",
    r"schlampe(?:n)?",
    r"hurensohn(?:es)?",
    r"verfickt(?:e|er|es)?",
]


def _load_cuss_patterns() -> List[str]:
    """Load patterns from the bundled data file or env override.

    Resolution order:
      1. $SIDECARAMEL_CUSSWORDS env var → path to JSON file
      2. <package>/data/cusswords.json (bundled with the package)
      3. _BUILTIN_CUSS_PATTERNS (hard-coded fallback)
    """
    env_path = os.environ.get("SIDECARAMEL_CUSSWORDS")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    bundled = Path(__file__).resolve().parent / "data" / "cusswords.json"
    candidates.append(bundled)
    for p in candidates:
        try:
            if p.is_file():
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                patterns = data.get("patterns")
                if isinstance(patterns, list) and patterns:
                    return [str(x) for x in patterns]
        except Exception:
            continue
    return list(_BUILTIN_CUSS_PATTERNS)


CUSS_PATTERNS = _load_cuss_patterns()

CUSS_RE = re.compile(
    r"\b(?:" + "|".join(CUSS_PATTERNS) + r")\b",
    re.IGNORECASE)


# ============================================================
# LRC parsing
# ============================================================

LINE_TS_RE = re.compile(
    r"\[(\d{1,3}):(\d{1,2})(?:\.(\d{1,3}))?\]")
# Extra LRC metadata markers to skip, on top of the generic
# `[tag:value]` line form the parser already drops. Empty by default —
# nothing producer-specific is built in. A pipeline that stamps its own
# marker registers it here, and the match is a substring so a marker
# that trails a line is caught too.
LRC_METADATA_PREFIXES: tuple = ()


def add_lrc_metadata_prefix(prefix: str) -> None:
    """Register an LRC metadata marker to skip, e.g. "[mytool:"."""
    global LRC_METADATA_PREFIXES
    if not prefix:
        raise ValueError("prefix must not be empty")
    if prefix not in LRC_METADATA_PREFIXES:
        LRC_METADATA_PREFIXES = LRC_METADATA_PREFIXES + (prefix,)


# Word-level LRC format: {00:37.98}word
WORD_TS_RE = re.compile(
    r"\{(\d{1,3}):(\d{1,2})(?:\.(\d{1,3}))?\}")
# Some tools also emit |00:16.22| as the re-aligned line timestamp;
# we treat it as just another line-level form.
ALT_LINE_TS_RE = re.compile(
    r"^\|(\d{1,3}):(\d{1,2})(?:\.(\d{1,3}))?\|")


def _ts(m: re.Match) -> float:
    mm = int(m.group(1))
    ss = int(m.group(2))
    frac = m.group(3) or "0"
    # frac string varies in width; treat as decimal portion
    return mm * 60 + ss + int(frac) / (10 ** len(frac))


@dataclass
class LineSpan:
    t_start: float
    t_end: float
    text_clean: str
    # If word-level timing exists, per-word (start_s, end_s, token)
    words: List[Tuple[float, float, str]]


def parse_lrc(text: str) -> List[LineSpan]:
    """Return time-ordered list of LineSpan.

    Word-level LRCs have three rows per logical line:
        [orig_ts] text          ← original lyric line + timestamp
        |aligned_ts| text       ← whisper-re-aligned line timestamp
        {word_ts}word ...        ← per-word timing for the same text

    We pull the per-word `{ts}word` rows when present (most useful for
    cuss-word windows), and fall back to line-level `[orig_ts]` /
    `|aligned_ts|` when no word-row exists for a given line.

    Returns one LineSpan per word-row found, plus one per orphan
    line-row that has no word-row companion.
    """
    pending: List[LineSpan] = []
    for raw in text.splitlines():
        # Drop metadata-only lines, e.g. [ar:Foo], [re:tool]
        if (raw.startswith("[")
                and "]" in raw
                and ":" in raw[:raw.find("]")]
                and not LINE_TS_RE.match(raw)):
            continue
        if any(p in raw for p in LRC_METADATA_PREFIXES):
            continue
        # Stop at JSON trailer blocks
        if raw.startswith("{["):
            break

        # Word-level row "{00:37.98}word ..."
        if raw.lstrip().startswith("{") and WORD_TS_RE.match(raw.lstrip()):
            rest = raw.lstrip()
            words: List[Tuple[float, float, str]] = []
            tokens: List[Tuple[float, str]] = []
            cur = 0
            last_t: Optional[float] = None
            for wm in WORD_TS_RE.finditer(rest):
                pre = rest[cur:wm.start()].strip()
                if pre and last_t is not None:
                    tokens.append((last_t, pre))
                last_t = _ts(wm)
                cur = wm.end()
            tail = rest[cur:].strip()
            if tail and last_t is not None:
                tokens.append((last_t, tail))
            if not tokens:
                continue
            for i, (s, w) in enumerate(tokens):
                e = (tokens[i + 1][0] if i + 1 < len(tokens)
                     else s + 0.6)
                words.append((s, e, w))
            text_clean = re.sub(WORD_TS_RE, " ", rest).strip()
            t_start = words[0][0]
            t_end = words[-1][1]
            pending.append(LineSpan(t_start=t_start, t_end=t_end,
                                       text_clean=text_clean,
                                       words=words))
            continue

        # Aligned-line row "|00:16.22| text"  → only used as fallback
        m = ALT_LINE_TS_RE.match(raw)
        if m:
            continue   # we prefer word-level rows for the same text

        # Original-line row "[00:01.58] text"
        m = LINE_TS_RE.match(raw)
        if not m:
            continue
        t = _ts(m)
        if t > 86400:
            continue
        rest = raw[m.end():].lstrip()
        pending.append(LineSpan(t_start=t, t_end=t + 4.0,
                                  text_clean=rest, words=[]))

    # Compute t_end from next line for any LineSpan that didn't get
    # one (the line-level fallback ones).
    for i, ls in enumerate(pending):
        if not ls.words:
            if i + 1 < len(pending):
                ls.t_end = pending[i + 1].t_start
            else:
                ls.t_end = ls.t_start + 4.0
    return pending


# ============================================================
# Cuss-window extraction
# ============================================================

@dataclass
class CussWindow:
    start_s: float
    end_s: float
    word: str
    line_text: str
    source: str   # "word_level" or "line_estimate"


def estimate_word_offsets(
        text: str, match: re.Match, line_start: float,
        line_duration: float) -> Tuple[float, float]:
    """Estimate (start_s, end_s) of a regex match within a
    line, using char-position / line-length to interpolate.

    line_duration is the gap to the next line. We treat the line
    text as covering ~80% of that gap (most lines have a short
    silence before the next), so estimate t = line_start +
    (char_pos / text_len) * (line_duration * 0.80)."""
    span = 0.80 * line_duration
    n = max(len(text), 1)
    a = line_start + (match.start() / n) * span
    word_chars = max(match.end() - match.start(), 1)
    word_span = max((word_chars / n) * span, 0.18)
    b = a + word_span
    return (a, b)


def extract_cuss_windows(
        spans: List[LineSpan],
        head_pad_s: float = 0.05,
        tail_pad_s: float = 0.05) -> List[CussWindow]:
    """Walk every line, find cuss-word matches, emit CussWindows."""
    out: List[CussWindow] = []
    for ls in spans:
        text = ls.text_clean
        if not CUSS_RE.search(text):
            continue
        line_dur = max(ls.t_end - ls.t_start, 0.5)
        # If we have word-level timing, prefer it
        if ls.words:
            for s, e, w in ls.words:
                clean = re.sub(r"[^A-Za-zÄÖÜäöüß']+", " ",
                                 w).strip()
                if not clean:
                    continue
                m = CUSS_RE.search(clean)
                if m:
                    out.append(CussWindow(
                        start_s=max(s - head_pad_s, 0.0),
                        end_s=e + tail_pad_s,
                        word=m.group(0),
                        line_text=ls.text_clean,
                        source="word_level"))
        else:
            for m in CUSS_RE.finditer(text):
                a, b = estimate_word_offsets(
                    text, m, ls.t_start, line_dur)
                out.append(CussWindow(
                    start_s=max(a - head_pad_s, 0.0),
                    end_s=b + tail_pad_s,
                    word=m.group(0),
                    line_text=ls.text_clean,
                    source="line_estimate"))
    out.sort(key=lambda w: w.start_s)

    # Dedup. The pipeline emits an [orig_ts] row AND a {word_ts}word
    # row for each logical line, which gets us both a line_estimate
    # and a word_level hit for the same word. Drop the line_estimate
    # whenever a word_level entry exists for the same (word, line).
    kept: List[CussWindow] = []
    word_level_lookup = {
        (w.word.lower(), w.line_text): w
        for w in out if w.source == "word_level"
    }
    for w in out:
        key = (w.word.lower(), w.line_text)
        if w.source == "line_estimate" and key in word_level_lookup:
            continue
        # Also drop a line_estimate whose line text differs only by
        # whitespace from a word-level line we've already accepted.
        if w.source == "line_estimate":
            squashed = " ".join(w.line_text.split()).lower()
            dup = False
            for wl_key, wl in word_level_lookup.items():
                if wl.word.lower() == w.word.lower() and \
                        " ".join(wl.line_text.split()).lower() == squashed:
                    dup = True
                    break
            if dup:
                continue
        kept.append(w)
    kept.sort(key=lambda w: w.start_s)
    return kept


# ============================================================
# CLI
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract cuss-word time windows from a word-level "
                    "LRC. Prints flip_writer --censor args.")
    ap.add_argument("lrc", help="Path to LRC file")
    ap.add_argument("--head-pad", type=float, default=0.05)
    ap.add_argument("--tail-pad", type=float, default=0.05)
    ap.add_argument("--format", choices=("table", "shell", "json"),
                     default="table")
    args = ap.parse_args()
    with open(args.lrc, "r", encoding="utf-8") as f:
        text = f.read()
    spans = parse_lrc(text)
    windows = extract_cuss_windows(spans,
                                       head_pad_s=args.head_pad,
                                       tail_pad_s=args.tail_pad)
    if args.format == "table":
        print(f"{'#':<3}  {'start_s':>8}  {'end_s':>8}  "
              f"{'word':<14}  {'source':<14}  line")
        print("-" * 100)
        for i, w in enumerate(windows):
            print(f"{i:<3}  {w.start_s:8.3f}  {w.end_s:8.3f}  "
                  f"{w.word:<14}  {w.source:<14}  "
                  f"{w.line_text[:60]}")
    elif args.format == "shell":
        for w in windows:
            print(f"--censor {w.start_s:.3f}:{w.end_s:.3f}",
                   end=" \\\n")
    elif args.format == "json":
        import json
        print(json.dumps([
            {"start_s": w.start_s, "end_s": w.end_s,
             "word": w.word, "source": w.source,
             "line": w.line_text}
            for w in windows], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
