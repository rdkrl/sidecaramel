# sidecaramel

> **Alpha — not thoroughly tested.  Use at your own risk!**

Read and write [Serato DJ Pro](https://serato.com/dj) tag formats, embedded lyrics, and cover-art across MP3, WAV, AIFF, MP4, M4A, M4V, FLAC, OGG, and Opus.  Plus a `database V2` / `.crate` reader and writer, a History `*.session` parser, a `.serato-stems` sidecar decoder, and an audio-direct waveform renderer.

**Status**: 0.1.0.  Alpha — not thoroughly tested; the API may change in any `0.x` bump.  Format support is reverse-engineered for interoperability.  Not affiliated with, endorsed by, or sponsored by Serato Limited.

**Platforms**: tag readers and the waveform renderer are cross-platform.  The `.crate` and `database V2` writers gate on a Serato-process check, implemented for **macOS** (via `pgrep`) and **Windows** (via `tasklist`).  On Linux and other hosts the probe raises `SeratoCheckUnavailableError`, and the writers refuse unless you pass `serato_known_closed=True`.

**License**: GPL-2.0-or-later — matching `mutagen`, the **core's** only runtime dependency.  The optional extras add their own libraries (`[render]` → numpy/scipy/soundfile/Pillow, `[gui]` → PySide6, `[connector]` → mcp), and stem extraction / audio-direct Overview also shell out to the external `ffmpeg` binary.  See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

---

## Install

```bash
pip install sidecaramel              # core
pip install 'sidecaramel[render]'    # + numpy/scipy/soundfile/Pillow for the waveform renderer
pip install 'sidecaramel[gui]'       # + PySide6 (render stack) for the drag-and-drop track viewer
pip install 'sidecaramel[all]'       # everything
```

From source:

```bash
cd sidecaramel
pip install -e .
```

---

## What it covers

| Surface | Read | Write |
|---|---|---|
| **Serato blobs** — Markers2, BeatGrid, Autotags, Analysis, Overview, FLIP | yes | yes (gated) |
| **Serato Markers_ V1** | yes (all containers) | yes for MP3/WAV/AIFF/MP4/M4A/M4V |
| **`.crate`** | yes | yes (gated) |
| **`database V2`** | yes | yes (gated, round-trip-safe via per-record `_raw` byte preservation) |
| **History `*.session`** | yes | – |
| **Embedded lyrics (untimed)** — USLT / ©lyr / Vorbis `LYRICS` | yes | yes (`write_lyrics`) |
| **Embedded lyrics (timed)** — SYLT | yes | yes (`write_sylt`, MP3/WAV/AIFF only) |
| **`.lrc` sidecar** — timed lyrics companion file | yes | yes (`read_lrc_sidecar` / `write_lrc_sidecar`, any container) |
| **Cover-art** — APIC / covr / FLAC Picture / Vorbis `METADATA_BLOCK_PICTURE` | yes | yes |
| **Audio-direct waveform** — per-column STFT, R=bass G=mid B=treble, any resolution | – | renders PNG |
| **Overview-blob waveform** — grayscale / serato_palette / serato_hue / rgb332 | renders | encodes |

Container coverage: MP3, WAV, AIFF (ID3 GEOB); MP4 / M4A / M4V (freeform atoms); FLAC / OGG / Opus (Vorbis `SERATO_*` comments).  Unified `*_any()` dispatcher routes per extension.

---

## Quick example

```python
import sidecaramel

# Read — returns None for a file with no Serato data (or an unsupported
# container); raises FileNotFoundError for a nonexistent path.
meta = sidecaramel.read_serato_metadata("track.mp3")
if meta:
    print(meta["bpm"], len(meta["cues"]), len(meta["loops"]))

# Write — every audio-writer requires confirm=True
sidecaramel.write_serato_markers_full_mp3(
    "track.mp3",
    cues=[{"idx": 0, "pos_ms": 0,    "color": (204, 136, 0), "label": "intro"}],
    loops=[{"idx": 0, "start_ms": 17890, "end_ms": 21320,
            "color": (39, 170, 225), "label": "drop", "locked": True}],
    confirm=True,
)

# Lyrics — untimed write is container-aware (USLT / ©lyr / Vorbis LYRICS).
from sidecaramel.lyrics import read_synced_lyrics_as_lrc, write_lyrics
print(read_synced_lyrics_as_lrc("track.flac"))   # → LRC text or None
write_lyrics("track.mp3", "Some lyrics", confirm=True)

# Timed lyrics — ID3 SYLT (MP3/WAV/AIFF) or a container-agnostic .lrc sidecar
from sidecaramel.lyrics import write_sylt, write_lrc_sidecar, read_lrc_sidecar
pairs = [(0, "Some lyrics"), (2000, "right on time")]
write_sylt("track.mp3", pairs, confirm=True)
write_lrc_sidecar("track.flac", pairs, confirm=True)   # → writes track.lrc
print(read_lrc_sidecar("track.flac"))                  # → same pairs back

# Cover-art
from sidecaramel.art import read_cover_art, write_cover_art
mime, data = read_cover_art("track.m4a") or (None, None)
write_cover_art("track.mp3", open("cover.jpg", "rb").read(), confirm=True)
```

Cue / loop dict schema for the markers writers:

```python
cue  = {"idx": int, "pos_ms": int, "color": (r, g, b),       "label": str}
loop = {"idx": int, "start_ms": int, "end_ms": int,
        "color": (r, g, b), "label": str, "locked": bool}
```

CLI mirrors the read side:

```bash
sidecaramel read   track.mp3
sidecaramel cues   track.m4a
sidecaramel loops  track.flac
sidecaramel lyrics track.wav
sidecaramel art    track.mp3 --out cover.jpg
sidecaramel blobs  track.mp3              # cross-container Serato-blob harvest
```

(`sidecaramel inspect` is a deprecated alias for `blobs`.)

---

## Audio-direct waveform — any resolution

`sidecaramel.audio_render.render_big_waveform` produces a full-track waveform straight from the audio at any `width × height`.  Per column it does a short FFT, splits the energy into bass / mid / treble bands, and paints them additively as R / G / B.

```python
from sidecaramel.audio_render import render_big_waveform_png

png = render_big_waveform_png("track.flac",
                                 width=1568, height=200,
                                 bass_boost=1.4, dynamics=1.7,
                                 axis_deg=135)
```

![A full-track waveform rendered by `render_big_waveform_png` — per-column bass / mid / treble energy painted additively as red / green / blue.](https://raw.githubusercontent.com/rdkrl/sidecaramel/main/docs/example-waveform.png)

`axis_deg` picks one of four distinct vinyl-axis projections (`0`, `45`, `90`, `135`, **default `135`**); under abs-amplitude rendering only those four are visually distinguishable.  The default is shared by `render_big_waveform`, `render_big_waveform_png`, the `sidecaramel-render` CLI, and the GUI viewer.  See the `audio_render` docstring for the projection math.

---

## Additional modules

Beyond the read/write core and the renderers, the package ships a few higher-level tools.  They are functional but younger than the Serato-IO core — **preview-grade, expect rougher edges**:

| Module | What it does | Extra deps |
|---|---|---|
| `sidecaramel.gui` | PySide6 drag-and-drop track viewer — waveform + beatgrid + cue overlay (`python -m sidecaramel.gui [track]`) | `sidecaramel[gui]` (PySide6 + render) |
| `sidecaramel.window_api` | Windowed-waveform interface for live overlays (e.g. an external `overlay.html` L-view consumer) | `sidecaramel[render]` |
| `sidecaramel.stems` | `.serato-stems` sidecar decoder — all four stem types | `sidecaramel[render]` |
| `sidecaramel.cuss_flip` + `sidecaramel.lrc_cusswords` | Auto-generate Serato CENSOR FLIPs from cusswords found in embedded / LRC lyrics | core |
| `sidecaramel.loops_extract` | Write each Serato loop region out as a separate audio file | `sidecaramel[render]` |
| `sidecaramel.flip_render` | Bake a Serato CENSOR FLIP into new audio | `sidecaramel[render]` |

---

## Extension points

Nothing producer-specific is built in. A pipeline that writes its own
sidecars registers them at start-up, and the defaults stay format-native:

```python
from sidecaramel import consolidate, lrc_cusswords, stems

consolidate.add_companion_suffix(".mytool-anchors.json")  # moved with the audio
stems.add_external_stem_suffix(".mytool-vocals.wav")      # swept up on cleanup
lrc_cusswords.add_lrc_metadata_prefix("[mytool:")         # skipped when parsing LRC
```

Defaults: `consolidate.SIDECAR_SUFFIXES` covers `.lrc`, `.serato-stems`
and the four `.serato-*.wav` stem artefacts (taken from
`sidecaramel.stems`, so the two lists cannot drift apart);
`lrc_cusswords.LRC_METADATA_PREFIXES` is empty, since the parser already
drops well-formed `[tag:value]` lines.

---

## Testing against real Serato data

Two opt-in environment variables point the golden tests at files Serato
itself wrote — the only way to check the decoders against something
other than this package's own output:

```bash
SIDECARAMEL_TEST_SIDECAR=/path/to/track.serato-stems \
SIDECARAMEL_TEST_AUDIO=/path/to/tagged-track.mp3 \
    pytest sidecaramel/tests -q
```

Without them those tests skip and the suite runs on synthetic fixtures.

A larger set of real Serato-tagged + blank baseline audio files lives on the
`gold-dropbox` branch (checked out to `roundtrip/`, git-ignored). When present,
`test_serato_audio_roundtrip.py` drives the full write path against them; it
skips when they are absent.

---

## Byte-for-byte fidelity

For the data-bearing blobs, `sidecaramel` reproduces the **exact bytes** Serato
DJ Pro wrote — checked against real Serato output, not just self-round-trips.
`test_serato_audio_roundtrip.py` goes end-to-end through real audio files: read a
Serato-tagged track, re-synthesise each blob from the decoded data, write it into
a copy of a *blank* baseline through the real (mutagen) writers, read the file
back, and byte-compare. Byte-identical after that round-trip, across MP3/WAV/MP4:

| blob | reproduced |
|------|-----------|
| `Serato Markers2` | the **full outer** GEOB/atom (base64 wrapped at 72 + NUL reserve) — cues, loops, labels, colours, BPM-lock, FLIPs |
| `Serato Markers_` (V1) | both variants: 7-bit-masked (MP3/WAV) and raw (MP4), incl. the per-loop `locked` flag |
| `Serato BeatGrid` | the exact f32 anchor position + BPM |
| `Serato Autotags` | BPM + auto-gain |
| `Serato Analysis` | the version marker — 2-byte (`02 01`, MP3/WAV) and 3-byte (`00 01 00`, MP4) |
| `.serato-stems` sidecar | the container: `srtshead` header + `stem` chunk framing |

A cue/loop edit is also **non-destructive**: rewriting only `Serato Markers2`
into a real tagged MP4 leaves every other blob — including the MP4-only
`RelVolAd` / `VidAssoc` / `playcount`, which have no builder — byte-identical, and
keeps all container atoms. And the sidecar's `+12` payload boundary is confirmed
by the LAME gapless identity holding on the real file:
`audio_frames × 1152 − encoder_delay − end_padding == total_samples`.

Library files round-trip too: `read_crate` → `write_crate` reproduces a real
Serato `.crate` byte-for-byte — the `vrsn` / `osrt` / `ovct` / `otrk`→`ptrk` TLV
envelope, the volume-relative path convention, and the column spec all
re-serialise exactly (`test_serato_crate_roundtrip.py`). Crates live in
`_Serato_/Subcrates/` (despite the name, these are the current crates; a
top-level `_Serato_/Crates/` folder is legacy ScratchLive-era).

### Round-trip vs. green-field

Byte-parity holds for everything that is a **function of the decoded data** —
parse → re-synthesise is a faithful inverse. It does *not* automatically hold for
the writer's **free parameters**: values that carry no data and that Serato
simply picks —

- the **Markers2 GEOB NUL reserve** (Serato leaves spare room in the tag: 426 B /
  200 B / 6 B in the sample MP3/WAV/MP4 — the amount doesn't follow from the
  cues/loops), and
- the **stem MP3 bytes** (they depend on the exact encoder, not on "the data").

A *round-trip* reproduces these only because it **copies them from the source it
just read** — the original blob's length fixes the reserve size, and Serato's stem
payloads are passed through verbatim. A *green-field* write (no prior tag) has
nothing to copy them from, so it must invent them, and Serato's choices aren't
predictable: the data would be identical, only the reserve size / encoder bytes
differ. `Serato Overview` is green-field by nature — recomputed from the audio by
Serato's waveform algorithm — which is why it is the one data-bearing blob not
reproduced byte-exact.

### The `.serato-stems` chunk length

Each `stem` chunk is `b"stem"` + a 4-byte big-endian length + that many bytes,
where the length **counts the 4-byte stem index/type that prefixes the MP3**
(`length = 4 + len(mp3)`, so the MP3 frames start at chunk offset `+12`, not `+8`
or `+16`). Reading the length as "MP3 only" eats the first frame — LAME's `Xing`
header — and silently drops 1152 samples (26 ms) of gapless alignment off the
front. Cross-checked against Good Clean Stems' own `serato_stems.py`, which frames
the identical bytes as `size = len(payload)` with the stem index at `payload[:4]`.

---

## Safety

Every audio-writer requires `confirm=True`.  Without it, `RuntimeError` is raised.  `.crate` and `database V2` writers additionally refuse to write while a Serato process is running.  The explicit-override kwarg is `serato_known_closed=True` (= "I have confirmed Serato is closed"); the legacy spelling `allow_serato_running=True` is still accepted but emits `DeprecationWarning`.  Back up any track you care about before pointing a writer at it.

---

## Preserving existing markers

`write_serato_markers_full_mp3` / `_mp4` default to **`preserve=True`**: when the
target already carries a Serato Markers2 blob, a cue/loop write MERGES rather
than rebuilds.  Only the entry types you explicitly pass are replaced — AUTOGAIN,
any unknown / future Serato entry, existing FLIPs, and a locked BPMLOCK survive
byte-for-byte.  `bpm_lock` now defaults to `None` ("leave the grid lock as-is");
it no longer writes an unlocked BPMLOCK by default.  Pass `preserve=False` to
rebuild the marker stream from scratch (the old behaviour), or call
`flip_writer.merge_markers2_inner()` to drive the splice directly.

```python
# Add cues to a track that already has a locked grid + AUTOGAIN —
# both survive; only the cues change.
sidecaramel.write_serato_markers_full_mp3(
    "track.mp3",
    cues=[{"idx": 0, "pos_ms": 0, "color": (204, 136, 0), "label": "intro"}],
    confirm=True)            # preserve=True is the default
```

---

## MCP connector

`pip install sidecaramel[connector]` adds a local MCP server
(`sidecaramel-mcp`, stdio) that exposes the library to MCP clients:
read tools for metadata, blobs, sidecars, database listings and
Overview renders, plus gated write tools. Every write refuses while
Serato runs and fails closed when that cannot be determined;
destructive operations require their own explicit confirmation on top.

### Hooking it up

`sidecaramel-mcp` is a local stdio server. Most MCP hosts launch it directly
as a command; hosts that speak only remote HTTP MCP (ChatGPT, the github.com
coding agent) need it fronted by an stdio→HTTP bridge. Generic config (put
`sidecaramel-mcp` on your `PATH`, or point `command` at your venv's `python`
with `"args": ["-m", "sidecaramel.connector"]`):

```json
{
  "mcpServers": {
    "sidecaramel": { "command": "sidecaramel-mcp" }
  }
}
```

| Host | Where the config lives |
|---|---|
| ChatGPT | remote/HTTP only — front it with an stdio→HTTP MCP bridge, then add that URL as a connector (Developer Mode) |
| Claude Code | `claude mcp add sidecaramel -- sidecaramel-mcp`, or `.mcp.json` in the project |
| Claude Desktop | `claude_desktop_config.json` (macOS `~/Library/Application Support/Claude/`, Windows `%APPDATA%\Claude\`) → `mcpServers`; restart the app |
| Cline / Roo (VS Code) | the extension's `cline_mcp_settings.json` → `mcpServers` |
| Continue.dev | the assistant config → `mcpServers` |
| Cursor | `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (project) → `mcpServers` |
| GitHub Copilot | in VS Code / JetBrains via the editor's `mcp.json` (stdio); the github.com coding agent takes remote MCP servers in repo settings |
| VS Code (Copilot agent) | `.vscode/mcp.json` → `"servers": { "sidecaramel": { "type": "stdio", "command": "sidecaramel-mcp" } }` |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` → `mcpServers` |
| Zed | `settings.json` → `context_servers` with `command` |

Config-file names and keys vary by host version (VS Code uses `servers`, not
`mcpServers`); the stdio `command` / `args` shape is the constant — check your
host's current docs if unsure.

## Serato DJ 4.0

Serato DJ 4.0 (2025) moved the library store to an SQLite database and
marks `database V2` and the crate files [LEGACY]. Per-file data — the
Serato blobs in tags and the `.serato-stems` sidecars — is unchanged,
so everything in `tags.py`, `stems.py` and the Overview tooling works
as before. The `db.py` / crate / consolidate paths target the LEGACY
store: they still read it, but on a 4.0 library their writes are not
authoritative. `db.detect_library_format()` (and the connector's
`serato_library_format` tool) classifies a folder before any write.

## Prior work

`sidecaramel`'s reverse-engineering builds on public prior art rather than starting from a blank binary dump:

- **[Holzhaus/serato-tags](https://github.com/Holzhaus/serato-tags)** — the primary reference for Serato's GEOB tag layouts (Markers2, Markers_ V1, BeatGrid, Overview, and more). Several layout constants and byte-offset choices in `blobs.py` and `overview.py` are credited to this documentation inline (`# github.com/Holzhaus/serato-tags`, `Holzhaus-verified`) and, in at least one case in `tags.py`, corrected against it where a real file disagreed with the published spec.
- **[Holzhaus/triseratops](https://github.com/Holzhaus/triseratops)** — a from-scratch Rust reimplementation of the same formats, by the same author; useful as a second independent parser to cross-check edge cases against.
- **[Mixxx](https://github.com/mixxxdj/mixxx)** — the open-source DJ application independently reverse-engineered Serato's Markers2/BeatGrid formats for its own Serato-library import feature ([2021 release notes](https://mixxx.org/news/2021-02-08-new-in-2-3-serato-support/); see also the project's [Reverse-Engineering wiki page](https://github.com/mixxxdj/mixxx/wiki/Reverse-Engineering)).
- **[Good Clean Stems](https://github.com/Fotsbeats/GOOD-CLEAN-STEMS)** — a separate, independently authored Serato-sidecar-building app. Its repository is public, but the software itself is licensed proprietary ("for evaluation and testing") per its own README — a different kind of debt than the three sources above. `stems_encode`'s payload-conditioning rules (bare MP3 stream, no ID3 wrapper, Xing frame at byte 0, uniform sample counts) were derived by measuring sidecars it produces that Serato accepts, alongside sidecars written by Serato itself. No code from it is used or vendored here.

None of the above is vendored — `sidecaramel` is an independent implementation informed by this public documentation and by measuring real sidecar output, not a fork or port of any of it.

---

## Known limitations

Honest scope statement before you wire `sidecaramel` into anything that matters.

- **Real-Serato verification covers one reference track (three containers); broad-library corroboration is still limited.**  Most of the suite is self-referential (writers ↔ parsers on synthetic fixtures), but the core data-bearing blobs are now checked **byte-for-byte against Serato's own output**: `test_serato_golden.py` against real `.blob` payloads and `test_serato_audio_roundtrip.py` end-to-end through real Serato-tagged MP3/WAV/MP4 files (see *Byte-for-byte fidelity* above) — Markers2 (full outer), Markers_ V1 (masked + raw), BeatGrid, Autotags and the `.serato-stems` container all round-trip exactly.  That corroboration comes from **one carefully-tagged reference track** across the three container envelopes; it is real third-party (Serato) output, not a broad multi-track / multi-Serato-version corpus.  `Serato Overview` correctness remains **author-validated only** (it is recomputed, not reproduced byte-exact — see the green-field note above).  The `roundtrip_safety` harness inventories and protects pre-existing Serato blobs; `python -m sidecaramel.roundtrip_safety --from-copy <file>` runs it against a copy of a real track.

- **Markers2 write coverage.**  The MP3 / WAV / AIFF (ID3) and MP4 / M4A / M4V (atom) Markers2 + V1 write paths are exercised in CI on the Ubuntu leg, where `ffmpeg` synthesizes the cross-container fixtures; the macOS and Windows legs skip those fixture tests (no `ffmpeg`).  Correctness against real Serato is still author-validated only (see the round-trip caveat above).

- **`ffmpeg` is required for the full test matrix.**  Several tests synthesize cross-container fixtures via `ffmpeg`; they `skip` silently when it isn't on `PATH`.  Install `ffmpeg` for the complete run.

- **SYLT write is ID3-only.**  `write_sylt()` covers MP3/WAV/AIFF (the only containers with a native timed-lyrics frame).  MP4 and the Vorbis containers have no directly analogous tag; use `write_lrc_sidecar()` for a container-agnostic timed-lyrics companion file instead.

- **Writers fail silently.**  Most writers `return False` (or `None`, or `[]`) on any error and swallow the underlying exception.  As a library consumer you currently cannot distinguish "unsupported container", "corrupt file", "mutagen rejected it", and "bug in sidecaramel" — they all look the same.  Refactoring the writer surface to raise typed exceptions is on the list for `0.2.0`.

- **Readers are also silent for non-fatal cases.**  `read_serato_metadata()` returns `None` for unsupported extensions, for files with no Serato data, and on internal parse errors — but **raises `FileNotFoundError` for a nonexistent path** so a typo is distinguishable from "no metadata".  A broader read-side error-shape pass is on the 0.2.0 list.

- **Audio writes are not atomic.**  Tag writers call `mutagen.save()` in place.  A crash mid-save can leave the audio file in an unrecoverable state.  The `.crate` and `database V2` writers DO use tempfile + `os.replace` for atomicity; the per-track audio writers do not yet.  Back up any track you care about before pointing a writer at it.

- **`.crate` / `database V2` write-gate covers macOS + Windows; Linux is "unknown".**  The Serato-running probe is `pgrep -ix` on macOS and `tasklist /NH /FO CSV` on Windows.  On Linux or other hosts the gate raises `SeratoCheckUnavailableError` → the writer refuses to write unless you pass `serato_known_closed=True` (legacy `allow_serato_running=True`).

---

## Tests

```bash
pip install pytest
pytest sidecaramel/tests -q
```

Install `ffmpeg` to enable the cross-container fixture tests (otherwise the `ffmpeg`-dependent fixtures `skip`).
