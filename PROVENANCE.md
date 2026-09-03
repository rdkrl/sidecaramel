# Provenance & Build — sidecaramel

This document answers the integration-material questions an evaluator
needs: who owns it, how to reproduce the build, and how mature each
module is.

## Authorship & license

- **Author / sole copyright holder:** Karl Yankovic. No third-party
  contributors, therefore no co-owned copyright.
- **License:** GPL-2.0-or-later. Chosen to match `mutagen` (GPL-2.0-or-later), the sole runtime dependency, rather than leave an MIT/GPL-dependency compatibility question unresolved. See `LICENSE` and `NOTICE`.
- **Format support** is reverse-engineered for interoperability with
  Serato DJ Pro. Not affiliated with, endorsed by, or sponsored by
  Serato Limited.

## Third-party dependencies

- **`mutagen` (GPL-2.0-or-later)** is the only runtime dependency and
  carries all container I/O — ID3, MP4, FLAC, WAVE, OggVorbis, OggOpus
  and AIFF — across 10 of this package's modules. It is imported, not
  vendored, but any distribution of sidecaramel has to account for its
  license terms.
- Optional extras: `numpy`, `scipy`, `soundfile`, `Pillow` (renderer),
  `PySide6` (GUI), `pytest` (tests). `ffmpeg` is invoked as an external
  binary by the stem extractor and the render helpers.

No third-party source is vendored into this package.

## Source identity

- Version is single-sourced: `sidecaramel/__init__.py:__version__`,
  `pyproject.toml:version`, and the wheel/sdist filenames all agree.

## Reproducible build

Standard PEP-517, no custom build steps:

```bash
python -m build          # builds dist/sidecaramel-<ver>-py3-none-any.whl + sdist
```

Verify a built wheel:

```bash
python - <<'PY'
import zipfile, hashlib, base64, csv, io
whl = "dist/sidecaramel-<ver>-py3-none-any.whl"
z = zipfile.ZipFile(whl)
rec = z.read([n for n in z.namelist() if n.endswith("RECORD")][0]).decode()
bad = 0
for row in csv.reader(io.StringIO(rec)):
    if not row or not row[1] or row[0].endswith("RECORD"): continue
    algo, _, want = row[1].partition("=")
    got = base64.urlsafe_b64encode(
        hashlib.new(algo, z.read(row[0])).digest()).rstrip(b"=").decode()
    bad += (got != want)
print("RECORD mismatches:", bad)
PY
```

Run the suite (install `ffmpeg` for the cross-container fixtures; point
`SIDECARAMEL_TEST_SIDECAR` at a real `.serato-stems` file to enable the
stem golden tests):

```bash
pip install pytest
pytest sidecaramel/tests -q
```

## Module maturity

Bounded contract per module — what is solid vs. preview.

| Module | Maturity | Notes |
|---|---|---|
| `tags` (Serato blob read + Markers2/V1 cue/loop encode) | **Stable core** | Read across all containers; writes confirm-gated; cue/loop writes are preservation-merging by default (`preserve=True`). Self-roundtrip tested. |
| `flip_writer` (FLIP splice, `merge_markers2_inner`) | **Stable core** | Byte-exact entry preservation. |
| `db` (`.crate` + `database V2` read/write) | **Stable core** | Atomic temp-file + `os.replace`; `.scrate` writes refused. |
| `blobs`, `overview`, `overview_palette`, `session` (read) | **Stable read** | Decoders / harvesters. |
| `lyrics`, `art` (read + write) | **Stable read; write OK** | Untimed (USLT/©lyr/Vorbis LYRICS) and timed (ID3 SYLT, MP3/WAV/AIFF only) lyric writes; `.lrc` sidecar read/write is container-agnostic. |
| `stems` | **Stable read** | `.serato-stems` decode/extract; chunk boundary and LAME gapless values verified against a real Serato 3.x sidecar. |
| `audio_render`, `window_api` | **Preview** | Renderers; needs `[render]`. |
| `cuss_flip`, `lrc_cusswords`, `loops_extract`, `flip_render` | **Preview** | Higher-level workflows; rougher edges. |
| `consolidate` | **Preview / dry-run-default** | Moves library audio (never deletes); `--apply` + interactive confirm required. |
| `roundtrip_safety` | **Tooling** | Offline write/strip/compare data-loss harness. Not a runtime safety net. |

## Known scope cuts (see README "Known limitations")

- Per-track **audio writes are not atomic** (no tempfile+replace yet) —
  contain behind an integration approval/rollback layer if used.
- Per-track tag writers **fail silently** (return `False`); typed
  exceptions are not implemented (`.crate`/`database V2` already raise
  typed errors).
- Tests are **self-referential** apart from the `.serato-stems` golden
  tests; run `python -m sidecaramel.roundtrip_safety --from-copy <file>`
  against a real track for an independent data-loss check.
