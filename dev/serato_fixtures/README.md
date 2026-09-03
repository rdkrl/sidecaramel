# Serato golden fixtures — extraction

Dev tooling (excluded from the wheel via `pyproject`'s `exclude = ["dev*"]`)
for turning a track tagged in **real Serato DJ Pro** into the small,
committable blob fixtures under `sidecaramel/tests/fixtures/serato_golden/`
that `test_serato_golden.py` asserts against.

```bash
# dump every Serato blob from a tagged file (no audio is written)
python dev/serato_fixtures/extract_serato_fixtures.py dump TAGGED.mp3 --out DIR

# diff a raw baseline against its Serato-tagged copy: shows exactly which
# blobs Serato added and their decoded values
python dev/serato_fixtures/extract_serato_fixtures.py diff RAW.mp3 TAGGED.mp3
```

The fixtures are the *blobs only* (ID3 GEOB / MP4 atom payloads, a few KB
each) — never the audio. `manifest.json` records the decoded ground-truth.
