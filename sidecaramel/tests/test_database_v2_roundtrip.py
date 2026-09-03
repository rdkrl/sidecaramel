"""Tests for `sidecaramel.db.write_database_v2`.

Coverage:
  R) round-trip — parse → mutate → write → reparse keeps known fields,
     preserved raw bytes, and the vrsn header byte-equal.
  S) `confirm=True` gate refuses without acknowledgment.
  T) `allow_serato_running` gate refuses while a Serato-name process
     would block writes (we monkey-patch the check to simulate this).
  U) Blob-level: writer produces a byte-identical file when there are
     no mutations between parse and write.
  V) Unknown 4-byte tags survive verbatim via the `_raw` sub-dict.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest


# ---- helpers -------------------------------------------------------

def _utf16(s: str) -> bytes:
    return s.encode("utf-16-be")


def _tlv(tag: bytes, payload: bytes) -> bytes:
    return tag + struct.pack(">I", len(payload)) + payload


def _build_fixture_blob() -> bytes:
    """Build a tiny in-memory DB blob covering all four type codes
    (s, u, h, b) plus a deliberately unknown tag."""
    blob = _tlv(b"vrsn", _utf16("@2.0/Serato Scratch LIVE Database"))
    rec1 = b""
    rec1 += _tlv(b"pfil", _utf16("Music/Test/track-one.mp3"))
    rec1 += _tlv(b"tsng", _utf16("Track One"))
    rec1 += _tlv(b"tart", _utf16("The Maintainers"))
    rec1 += _tlv(b"tbpm", _utf16("128.00"))
    rec1 += _tlv(b"utpc", struct.pack(">I", 42))     # play count
    rec1 += _tlv(b"bovc", b"\x01")                    # analysed
    rec1 += _tlv(b"bstm", b"\x00")                    # no stems
    rec1 += _tlv(b"sbav", struct.pack(">H", 7))       # uint16
    rec1 += _tlv(b"XXXX", b"\xde\xad\xbe\xef")        # unknown
    blob += _tlv(b"otrk", rec1)
    rec2 = _tlv(b"pfil", _utf16("Music/Other/two.mp3"))
    rec2 += _tlv(b"tsng", _utf16("Two"))
    blob += _tlv(b"otrk", rec2)
    return blob


@pytest.fixture
def db_v2_fixture(tmp_path: Path) -> Path:
    p = tmp_path / "database V2"
    p.write_bytes(_build_fixture_blob())
    return p


# ---- R) round-trip --------------------------------------------------

def test_R_roundtrip_known_fields_and_raw(db_v2_fixture: Path,
                                            tmp_path: Path):
    from sidecaramel.db import (parse_database_v2,
                                       write_database_v2)
    recs = parse_database_v2(str(db_v2_fixture))
    assert len(recs) == 2
    assert recs[0]["title"] == "Track One"
    assert recs[0]["play_count"] == 42
    assert recs[0]["analyzed"] is True
    assert recs[0]["has_stems"] is False
    assert recs[0]["sbav"] == 7
    assert recs[0]["_raw"]["XXXX"] == b"\xde\xad\xbe\xef"

    # Mutate
    recs[0]["play_count"] = 99
    recs[0]["title"] = "Track One (Renamed)"

    out = tmp_path / "database V2.out"
    ok = write_database_v2(recs, str(out),
                             confirm=True, allow_serato_running=True)
    assert ok is True

    recs2 = parse_database_v2(str(out))
    assert len(recs2) == 2
    assert recs2[0]["title"] == "Track One (Renamed)"
    assert recs2[0]["play_count"] == 99
    assert recs2[0]["artist"] == "The Maintainers"
    assert recs2[0]["sbav"] == 7
    assert recs2[0]["_raw"]["XXXX"] == b"\xde\xad\xbe\xef"
    assert recs2[0]["analyzed"] is True
    assert recs2[0]["has_stems"] is False
    assert recs2[1]["title"] == "Two"


# ---- S) confirm gate -----------------------------------------------

def test_S_confirm_gate_required(db_v2_fixture: Path, tmp_path: Path):
    from sidecaramel.db import (parse_database_v2,
                                       write_database_v2,
                                       DatabaseV2WriteError)
    recs = parse_database_v2(str(db_v2_fixture))
    out = tmp_path / "database V2.out"

    # Default: no confirm → raises, leaves file alone
    with pytest.raises(DatabaseV2WriteError) as exc_info:
        write_database_v2(recs, str(out), allow_serato_running=True)
    assert "confirm=True" in str(exc_info.value)
    assert not out.exists()

    # Explicit confirm=False also raises
    with pytest.raises(DatabaseV2WriteError):
        write_database_v2(recs, str(out),
                            confirm=False,
                            allow_serato_running=True)

    # confirm=True succeeds
    assert write_database_v2(recs, str(out),
                               confirm=True,
                               allow_serato_running=True) is True
    assert out.exists()


# ---- T) Serato-not-running gate ------------------------------------

def test_T_refuses_while_serato_running(db_v2_fixture: Path,
                                          tmp_path: Path,
                                          monkeypatch):
    """When the optional `sidecaramel.check` module says Serato is
    running, the writer must refuse unless `allow_serato_running=True`."""
    from sidecaramel.db import (parse_database_v2,
                                       write_database_v2)
    recs = parse_database_v2(str(db_v2_fixture))
    out = tmp_path / "database V2.out"

    # Monkey-patch the assert helper to simulate "Serato is running"
    try:
        import sidecaramel.check as check_mod
    except ImportError:
        pytest.skip("sidecaramel.check not installed in this env")

    def _fake_assert_running(*args, **kwargs):
        raise RuntimeError("simulated: Serato is running")

    monkeypatch.setattr(check_mod, "assert_not_running",
                          _fake_assert_running)

    with pytest.raises(RuntimeError, match="Serato is running"):
        write_database_v2(recs, str(out), confirm=True)
    assert not out.exists()

    # Override works
    assert write_database_v2(recs, str(out),
                               confirm=True,
                               allow_serato_running=True) is True
    assert out.exists()


# ---- U) byte-equal when nothing mutates ----------------------------

def test_U_byte_equal_no_mutation(db_v2_fixture: Path,
                                     tmp_path: Path):
    """parse → write without any mutation produces a byte-identical
    file, provided the source had no fields we drop.  Our fixture is
    constructed exactly that way (all fields known or in _raw)."""
    from sidecaramel.db import (parse_database_v2,
                                       write_database_v2)
    recs = parse_database_v2(str(db_v2_fixture))
    out = tmp_path / "database V2.out"
    write_database_v2(recs, str(out),
                        confirm=True, allow_serato_running=True)

    src = db_v2_fixture.read_bytes()
    dst = out.read_bytes()
    assert dst == src, (
        f"byte-equal roundtrip failed:\n"
        f"  src len={len(src)}  dst len={len(dst)}\n"
        f"  first-diff offset = "
        f"{next((i for i, (a, b) in enumerate(zip(src, dst)) if a != b), 'n/a')}")


# ---- V) unknown-tag preservation -----------------------------------

def test_V_unknown_tag_round_trips_verbatim(db_v2_fixture: Path,
                                                  tmp_path: Path):
    from sidecaramel.db import (parse_database_v2,
                                       write_database_v2)
    recs = parse_database_v2(str(db_v2_fixture))
    # Inject another unknown tag programmatically — this is the path
    # callers will use to graft custom data onto records.
    recs[0]["_raw"]["YYYY"] = b"\x01\x02\x03\x04\x05\x06\x07\x08"

    out = tmp_path / "database V2.out"
    write_database_v2(recs, str(out),
                        confirm=True, allow_serato_running=True)
    recs2 = parse_database_v2(str(out))
    assert recs2[0]["_raw"]["XXXX"] == b"\xde\xad\xbe\xef"
    assert recs2[0]["_raw"]["YYYY"] == b"\x01\x02\x03\x04\x05\x06\x07\x08"
