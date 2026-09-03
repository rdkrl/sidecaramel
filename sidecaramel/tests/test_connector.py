"""The MCP connector: tool inventory, annotation truthfulness, and —
most important — that the write gate fails CLOSED."""
from __future__ import annotations

import json

import pytest

# Both imports must be import-or-skip: `pytest sidecaramel/tests -q` should
# skip this module when the optional `connector` extra isn't installed, not
# hard-error the whole collection. `anyio` used to be a bare `import anyio`
# ahead of the mcp check, so a missing anyio turned a skip into a collection
# error — the first command a stranger runs, failing on an optional extra.
anyio = pytest.importorskip("anyio")
pytest.importorskip("mcp")

from sidecaramel import connector                            # noqa: E402


def _tools():
    return {t.name: t for t in anyio.run(connector.mcp.list_tools)}


def test_tool_inventory_and_annotations():
    tools = _tools()
    read_only = {"serato_track_metadata", "serato_blob_inventory",
                 "serato_sidecar_info", "serato_library_tracks"}
    # serato_render_overview WRITES a file (its out_png), so it must not
    # advertise readOnlyHint=True — a false read-only hint lets MCP hosts
    # auto-approve a tool that can overwrite a caller-chosen path.
    # serato_extract_stems WRITES WAV files (force overwrites), so it is a
    # writer too — not read-only, and flagged destructive so hosts prompt.
    # (It isn't Serato-running-gated because it never touches Serato data.)
    writers = {"serato_write_overview", "serato_build_stems_sidecar",
               "serato_wipe_blobs", "serato_library_consolidate",
               "serato_render_overview", "serato_extract_stems"}
    assert read_only | writers <= set(tools)
    for name in read_only:
        assert tools[name].annotations.read_only_hint is True, name
    for name in writers:
        a = tools[name].annotations
        assert a.read_only_hint is False and a.destructive_hint is True, name


def test_gate_refuses_when_probe_unavailable(monkeypatch):
    """Unavailable probe == Serato running, unless explicitly overridden."""
    def boom():
        raise connector.SeratoCheckUnavailableError("no probe here")
    monkeypatch.setattr(connector, "is_running", boom)
    with pytest.raises(connector.WriteRefused):
        connector._gate_write(False)
    connector._gate_write(True)          # explicit override passes


def test_gate_running_is_never_overridable(monkeypatch):
    monkeypatch.setattr(connector, "is_running",
                        lambda: (True, ["serato dj pro"]))
    with pytest.raises(connector.WriteRefused):
        connector._gate_write(True)


def test_extract_stems_is_gated(monkeypatch, tmp_path):
    """serato_extract_stems writes WAVs, so — like every other write tool —
    it fails closed when the Serato-running probe is unavailable, before any
    file IO."""
    def boom():
        raise connector.SeratoCheckUnavailableError("no probe")
    monkeypatch.setattr(connector, "is_running", boom)
    with pytest.raises(connector.WriteRefused):
        connector.serato_extract_stems(str(tmp_path / "x.mp3"))


def test_wipe_needs_both_confirmations(monkeypatch, tmp_path):
    """confirm_wipe is checked BEFORE the gate — a destructive call
    without it must not even probe."""
    called = []
    monkeypatch.setattr(connector, "is_running",
                        lambda: called.append(1) or (False, []))
    with pytest.raises(connector.WriteRefused):
        connector.serato_wipe_blobs(str(tmp_path / "x.mp3"),
                                    confirm_wipe=False)
    assert called == [], "gate must not run before confirm_wipe"


def test_consolidate_live_run_needs_confirm_move(monkeypatch, tmp_path):
    """confirm_move is checked BEFORE the gate — a live (non-dry-run)
    consolidate without it must not even probe, same order as
    test_wipe_needs_both_confirmations."""
    called = []
    monkeypatch.setattr(connector, "is_running",
                        lambda: called.append(1) or (False, []))
    with pytest.raises(connector.WriteRefused):
        connector.serato_library_consolidate(
            str(tmp_path / "canonical"), [str(tmp_path / "secondary")],
            dry_run=False, confirm_move=False)
    assert called == [], "gate must not run before confirm_move"


def test_consolidate_dry_run_needs_no_confirm(tmp_path):
    """dry_run=true (the default) still just reports — no confirm_move,
    no gate, no error."""
    out = json.loads(connector.serato_library_consolidate(
        str(tmp_path / "canonical"), [str(tmp_path / "secondary")]))
    assert isinstance(out, dict)


def test_tool_errors_carry_their_message():
    """Anticipated failures must reach the MCP client as a ToolError with
    the guidance intact — not a generic 'Error executing tool X'. Regression
    guard: WriteRefused/ValueError/FileNotFoundError used to be wrapped as
    UnexpectedToolError, which strips the message."""
    assert issubclass(connector.WriteRefused, connector.ToolError)

    async def _call(name, args):
        try:
            await connector.mcp.call_tool(name, args)
            return ""
        except Exception as e:               # noqa: BLE001 - want the text
            return str(e)

    msg = anyio.run(_call, "serato_track_metadata",
                    {"audio_path": "/no/such/file.mp3"})
    assert "no such file" in msg, msg

    msg = anyio.run(_call, "serato_wipe_blobs",
                    {"audio_path": "/no/such/file.mp3", "confirm_wipe": False})
    assert "confirm_wipe=true" in msg, msg


def test_metadata_tool_roundtrip(minimal_mp3_with_id3):
    out = json.loads(connector.serato_track_metadata(
        str(minimal_mp3_with_id3)))
    assert out is None or isinstance(out, dict)
    # A missing file is an anticipated failure: it now raises ToolError so
    # the message ("no such file: …") reaches the MCP client, rather than a
    # bare FileNotFoundError the SDK would strip to a generic error.
    with pytest.raises(connector.ToolError, match="no such file"):
        connector.serato_track_metadata("/nope/definitely_missing.mp3")
