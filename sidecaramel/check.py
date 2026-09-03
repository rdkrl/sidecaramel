"""sidecaramel.check — "is a Serato app currently running?"

**Platform support**: process detection is implemented for **macOS**
(via ``pgrep -ix``) and **Windows** (via ``tasklist /NH /FO CSV``).
On Linux and other hosts :func:`is_running` raises
:class:`SeratoCheckUnavailableError` and the caller must either
ask the user to confirm Serato is closed, or pass
``serato_known_closed=True`` (legacy alias: ``allow_serato_running``)
to override the gate on the ``.crate`` and ``database V2`` writers.

Why the writers gate on this: Serato re-reads its Subcrates folder
on app-quit from its in-memory snapshot.  Any crate file edited
while Serato is alive gets clobbered.  The check is the writers'
opt-out: refuse to write if the app is open.

Usage:

    from sidecaramel.check import is_running, assert_not_running
    running, processes = is_running()
    if running:
        for p in processes: print(p)
    # or, raise instead of returning:
    assert_not_running(log=print)   # raises SeratoRunningError
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Tuple


# Canonical Serato app process names. EXACT match (pgrep -ix), not
# substring — substring pgrep -i "serato" false-positived on
# CoreAudio helpers (e.g. com.apple.audio.*) whose paths/names
# contained "serato" somewhere unrelated.  Windows ``tasklist``
# matches the executable name (case-insensitive), so the same list
# works there with an ``.exe`` suffix appended at probe time.
SERATO_APP_NAMES: List[str] = [
    "Serato DJ Pro",
    "Serato DJ Lite",
    "Serato DJ Intro",
    "Serato Studio",
    "Serato ITCH",
    "ScratchLIVE",
    "Scratch Live",
]


class SeratoRunningError(RuntimeError):
    """Raised when a crate-write is attempted while Serato is alive."""


class SeratoCheckUnavailableError(RuntimeError):
    """Raised when we cannot determine Serato status — e.g. running
    on a host without a supported probe (currently only macOS and
    Windows are supported).

    A silent "nothing found" return from an unsupported probe must
    not be treated as "Serato is off".  Treat this error as
    "unknown / block by default" and require the user to confirm.
    """


def _platform() -> str:
    """Return ``"darwin"`` on macOS, ``"win"`` on Windows, ``""``
    elsewhere.  Used to pick the right process probe.
    """
    try:
        name = os.uname().sysname.lower()
        if name == "darwin":
            return "darwin"
    except (AttributeError, OSError):
        # os.uname doesn't exist on Windows.
        pass
    if sys.platform.startswith("win"):
        return "win"
    return ""


def _probe_supported() -> bool:
    """Backward-compat helper kept under its public-looking name for
    one minor.  Prefer :func:`_platform`.  Returns ``True`` iff a
    process probe is implemented for the current host.
    """
    return _platform() in ("darwin", "win")


# Legacy private name kept as an alias — older imports may grab it.
def _running_in_sandbox() -> bool:
    """Backward-compat alias for ``not _probe_supported()``.
    Superseded by ``_probe_supported`` (with inverted polarity)
    because "sandbox" was a dishonest name — a Windows desktop is
    not a sandbox.  Kept for one minor; will be removed in 0.2.0.
    """
    return not _probe_supported()


def _probe_macos() -> List[str]:
    """Run ``pgrep -ix`` for each canonical Serato app name."""
    matches: List[str] = []
    for app in SERATO_APP_NAMES:
        try:
            r = subprocess.run(
                ["pgrep", "-ix", app],
                capture_output=True, text=True, timeout=5)
        except (FileNotFoundError, OSError):
            continue
        if r.returncode == 0 and r.stdout.strip():
            for pid_str in r.stdout.strip().splitlines():
                pid_str = pid_str.strip()
                if pid_str.isdigit():
                    matches.append(f"{app}@{pid_str}")
    return matches


def _probe_windows() -> List[str]:
    """Run ``tasklist /NH /FO CSV`` and grep for Serato apps.

    Windows imagename matches against ``<AppName>.exe`` (case-
    insensitive).  ``tasklist`` is a built-in Windows utility so no
    extra dependency.
    """
    matches: List[str] = []
    try:
        r = subprocess.run(
            ["tasklist", "/NH", "/FO", "CSV"],
            capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, OSError):
        return matches
    if r.returncode != 0 or not r.stdout:
        return matches
    # CSV columns: "Image Name","PID","Session Name","Session#","Mem Usage"
    targets = {(name + ".exe").lower(): name for name in SERATO_APP_NAMES}
    for line in r.stdout.splitlines():
        # Naive CSV split — image names don't contain commas.
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) < 2:
            continue
        img = parts[0].lower()
        if img in targets and parts[1].isdigit():
            matches.append(f"{targets[img]}@{parts[1]}")
    return matches


def is_running() -> Tuple[bool, List[str]]:
    """Return ``(is_running, ["AppName@PID", ...])``.

    Probes:
        macOS:   ``pgrep -ix`` per canonical Serato app name.
        Windows: ``tasklist /NH /FO CSV`` + case-insensitive
                 ImageName grep on ``<AppName>.exe``.
        else:    raises :class:`SeratoCheckUnavailableError`.

    A non-macOS-non-Windows host cannot tell whether Serato is
    running, so silently returning ``(False, [])`` would be a
    Serato-running check false-negative.  Callers that catch the
    error should fall back to manual confirmation (= ask the user
    before edit).
    """
    plat = _platform()
    if plat == "darwin":
        matches = _probe_macos()
    elif plat == "win":
        matches = _probe_windows()
    else:
        raise SeratoCheckUnavailableError(
            "Cannot check Serato status on this platform — only "
            "macOS (pgrep) and Windows (tasklist) probes are "
            "implemented.  Caller should either ask the user to "
            "confirm Serato is closed, or pass "
            "`serato_known_closed=True` (legacy alias: "
            "`allow_serato_running=True`) to override the gate."
        )
    return (len(matches) > 0, matches)


def assert_not_running(log=None) -> None:
    """Raise :class:`SeratoRunningError` if Serato is detected.

    When the probe-unsupported error fires (e.g. on Linux), this
    re-raises as ``SeratoRunningError`` = "treat as running"
    because that's the safer default.  The caller's existing
    handling (block crate-writes, prompt user) does the right
    thing either way.

    Optional ``log`` is a callable that receives a one-line warning
    string before the raise (e.g. ``print`` or a pipeline logger).
    """
    try:
        running, processes = is_running()
    except SeratoCheckUnavailableError as e:
        msg = (f"Serato status unknown ({e}) — refusing to "
               f"proceed. Manual confirmation required.")
        if log:
            try:
                log(f"[serato-check] {msg}")
            except Exception:
                pass
        raise SeratoRunningError(msg)
    if running:
        msg = ("Serato is running (" + ", ".join(processes) +
                ") — refusing to write .crate files. Quit Serato "
                "first, then re-run.")
        if log:
            try:
                log(f"[serato-check] {msg}")
            except Exception:
                pass
        raise SeratoRunningError(msg)


# ============================================================
# CLI for shell wrappers
# ============================================================

def _cli_main() -> int:
    """When run as ``python -m sidecaramel.check``:

      exit 0 = Serato not running (safe to proceed)
      exit 1 = Serato running (PID/AppName on stdout)
      exit 2 = could not determine (platform without a probe) —
              caller MUST treat as "running" / block.
    """
    try:
        running, processes = is_running()
    except SeratoCheckUnavailableError as e:
        print(f"[serato-check] {e}", file=sys.stderr)
        return 2
    if not running:
        return 0
    for p in processes:
        # Re-format AppName@PID → "PID  AppName" for the shell's
        # echo loop.
        if "@" in p:
            name, pid = p.rsplit("@", 1)
            print(f"{pid}  {name}")
        else:
            print(p)
    return 1


if __name__ == "__main__":
    sys.exit(_cli_main())
