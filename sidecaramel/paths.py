"""sidecaramel.paths — volume-root resolution for Serato-relative paths.

Serato stores file paths inside its `database V2` and `.crate` files
as volume-relative strings ("Music/Foo.mp3"), not absolute paths.
Resolving them back to absolute paths requires knowing which volume
the DB/crate lives on.

On macOS, Serato's library lives on either:
  * The boot volume: paths are relative to `/`.
  * An external volume `/Volumes/X/_Serato_/...`: paths are relative
    to `/Volumes/X/`.

`volume_root_for_path` returns the volume-root that the given
absolute path lives under, so the caller can compute the inverse
mapping when reading a crate or DB record back.
"""
from __future__ import annotations

import os


def volume_root_for_path(path: str) -> str:
    """Given an absolute audio-file path, return the volume-root
    string Serato stores paths relative to: `'/'` for the boot
    volume, `'/Volumes/X'` for an external drive mounted at
    `/Volumes/X`.
    """
    parts = os.path.abspath(path).split(os.sep)
    if len(parts) > 2 and parts[1] == "Volumes":
        return os.sep + os.path.join(parts[1], parts[2])
    return os.sep


__all__ = [
    "volume_root_for_path",
]
