"""sidecaramel.gui_plugins — base classes for the modular plugin
system of the GUI app.

A plugin = anything that draws on top of the waveform view OR adds
a side panel / toolbar action.  Subclass `OverlayPlugin` or
`PanelPlugin` and register it via `register_plugin(...)`.

Built-in plugins live in `gui.py` directly:
- `BeatgridOverlay`  (renders Serato BeatGrid markers)
- `CueOverlay`        (renders Serato cue triangles)
- `WaveformOverlay`   (the base waveform — always present)

Third-party / future plugins:
- Section / structure overlay 
- Lyrics line overlay
- FX-band indicator overlay
"""
from __future__ import annotations

from typing import List, Optional


_REGISTRY: List["OverlayPlugin"] = []


def register_plugin(plugin: "OverlayPlugin") -> None:
    """Add a plugin instance to the global registry.  Plugins are
    drawn in registration order (later plugins draw on top)."""
    _REGISTRY.append(plugin)


def all_plugins() -> List["OverlayPlugin"]:
    return list(_REGISTRY)


class OverlayPlugin:
    """Base class for plugins that draw on top of the waveform.

    Subclasses override `paint(painter, view_rect, duration_sec)`.
    `view_rect` is the QRectF that's currently visible (in scene
    coordinates).  `duration_sec` is the total track duration so
    plugins can translate seconds → x-pixels.
    """

    name: str = "anonymous"
    enabled: bool = True

    def __init__(self, name: str = "anonymous"):
        self.name = name

    def on_track_loaded(self, audio_path: str, duration_sec: float,
                        meta: Optional[dict]) -> None:
        """Called when a new track is loaded into the view.  Override
        to pre-compute overlay data (e.g. fetch beats from `meta`)."""
        pass

    def paint(self, painter, view_rect, duration_sec: float) -> None:
        """Draw on the QPainter clipped to the visible view_rect.
        Override in subclass."""
        pass


class PanelPlugin:
    """Base class for plugins that add a side-panel widget."""

    name: str = "anonymous"

    def __init__(self, name: str = "anonymous"):
        self.name = name

    def make_widget(self, parent):
        """Return a QWidget that the app will dock into the panel."""
        raise NotImplementedError
