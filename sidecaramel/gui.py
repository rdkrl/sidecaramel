"""sidecaramel.gui — drag-and-drop track viewer.

Drop an audio file onto the window, see its waveform, beatgrid,
and cue points; scroll and zoom with the trackpad.  A side panel
shows the parsed metadata: cover art, tags, BPM, cues and loops,
lyrics, and the Serato blob inventory.

Architecture:

    ┌─ MainWindow ─────────────────────────────────────────────────┐
    │  drag-drop accepts audio files                                │
    │  ┌─ MetadataPanel ─┐┌─ TrackView (QGraphicsView) ───────────┐ │
    │  │  cover art       ││  ┌─ scene ─────────────────────────┐ │ │
    │  │  title / artist  ││  │  WaveformItem  (@ 48.5 px/s)     │ │ │
    │  │  audio props     ││  │  BeatgridOverlay (over waveform) │ │ │
    │  │  cues / loops    ││  │  CueOverlay      (over beatgrid) │ │ │
    │  │  lyrics          ││  └─────────────────────────────────┘ │ │
    │  │  blob inventory  ││  trackpad: pinch = zoom, scroll = pan │ │
    │  └─────────────────┘└───────────────────────────────────────┘ │
    │  status: BPM, length, cues, sample-rate                       │
    └───────────────────────────────────────────────────────────────┘

Plugin contract: `sidecaramel.gui_plugins.OverlayPlugin` subclasses
register via `register_plugin(plugin)`.  The view calls each
plugin's `paint(painter, visible_rect, duration_sec)` after the
waveform.

Run: `python -m sidecaramel.gui [optional_initial_track.flac]`

Requires PySide6 + PIL.  PySide6 ships with `pip install
PySide6` (also needs Qt runtime, included).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

try:
    from PySide6.QtCore import (Qt, QRectF, QPointF, QPoint, QSize,
                                    QEvent, QTimer, Signal)
    from PySide6.QtGui import (QPixmap, QImage, QPainter, QColor,
                                   QPen, QBrush, QFont, QGuiApplication,
                                   QPolygonF, QWheelEvent)
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
        QGraphicsItem, QGraphicsPixmapItem, QStatusBar, QLabel,
        QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QToolBar,
        QDockWidget, QSplitter, QFrame, QSizePolicy)
except ImportError as e:
    # A library module must never sys.exit() at import time: that raises
    # SystemExit (a BaseException), which escapes the optional-dependency
    # `except ImportError` guards in importers — e.g. the
    # test_every_submodule_imports_clean smoke test — and crashes them
    # instead of skipping. Re-raise as a plain ImportError so that
    # `import sidecaramel.gui` degrades cleanly to "optional dep missing".
    raise ImportError(
        "sidecaramel.gui requires PySide6 — install the optional GUI extra:\n"
        "    pip install 'sidecaramel[gui]'\n"
        "(or directly: pip install PySide6 Pillow soundfile scipy numpy)"
    ) from e

from sidecaramel.audio_render import render_big_waveform
from sidecaramel.tags import (read_serato_metadata,
                                  serato_beat_grid_seconds,
                                  harvest_serato_blobs,
                                  _parse_serato_beatgrid)
try:
    from sidecaramel.lyrics import read_embedded_lyrics_with_format
    from sidecaramel.art import read_cover_art
except ImportError:
    read_embedded_lyrics_with_format = lambda p: None
    read_cover_art = lambda p: None
try:
    import mutagen
except ImportError:
    mutagen = None
from sidecaramel.gui_plugins import (OverlayPlugin, register_plugin,
                                          all_plugins)


# Canonical pixel-density (Serato extended view) from PROJECT FACTS.
PX_PER_SEC = 48.5
VIEW_HEIGHT = 300
HEADER_H = 30   # black margin reserved for bar numbers / cue tris
FOOTER_H = 18


def _audio_duration(audio_path: str) -> float:
    """Get audio duration via ffprobe (fast, no decode)."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1", audio_path])
        return float(out.decode().strip())
    except Exception:
        return 0.0


# =====================================================================
# Built-in overlays
# =====================================================================

class BeatgridOverlay(OverlayPlugin):
    """Render Serato BeatGrid markers as vertical lines.  Manual-
    anchor downbeats are RED, others WHITE, normal beats GREY.  Bar
    numbers go in the header strip."""

    def __init__(self):
        super().__init__(name="beatgrid")
        self.beats = []
        self.anchors = []
        self.bpm = None
        self.downbeat_t = 0.0

    def on_track_loaded(self, audio_path, duration_sec, meta):
        self.beats = []
        self.anchors = []
        if not meta:
            return
        self.beats = serato_beat_grid_seconds(meta, duration_sec)
        self.bpm = meta.get("bpm") or 120.0
        # Pull anchor positions from raw BeatGrid blob markers.
        for desc, payload, _ in harvest_serato_blobs(audio_path):
            if desc == "Serato BeatGrid":
                bg = _parse_serato_beatgrid(payload)
                if bg:
                    self.anchors = [m[0] for m in bg.get("markers", [])]
                break
        cues = meta.get("cues") or []
        self.downbeat_t = (self.anchors[0] if self.anchors
                              else (cues[0]["pos_ms"] / 1000.0
                                     if cues else
                                     (self.beats[0] if self.beats
                                      else 0.0)))

    def paint(self, painter, view_rect: QRectF, duration_sec: float):
        if not self.beats or duration_sec <= 0:
            return
        beat_period = 60.0 / self.bpm
        scene_w = duration_sec * PX_PER_SEC * 1.0  # base scale 1× = our world units
        # Actually we render in seconds-coords, painter is already
        # transformed.  Iterate beats and draw.
        # scene_w is just for normalization
        ANCHOR_C = QColor(255, 30, 30)
        DOWNB_C = QColor(240, 240, 240)
        BEAT_C = QColor(130, 130, 130, 180)
        for bt in self.beats:
            x_sec = bt
            if x_sec < view_rect.left() / PX_PER_SEC - 1:
                continue
            if x_sec > view_rect.right() / PX_PER_SEC + 1:
                break
            x_px = x_sec * PX_PER_SEC
            n_from = round((bt - self.downbeat_t) / beat_period)
            is_anchor = any(abs(bt - a) < beat_period * 0.5
                               for a in self.anchors)
            is_downbeat = (n_from % 4) == 0
            if is_anchor:
                pen = QPen(ANCHOR_C, 3); painter.setPen(pen)
            elif is_downbeat:
                pen = QPen(DOWNB_C, 2); painter.setPen(pen)
            else:
                pen = QPen(BEAT_C, 1); painter.setPen(pen)
            painter.drawLine(QPointF(x_px, 0), QPointF(x_px, VIEW_HEIGHT))

        # Bar numbers in header strip.
        bar = 1
        painter.setPen(QPen(QColor(220, 220, 220)))
        font = QFont("DejaVu Sans Mono", 11, QFont.Bold)
        painter.setFont(font)
        for bt in self.beats:
            n_from = round((bt - self.downbeat_t) / beat_period)
            if n_from % 4 != 0:
                continue
            x_px = bt * PX_PER_SEC
            if x_px < view_rect.left() - 30 or x_px > view_rect.right() + 30:
                bar += 1
                continue
            painter.drawText(QPointF(x_px + 3, 14), str(bar))
            bar += 1


class CueOverlay(OverlayPlugin):
    """Render Serato cue points as triangles dropping from the top
    edge in their own color."""

    def __init__(self):
        super().__init__(name="cues")
        self.cues = []

    def on_track_loaded(self, audio_path, duration_sec, meta):
        self.cues = (meta or {}).get("cues") or []

    def paint(self, painter, view_rect: QRectF, duration_sec: float):
        if not self.cues:
            return
        for cue in self.cues:
            x_px = (cue["pos_ms"] / 1000.0) * PX_PER_SEC
            if x_px < view_rect.left() - 20 or x_px > view_rect.right() + 20:
                continue
            color = QColor(*cue.get("color", (255, 255, 0)))
            tri = QPolygonF([QPointF(x_px, HEADER_H - 2),
                                QPointF(x_px - 7, 2),
                                QPointF(x_px + 7, 2)])
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawPolygon(tri)


# =====================================================================
# Waveform scene item — renders cached PNG into the scene
# =====================================================================

class WaveformPixmapItem(QGraphicsPixmapItem):
    """Holds the rendered audio waveform pixmap.  Position (0,0) =
    track start, x-axis = seconds * PX_PER_SEC."""
    pass


class OverlayItem(QGraphicsItem):
    """A QGraphicsItem that delegates painting to all registered
    OverlayPlugin instances.  Sits ON TOP of the WaveformPixmapItem
    in the scene's Z-order."""

    def __init__(self, duration_sec: float, total_width: float):
        super().__init__()
        self.duration_sec = duration_sec
        self.total_width = total_width
        self.setZValue(10)  # above waveform

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.total_width, VIEW_HEIGHT)

    def paint(self, painter: QPainter, option, widget=None):
        rect = painter.viewport()
        # painter is in scene coords; visible_rect is the area we
        # actually need to draw.  We can ask the parent view for it.
        view = self.scene().views()[0] if self.scene().views() else None
        if view:
            visible = view.mapToScene(view.viewport().rect()
                                          ).boundingRect()
        else:
            visible = self.boundingRect()
        for plugin in all_plugins():
            if not plugin.enabled:
                continue
            painter.save()
            try:
                plugin.paint(painter, visible, self.duration_sec)
            except Exception as e:
                print(f"plugin {plugin.name} paint error: {e}")
            painter.restore()


# =====================================================================
# Main view with trackpad zoom + pan
# =====================================================================

class TrackView(QGraphicsView):
    """QGraphicsView with trackpad pinch-zoom and 2-finger pan.

    Cmd / Ctrl + scroll = zoom (alternative for mouse users)
    Plain horizontal scroll = pan track
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QBrush(QColor(15, 15, 15)))
        self.setMinimumHeight(VIEW_HEIGHT + 40)
        self.scale_factor = 1.0
        # Accept pinch-zoom. On a QGraphicsView the touch/gesture events are
        # delivered to the viewport widget, not the view itself, so the
        # gesture must be grabbed on the viewport and handled in
        # viewportEvent(). Grabbing it on the view (self.grabGesture) instead
        # logs "QGestureManager: could not find the target for gesture" and
        # never zooms.
        self.viewport().grabGesture(Qt.PinchGesture)

        # Drag-and-drop. This QGraphicsView is the central widget and covers
        # the whole window, so it — not the QMainWindow — receives the drag
        # events; a QGraphicsView otherwise forwards them to its scene, whose
        # items don't accept drops, and the drop silently fails. Handle it
        # here (without calling super(), which would re-forward to the scene)
        # and hand the dropped file to the window's load_track.
        self.setAcceptDrops(True)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev):
        # macOS rejects the drop unless dragMove also accepts it.
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        urls = ev.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path and os.path.isfile(path):
                win = self.window()
                if hasattr(win, "load_track"):
                    win.load_track(path)
        ev.acceptProposedAction()

    def viewportEvent(self, ev: QEvent) -> bool:
        # Pinch-zoom arrives at the viewport two different ways:
        #  - macOS trackpads send a native "zoom" gesture
        #    (QNativeGestureEvent, ZoomNativeGesture) — NOT a QPinchGesture,
        #    because the OS reports a high-level gesture rather than raw
        #    touch points. This is the path Macs actually take.
        #  - touchscreens on Linux/Windows drive the QGesture framework, so a
        #    grabbed QPinchGesture shows up as a QEvent.Gesture.
        # Handle both; swallow the event when we zoom.
        if ev.type() == QEvent.NativeGesture:
            if ev.gestureType() == Qt.ZoomNativeGesture:
                factor = 1.0 + ev.value()
                self.scale_factor *= factor
                self.scale(factor, 1.0)  # zoom only horizontally
                return True
        elif ev.type() == QEvent.Gesture:
            pinch = ev.gesture(Qt.PinchGesture)
            if pinch and pinch.changeFlags() & pinch.ScaleFactorChanged:
                s = pinch.scaleFactor()
                self.scale_factor *= s
                self.scale(s, 1.0)  # zoom only horizontally
                return True
        return super().viewportEvent(ev)

    def wheelEvent(self, ev: QWheelEvent):
        mods = ev.modifiers()
        if mods & (Qt.ControlModifier | Qt.MetaModifier):
            # Cmd/Ctrl + scroll = zoom
            dy = ev.angleDelta().y()
            factor = 1.0 + (dy / 1200.0)
            self.scale_factor *= factor
            self.scale(factor, 1.0)
            ev.accept()
            return
        # Default: horizontal pan via wheel (trackpad sends dx, dy)
        dx = ev.angleDelta().x() or ev.pixelDelta().x()
        dy = ev.angleDelta().y() or ev.pixelDelta().y()
        # Convert vertical scroll to horizontal pan if no h-component
        if abs(dx) < abs(dy) // 2:
            dx = dy
        bar = self.horizontalScrollBar()
        bar.setValue(bar.value() - dx)
        ev.accept()


# =====================================================================
# Metadata side panel
# =====================================================================

def _get_id3_text(m, *keys, default=""):
    """Pull first non-empty text value from an mutagen audio file
    across several possible key names."""
    if m is None:
        return default
    for k in keys:
        v = m.get(k)
        if v:
            if isinstance(v, list):
                v = v[0] if v else None
            s = str(v) if v else ""
            if s:
                return s
    return default


class MetadataPanel(QWidget):
    """Left side panel showing all parsed track metadata: title,
    artist, album, BPM, key, length, sample-rate, bit-rate, cues,
    loops, lyrics, cover-art preview, Serato blob inventory."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(420)
        self.setStyleSheet(
            "QWidget { background-color: #1a1a1a; color: #d8d8d8; "
            "font: 12px -apple-system; }"
            "QLabel.title { font: bold 14px -apple-system; "
            "color: #ffd870; }"
            "QLabel.section { font: bold 11px -apple-system; "
            "color: #6ec0ff; margin-top: 8px; }"
            "QLabel.value { color: #ffffff; }"
            "QLabel.muted { color: #888888; }"
            "QFrame.sep { background-color: #2a2a2a; min-height: 1px; "
            "max-height: 1px; }"
        )
        self.outer = QVBoxLayout(self)
        self.outer.setContentsMargins(10, 10, 10, 10)
        self.outer.setSpacing(2)
        # Cover art
        self.cover = QLabel("(no cover)")
        self.cover.setAlignment(Qt.AlignCenter)
        self.cover.setFixedHeight(220)
        self.cover.setStyleSheet("background:#222;color:#777;"
                                     "border:1px solid #333;")
        self.outer.addWidget(self.cover)
        self.outer.addSpacing(6)
        # Title block
        self.title_lbl = QLabel("— no track —")
        self.title_lbl.setProperty("class", "title")
        self.title_lbl.setStyleSheet(
            "font: bold 14px -apple-system; color: #ffd870;")
        self.title_lbl.setWordWrap(True)
        self.outer.addWidget(self.title_lbl)
        self.artist_lbl = QLabel("")
        self.artist_lbl.setWordWrap(True)
        self.outer.addWidget(self.artist_lbl)
        self.album_lbl = QLabel("")
        self.album_lbl.setStyleSheet("color:#aaa;")
        self.album_lbl.setWordWrap(True)
        self.outer.addWidget(self.album_lbl)

        self._sep()
        # Audio properties
        self._section("AUDIO")
        self.audio_lbl = QLabel("")
        self.audio_lbl.setStyleSheet("color:#ccc;font-family:Menlo;font-size:11px;")
        self.outer.addWidget(self.audio_lbl)

        self._sep()
        # Serato
        self._section("SERATO")
        self.serato_lbl = QLabel("")
        self.serato_lbl.setStyleSheet("color:#ccc;font-family:Menlo;font-size:11px;")
        self.outer.addWidget(self.serato_lbl)

        self._sep()
        # Lyrics summary
        self._section("LYRICS")
        self.lyrics_lbl = QLabel("(no lyrics)")
        self.lyrics_lbl.setStyleSheet("color:#aaa;font-size:11px;")
        self.lyrics_lbl.setWordWrap(True)
        self.outer.addWidget(self.lyrics_lbl)

        self._sep()
        # Blobs inventory
        self._section("SERATO BLOBS")
        self.blobs_lbl = QLabel("")
        self.blobs_lbl.setStyleSheet("color:#9c9;font-family:Menlo;font-size:10px;")
        self.blobs_lbl.setWordWrap(True)
        self.outer.addWidget(self.blobs_lbl)

        self.outer.addStretch(1)

    def _section(self, label: str):
        lbl = QLabel(label)
        lbl.setStyleSheet("font: bold 11px -apple-system; color: #6ec0ff; "
                            "margin-top: 6px; letter-spacing: 1px;")
        self.outer.addWidget(lbl)

    def _sep(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color:#2a2a2a;color:#2a2a2a;"
                             "max-height:1px;")
        self.outer.addWidget(sep)

    def load(self, audio_path: str, meta: Optional[dict],
                duration_sec: float):
        # mutagen ID3 / atoms
        m = None
        if mutagen is not None:
            try:
                m = mutagen.File(audio_path, easy=True)
            except Exception:
                m = None
        title = _get_id3_text(m, "title") or Path(audio_path).stem
        artist = _get_id3_text(m, "artist", "albumartist")
        album = _get_id3_text(m, "album")
        self.title_lbl.setText(title)
        self.artist_lbl.setText(artist if artist else "")
        self.album_lbl.setText(album if album else "")

        # Audio properties
        info_lines = []
        info_lines.append(f"length:      {duration_sec:.1f} s")
        if m is not None:
            try:
                info = m.info
                if hasattr(info, "sample_rate"):
                    info_lines.append(f"sample rate: {info.sample_rate} Hz")
                if hasattr(info, "channels"):
                    info_lines.append(f"channels:    {info.channels}")
                if hasattr(info, "bitrate"):
                    info_lines.append(f"bitrate:     {info.bitrate // 1000} kbps")
                if hasattr(info, "bits_per_sample"):
                    info_lines.append(f"bit depth:   {info.bits_per_sample}")
            except Exception:
                pass
        info_lines.append(f"file size:   "
                            f"{os.path.getsize(audio_path) // 1024} KB")
        info_lines.append(f"format:      {Path(audio_path).suffix.lstrip('.').upper()}")
        self.audio_lbl.setText("\n".join(info_lines))

        # Serato
        serato_lines = []
        if meta:
            bpm = meta.get("bpm")
            serato_lines.append(f"BPM:         {bpm:.2f}" if bpm else "BPM:         —")
            serato_lines.append(f"bpm_lock:    {meta.get('bpm_lock')}")
            tcol = meta.get("color")
            if tcol:
                serato_lines.append(f"track color: #{tcol[0]:02x}"
                                       f"{tcol[1]:02x}{tcol[2]:02x}")
            cues = meta.get("cues") or []
            loops = meta.get("loops") or []
            serato_lines.append(f"cues:        {len(cues)}")
            for c in cues:
                ts = c.get("pos_ms", 0) / 1000.0
                lbl = c.get("label") or ""
                r, g, b = c.get("color", (255, 255, 0))
                serato_lines.append(f"  [{c['idx']}] "
                                       f"{ts:>6.2f}s "
                                       f"#{r:02x}{g:02x}{b:02x} {lbl}")
            serato_lines.append(f"loops:       {len(loops)}")
            for l in loops[:8]:
                s = l.get("start_ms", 0) / 1000.0
                e = l.get("end_ms", 0) / 1000.0
                serato_lines.append(f"  [{l['idx']}] "
                                       f"{s:>6.2f}→{e:>6.2f}s")
        else:
            serato_lines.append("— no Serato metadata —")
        self.serato_lbl.setText("\n".join(serato_lines))

        # Lyrics
        try:
            lyr = read_embedded_lyrics_with_format(audio_path)
        except Exception:
            lyr = None
        if lyr:
            text, fmt = lyr
            n_lines = len(text.splitlines())
            preview = text[:120].replace("\n", " ")
            self.lyrics_lbl.setText(
                f"format: {fmt}, {n_lines} lines, "
                f"{len(text)} chars\n\n{preview}…")
        else:
            self.lyrics_lbl.setText("(no embedded lyrics)")

        # Cover art
        try:
            art = read_cover_art(audio_path)
        except Exception:
            art = None
        if art:
            mime, data = art
            pix = QPixmap()
            pix.loadFromData(data)
            if not pix.isNull():
                self.cover.setPixmap(pix.scaled(
                    self.cover.width(), self.cover.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.cover.setText(f"(cover: {mime}, "
                                      f"{len(data)//1024}KB)")
        else:
            self.cover.setText("(no cover art)")
            self.cover.setPixmap(QPixmap())

        # Blob inventory
        try:
            blobs = list(harvest_serato_blobs(audio_path))
        except Exception:
            blobs = []
        blob_lines = []
        for desc, payload, src in blobs:
            blob_lines.append(f"  {desc:<22} {len(payload):>6} B")
        if not blob_lines:
            blob_lines.append("(no Serato blobs)")
        self.blobs_lbl.setText("\n".join(blob_lines))


# =====================================================================
# Main window — drag-drop entry point
# =====================================================================

class SidecaramelMain(QMainWindow):
    """Drag-drop main window.  Accept audio files anywhere on the
    window; load them into the view."""

    track_loaded = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("sidecaramel — track viewer")
        self.resize(1600, 600)
        self.setAcceptDrops(True)

        # Metadata side panel on the left, waveform view on the right,
        # in a splitter so the panel can be dragged wider/narrower. The
        # panel doesn't accept drops, so a drop over it bubbles up to this
        # window's dropEvent; the view handles drops over itself.
        self.view = TrackView(self)
        self.panel = MetadataPanel(self)
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(self.panel)
        splitter.addWidget(self.view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1260])
        self.setCentralWidget(splitter)

        self.status = QStatusBar(self)
        self.setStatusBar(self.status)
        self.status.showMessage("Drop an audio file (or use File → Open) "
                                  "— supports MP3 / WAV / FLAC / M4A / "
                                  "AIFF / OGG.")

        # Toolbar with Open and Reset Zoom
        tb = QToolBar("main", self)
        self.addToolBar(tb)
        tb.addAction("Open…", self.open_dialog)
        tb.addAction("Reset Zoom", self.reset_zoom)

        # Register built-in plugins
        register_plugin(BeatgridOverlay())
        register_plugin(CueOverlay())

        self.current_track: Optional[str] = None
        self.duration: float = 0.0

    # -- drag-drop ----------------------------------------------------
    # The central TrackView covers the whole window and handles drops over
    # itself; these catch drops that land on the chrome (toolbar / statusbar).
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev):
        # macOS rejects the drop unless dragMove also accepts it.
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        urls = ev.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path and os.path.isfile(path):
            self.load_track(path)

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open audio",
            os.path.expanduser("~/Music"),
            "Audio (*.mp3 *.flac *.wav *.aiff *.m4a *.ogg *.mp4)")
        if path:
            self.load_track(path)

    def reset_zoom(self):
        self.view.resetTransform()
        self.view.scale_factor = 1.0

    # -- track loading ------------------------------------------------
    def load_track(self, audio_path: str):
        self.setWindowTitle(f"sidecaramel — {os.path.basename(audio_path)}")
        self.status.showMessage(f"Loading {os.path.basename(audio_path)}…")
        QApplication.processEvents()
        self.current_track = audio_path
        self.duration = _audio_duration(audio_path)
        if self.duration <= 0:
            self.status.showMessage("Could not read audio.")
            return
        # Render waveform PNG to a temp file via audio_render.
        width_px = int(self.duration * PX_PER_SEC)
        tmp_png = Path(tempfile.gettempdir()) / (
            f"sidecaramel_wf_{abs(hash(audio_path))}_{int(self.duration)}.png")
        if not tmp_png.exists():
            ok = render_big_waveform(
                audio_path, str(tmp_png),
                width=width_px, height=VIEW_HEIGHT,
                bass_boost=1.4, dynamics=1.7,
                show_beatgrid=False,  # we draw beatgrid in overlay
                oversample=2)
            if not ok:
                self.status.showMessage("Render failed.")
                return
        pixmap = QPixmap(str(tmp_png))
        # Rebuild scene
        scene = self.view.scene()
        scene.clear()
        wf_item = WaveformPixmapItem(pixmap)
        wf_item.setPos(0, 0)
        scene.addItem(wf_item)
        # Add overlay layer
        overlay = OverlayItem(self.duration, width_px)
        scene.addItem(overlay)
        scene.setSceneRect(0, 0, width_px, VIEW_HEIGHT)

        # Load track-level metadata + notify plugins
        try:
            meta = read_serato_metadata(audio_path)
        except Exception:
            meta = None
        for plugin in all_plugins():
            try:
                plugin.on_track_loaded(audio_path, self.duration, meta)
            except Exception as e:
                print(f"plugin {plugin.name} on_track_loaded error: {e}")
        overlay.update()

        # Fill the metadata side panel (cover art, tags, cues/loops,
        # lyrics, Serato blob inventory).
        try:
            self.panel.load(audio_path, meta, self.duration)
        except Exception as e:
            print(f"metadata panel load error: {e}")

        bpm = (meta or {}).get("bpm")
        ncues = len((meta or {}).get("cues", []))
        self.status.showMessage(
            f"{os.path.basename(audio_path)} — "
            f"{self.duration:.1f}s, BPM={bpm or '?'}, "
            f"{ncues} cues  (trackpad: pinch=zoom, 2-finger=pan; "
            f"Cmd+scroll=zoom)")
        self.view.fitInView(QRectF(0, 0, min(width_px, 2400),
                                       VIEW_HEIGHT),
                                Qt.IgnoreAspectRatio)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("sidecaramel")
    w = SidecaramelMain()
    w.show()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        QTimer.singleShot(100, lambda: w.load_track(sys.argv[1]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
