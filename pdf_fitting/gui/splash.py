"""
Start-up loading screen.

Every heavy library is imported here, before the main window opens, so later
actions (Calculate, Structure 3D, GPU) do not stop to import anything. The
imports run in a background thread; the screen (logo, progress bar, the name
of the library being loaded) is repainted by a timer on the GUI thread, so it
stays alive during long imports. Only imports happen off the GUI thread — no
widget is created there.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import threading
import time
from typing import List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

# (module, label shown to the user, typical seconds — used only to pace the bar)
MODULES: Tuple[Tuple[str, str, float], ...] = (
    ("numpy", "NumPy", 2.5),
    ("scipy.optimize", "SciPy optimizers", 9.0),
    ("scipy.sparse", "SciPy sparse matrices", 0.1),
    ("numba", "Numba compiler", 4.0),
    ("pandas", "pandas", 19.0),
    ("matplotlib", "Matplotlib", 1.6),
    ("matplotlib.backends.backend_qtagg", "Matplotlib Qt canvas", 1.4),
    ("pymatgen.core", "pymatgen (structures)", 1.6),
    ("vtk", "VTK (3D graphics)", 18.0),
    ("pyvista", "PyVista", 3.7),
    ("pyvistaqt", "PyVista Qt view", 2.4),
    ("cupy", "CuPy (NVIDIA GPU)", 6.2),
    ("cupyx.scipy.special", "CuPy special functions", 1.3),
    ("pdf_fitting.models.gr_model", "HKL-PDF model", 0.6),
    ("pdf_fitting.models.model_factory", "HKL-PDF model factory", 0.1),
    ("pdf_fitting.fit_engine", "HKL-PDF fit engine", 0.1),
    ("pdf_fitting.uncertainty", "HKL-PDF esd engine", 0.1),
    ("pdf_fitting.project.multiphase", "HKL-PDF projects", 0.1),
    ("pdf_fitting.project.free_peaks", "HKL-PDF free-peak fit", 0.1),
    ("pdf_fitting.models.gpu.generic_stacked_pdf", "HKL-PDF GPU kernels", 0.1),
    ("pdf_fitting.models.dynamic_stacked_model", "HKL-PDF stacking models", 0.2),
    ("pdf_fitting.models.stochastic_stacked_pdf_model", "HKL-PDF stochastic stacking", 0.1),
    ("pdf_fitting.gui.structure_studio", "HKL-PDF structure viewer", 0.1),
    ("pdf_fitting.gui.mainwindow", "HKL-PDF classic window", 3.3),
    ("pdf_fitting.gui.project_window", "HKL-PDF project window", 2.0),
)

ACCENT = QtGui.QColor("#1d4ed8")


# -----------------------------------------------------------------------------
# Logo
# -----------------------------------------------------------------------------


def _gr_curve(rect: QtCore.QRectF, phase: float = 0.0, n: int = 220) -> QtGui.QPainterPath:
    """A damped G(r)-like oscillation across ``rect``."""
    path = QtGui.QPainterPath()
    for i in range(n):
        t = i / (n - 1)
        r = 0.6 + 9.0 * t
        g = math.sin(2.6 * r - phase) * math.exp(-0.22 * r) * (1.0 - math.exp(-3.0 * t))
        x = rect.left() + t * rect.width()
        y = rect.center().y() - g * rect.height() * 0.48
        if i == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    return path


def paint_logo(p: QtGui.QPainter, rect: QtCore.QRectF, phase: float = 0.0) -> None:
    """Rounded badge with a lattice motif and a G(r) trace."""
    p.save()
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    grad = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
    grad.setColorAt(0.0, QtGui.QColor("#2563eb"))
    grad.setColorAt(1.0, QtGui.QColor("#1e3a8a"))
    p.setPen(Qt.NoPen)
    p.setBrush(grad)
    radius = rect.width() * 0.22
    p.drawRoundedRect(rect, radius, radius)
    # lattice points
    p.setBrush(QtGui.QColor(255, 255, 255, 70))
    d = rect.width() * 0.055
    for i in range(4):
        for j in range(4):
            cx = rect.left() + rect.width() * (0.2 + 0.2 * i)
            cy = rect.top() + rect.height() * (0.2 + 0.2 * j)
            p.drawEllipse(QtCore.QPointF(cx, cy), d, d)
    # G(r)
    inner = rect.adjusted(rect.width() * 0.1, rect.height() * 0.22, -rect.width() * 0.1, -rect.height() * 0.22)
    pen = QtGui.QPen(QtGui.QColor("white"), max(1.5, rect.width() * 0.045))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawPath(_gr_curve(inner, phase))
    p.restore()


def app_icon() -> QtGui.QIcon:
    icon = QtGui.QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pm = QtGui.QPixmap(size, size)
        pm.fill(Qt.transparent)
        painter = QtGui.QPainter(pm)
        paint_logo(painter, QtCore.QRectF(0, 0, size, size))
        painter.end()
        icon.addPixmap(pm)
    return icon


# -----------------------------------------------------------------------------
# Loader
# -----------------------------------------------------------------------------


class _Loader:
    """Imports MODULES in a thread; the GUI polls its state."""

    def __init__(self, modules):
        # Optional packages that are not installed are skipped (no error).
        self.modules = [m for m in modules if self._installed(m[0])]
        self.index = 0
        self.started_current = time.perf_counter()
        self.done = False
        self.errors: List[str] = []
        self.timings: List[Tuple[str, float]] = []

    @staticmethod
    def _installed(name: str) -> bool:
        try:
            return importlib.util.find_spec(name.split(".")[0]) is not None
        except (ImportError, ValueError):
            return False

    def start(self):
        threading.Thread(target=self._run, name="hklpdf-startup-imports", daemon=True).start()

    def _run(self):
        for i, (name, _label, _w) in enumerate(self.modules):
            self.index = i
            self.started_current = t0 = time.perf_counter()
            try:
                importlib.import_module(name)
            except Exception as exc:  # noqa: BLE001 — imported again (and reported) when used
                self.errors.append(f"{name}: {exc}")
            self.timings.append((name, time.perf_counter() - t0))
        self.index = len(self.modules)
        self.done = True

    def fraction(self) -> float:
        total = sum(w for _n, _l, w in self.modules) or 1.0
        i = min(self.index, len(self.modules))
        done = sum(w for _n, _l, w in self.modules[:i])
        if i < len(self.modules) and not self.done:
            w = self.modules[i][2]
            elapsed = time.perf_counter() - self.started_current
            # Ease towards (never past) the expected share of this module.
            done += w * (1.0 - math.exp(-elapsed / max(w, 0.05)))
        return 1.0 if self.done else min(done / total, 0.99)

    def current_label(self) -> str:
        if self.done:
            return "Opening the main window…"
        i = min(self.index, len(self.modules) - 1)
        name, label, _w = self.modules[i]
        return f"Loading {label}  ·  {name}"


# -----------------------------------------------------------------------------
# Screen
# -----------------------------------------------------------------------------


class LoadingSplash(QtWidgets.QWidget):
    W, H = 640, 360

    def __init__(self, loader: _Loader, version: str = ""):
        super().__init__(None, Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(self.W, self.H)
        self.loader = loader
        self.version = version
        self.t0 = time.perf_counter()
        self.phase = 0.0
        self.shown_fraction = 0.0
        self.message: Optional[str] = None
        screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.move(screen.center() - QtCore.QPoint(self.W // 2, self.H // 2))
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _tick(self):
        self.phase += 0.12
        target = self.loader.fraction()
        self.shown_fraction += (target - self.shown_fraction) * 0.25
        self.update()

    def set_message(self, text: str):
        self.message = text
        self.shown_fraction = 1.0
        self.repaint()

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = QtCore.QRectF(self.rect()).adjusted(1, 1, -1, -1)

        # card
        bg = QtGui.QLinearGradient(r.topLeft(), r.bottomLeft())
        bg.setColorAt(0.0, QtGui.QColor("#0f172a"))
        bg.setColorAt(1.0, QtGui.QColor("#172554"))
        p.setPen(QtGui.QPen(QtGui.QColor("#1e293b"), 1))
        p.setBrush(bg)
        p.drawRoundedRect(r, 14, 14)

        # faint G(r) across the background
        wave = QtCore.QRectF(r.left() + 20, r.top() + 150, r.width() - 40, 110)
        pen = QtGui.QPen(QtGui.QColor(96, 165, 250, 55), 2)
        p.setPen(pen)
        p.drawPath(_gr_curve(wave, self.phase * 0.5))

        # logo + title
        paint_logo(p, QtCore.QRectF(r.left() + 36, r.top() + 40, 88, 88), self.phase)
        p.setPen(QtGui.QColor("white"))
        f = QtGui.QFont(self.font())
        f.setPointSize(30)
        f.setBold(True)
        p.setFont(f)
        p.drawText(QtCore.QRectF(r.left() + 144, r.top() + 44, 460, 50), Qt.AlignLeft | Qt.AlignVCenter, "HKL-PDF")
        f.setPointSize(11)
        f.setBold(False)
        p.setFont(f)
        p.setPen(QtGui.QColor("#93c5fd"))
        p.drawText(QtCore.QRectF(r.left() + 146, r.top() + 94, 460, 24), Qt.AlignLeft | Qt.AlignVCenter,
                   "Pair distribution function fitting · microstructure · stacking faults")
        if self.version:
            f.setPointSize(9)
            p.setFont(f)
            p.setPen(QtGui.QColor("#64748b"))
            p.drawText(QtCore.QRectF(r.right() - 120, r.top() + 14, 104, 18), Qt.AlignRight, f"v{self.version}")

        # progress bar
        bar = QtCore.QRectF(r.left() + 36, r.bottom() - 74, r.width() - 72, 10)
        p.setPen(Qt.NoPen)
        p.setBrush(QtGui.QColor("#1e293b"))
        p.drawRoundedRect(bar, 5, 5)
        frac = max(0.0, min(1.0, self.shown_fraction))
        if frac > 0:
            fill = QtCore.QRectF(bar.left(), bar.top(), max(10.0, bar.width() * frac), bar.height())
            g = QtGui.QLinearGradient(fill.topLeft(), fill.topRight())
            g.setColorAt(0.0, QtGui.QColor("#3b82f6"))
            g.setColorAt(1.0, QtGui.QColor("#60a5fa"))
            p.setBrush(g)
            p.drawRoundedRect(fill, 5, 5)

        # status line (bottom)
        f.setPointSize(9)
        p.setFont(f)
        p.setPen(QtGui.QColor("#cbd5e1"))
        text = self.message or self.loader.current_label()
        p.drawText(QtCore.QRectF(bar.left(), bar.bottom() + 12, bar.width() - 90, 20), Qt.AlignLeft | Qt.AlignVCenter, text)
        p.setPen(QtGui.QColor("#64748b"))
        elapsed = time.perf_counter() - self.t0
        p.drawText(QtCore.QRectF(bar.right() - 90, bar.bottom() + 12, 90, 20), Qt.AlignRight | Qt.AlignVCenter,
                   f"{int(frac * 100):d} %  ·  {elapsed:.0f} s")
        p.end()


def load_with_splash(app: QtWidgets.QApplication, version: str = "", *, show: bool = True, show_when=None) -> LoadingSplash:
    """
    Show the loading screen and import every heavy library, keeping the
    screen (and the event loop) alive. Returns the splash so the caller can
    show a last message while the main window is built, then close it.

    show=False loads silently (login start); ``show_when()`` returning True
    makes the screen appear anyway (the user asked for a window meanwhile).
    """
    loader = _Loader(MODULES)
    splash = LoadingSplash(loader, version)
    if show:
        splash.show()
    app.processEvents()
    loader.start()
    while not loader.done:
        app.processEvents(QtCore.QEventLoop.AllEvents, 30)
        if not splash.isVisible() and show_when is not None and show_when():
            splash.show()
        time.sleep(0.01)
    total = time.perf_counter() - splash.t0
    print(f"[STARTUP] libraries loaded in {total:.1f} s", flush=True)
    for err in loader.errors:
        print(f"[STARTUP] could not preload {err}", flush=True)
    return splash
