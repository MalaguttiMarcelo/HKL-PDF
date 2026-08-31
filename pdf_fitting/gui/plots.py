from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np
from PySide6 import QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MplPlot(QtWidgets.QWidget):
    """A Matplotlib plot widget with a toolbar."""

    def __init__(self, parent=None, *, title: str = ""):
        super().__init__(parent)
        self.fig = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.fig.add_subplot(111)
        if title:
            self.ax.set_title(title)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def clear(self):
        self.ax.clear()
        self.canvas.draw_idle()


class ParamEvolutionTabs(QtWidgets.QTabWidget):
    """Per-parameter evolution plots (value vs nfev)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plots: Dict[str, MplPlot] = {}
        self._lines: Dict[str, Any] = {}
        self._series: Dict[str, Dict[str, List[float]]] = {}
        self._zoom_last_n: int = 0

    def set_zoom_last_n(self, n: int) -> None:
        self._zoom_last_n = max(0, int(n))

    def clear_all(self) -> None:
        self._plots.clear()
        self._lines.clear()
        self._series.clear()
        self.clear()

    def ensure_param(self, name: str) -> None:
        if name in self._plots:
            return
        plot = MplPlot(title=f"{name} vs nfev")
        plot.ax.set_xlabel("nfev")
        plot.ax.set_ylabel(name)
        (line,) = plot.ax.plot([], [])
        plot.canvas.draw_idle()

        self._plots[name] = plot
        self._lines[name] = line
        self._series[name] = {"x": [], "y": []}
        self.addTab(plot, name)

    def push(self, nfev: int, params: Dict[str, Any], *, allowed: Optional[set[str]] = None) -> None:
        if nfev <= 0 or not isinstance(params, dict):
            return
        for k, v in params.items():
            name = str(k)
            if allowed is not None and name not in allowed:
                continue
            try:
                y = float(v)
            except Exception:
                continue

            self.ensure_param(name)
            s = self._series[name]
            s["x"].append(float(nfev))
            s["y"].append(float(y))

            plot = self._plots[name]
            line = self._lines[name]
            line.set_data(s["x"], s["y"])
            plot.ax.relim()
            plot.ax.autoscale_view(True, True, True)

            if self._zoom_last_n and len(s["x"]) >= 2:
                xmax = float(s["x"][-1])
                xmin = max(0.0, xmax - float(self._zoom_last_n))
                plot.ax.set_xlim(xmin, xmax)
                yy = [yy for xx, yy in zip(s["x"], s["y"]) if xx >= xmin]
                if yy:
                    y0, y1 = min(yy), max(yy)
                    if np.isfinite(y0) and np.isfinite(y1):
                        if y0 == y1:
                            y0 -= 1e-12
                            y1 += 1e-12
                        pad = 0.05 * (y1 - y0)
                        plot.ax.set_ylim(y0 - pad, y1 + pad)

            plot.canvas.draw_idle()
