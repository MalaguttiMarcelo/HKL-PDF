"""
Bridge to the classic single-phase window.

The classic MainWindow already implements the PDF plot with tick markers
(hkl / uvw, multiplicity, filters, styles), pair contributions, colour and
curve-style dialogs, pop-out / maximize / full screen, and the Warren,
size-distribution, local-trends, shape-factor and CS-grid outputs. Its
Builder holds the finite-crystallite / stacking-fault editors with their 3D
views.

A LegacyHost owns one *hidden* classic MainWindow loaded with one phase's
single-phase config (or, for a dataset, with a merged view of its phases)
and lends its widgets to the project window:

    host.pdf_container   PDF plot + its toolbar
    host.output_tabs     Warren / size / local trends / shape factors / CS grid

The classic window is never shown and never gets a file path, so it can
never write an input file.
"""

from __future__ import annotations

import configparser
import copy
import io
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

from pdf_fitting.io_handler import read_input_file

# Sections the Builder writes itself (complex, JSON-encoded state).
_BUILDER_OWNED_SECTIONS = (
    "layer_stacking",
    "stacking_search",
    "stacking_vector_search",
    "stacking_refinement",
    "stacking_faults",
    "contrast_factors",
)


def _ini_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    return str(v)


def legacy_cfg_to_ini_text(cfg: Dict[str, Any]) -> str:
    """
    INI text for the simple sections of a single-phase config. Complex
    sections (stacking, contrast factors) are produced by the classic
    Builder from the same config.
    """
    cp = configparser.RawConfigParser()
    cp.optionxform = str
    cp["files"] = {
        "structure_file": str(cfg.get("structure_file", "")),
        "gr_data_file": str(cfg.get("gr_data_file", "")),
    }
    cp["range"] = {"r_min": _ini_value(cfg.get("r_min", 1.0)), "r_max": _ini_value(cfg.get("r_max", 10.0))}
    cp["pair_generation"] = {"r_extension": _ini_value(cfg.get("r_extension", 1.2))}
    cp["refinement"] = {
        str(k): _ini_value(v)
        for k, v in (cfg.get("refinement", {}) or {}).items()
        if not isinstance(v, (dict, list))
    }
    cp["initial_values"] = {str(k): _ini_value(v) for k, v in (cfg.get("initial", {}) or {}).items()}
    cp["refinable_parameters"] = {str(k): _ini_value(bool(v)) for k, v in (cfg.get("refinable", {}) or {}).items()}
    bounds = {}
    for k, b in (cfg.get("bounds", {}) or {}).items():
        try:
            lo, hi = b
        except (TypeError, ValueError):
            continue
        bounds[str(k)] = f"{_ini_value(lo)}, {_ini_value(hi)}"
    cp["bounds"] = bounds
    cp["constraints"] = {str(k): str(v) for k, v in (cfg.get("constraints", {}) or {}).items()}
    shape = {str(k): _ini_value(v) for k, v in (cfg.get("crystallite_shape", {}) or {}).items() if not isinstance(v, (dict, list))}
    if shape:
        cp["crystallite_shape"] = shape
    buf = io.StringIO()
    cp.write(buf)
    return buf.getvalue()


def ini_text_to_cfg(text: str) -> Dict[str, Any]:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".inp", mode="w", encoding="utf-8")
    try:
        tmp.write(text)
        tmp.close()
        return read_input_file(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def size_distribution(params: Dict[str, Any]):
    """Classic size-distribution diagnostic (FitWorker._build_size_distribution)."""
    try:
        from pdf_fitting.gui.worker import FitWorker

        return FitWorker._build_size_distribution(None, params)
    except Exception:
        return None


class LegacyHost(QtCore.QObject):
    """A hidden classic MainWindow serving one phase or one dataset."""

    configChanged = QtCore.Signal()  # the Builder state changed (e.g. shape dialog)

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        from pdf_fitting.gui.mainwindow import MainWindow

        self.win = MainWindow()
        self.win.setWindowTitle(title or "classic host")
        self.win._current_path = None
        self._loading = False
        self.popped_out = False
        self.user_session = False  # set when the user edits through this host
        self._dock_layout: Optional[QtWidgets.QLayout] = None
        self._placeholder: Optional[QtWidgets.QWidget] = None

        # The classic "Close PDF plot" hides its pop-out window but keeps the
        # plot inside it; here it docks the plot back into the project window.
        try:
            self.win.btn_detach_pdf.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.win.btn_detach_pdf.setText("Dock back")
        self.win.btn_detach_pdf.setToolTip("Put the plot back into the main window.")
        self.win.btn_detach_pdf.clicked.connect(self.dock_back)
        self.win.attach_pdf_plot = self.dock_back  # used by the dialog's closeEvent

        self._change_timer = QtCore.QTimer(self)
        self._change_timer.setSingleShot(True)
        self._change_timer.setInterval(400)
        self._change_timer.timeout.connect(self.configChanged)
        self.win.builder.changed.connect(self._builder_changed)

    # ------------------------------------------------------------------
    # Config in / out
    # ------------------------------------------------------------------

    def load_config(self, cfg: Dict[str, Any], params: Optional[Dict[str, Any]] = None):
        cfg = copy.deepcopy(cfg)
        self._loading = True
        w = self.win
        w._sync_block = True
        try:
            w.raw_editor.setPlainText(legacy_cfg_to_ini_text(cfg))
            w.builder.set_from_config(cfg)
        finally:
            w._sync_block = False
            QtCore.QTimer.singleShot(0, self._end_loading)
        w._current_path = None
        if params is not None:
            w._last_refined_params = dict(params)
        self._cfg = cfg

    def _end_loading(self):
        self._loading = False
        self._change_timer.stop()

    def _builder_changed(self):
        if not self._loading:
            self._change_timer.start()

    def compose_config(self) -> Dict[str, Any]:
        """Current Builder state as a single-phase config dict."""
        return ini_text_to_cfg(self.win._compose_input_text_for_run())

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def show_result(self, payload: Dict[str, Any]):
        w = self.win
        w._current_path = None
        payload = dict(payload)
        payload.pop("input_path", None)
        payload.pop("refined_params", None)  # avoid the classic "save to file" prompt
        try:
            w.on_finished(payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[PROJECT] classic result display failed: {exc}")

    # ------------------------------------------------------------------
    # Widgets
    # ------------------------------------------------------------------

    @property
    def pdf_container(self) -> QtWidgets.QWidget:
        return self.win.pdf_container

    @property
    def output_tabs(self) -> QtWidgets.QTabWidget:
        return self.win.tabs

    def dock_into(self, layout: QtWidgets.QLayout, placeholder: QtWidgets.QWidget):
        """
        Show the PDF plot inside ``layout``; ``placeholder`` shows while the
        plot is popped out. A popped-out plot stays out.
        """
        self._dock_layout = layout
        self._placeholder = placeholder
        if self.popped_out:
            placeholder.setVisible(True)
        else:
            self.dock_back()

    def pop_out(self, force_show: bool = False):
        self.popped_out = True
        if self._placeholder is not None:
            self._placeholder.setVisible(True)
        dlg = getattr(self.win, "_pdf_dialog", None)
        dialog_open = dlg is not None and dlg.isVisible() and not dlg.isMinimized()
        if not dialog_open or force_show:
            self.win.detach_pdf_plot()
        ensure_on_screen(getattr(self.win, "_pdf_dialog", None))

    def update_live(self, r, G_exp, G_fit, title: str = ""):
        """Redraw the fit during a refinement (classic plot, keeps styles)."""
        w = self.win
        try:
            w._update_pdf_plot(r, G_exp, G_fit, G_exp - G_fit, title_suffix="(refining)")
            if title:
                w.plot_pdf.ax.set_title(title, fontsize=10)
            w.plot_pdf.canvas.draw_idle()
        except Exception as exc:  # noqa: BLE001
            print(f"[PROJECT] live plot update failed: {exc}")

    def dock_back(self):
        self.popped_out = False
        w = self.win
        w._returning_pdf_to_main = True
        try:
            if getattr(w, "_pdf_dialog", None) is not None:
                w._pdf_dialog.hide()
            w._pdf_detached = False
            if self._dock_layout is not None:
                self._dock_layout.addWidget(w.pdf_container, 1)
                w.pdf_container.setVisible(True)
            if self._placeholder is not None:
                self._placeholder.setVisible(False)
        finally:
            w._returning_pdf_to_main = False

    def shutdown(self):
        try:
            if getattr(self.win, "_pdf_dialog", None) is not None:
                self.win._pdf_dialog.hide()
            for studio in (getattr(self.win, "_structure_studio", None),):
                if studio is not None:
                    studio.shutdown()
        except Exception:  # noqa: BLE001
            pass


class DatasetPlotHost(LegacyHost):
    """
    Classic PDF plot for a whole dataset: observed vs total fit, pair
    contributions and tick markers of every phase, prefixed with the phase
    name and coloured per phase by default.
    """

    def __init__(self, title: str, phases_provider: Callable[[], List[Dict[str, Any]]], parent=None):
        super().__init__(title, parent)
        self._phases_provider = phases_provider
        # The tick-marker toggle looks this up at call time.
        self.win._calculate_tick_markers_to_r = self._calculate_merged_ticks

    def _calculate_merged_ticks(self, max_r: float) -> bool:
        """Tick markers of every active phase, merged with a phase prefix."""
        from pdf_fitting.models.gr_model import PDFCalculator

        w = self.win
        try:
            max_r = float(max_r)
            merged: Dict[str, Any] = {"pairs": {}}
            for entry in self._phases_provider():
                cfg, params, name, color = entry["cfg"], entry["params"], entry["name"], entry["color"]
                w.statusBar().showMessage(f"Tick markers: {name} up to {max_r:.3g} Å…")
                QtWidgets.QApplication.processEvents()
                r_dummy = np.linspace(float(cfg.get("r_min", 0.0)), max_r, 10)
                calc = PDFCalculator(cfg, cfg["structure_file"], r_dummy, pair_cutoff=max_r)
                ticks = calc.build_pdf_tick_data(params, max_r=max_r, max_ticks_per_pair=None)
                for pair, data in (ticks.get("pairs") or {}).items():
                    key = f"{name}: {pair}"
                    merged["pairs"][key] = data
                    w._tick_pair_style.setdefault(key, {"color": color, "linewidth": 0.7, "alpha": 0.35})
            w._set_pdf_tick_data(merged)
            return True
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(w.window(), "Tick marker calculation failed", str(exc))
            return False


def ensure_on_screen(window) -> None:
    """
    Make a top-level window visible: restore it when minimized, give it a
    sensible size and move it onto a screen when it is (partly) off-screen
    or collapsed, then raise it.
    """
    if window is None:
        return
    try:
        if window.isMinimized():
            window.showNormal()
        if not window.isVisible():
            window.show()
        screen = QtWidgets.QApplication.screenAt(window.frameGeometry().center()) or QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        geo = window.frameGeometry()
        too_small = geo.width() < 400 or geo.height() < 300
        off = not avail.intersects(geo) or not avail.contains(geo.center())
        if (too_small or off) and not window.isMaximized():
            w = max(800, int(avail.width() * 0.6))
            h = max(600, int(avail.height() * 0.6))
            window.resize(min(w, avail.width()), min(h, avail.height()))
            window.move(avail.center().x() - window.width() // 2, avail.center().y() - window.height() // 2)
        window.raise_()
        window.activateWindow()
    except RuntimeError:
        pass
