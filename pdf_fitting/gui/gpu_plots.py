from __future__ import annotations

import os
import html
from typing import Any, Dict, List, Optional

import numpy as np

from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets

import pyqtgraph as pg


def _environment_bool(
    name: str,
    default: bool,
) -> bool:
    value = os.environ.get(
        name,
        None,
    )

    if value is None:
        return bool(default)

    text = str(value).strip().lower()

    if text in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    ):
        return True

    if text in (
        "0",
        "false",
        "no",
        "n",
        "off",
    ):
        return False

    return bool(default)


GPU_PLOTS_USE_OPENGL = _environment_bool(
    "HKL_PDF_GPU_PLOTS",
    True,
)

GPU_PLOTS_ANTIALIAS = _environment_bool(
    "HKL_PDF_GPU_ANTIALIAS",
    True,
)


try:
    pg.setConfigOptions(
        useOpenGL=GPU_PLOTS_USE_OPENGL,
        antialias=GPU_PLOTS_ANTIALIAS,
    )
except Exception:
    pg.setConfigOptions(
        useOpenGL=False,
        antialias=True,
    )

    GPU_PLOTS_USE_OPENGL = False


def _normalise_color(
    color: Any,
    default: str = "#1f77b4",
):
    if color is None:
        return default

    if isinstance(
        color,
        str,
    ):
        text = color.strip()

        matplotlib_names = {
            "tab:blue": "#1f77b4",
            "tab:orange": "#ff7f0e",
            "tab:green": "#2ca02c",
            "tab:red": "#d62728",
            "tab:purple": "#9467bd",
            "tab:brown": "#8c564b",
            "tab:pink": "#e377c2",
            "tab:gray": "#7f7f7f",
            "tab:grey": "#7f7f7f",
            "tab:olive": "#bcbd22",
            "tab:cyan": "#17becf",
        }

        return matplotlib_names.get(
            text.lower(),
            text or default,
        )

    return color


class FastPlotCurve:
    """
    Small compatibility wrapper around pyqtgraph.PlotDataItem.

    The existing MainWindow code calls line.set_data(x, y), so this wrapper
    provides the same method name.
    """

    def __init__(
        self,
        item,
    ):
        self.item = item
        self._x = np.asarray(
            [],
            dtype=float,
        )
        self._y = np.asarray(
            [],
            dtype=float,
        )

    def set_data(
        self,
        x,
        y,
    ) -> None:
        self._x = np.asarray(
            x,
            dtype=float,
        )

        self._y = np.asarray(
            y,
            dtype=float,
        )

        self.item.setData(
            self._x,
            self._y,
        )

    def setData(
        self,
        x,
        y,
    ) -> None:
        self.set_data(
            x,
            y,
        )

    def get_xdata(self):
        return self._x

    def get_ydata(self):
        return self._y

    def set_visible(
        self,
        visible: bool,
    ) -> None:
        self.item.setVisible(
            bool(visible)
        )

    def setVisible(
        self,
        visible: bool,
    ) -> None:
        self.set_visible(
            visible
        )


class FastPlotCanvasAdapter:
    """
    Compatibility adapter for existing canvas.draw_idle() calls.
    """

    def __init__(
        self,
        owner,
    ):
        self.owner = owner

    def draw_idle(self) -> None:
        try:
            self.owner.plot_widget.update()
        except Exception:
            pass

    def draw(self) -> None:
        self.draw_idle()


class FastPlotAxisAdapter:
    """
    Minimal Matplotlib-like axis interface used by the current Rwp code.

    This is intentionally limited to the methods currently required by the
    Rwp plot. Other scientific plots remain on Matplotlib for now.
    """

    def __init__(
        self,
        owner,
    ):
        self.owner = owner

    def clear(self) -> None:
        self.owner.clear()

    def set_title(
        self,
        title: str,
    ) -> None:
        self.owner.set_title(
            title
        )

    def set_xlabel(
        self,
        label: str,
    ) -> None:
        self.owner.set_xlabel(
            label
        )

    def set_ylabel(
        self,
        label: str,
    ) -> None:
        self.owner.set_ylabel(
            label
        )

    def plot(
        self,
        x,
        y,
        *args,
        **kwargs,
    ):
        curve = self.owner.add_curve(
            x,
            y,
            color=kwargs.get(
                "color",
                "#1f77b4",
            ),
            width=kwargs.get(
                "linewidth",
                1.5,
            ),
            name=kwargs.get(
                "label",
                None,
            ),
            symbol=kwargs.get(
                "marker",
                None,
            ),
        )

        return (curve,)

    def relim(
        self,
        *args,
        **kwargs,
    ) -> None:
        return

    def autoscale_view(
        self,
        *args,
        **kwargs,
    ) -> None:
        self.owner.auto_range()

    def autoscale(
        self,
        *args,
        **kwargs,
    ) -> None:
        self.owner.auto_range()

    def set_xlim(
        self,
        minimum,
        maximum,
    ) -> None:
        self.owner.set_x_range(
            minimum,
            maximum,
        )

    def set_ylim(
        self,
        minimum,
        maximum,
    ) -> None:
        self.owner.set_y_range(
            minimum,
            maximum,
        )

    def get_xlim(self):
        ranges = self.owner.plot_widget.viewRange()
        return tuple(ranges[0])

    def get_ylim(self):
        ranges = self.owner.plot_widget.viewRange()
        return tuple(ranges[1])

    def grid(
        self,
        enabled=True,
        *args,
        **kwargs,
    ) -> None:
        self.owner.plot_widget.showGrid(
            x=bool(enabled),
            y=bool(enabled),
            alpha=0.25,
        )

class FastPlotToolbar(QtWidgets.QWidget):
    """
    Lightweight toolbar used by FastLinePlot.

    It provides the addWidget() and addSeparator() methods used by the existing
    MainWindow code, allowing PyQtGraph plots to accept the same custom controls
    previously inserted into Matplotlib toolbars.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._layout = QtWidgets.QHBoxLayout(self)
        self._layout.setContentsMargins(4, 2, 4, 2)
        self._layout.setSpacing(4)
        self._layout.addStretch(1)

    def addWidget(self, widget):
        stretch_item = self._layout.takeAt(self._layout.count() - 1)

        self._layout.addWidget(widget)

        if stretch_item is not None:
            self._layout.addItem(stretch_item)

        return widget

    def addSeparator(self):
        separator = QtWidgets.QFrame(self)
        separator.setFrameShape(QtWidgets.QFrame.VLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        separator.setFixedWidth(8)

        self.addWidget(separator)

        return separator
    

class FastLinePlot(QtWidgets.QWidget):
    """
    Fast Qt-native line plot using PyQtGraph.

    OpenGL use is controlled by:

        HKL_PDF_GPU_PLOTS=1

    Set it to zero to disable OpenGL:

        HKL_PDF_GPU_PLOTS=0
    """

    def __init__(
        self,
        parent=None,
        *,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ):
        super().__init__(
            parent
        )

        self._title = str(
            title
        )

        self._curves: List[FastPlotCurve] = []

        self.toolbar = FastPlotToolbar(self)

        self.plot_widget = pg.PlotWidget(
            parent=self,
        )

        self.plot_widget.setBackground(
            "w"
        )

        self.plot_widget.showGrid(
            x=True,
            y=True,
            alpha=0.20,
        )

        self.plot_widget.setMouseEnabled(
            x=True,
            y=True,
        )

        self.plot_widget.setMenuEnabled(
            True,
        )

        self.plot_widget.getPlotItem().setClipToView(
            True
        )

        try:
            self.legend = self.plot_widget.getPlotItem().addLegend(
                offset=(-10, 10),
                labelTextColor="#202020",
                brush=pg.mkBrush(
                    255,
                    255,
                    255,
                    210,
                ),
                pen=pg.mkPen(
                    "#B0B0B0",
                    width=1.0,
                ),
            )
        except Exception:
            self.legend = None

        try:
            self.plot_widget.getPlotItem().setDownsampling(
                auto=True,
                mode="peak",
            )
        except Exception:
            pass

        if title:
            self.set_title(
                title
            )

        if xlabel:
            self.set_xlabel(
                xlabel
            )

        if ylabel:
            self.set_ylabel(
                ylabel
            )

        self.backend_label = QtWidgets.QLabel()

        if GPU_PLOTS_USE_OPENGL:
            self.backend_label.setText(
                "PyQtGraph / OpenGL"
            )
        else:
            self.backend_label.setText(
                "PyQtGraph / Qt"
            )

        self.backend_label.setStyleSheet(
            "color: #707070; padding: 2px 6px;"
        )

        self.backend_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
        )

        self.info_label = QtWidgets.QLabel()
        self.info_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self.info_label.setWordWrap(
            True
        )
        self.info_label.setStyleSheet(
            """
            QLabel {
                color: #303030;
                background-color: #F4F4F4;
                border: 1px solid #D0D0D0;
                border-radius: 3px;
                padding: 4px 7px;
            }
            """
        )
        self.info_label.setVisible(
            False
        )

        self.message_label = QtWidgets.QLabel()
        self.message_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.message_label.setWordWrap(
            True
        )
        self.message_label.setStyleSheet(
            """
            QLabel {
                color: #606060;
                font-size: 12pt;
                padding: 12px;
            }
            """
        )
        self.message_label.setVisible(
            False
        )

        layout = QtWidgets.QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        layout.setSpacing(
            0
        )

        layout.addWidget(
            self.toolbar
        )

        layout.addWidget(
            self.backend_label
        )

        layout.addWidget(
            self.info_label
        )

        layout.addWidget(
            self.message_label
        )

        layout.addWidget(
            self.plot_widget,
            1,
        )

        self.ax = FastPlotAxisAdapter(
            self
        )

        self.canvas = FastPlotCanvasAdapter(
            self
        )

    def set_info(
        self,
        text: str,
    ) -> None:
        text = str(
            text or ""
        ).strip()

        self.info_label.setText(
            text
        )

        self.info_label.setVisible(
            bool(text)
        )


    def show_message(
        self,
        text: str,
    ) -> None:
        text = str(
            text or ""
        ).strip()

        self.message_label.setText(
            text
        )

        self.message_label.setVisible(
            bool(text)
        )

        self.plot_widget.setVisible(
            not bool(text)
        )


    def hide_message(self) -> None:
        self.message_label.setVisible(
            False
        )

        self.plot_widget.setVisible(
            True
        )

    def clear(self) -> None:
        self.plot_widget.clear()
        self._curves = []

        self.set_info(
            ""
        )

        self.hide_message()

        if self._title:
            self.set_title(
                self._title
            )


    def set_title(
        self,
        title: str,
    ) -> None:
        self._title = str(
            title
        )

        self.plot_widget.setTitle(
            self._title,
            color="#202020",
            size="11pt",
        )

    def set_xlabel(
        self,
        label: str,
    ) -> None:
        self.plot_widget.setLabel(
            "bottom",
            str(label),
        )

    def set_ylabel(
        self,
        label: str,
    ) -> None:
        self.plot_widget.setLabel(
            "left",
            str(label),
        )

    def add_curve(
        self,
        x=None,
        y=None,
        *,
        color="#1f77b4",
        width: float = 1.5,
        name: Optional[str] = None,
        symbol=None,
        symbol_size: float = 6.0,
        fill_level=None,
        fill_color=None,
        fill_alpha: int = 45,
        style: str = "-",
        alpha: float = 1.0,
    ) -> FastPlotCurve:
        if x is None:
            x = np.asarray(
                [],
                dtype=float,
            )

        if y is None:
            y = np.asarray(
                [],
                dtype=float,
            )

        color = _normalise_color(
            color,
            "#1f77b4",
        )

        qcolor = pg.mkColor(color)
        qcolor.setAlphaF(
            max(
                0.0,
                min(
                    1.0,
                    float(alpha),
                ),
            )
        )

        style_text = str(style or "-").strip()

        qt_pen_styles = {
            "-": QtCore.Qt.PenStyle.SolidLine,
            "--": QtCore.Qt.PenStyle.DashLine,
            "-.": QtCore.Qt.PenStyle.DashDotLine,
            ":": QtCore.Qt.PenStyle.DotLine,
            "None": QtCore.Qt.PenStyle.NoPen,
            "none": QtCore.Qt.PenStyle.NoPen,
            "": QtCore.Qt.PenStyle.NoPen,
        }

        pen = pg.mkPen(
            color=qcolor,
            width=float(width),
            style=qt_pen_styles.get(
                style_text,
                QtCore.Qt.PenStyle.SolidLine,
            ),
        )

        symbol_map = {
            "o": "o",
            ".": "o",
            "s": "s",
            "^": "t",
            "v": "t1",
            "D": "d",
            "d": "d",
            "x": "x",
            "+": "+",
        }

        if symbol in (
            None,
            "",
            "None",
            "none",
        ):
            symbol_value = None
        else:
            symbol_value = symbol_map.get(
                str(symbol),
                str(symbol),
            )

        # PyQtGraph legend labels use rich-text/HTML rendering.
        #
        # Warren direction labels are written as:
        #
        #     <1 0 0>
        #
        # Without HTML escaping, PyQtGraph interprets this as an HTML tag,
        # producing an empty legend label. Escape all labels before passing
        # them to PlotDataItem.
        if name is None:
            legend_name = None
        else:
            legend_name = html.escape(
                str(name)
            )

        plot_arguments = {
            "pen": pen,
            "symbol": symbol_value,
            "symbolSize": float(symbol_size),
            "symbolBrush": qcolor,
            "symbolPen": qcolor,
            "name": legend_name,
        }

        if fill_level is not None:
            plot_arguments["fillLevel"] = float(
                fill_level
            )

            if fill_color is None:
                fill_color = color

            fill_qcolor = pg.mkColor(
                _normalise_color(
                    fill_color,
                    color,
                )
            )

            fill_qcolor.setAlpha(
                max(
                    0,
                    min(
                        255,
                        int(fill_alpha),
                    ),
                )
            )

            plot_arguments["brush"] = pg.mkBrush(
                fill_qcolor
            )

        item = self.plot_widget.plot(
            np.asarray(
                x,
                dtype=float,
            ),
            np.asarray(
                y,
                dtype=float,
            ),
            **plot_arguments,
        )

        try:
            item.setClipToView(
                True
            )
        except Exception:
            pass

        try:
            item.setDownsampling(
                auto=True,
                method="peak",
            )
        except Exception:
            pass

        curve = FastPlotCurve(
            item
        )

        curve.set_data(
            x,
            y,
        )

        self._curves.append(
            curve
        )

        return curve
    
    def auto_range(self) -> None:
        try:
            self.plot_widget.enableAutoRange(
                axis="xy",
                enable=True,
            )

            self.plot_widget.autoRange()

        except Exception:
            try:
                self.plot_widget.autoRange()
            except Exception:
                pass

    def set_x_range(
        self,
        minimum,
        maximum,
    ) -> None:
        self.plot_widget.setXRange(
            float(minimum),
            float(maximum),
            padding=0.0,
        )

    def set_y_range(
        self,
        minimum,
        maximum,
    ) -> None:
        self.plot_widget.setYRange(
            float(minimum),
            float(maximum),
            padding=0.0,
        )


class FastParamEvolutionTabs(QtWidgets.QTabWidget):
    """
    GPU-capable parameter-evolution plots using PyQtGraph.

    This has the same public methods used by MainWindow:

        set_zoom_last_n()
        clear_all()
        ensure_param()
        push()
    """

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(
            parent
        )

        self._plots: Dict[str, FastLinePlot] = {}
        self._lines: Dict[str, FastPlotCurve] = {}
        self._series: Dict[str, Dict[str, List[float]]] = {}
        self._zoom_last_n = 0

    def set_zoom_last_n(
        self,
        number_of_points: int,
    ) -> None:
        self._zoom_last_n = max(
            0,
            int(number_of_points),
        )

    def clear_all(self) -> None:
        for plot in self._plots.values():
            try:
                plot.deleteLater()
            except Exception:
                pass

        self._plots = {}
        self._lines = {}
        self._series = {}

        self.clear()

    def ensure_param(
        self,
        name: str,
    ) -> None:
        name = str(
            name
        )

        if name in self._plots:
            return

        plot = FastLinePlot(
            title=f"{name} vs nfev",
            xlabel="nfev",
            ylabel=name,
        )

        curve = plot.add_curve(
            [],
            [],
            color="#1f77b4",
            width=1.6,
        )

        self._plots[name] = plot
        self._lines[name] = curve
        self._series[name] = {
            "x": [],
            "y": [],
        }

        self.addTab(
            plot,
            name,
        )

    def push(
        self,
        nfev: int,
        params: Dict[str, Any],
        *,
        allowed: Optional[set[str]] = None,
    ) -> None:
        if nfev <= 0:
            return

        if not isinstance(
            params,
            dict,
        ):
            return

        for key, value in params.items():
            name = str(
                key
            )

            if (
                allowed is not None
                and name not in allowed
            ):
                continue

            try:
                y_value = float(
                    value
                )
            except Exception:
                continue

            self.ensure_param(
                name
            )

            series = self._series[
                name
            ]

            series["x"].append(
                float(nfev)
            )

            series["y"].append(
                y_value
            )

            x_values = np.asarray(
                series["x"],
                dtype=float,
            )

            y_values = np.asarray(
                series["y"],
                dtype=float,
            )

            curve = self._lines[
                name
            ]

            curve.set_data(
                x_values,
                y_values,
            )

            plot = self._plots[
                name
            ]

            if (
                self._zoom_last_n > 0
                and x_values.size >= 2
            ):
                x_maximum = float(
                    x_values[-1]
                )

                x_minimum = max(
                    0.0,
                    x_maximum
                    - float(
                        self._zoom_last_n
                    ),
                )

                visible = (
                    x_values >= x_minimum
                )

                plot.set_x_range(
                    x_minimum,
                    x_maximum,
                )

                visible_y = y_values[
                    visible
                ]

                visible_y = visible_y[
                    np.isfinite(
                        visible_y
                    )
                ]

                if visible_y.size > 0:
                    y_minimum = float(
                        np.min(
                            visible_y
                        )
                    )

                    y_maximum = float(
                        np.max(
                            visible_y
                        )
                    )

                    if y_minimum == y_maximum:
                        padding = max(
                            abs(y_minimum) * 1.0e-6,
                            1.0e-12,
                        )
                    else:
                        padding = 0.05 * (
                            y_maximum
                            - y_minimum
                        )

                    plot.set_y_range(
                        y_minimum - padding,
                        y_maximum + padding,
                    )

            else:
                plot.auto_range()

class FastGridMap(QtWidgets.QWidget):
    """
    PyQtGraph/OpenGL grid-map widget for crystallite-shape grid searches.

    The map uses ImageItem for regular two-dimensional grids and a GPU-capable
    ScatterPlotItem fallback when only one diameter or one height is available.
    """

    def __init__(
        self,
        parent=None,
        *,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ):
        super().__init__(parent)

        self._title = str(title)
        self._xlabel = str(xlabel)
        self._ylabel = str(ylabel)

        self.toolbar = FastPlotToolbar(self)

        self.backend_label = QtWidgets.QLabel(
            "PyQtGraph / OpenGL"
            if GPU_PLOTS_USE_OPENGL
            else "PyQtGraph / Qt"
        )
        self.backend_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
        )
        self.backend_label.setStyleSheet(
            "color: #707070; padding: 2px 6px;"
        )

        self.plot_widget = pg.PlotWidget(self)
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(
            x=True,
            y=True,
            alpha=0.20,
        )
        self.plot_widget.setMenuEnabled(True)
        self.plot_widget.setMouseEnabled(
            x=True,
            y=True,
        )

        try:
            self.plot_widget.getPlotItem().setClipToView(True)
        except Exception:
            pass

        self.histogram = pg.HistogramLUTWidget(self)
        self.histogram.setMinimumWidth(100)
        self.histogram.setMaximumWidth(145)

        graph_row = QtWidgets.QHBoxLayout()
        graph_row.setContentsMargins(0, 0, 0, 0)
        graph_row.setSpacing(2)
        graph_row.addWidget(self.plot_widget, 1)
        graph_row.addWidget(self.histogram)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.backend_label)
        layout.addLayout(graph_row, 1)

        self.image_item = None
        self.scatter_item = None

        self.set_title(title)
        self.set_xlabel(xlabel)
        self.set_ylabel(ylabel)

    def set_title(self, title: str) -> None:
        self._title = str(title)

        self.plot_widget.setTitle(
            self._title,
            color="#202020",
            size="11pt",
        )

    def set_xlabel(self, label: str) -> None:
        self._xlabel = str(label)
        self.plot_widget.setLabel(
            "bottom",
            self._xlabel,
        )

    def set_ylabel(self, label: str) -> None:
        self._ylabel = str(label)
        self.plot_widget.setLabel(
            "left",
            self._ylabel,
        )

    def clear(self) -> None:
        try:
            self.plot_widget.clear()
        except Exception:
            pass

        self.image_item = None
        self.scatter_item = None

        try:
            self.histogram.setImageItem(None)
        except Exception:
            pass

        self.set_title(self._title)
        self.set_xlabel(self._xlabel)
        self.set_ylabel(self._ylabel)

    def _spacing_edges(self, values: np.ndarray):
        values = np.asarray(values, dtype=float)

        if values.size <= 1:
            center = float(values[0]) if values.size else 0.0
            return center - 0.5, center + 0.5

        differences = np.diff(values)

        first = float(values[0] - 0.5 * differences[0])
        last = float(values[-1] + 0.5 * differences[-1])

        return first, last

    def set_points(
        self,
        diameter,
        height,
        rwp,
    ) -> None:
        self.clear()

        diameter = np.asarray(diameter, dtype=float)
        height = np.asarray(height, dtype=float)
        rwp = np.asarray(rwp, dtype=float)

        valid = (
            np.isfinite(diameter)
            & np.isfinite(height)
            & np.isfinite(rwp)
        )

        if not np.any(valid):
            return

        diameter = diameter[valid]
        height = height[valid]
        rwp = rwp[valid]

        try:
            colour_map = pg.colormap.get("viridis")
            colours = colour_map.map(
                rwp,
                mode="qcolor",
            )
        except Exception:
            minimum = float(np.min(rwp))
            maximum = float(np.max(rwp))
            span = max(maximum - minimum, 1.0e-12)
            normalised = (rwp - minimum) / span

            colours = [
                pg.intColor(
                    int(value * 255),
                    hues=256,
                )
                for value in normalised
            ]

        self.scatter_item = pg.ScatterPlotItem(
            x=diameter,
            y=height,
            size=13,
            brush=colours,
            pen=pg.mkPen("#303030", width=1.0),
            pxMode=True,
        )

        self.plot_widget.addItem(
            self.scatter_item
        )

        self.plot_widget.autoRange()

    def set_grid(
        self,
        diameter_values,
        height_values,
        z_values,
    ) -> None:
        self.clear()

        diameter_values = np.asarray(
            diameter_values,
            dtype=float,
        )

        height_values = np.asarray(
            height_values,
            dtype=float,
        )

        z_values = np.asarray(
            z_values,
            dtype=float,
        )

        if (
            diameter_values.size < 2
            or height_values.size < 2
        ):
            dd, hh = np.meshgrid(
                diameter_values,
                height_values,
            )

            self.set_points(
                dd.ravel(),
                hh.ravel(),
                z_values.ravel(),
            )
            return

        if z_values.shape != (
            height_values.size,
            diameter_values.size,
        ):
            raise ValueError(
                "z_values shape must be "
                "(len(height_values), len(diameter_values))."
            )

        finite = z_values[
            np.isfinite(z_values)
        ]

        if finite.size == 0:
            return

        z_display = np.asarray(
            z_values.T,
            dtype=float,
        )

        self.image_item = pg.ImageItem(
            z_display
        )

        d0, d1 = self._spacing_edges(
            diameter_values
        )

        h0, h1 = self._spacing_edges(
            height_values
        )

        self.image_item.setRect(
            QtCore.QRectF(
                float(d0),
                float(h0),
                float(d1 - d0),
                float(h1 - h0),
            )
        )

        try:
            colour_map = pg.colormap.get("viridis")
            self.image_item.setLookupTable(
                colour_map.getLookupTable(
                    0.0,
                    1.0,
                    256,
                )
            )
        except Exception:
            pass

        minimum = float(np.nanmin(finite))
        maximum = float(np.nanmax(finite))

        if minimum == maximum:
            padding = max(
                abs(minimum) * 1.0e-6,
                1.0e-9,
            )
            minimum -= padding
            maximum += padding

        self.image_item.setLevels(
            (
                minimum,
                maximum,
            )
        )

        self.plot_widget.addItem(
            self.image_item
        )

        try:
            self.histogram.setImageItem(
                self.image_item
            )
            self.histogram.setLevels(
                minimum,
                maximum,
            )
        except Exception:
            pass

        self.plot_widget.setXRange(
            float(d0),
            float(d1),
            padding=0.0,
        )

        self.plot_widget.setYRange(
            float(h0),
            float(h1),
            padding=0.0,
        )

    @property
    def canvas(self):
        return FastPlotCanvasAdapter(self)

    def render(self):
        try:
            self.plot_widget.update()
        except Exception:
            pass