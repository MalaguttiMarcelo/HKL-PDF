from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg

from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets

from .gpu_plots import (
    FastLinePlot,
)
from .plots import (
    MplPlot,
)


def _float_value(
    value,
    default=np.nan,
) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _finite_float(
    value,
) -> bool:
    try:
        return bool(
            np.isfinite(
                float(value)
            )
        )
    except Exception:
        return False


def _colour_values(
    values: np.ndarray,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    finite = values[
        np.isfinite(values)
    ]

    if finite.size == 0:
        return [
            QtGui.QColor("#808080")
            for _ in values
        ]

    minimum = float(
        np.min(finite)
    )

    maximum = float(
        np.max(finite)
    )

    if maximum <= minimum:
        normalised = np.zeros_like(
            values,
            dtype=float,
        )
    else:
        normalised = (
            values
            - minimum
        ) / (
            maximum
            - minimum
        )

    normalised = np.nan_to_num(
        normalised,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    try:
        colour_map = pg.colormap.get(
            "viridis"
        )

        return colour_map.map(
            normalised,
            mode="qcolor",
        )

    except Exception:
        return [
            pg.intColor(
                int(
                    value
                    * 255
                ),
                hues=256,
            )
            for value in normalised
        ]

class ClickableGridMap(
    QtWidgets.QWidget
):
    """
    Discrete grid-search colour map.

    Features
    --------
    - white background;
    - one bordered square per completed simulation;
    - Rwp or mapped value written inside each square;
    - right-side colour scale;
    - click a square to select its candidate;
    - optional automatic view fitting as new results arrive.
    """

    cellSelected = QtCore.Signal(object)

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

        self._rows_by_cell: Dict[
            Tuple[int, int],
            Dict[str, Any],
        ] = {}

        self._x_values = np.asarray(
            [],
            dtype=float,
        )

        self._y_values = np.asarray(
            [],
            dtype=float,
        )

        self._cell_items = []
        self._label_items = []

        self.graphics = pg.GraphicsLayoutWidget(
            self
        )

        self.graphics.setBackground(
            "w"
        )

        self.plot = self.graphics.addPlot(
            row=0,
            col=0,
        )

        self.plot.setTitle(
            title,
            color="#202020",
        )

        self.plot.setLabel(
            "bottom",
            xlabel,
            color="#202020",
        )

        self.plot.setLabel(
            "left",
            ylabel,
            color="#202020",
        )

        self.plot.showGrid(
            x=True,
            y=True,
            alpha=0.15,
        )

        self.plot.getAxis(
            "bottom"
        ).setPen(
            pg.mkPen(
                "#404040",
                width=1.0,
            )
        )

        self.plot.getAxis(
            "left"
        ).setPen(
            pg.mkPen(
                "#404040",
                width=1.0,
            )
        )

        self.plot.getAxis(
            "bottom"
        ).setTextPen(
            "#303030"
        )

        self.plot.getAxis(
            "left"
        ).setTextPen(
            "#303030"
        )

        # This image is hidden. It exists only to drive HistogramLUTItem,
        # which supplies the colour scale at the right side.
        self._colourbar_image = pg.ImageItem()

        self._colourbar_image.setVisible(
            False
        )

        self.plot.addItem(
            self._colourbar_image
        )

        self.colorbar = pg.HistogramLUTItem(
            image=self._colourbar_image,
            fillHistogram=False,
        )

        self.graphics.addItem(
            self.colorbar,
            row=0,
            col=1,
        )

        try:
            self.colour_map = pg.colormap.get(
                "magma"
            )

            self.colorbar.gradient.setColorMap(
                self.colour_map
            )

        except Exception:
            self.colour_map = pg.colormap.get(
                "viridis"
            )

            self.colorbar.gradient.setColorMap(
                self.colour_map
            )

        self.info_label = QtWidgets.QLabel()

        self.info_label.setWordWrap(
            True
        )

        self.chk_auto_fit = QtWidgets.QCheckBox(
            "Auto-fit new results"
        )

        self.chk_auto_fit.setChecked(
            True
        )

        self.btn_reset_view = QtWidgets.QPushButton(
            "Reset zoom"
        )

        self.btn_reset_view.setToolTip(
            "Reset the map dimensions so all currently completed "
            "grid-search cells are visible."
        )

        controls = QtWidgets.QHBoxLayout()

        controls.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        controls.addWidget(
            self.chk_auto_fit
        )

        controls.addWidget(
            self.btn_reset_view
        )

        controls.addStretch(
            1
        )

        layout = QtWidgets.QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            4,
            4,
            4,
            4,
        )

        layout.addWidget(
            self.info_label
        )

        layout.addLayout(
            controls
        )

        layout.addWidget(
            self.graphics,
            1,
        )

        self.btn_reset_view.clicked.connect(
            self.reset_view
        )

        self.plot.scene().sigMouseClicked.connect(
            self._plot_clicked
        )

    def clear(
        self,
    ) -> None:
        self._clear_cells()

        self._rows_by_cell = {}

        self._x_values = np.asarray(
            [],
            dtype=float,
        )

        self._y_values = np.asarray(
            [],
            dtype=float,
        )

        self.info_label.setText(
            ""
        )

    def _clear_cells(
        self,
    ) -> None:
        for item in self._cell_items:
            try:
                self.plot.removeItem(
                    item
                )
            except Exception:
                pass

        for item in self._label_items:
            try:
                self.plot.removeItem(
                    item
                )
            except Exception:
                pass

        self._cell_items = []
        self._label_items = []

    @staticmethod
    def _cell_edges(
        values: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(
            values,
            dtype=float,
        )

        if values.size == 1:
            delta = max(
                abs(
                    float(
                        values[0]
                    )
                )
                * 0.05,
                0.5,
            )

            return np.asarray(
                [
                    values[0] - delta,
                    values[0] + delta,
                ],
                dtype=float,
            )

        middle = 0.5 * (
            values[:-1]
            + values[1:]
        )

        first = values[0] - (
            middle[0]
            - values[0]
        )

        last = values[-1] + (
            values[-1]
            - middle[-1]
        )

        return np.concatenate(
            (
                [first],
                middle,
                [last],
            )
        )

    @staticmethod
    def _text_colour(
        colour: QtGui.QColor,
    ) -> str:
        """
        Choose black or white text depending on cell brightness.
        """
        brightness = (
            0.299 * colour.red()
            + 0.587 * colour.green()
            + 0.114 * colour.blue()
        )

        if brightness >= 155.0:
            return "#111111"

        return "#FFFFFF"

    def reset_view(
        self,
    ) -> None:
        """
        Reset dimensions to include all currently shown simulations.
        """
        try:
            self.plot.autoRange(
                padding=0.05,
            )
        except Exception:
            pass

    def set_grid(
        self,
        x_values,
        y_values,
        mapped_values,
        rows: List[Dict[str, Any]],
        *,
        information: str = "",
        colorbar_label: str = "Rwp (%)",
        cell_text_values=None,
    ) -> None:
        """
        Draw a bordered colour-cell map.

        Parameters
        ----------
        x_values, y_values
            Grid coordinates.

        mapped_values
            Values controlling the cell colour, normally Rwp.

        rows
            Candidate result dictionaries. Clicking a cell emits its row.

        cell_text_values
            Values shown inside the cells. When None, mapped_values are shown.
        """
        x_values = np.asarray(
            x_values,
            dtype=float,
        )

        y_values = np.asarray(
            y_values,
            dtype=float,
        )

        mapped_values = np.asarray(
            mapped_values,
            dtype=float,
        )

        if cell_text_values is None:
            cell_text_values = mapped_values

        cell_text_values = np.asarray(
            cell_text_values,
            dtype=float,
        )

        valid = (
            np.isfinite(
                x_values
            )
            & np.isfinite(
                y_values
            )
            & np.isfinite(
                mapped_values
            )
        )

        x_values = x_values[
            valid
        ]

        y_values = y_values[
            valid
        ]

        mapped_values = mapped_values[
            valid
        ]

        cell_text_values = cell_text_values[
            valid
        ]

        rows = [
            row
            for row, keep in zip(
                rows,
                valid,
            )
            if bool(keep)
        ]

        self._clear_cells()

        self._rows_by_cell = {}

        if (
            x_values.size == 0
            or y_values.size == 0
            or not rows
        ):
            self.info_label.setText(
                information
                or "No completed grid-search results are available."
            )

            return

        unique_x = np.unique(
            x_values
        )

        unique_y = np.unique(
            y_values
        )

        unique_x.sort()
        unique_y.sort()

        x_edges = self._cell_edges(
            unique_x
        )

        y_edges = self._cell_edges(
            unique_y
        )

        minimum = float(
            np.min(
                mapped_values
            )
        )

        maximum = float(
            np.max(
                mapped_values
            )
        )

        if maximum <= minimum:
            maximum = minimum + 1.0e-12

        # Hidden image used only for the right-side LUT / colourbar.
        colourbar_values = np.asarray(
            [
                [
                    minimum,
                    maximum,
                ]
            ],
            dtype=float,
        )

        self._colourbar_image.setImage(
            colourbar_values,
            autoLevels=False,
        )

        self._colourbar_image.setLevels(
            (
                minimum,
                maximum,
            )
        )

        self.colorbar.setLevels(
            minimum,
            maximum,
        )

        self.colorbar.axis.setLabel(
            colorbar_label
        )

        self._x_values = unique_x
        self._y_values = unique_y

        # Keep the lowest mapped value when several simulations map to
        # the same (x, y) grid cell.
        selected_cells: Dict[
            Tuple[int, int],
            Tuple[float, float, Dict[str, Any]],
        ] = {}

        for x_value, y_value, mapped_value, text_value, row in zip(
            x_values,
            y_values,
            mapped_values,
            cell_text_values,
            rows,
        ):
            x_index = int(
                np.searchsorted(
                    unique_x,
                    x_value,
                )
            )

            y_index = int(
                np.searchsorted(
                    unique_y,
                    y_value,
                )
            )

            key = (
                x_index,
                y_index,
            )

            previous = selected_cells.get(
                key
            )

            if (
                previous is None
                or float(
                    mapped_value
                )
                < float(
                    previous[0]
                )
            ):
                selected_cells[
                    key
                ] = (
                    float(
                        mapped_value
                    ),
                    float(
                        text_value
                    ),
                    row,
                )

        for (
            x_index,
            y_index,
        ), (
            mapped_value,
            text_value,
            row,
        ) in selected_cells.items():
            normalised = (
                mapped_value
                - minimum
            ) / (
                maximum
                - minimum
            )

            normalised = min(
                1.0,
                max(
                    0.0,
                    float(
                        normalised
                    ),
                ),
            )

            colour = self.colour_map.map(
                normalised,
                mode="qcolor",
            )

            rectangle = QtWidgets.QGraphicsRectItem(
                QtCore.QRectF(
                    float(
                        x_edges[
                            x_index
                        ]
                    ),
                    float(
                        y_edges[
                            y_index
                        ]
                    ),
                    float(
                        x_edges[
                            x_index + 1
                        ]
                        - x_edges[
                            x_index
                        ]
                    ),
                    float(
                        y_edges[
                            y_index + 1
                        ]
                        - y_edges[
                            y_index
                        ]
                    ),
                )
            )

            rectangle.setBrush(
                pg.mkBrush(
                    colour
                )
            )

            rectangle.setPen(
                pg.mkPen(
                    "#404040",
                    width=1.0,
                )
            )

            rectangle.setZValue(
                2
            )

            self.plot.addItem(
                rectangle
            )

            self._cell_items.append(
                rectangle
            )

            text = pg.TextItem(
                text=f"{text_value:.3f}",
                color=self._text_colour(
                    colour
                ),
                anchor=(
                    0.5,
                    0.5,
                ),
            )

            text.setFont(
                QtGui.QFont(
                    "Arial",
                    9,
                )
            )

            text.setPos(
                float(
                    unique_x[
                        x_index
                    ]
                ),
                float(
                    unique_y[
                        y_index
                    ]
                ),
            )

            text.setZValue(
                3
            )

            self.plot.addItem(
                text
            )

            self._label_items.append(
                text
            )

            self._rows_by_cell[
                (
                    x_index,
                    y_index,
                )
            ] = row

        self.info_label.setText(
            information
        )

        if self.chk_auto_fit.isChecked():
            self.reset_view()

    def _plot_clicked(
        self,
        event,
    ) -> None:
        if not self._rows_by_cell:
            return

        if event.button() != QtCore.Qt.LeftButton:
            return

        point = self.plot.vb.mapSceneToView(
            event.scenePos()
        )

        x_value = float(
            point.x()
        )

        y_value = float(
            point.y()
        )

        if (
            self._x_values.size == 0
            or self._y_values.size == 0
        ):
            return

        x_index = int(
            np.argmin(
                np.abs(
                    self._x_values
                    - x_value
                )
            )
        )

        y_index = int(
            np.argmin(
                np.abs(
                    self._y_values
                    - y_value
                )
            )
        )

        row = self._rows_by_cell.get(
            (
                x_index,
                y_index,
            )
        )

        if isinstance(
            row,
            dict,
        ):
            self.cellSelected.emit(
                row
            )


class NumericTableItem(
    QtWidgets.QTableWidgetItem
):
    def __init__(
        self,
        value,
    ):
        self.numeric_value = _float_value(
            value
        )

        if _finite_float(value):
            text = f"{float(value):.10g}"
        elif value is None:
            text = ""
        else:
            text = str(value)

        super().__init__(
            text
        )

    def __lt__(
        self,
        other,
    ):
        if isinstance(
            other,
            NumericTableItem,
        ):
            if np.isfinite(
                self.numeric_value
            ) and np.isfinite(
                other.numeric_value
            ):
                return (
                    self.numeric_value
                    < other.numeric_value
                )

        return super().__lt__(
            other
        )


class ClickableScatterPlot(
    QtWidgets.QWidget
):
    pointSelected = QtCore.Signal(object)

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

        self._rows: List[
            Dict[str, Any]
        ] = []

        self.plot_widget = pg.PlotWidget(
            self
        )

        self.plot_widget.setBackground(
            "w"
        )

        self.plot_widget.showGrid(
            x=True,
            y=True,
            alpha=0.20,
        )

        self.plot_widget.setTitle(
            title
        )

        self.plot_widget.setLabel(
            "bottom",
            xlabel,
        )

        self.plot_widget.setLabel(
            "left",
            ylabel,
        )

        self.scatter = pg.ScatterPlotItem(
            pxMode=True,
            size=13,
            pen=pg.mkPen(
                "#303030",
                width=1.0,
            ),
        )

        self.scatter.sigClicked.connect(
            self._points_clicked
        )

        self.plot_widget.addItem(
            self.scatter
        )

        self.info_label = QtWidgets.QLabel()
        self.info_label.setWordWrap(
            True
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

        layout.addWidget(
            self.info_label
        )

        layout.addWidget(
            self.plot_widget,
            1,
        )

    def clear(
        self,
    ) -> None:
        self._rows = []

        self.scatter.setData(
            []
        )

        self.info_label.setText(
            ""
        )

    def set_points(
        self,
        x_values,
        y_values,
        colour_values,
        rows: List[Dict[str, Any]],
        *,
        information: str = "",
    ) -> None:
        x_values = np.asarray(
            x_values,
            dtype=float,
        )

        y_values = np.asarray(
            y_values,
            dtype=float,
        )

        colour_values = np.asarray(
            colour_values,
            dtype=float,
        )

        valid = (
            np.isfinite(x_values)
            & np.isfinite(y_values)
            & np.isfinite(colour_values)
        )

        x_values = x_values[
            valid
        ]

        y_values = y_values[
            valid
        ]

        colour_values = colour_values[
            valid
        ]

        rows = [
            row
            for row, keep
            in zip(
                rows,
                valid,
            )
            if bool(keep)
        ]

        self._rows = list(
            rows
        )

        colours = _colour_values(
            colour_values
        )

        spots = []

        for index, (
            x_value,
            y_value,
            colour,
            row,
        ) in enumerate(
            zip(
                x_values,
                y_values,
                colours,
                rows,
            )
        ):
            spots.append(
                {
                    "pos": (
                        float(x_value),
                        float(y_value),
                    ),
                    "brush": pg.mkBrush(
                        colour
                    ),
                    "pen": pg.mkPen(
                        "#202020",
                        width=1.0,
                    ),
                    "size": 13,
                    "data": {
                        "index": index,
                        "row": row,
                    },
                }
            )

        self.scatter.setData(
            spots
        )

        self.info_label.setText(
            information
        )

        self.plot_widget.autoRange()

    def _points_clicked(
        self,
        scatter,
        points,
        event=None,
    ) -> None:
        if not points:
            return

        data = points[0].data()

        if not isinstance(
            data,
            dict,
        ):
            return

        row = data.get(
            "row"
        )

        if isinstance(
            row,
            dict,
        ):
            self.pointSelected.emit(
                row
            )

class CSResultsExplorer(
    QtWidgets.QWidget
):
    candidateSelected = QtCore.Signal(object)
    applyCandidateRequested = QtCore.Signal(object)
    openSeparateWindowRequested = QtCore.Signal(object)

    def __init__(
        self,
        parent=None,
        *,
        allow_separate_window: bool = True,
    ):
        super().__init__(
            parent
        )

        self.results: List[
            Dict[str, Any]
        ] = []

        self.selected_candidate: Optional[
            Dict[str, Any]
        ] = None

        self.allow_separate_window = bool(
            allow_separate_window
        )

        self._refresh_timer = QtCore.QTimer(
            self
        )

        self._refresh_timer.setSingleShot(
            True
        )

        self._refresh_timer.setInterval(
            120
        )

        self._refresh_timer.timeout.connect(
            self.refresh
        )

        self._build_ui()

    def _build_ui(
        self,
    ) -> None:
        layout = QtWidgets.QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        summary_group = QtWidgets.QGroupBox(
            "Stacking-search result"
        )

        summary_layout = QtWidgets.QGridLayout(
            summary_group
        )

        self.global_summary = QtWidgets.QLabel(
            "No completed candidates."
        )

        self.global_summary.setWordWrap(
            True
        )

        self.selected_summary = QtWidgets.QLabel(
            "No candidate selected."
        )

        self.selected_summary.setWordWrap(
            True
        )

        self.combo_cs = QtWidgets.QComboBox()
        self.combo_cs.setMinimumWidth(
            320
        )

        self.combo_spacing = QtWidgets.QComboBox()
        self.combo_spacing.addItem(
            "Best over all spacings",
            None,
        )

        self.button_apply = QtWidgets.QPushButton(
            "Apply selected candidate"
        )

        self.button_open_input = QtWidgets.QPushButton(
            "Open candidate input"
        )

        self.button_open_folder = QtWidgets.QPushButton(
            "Open result folder"
        )

        self.button_separate = QtWidgets.QPushButton(
            "Open stacking results window"
        )

        self.button_separate.setVisible(
            self.allow_separate_window
        )


        summary_layout.addWidget(
            QtWidgets.QLabel("Global best:"),
            0,
            0,
        )

        summary_layout.addWidget(
            self.global_summary,
            0,
            1,
            1,
            5,
        )

        summary_layout.addWidget(
            QtWidgets.QLabel("Selected:"),
            1,
            0,
        )

        summary_layout.addWidget(
            self.selected_summary,
            1,
            1,
            1,
            5,
        )

        summary_layout.addWidget(
            QtWidgets.QLabel("Crystallite size:"),
            2,
            0,
        )

        summary_layout.addWidget(
            self.combo_cs,
            2,
            1,
        )

        summary_layout.addWidget(
            QtWidgets.QLabel("Spacing filter:"),
            2,
            2,
        )

        summary_layout.addWidget(
            self.combo_spacing,
            2,
            3,
        )

        summary_layout.addWidget(
            self.button_apply,
            3,
            0,
        )
        summary_layout.addWidget(
            self.button_open_input,
            3,
            1,
        )
        summary_layout.addWidget(
            self.button_open_folder,
            3,
            2,
        )
        summary_layout.addWidget(
            self.button_separate,
            3,
            3,
            1,
            3,
        )

        layout.addWidget(
            summary_group
        )

        self.tabs = QtWidgets.QTabWidget()

        self.size_map = ClickableGridMap(
            title="Crystallite-size Rwp map",
            xlabel="Axis 1 / cells",
            ylabel="Axis 3 / cells or layers",
        )

        self.uv_map = ClickableGridMap(
            title="Stacking u-v Rwp map",
            xlabel="u / parent a",
            ylabel="v / parent b",
        )

        self.spacing_plot = ClickableGridMap(
            title="Layer-spacing Rwp map",
            xlabel="Stacking distance / Å",
            ylabel="",
        )

        self.size_map.cellSelected.connect(
            self._size_point_selected
        )

        self.uv_map.cellSelected.connect(
            self.select_candidate
        )

        self.spacing_plot.cellSelected.connect(
            self.select_candidate
        )

        # ---------------------------------------------------------
        # Probability-scan page
        #
        # The scrollable list supports both:
        #
        #   correlated A/B:
        #       P(alternate)
        #
        #   generic Markov:
        #       P(state_i -> state_j)
        # ---------------------------------------------------------
        self.probability_page = QtWidgets.QWidget()

        probability_layout = QtWidgets.QHBoxLayout(
            self.probability_page
        )

        probability_layout.setContentsMargins(
            4,
            4,
            4,
            4,
        )

        probability_selector_panel = QtWidgets.QWidget()

        probability_selector_layout = QtWidgets.QVBoxLayout(
            probability_selector_panel
        )

        probability_selector_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        probability_selector_label = QtWidgets.QLabel(
            "Select probability:"
        )

        self.probability_selector = QtWidgets.QListWidget()

        self.probability_selector.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection
        )

        self.probability_selector.setMinimumWidth(
            240
        )

        self.probability_selector.setMaximumWidth(
            340
        )

        probability_selector_layout.addWidget(
            probability_selector_label
        )

        probability_selector_layout.addWidget(
            self.probability_selector,
            1,
        )

        self.probability_plot = FastLinePlot(
            title="Rwp vs selected probability",
            xlabel="Probability",
            ylabel="Best Rwp (%)",
        )

        probability_layout.addWidget(
            probability_selector_panel
        )

        probability_layout.addWidget(
            self.probability_plot,
            1,
        )


        self.candidate_fit = MplPlot(
            title="Candidate PDF fit"
        )

        self.parameter_page = QtWidgets.QWidget()
        parameter_layout = QtWidgets.QVBoxLayout(
            self.parameter_page
        )

        self.chk_show_fixed_parameters = QtWidgets.QCheckBox(
            "Show fixed parameters"
        )

        self.parameter_table = QtWidgets.QTableWidget(
            0,
            5,
        )

        self.parameter_table.setHorizontalHeaderLabels(
            [
                "Parameter",
                "Initial",
                "Final",
                "Change",
                "Refined?",
            ]
        )

        self.parameter_table.setSortingEnabled(
            True
        )

        self.parameter_table.horizontalHeader().setStretchLastSection(
            True
        )

        parameter_layout.addWidget(
            self.chk_show_fixed_parameters
        )

        parameter_layout.addWidget(
            self.parameter_table,
            1,
        )

        self.results_table = QtWidgets.QTableWidget(
            0,
            14,
        )

        self.results_table.setHorizontalHeaderLabels(
            [
                "Candidate",
                "CS",
                "Axis 1",
                "Axis 2",
                "Layers",
                "P_alt",
                "Input spacing",
                "Refined spacing",
                "u",
                "v",
                "Rwp",
                "Fit time / s",
                "Detail file",
                "Input file",
            ]
        )

        self.results_table.setSortingEnabled(
            True
        )

        self.results_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )

        self.results_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection
        )

        self.results_table.horizontalHeader().setStretchLastSection(
            True
        )

        self.tabs.addTab(
            self.size_map,
            "CS overview",
        )
        self.tabs.addTab(
            self.uv_map,
            "Stacking u-v",
        )
        self.tabs.addTab(
            self.spacing_plot,
            "Layer spacing",
        )
        self.tabs.addTab(
            self.probability_page,
            "Probability scan",
        )
        self.tabs.addTab(
            self.candidate_fit,
            "Candidate fit",
        )
        self.tabs.addTab(
            self.parameter_page,
            "Refined parameters",
        )
        self.tabs.addTab(
            self.results_table,
            "Results table",
        )

        layout.addWidget(
            self.tabs,
            1,
        )

        self.combo_cs.currentIndexChanged.connect(
            self._selected_cs_changed
        )

        self.combo_spacing.currentIndexChanged.connect(
            self.refresh_selected_cs
        )

        self.probability_selector.currentItemChanged.connect(
            self._on_probability_selection_changed
        )

        self.chk_show_fixed_parameters.toggled.connect(
            self._refresh_parameter_table
        )

        self.size_map.cellSelected.connect(
            self._size_point_selected
        )

        self.uv_map.cellSelected.connect(
            self.select_candidate
        )

        self.results_table.itemSelectionChanged.connect(
            self._table_selection_changed
        )

        self.results_table.itemDoubleClicked.connect(
            self._table_item_double_clicked
        )

        self.button_apply.clicked.connect(
            self._apply_selected
        )

        self.button_open_input.clicked.connect(
            self._open_selected_input
        )

        self.button_open_folder.clicked.connect(
            self._open_selected_folder
        )

        self.button_separate.clicked.connect(
            self._open_separate_window
        )

        self._update_action_buttons()

    def _on_probability_selection_changed(
        self,
        current_item,
        previous_item,
    ) -> None:
        """
        Redraw the probability plot when the selected probability changes.
        """
        _ = current_item
        _ = previous_item

        self._refresh_probability_plot(
            self.selected_cs_rows()
        )


    @staticmethod
    def cs_key(
        row: Dict[str, Any],
    ) -> Tuple[float, float, float]:
        return (
            round(
                _float_value(
                    row.get(
                        "axis1_cells"
                    )
                ),
                8,
            ),
            round(
                _float_value(
                    row.get(
                        "axis2_cells"
                    )
                ),
                8,
            ),
            round(
                _float_value(
                    row.get(
                        "axis3_cells"
                    )
                ),
                8,
            ),
        )

    def clear_results(
        self,
    ) -> None:
        self.results = []
        self.selected_candidate = None

        self.combo_cs.clear()
        self.combo_spacing.clear()
        self.combo_spacing.addItem(
            "Best over all spacings",
            None,
        )

        self.size_map.clear()
        self.uv_map.clear()
        self.spacing_plot.clear()
        self.probability_selector.clear()
        self.probability_plot.clear()
        self.results_table.setRowCount(
            0
        )
        self.parameter_table.setRowCount(
            0
        )
        self.candidate_fit.clear()

        self.global_summary.setText(
            "No completed candidates."
        )

        self.selected_summary.setText(
            "No candidate selected."
        )

        self._update_action_buttons()

    def set_results(
        self,
        rows: List[Dict[str, Any]],
    ) -> None:
        self.results = [
            dict(row)
            for row in (
                rows
                or []
            )
        ]

        self.refresh()

    def append_result(
        self,
        row: Dict[str, Any],
    ) -> None:
        self.results.append(
            dict(row)
        )

        self._refresh_timer.start()

    def refresh(
        self,
    ) -> None:
        self._refresh_selectors()
        self._refresh_global_summary()
        self._refresh_size_map()
        self._refresh_results_table()
        self.refresh_selected_cs()

        if (
            self.selected_candidate is None
            and self.results
        ):
            best = min(
                self.results,
                key=lambda row: _float_value(
                    row.get(
                        "rwp"
                    ),
                    np.inf,
                ),
            )

            self.select_candidate(
                best
            )

    def _refresh_selectors(
        self,
    ) -> None:
        current_key = self.combo_cs.currentData()

        keys = sorted(
            {
                self.cs_key(row)
                for row in self.results
            }
        )

        self.combo_cs.blockSignals(
            True
        )
        self.combo_cs.clear()

        for index, key in enumerate(keys):
            self.combo_cs.addItem(
                (
                    f"CS {index + 1}: "
                    f"axis1={key[0]:g}, "
                    f"axis2={key[1]:g}, "
                    f"axis3/layers={key[2]:g}"
                ),
                key,
            )

        if current_key in keys:
            self.combo_cs.setCurrentIndex(
                keys.index(
                    current_key
                )
            )

        self.combo_cs.blockSignals(
            False
        )

        current_spacing = self.combo_spacing.currentData()

        spacing_values = sorted(
            {
                _float_value(
                    row.get(
                        "stacking_distance_grid"
                    )
                )
                for row in self.results
                if _finite_float(
                    row.get(
                        "stacking_distance_grid"
                    )
                )
            }
        )

        self.combo_spacing.blockSignals(
            True
        )
        self.combo_spacing.clear()
        self.combo_spacing.addItem(
            "Best over all spacings",
            None,
        )

        for value in spacing_values:
            self.combo_spacing.addItem(
                f"{value:.6g} Å",
                value,
            )

        if current_spacing is not None:
            for index in range(
                self.combo_spacing.count()
            ):
                value = self.combo_spacing.itemData(
                    index
                )

                if (
                    value is not None
                    and np.isclose(
                        float(value),
                        float(current_spacing),
                    )
                ):
                    self.combo_spacing.setCurrentIndex(
                        index
                    )
                    break

        self.combo_spacing.blockSignals(
            False
        )




    def _refresh_global_summary(
        self,
    ) -> None:
        valid_rows = [
            row
            for row in self.results
            if _finite_float(
                row.get(
                    "rwp"
                )
            )
        ]

        if not valid_rows:
            self.global_summary.setText(
                "No valid completed candidates."
            )
            return

        best = min(
            valid_rows,
            key=lambda row: float(
                row["rwp"]
            ),
        )

        self.global_summary.setText(
            self._candidate_text(
                best
            )
        )

    @staticmethod
    def _candidate_text(
        row: Dict[str, Any],
    ) -> str:
        input_spacing = row.get(
            "stacking_distance_grid"
        )

        refined_spacing = row.get(
            "stacking_distance"
        )

        p_alternate = _float_value(
            row.get(
                "p_alternate",
                np.nan,
            )
        )

        p_text = ""

        if np.isfinite(
            p_alternate
        ):
            p_text = (
                f"P_alt={p_alternate:.7g} | "
            )

        return (
            f"Rwp={_float_value(row.get('rwp')):.5g}% | "
            + p_text
            + f"axis1={_float_value(row.get('axis1_cells')):.5g}, "
            f"axis2={_float_value(row.get('axis2_cells')):.5g}, "
            f"layers={row.get('number_of_layers', row.get('axis3_cells'))} | "
            f"u={_float_value(row.get('stacking_u')):.7g}, "
            f"v={_float_value(row.get('stacking_v')):.7g} | "
            f"input spacing={_float_value(input_spacing):.7g} Å, "
            f"refined spacing={_float_value(refined_spacing):.7g} Å | "
            f"candidate={row.get('stacking_candidate_index', row.get('index', ''))}"
        )

    def _refresh_size_map(
        self,
    ) -> None:
        grouped: Dict[
            Tuple[float, float, float],
            Dict[str, Any],
        ] = {}

        for row in self.results:
            key = self.cs_key(
                row
            )

            current = grouped.get(
                key
            )

            if (
                current is None
                or _float_value(
                    row.get(
                        "rwp"
                    ),
                    np.inf,
                )
                < _float_value(
                    current.get(
                        "rwp"
                    ),
                    np.inf,
                )
            ):
                grouped[key] = row

        rows = list(
            grouped.values()
        )

        self.size_map.set_grid(
            [
                row.get(
                    "axis1_cells"
                )
                for row in rows
            ],
            [
                row.get(
                    "axis3_cells"
                )
                for row in rows
            ],
            [
                row.get(
                    "rwp"
                )
                for row in rows
            ],
            rows,
            information=(
                "Each square shows the lowest Rwp for one crystallite-size "
                "combination. Click a square to select its best candidate."
            ),
            colorbar_label="Best Rwp (%)",
            cell_text_values=[
                row.get(
                    "rwp"
                )
                for row in rows
            ],
        )

        

    def _selected_cs_changed(
        self,
    ) -> None:
        self.refresh_selected_cs()

    def selected_cs_rows(
        self,
    ) -> List[Dict[str, Any]]:
        key = self.combo_cs.currentData()

        if key is None:
            return []

        key = tuple(
            key
        )

        return [
            row
            for row in self.results
            if self.cs_key(row) == key
        ]

    def refresh_selected_cs(
        self,
    ) -> None:
        rows = self.selected_cs_rows()

        if not rows:
            self.uv_map.clear()
            self.spacing_plot.clear()

            self.probability_selector.blockSignals(
                True
            )

            try:
                self.probability_selector.clear()

            finally:
                self.probability_selector.blockSignals(
                    False
                )

            self.probability_plot.clear()
            self.probability_plot.show_message(
                "No probability-grid results are available."
            )

            return

        selected_spacing = self.combo_spacing.currentData()

        uv_rows = list(
            rows
        )

        if selected_spacing is not None:
            uv_rows = [
                row
                for row in rows
                if np.isclose(
                    _float_value(
                        row.get(
                            "stacking_distance_grid"
                        )
                    ),
                    float(selected_spacing),
                )
            ]

        if selected_spacing is None:
            best_by_uv: Dict[
                Tuple[float, float],
                Dict[str, Any],
            ] = {}

            for row in uv_rows:
                key = (
                    round(
                        _float_value(
                            row.get(
                                "stacking_u"
                            )
                        ),
                        10,
                    ),
                    round(
                        _float_value(
                            row.get(
                                "stacking_v"
                            )
                        ),
                        10,
                    ),
                )

                current = best_by_uv.get(
                    key
                )

                if (
                    current is None
                    or _float_value(
                        row.get(
                            "rwp"
                        ),
                        np.inf,
                    )
                    < _float_value(
                        current.get(
                            "rwp"
                        ),
                        np.inf,
                    )
                ):
                    best_by_uv[key] = row

            uv_rows = list(
                best_by_uv.values()
            )



        self.uv_map.set_grid(
            [
                row.get(
                    "stacking_u"
                )
                for row in uv_rows
            ],
            [
                row.get(
                    "stacking_v"
                )
                for row in uv_rows
            ],
            [
                row.get(
                    "rwp"
                )
                for row in uv_rows
            ],
            uv_rows,
            information=(
                "Each square is one stacking u-v simulation. "
                "Colour and number both show Rwp. "
                "Click a square to inspect or apply the candidate."
            ),
            colorbar_label="Rwp (%)",
            cell_text_values=[
                row.get(
                    "rwp"
                )
                for row in uv_rows
            ],
        )

        self._refresh_spacing_plot(
            rows
        )

        self._refresh_probability_selector(
            rows
        )

        self._refresh_probability_plot(
            rows
        )

    def _refresh_spacing_plot(
        self,
        rows: List[Dict[str, Any]],
    ) -> None:
        grouped: Dict[
            float,
            Dict[str, Any],
        ] = {}

        for row in rows:
            spacing = _float_value(
                row.get(
                    "stacking_distance_grid"
                )
            )

            if not np.isfinite(
                spacing
            ):
                continue

            current = grouped.get(
                spacing
            )

            if (
                current is None
                or _float_value(
                    row.get(
                        "rwp"
                    ),
                    np.inf,
                )
                < _float_value(
                    current.get(
                        "rwp"
                    ),
                    np.inf,
                )
            ):
                grouped[
                    spacing
                ] = row

        if not grouped:
            self.spacing_plot.clear()

            self.spacing_plot.info_label.setText(
                "No layer-spacing results are available."
            )

            return

        spacing_values = sorted(
            grouped.keys()
        )

        best_rows = [
            grouped[
                spacing
            ]
            for spacing in spacing_values
        ]

        self.spacing_plot.set_grid(
            spacing_values,
            [
                0.0
                for _ in spacing_values
            ],
            [
                row.get(
                    "rwp"
                )
                for row in best_rows
            ],
            best_rows,
            information=(
                "Each square shows the lowest Rwp at one layer spacing, "
                "minimized over all u-v candidates."
            ),
            colorbar_label="Best Rwp (%)",
            cell_text_values=[
                row.get(
                    "rwp"
                )
                for row in best_rows
            ],
        )


    def _refresh_probability_selector(
        self,
        rows: List[
            Dict[str, Any]
        ],
    ) -> None:
        """
        Populate the scrolling probability selector.

        The current selection is preserved while new candidates arrive.
        """
        previous_data = None

        current_item = self.probability_selector.currentItem()

        if current_item is not None:
            previous_data = current_item.data(
                QtCore.Qt.UserRole
            )

        probability_options = []

        # Correlated A/B P(alternate).
        if any(
            _finite_float(
                row.get(
                    "p_alternate",
                    np.nan,
                )
            )
            for row in rows
        ):
            probability_options.append(
                (
                    "P(alternate)",
                    {
                        "kind": "p_alternate",
                    },
                )
            )

        # Generic Markov transition probabilities.
        state_count = 0
        state_labels = []

        for row in rows:
            matrix = row.get(
                "markov_transition_matrix",
                None,
            )

            if not isinstance(
                matrix,
                (
                    list,
                    tuple,
                    np.ndarray,
                ),
            ):
                continue

            try:
                matrix_array = np.asarray(
                    matrix,
                    dtype=float,
                )

            except Exception:
                continue

            if (
                matrix_array.ndim != 2
                or matrix_array.shape[0]
                != matrix_array.shape[1]
                or matrix_array.shape[0] <= 0
            ):
                continue

            current_state_count = int(
                matrix_array.shape[0]
            )

            if current_state_count > state_count:
                state_count = current_state_count

                candidate_labels = row.get(
                    "markov_state_labels",
                    [],
                )

                if (
                    isinstance(
                        candidate_labels,
                        (
                            list,
                            tuple,
                        ),
                    )
                    and len(
                        candidate_labels
                    )
                    == current_state_count
                ):
                    state_labels = [
                        str(
                            label
                        )
                        for label in candidate_labels
                    ]

                else:
                    state_labels = [
                        f"S{index}"
                        for index in range(
                            current_state_count
                        )
                    ]

        if state_count > 0:
            if len(
                state_labels
            ) != state_count:
                state_labels = [
                    f"S{index}"
                    for index in range(
                        state_count
                    )
                ]

            for source_index in range(
                state_count
            ):
                for target_index in range(
                    state_count
                ):
                    probability_options.append(
                        (
                            (
                                f"P({state_labels[source_index]} "
                                f"→ {state_labels[target_index]})"
                            ),
                            {
                                "kind": "markov",
                                "source": int(
                                    source_index
                                ),
                                "target": int(
                                    target_index
                                ),
                                "source_label": str(
                                    state_labels[
                                        source_index
                                    ]
                                ),
                                "target_label": str(
                                    state_labels[
                                        target_index
                                    ]
                                ),
                            },
                        )
                    )

        # Do not rebuild the widget when its options have not changed.
        existing_options = []

        for item_index in range(
            self.probability_selector.count()
        ):
            item = self.probability_selector.item(
                item_index
            )

            existing_options.append(
                (
                    item.text(),
                    item.data(
                        QtCore.Qt.UserRole
                    ),
                )
            )

        if existing_options == probability_options:
            return

        self.probability_selector.blockSignals(
            True
        )

        try:
            self.probability_selector.clear()

            selected_index = -1

            for option_index, (
                label,
                option_data,
            ) in enumerate(
                probability_options
            ):
                item = QtWidgets.QListWidgetItem(
                    label
                )

                item.setData(
                    QtCore.Qt.UserRole,
                    option_data,
                )

                self.probability_selector.addItem(
                    item
                )

                if option_data == previous_data:
                    selected_index = option_index

            if self.probability_selector.count() > 0:
                if selected_index < 0:
                    selected_index = 0

                self.probability_selector.setCurrentRow(
                    selected_index
                )

        finally:
            self.probability_selector.blockSignals(
                False
            )

            

    def _refresh_probability_plot(
        self,
        rows: List[
            Dict[str, Any]
        ],
    ) -> None:
        """
        Plot best Rwp against the probability selected in the scrolling list.

        For each probability value, retain the lowest Rwp over all other
        candidate variables such as spacing and lateral shifts.
        """
        self.probability_plot.clear()

        selected_item = self.probability_selector.currentItem()

        if selected_item is None:
            self.probability_plot.show_message(
                "No probability-grid results are available."
            )
            return

        option = selected_item.data(
            QtCore.Qt.UserRole
        )

        if not isinstance(
            option,
            dict,
        ):
            self.probability_plot.show_message(
                "No probability is selected."
            )
            return

        option_kind = str(
            option.get(
                "kind",
                "",
            )
        )

        best_by_probability: Dict[
            float,
            Dict[str, Any],
        ] = {}

        for row in rows:
            if option_kind == "p_alternate":
                probability = _float_value(
                    row.get(
                        "p_alternate",
                        np.nan,
                    )
                )

            elif option_kind == "markov":
                matrix = row.get(
                    "markov_transition_matrix",
                    None,
                )

                if not isinstance(
                    matrix,
                    (
                        list,
                        tuple,
                    ),
                ):
                    continue

                try:
                    matrix_array = np.asarray(
                        matrix,
                        dtype=float,
                    )

                    source_index = int(
                        option[
                            "source"
                        ]
                    )

                    target_index = int(
                        option[
                            "target"
                        ]
                    )

                    probability = float(
                        matrix_array[
                            source_index,
                            target_index,
                        ]
                    )

                except Exception:
                    continue

            else:
                continue

            rwp = _float_value(
                row.get(
                    "rwp",
                    np.nan,
                )
            )

            if not (
                np.isfinite(
                    probability
                )
                and np.isfinite(
                    rwp
                )
            ):
                continue

            probability_key = round(
                float(
                    probability
                ),
                12,
            )

            current_best = best_by_probability.get(
                probability_key
            )

            if (
                current_best is None
                or rwp
                < _float_value(
                    current_best.get(
                        "rwp",
                        np.inf,
                    )
                )
            ):
                best_by_probability[
                    probability_key
                ] = row

        if not best_by_probability:
            self.probability_plot.show_message(
                "No completed candidates contain the selected probability."
            )
            return

        probability_values = np.asarray(
            sorted(
                best_by_probability.keys()
            ),
            dtype=float,
        )

        rwp_values = np.asarray(
            [
                _float_value(
                    best_by_probability[
                        value
                    ].get(
                        "rwp",
                        np.nan,
                    )
                )
                for value in probability_values
            ],
            dtype=float,
        )

        selected_label = selected_item.text()

        self.probability_plot.set_title(
            f"Best Rwp vs {selected_label}"
        )

        self.probability_plot.set_xlabel(
            selected_label
        )

        self.probability_plot.set_ylabel(
            "Best Rwp (%)"
        )

        self.probability_plot.add_curve(
            probability_values,
            rwp_values,
            color="#7b2cbf",
            width=2.0,
            symbol="o",
            symbol_size=7.0,
            name="Best over other candidates",
        )

        best_index = int(
            np.argmin(
                rwp_values
            )
        )

        self.probability_plot.set_info(
            f"Selected probability: {selected_label}\n"
            f"Best value = "
            f"{probability_values[best_index]:.8g}; "
            f"best Rwp = "
            f"{rwp_values[best_index]:.8g}%."
        )

        self.probability_plot.auto_range()


    def _size_point_selected(
        self,
        row: Dict[str, Any],
    ) -> None:
        key = self.cs_key(
            row
        )

        for index in range(
            self.combo_cs.count()
        ):
            if tuple(
                self.combo_cs.itemData(
                    index
                )
            ) == key:
                self.combo_cs.setCurrentIndex(
                    index
                )
                break

        selected_rows = self.selected_cs_rows()

        if selected_rows:
            best = min(
                selected_rows,
                key=lambda result: _float_value(
                    result.get(
                        "rwp"
                    ),
                    np.inf,
                ),
            )

            self.select_candidate(
                best
            )

    def select_candidate(
        self,
        row: Dict[str, Any],
    ) -> None:
        if not isinstance(
            row,
            dict,
        ):
            return

        self.selected_candidate = row

        self.selected_summary.setText(
            self._candidate_text(
                row
            )
        )

        self._refresh_candidate_fit()
        self._refresh_parameter_table()
        self._select_result_table_row(
            row
        )
        self._update_action_buttons()

        self.candidateSelected.emit(
            row
        )


    def _refresh_candidate_fit(
        self,
    ) -> None:
        ax = self.candidate_fit.ax
        ax.clear()

        row = self.selected_candidate

        if not isinstance(
            row,
            dict,
        ):
            ax.text(
                0.5,
                0.5,
                "No candidate selected.",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            self.candidate_fit.canvas.draw_idle()
            return

        visual = row.get(
            "visual"
        )

        if isinstance(
            visual,
            dict,
        ):
            r = np.asarray(
                visual.get(
                    "r",
                    [],
                ),
                dtype=float,
            )
            experimental = np.asarray(
                visual.get(
                    "G_exp",
                    [],
                ),
                dtype=float,
            )
            calculated = np.asarray(
                visual.get(
                    "G_fit",
                    [],
                ),
                dtype=float,
            )
            residual = np.asarray(
                visual.get(
                    "residual",
                    [],
                ),
                dtype=float,
            )

        else:
            detail_file = str(
                row.get(
                    "detail_file",
                    "",
                )
                or ""
            )

            if not detail_file or not os.path.exists(
                detail_file
            ):
                ax.text(
                    0.5,
                    0.5,
                    "Detailed fit data were not stored for this candidate.",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                self.candidate_fit.canvas.draw_idle()
                return

            try:
                data = np.load(
                    detail_file,
                    allow_pickle=False,
                )

                r = np.asarray(
                    data["r"],
                    dtype=float,
                )
                experimental = np.asarray(
                    data["G_exp"],
                    dtype=float,
                )
                calculated = np.asarray(
                    data["G_fit"],
                    dtype=float,
                )
                residual = np.asarray(
                    data["residual"],
                    dtype=float,
                )

            except Exception as exc:
                ax.text(
                    0.5,
                    0.5,
                    f"Could not load candidate detail:\n{exc}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                self.candidate_fit.canvas.draw_idle()
                return

        if (
            r.size == 0
            or experimental.size != r.size
            or calculated.size != r.size
        ):
            return

        if residual.size != r.size:
            residual = (
                experimental
                - calculated
            )

        amplitude = max(
            float(
                np.nanmax(experimental)
                - np.nanmin(experimental)
            ),
            1.0e-12,
        )

        offset = (
            -0.25
            * amplitude
        )

        ax.plot(
            r,
            experimental,
            color="black",
            linewidth=1.0,
            label="G_exp",
        )

        ax.plot(
            r,
            calculated,
            color="red",
            linewidth=1.5,
            label="G_fit",
        )

        ax.plot(
            r,
            residual + offset,
            color="green",
            linewidth=1.0,
            label="Residual",
        )

        ax.set_xlabel(
            "r (Å)"
        )
        ax.set_ylabel(
            "G(r)"
        )
        ax.set_title(
            self._candidate_text(
                row
            )
        )
        ax.legend(
            loc="best"
        )

        self.candidate_fit.canvas.draw_idle()

    def _refresh_parameter_table(
        self,
    ) -> None:
        row = self.selected_candidate

        if not isinstance(
            row,
            dict,
        ):
            self.parameter_table.setRowCount(
                0
            )
            return

        initial_values = dict(
            row.get(
                "initial_params",
                {},
            )
            or {}
        )

        final_values = dict(
            row.get(
                "params",
                {},
            )
            or {}
        )

        refinable_flags = dict(
            row.get(
                "refinable_flags",
                {},
            )
            or {}
        )

        show_fixed = bool(
            self.chk_show_fixed_parameters.isChecked()
        )

        parameter_names = sorted(
            set(initial_values.keys())
            | set(final_values.keys())
            | set(refinable_flags.keys())
        )

        if not show_fixed:
            parameter_names = [
                parameter_name
                for parameter_name in parameter_names
                if bool(
                    refinable_flags.get(
                        parameter_name,
                        False,
                    )
                )
            ]

        self.parameter_table.setSortingEnabled(
            False
        )

        self.parameter_table.setRowCount(
            len(parameter_names)
        )

        for table_row, parameter_name in enumerate(
            parameter_names
        ):
            initial_value = initial_values.get(
                parameter_name,
                np.nan,
            )

            final_value = final_values.get(
                parameter_name,
                initial_value,
            )

            if (
                _finite_float(initial_value)
                and _finite_float(final_value)
            ):
                change = (
                    float(final_value)
                    - float(initial_value)
                )
            else:
                change = ""

            values = [
                parameter_name,
                initial_value,
                final_value,
                change,
                (
                    "Yes"
                    if bool(
                        refinable_flags.get(
                            parameter_name,
                            False,
                        )
                    )
                    else "No"
                ),
            ]

            for column, value in enumerate(values):
                if column in {
                    1,
                    2,
                    3,
                }:
                    item = NumericTableItem(
                        value
                    )
                else:
                    item = QtWidgets.QTableWidgetItem(
                        str(value)
                    )

                self.parameter_table.setItem(
                    table_row,
                    column,
                    item,
                )

        self.parameter_table.setSortingEnabled(
            True
        )
        self.parameter_table.resizeColumnsToContents()

    def _refresh_results_table(
        self,
    ) -> None:
        self.results_table.setSortingEnabled(
            False
        )

        self.results_table.setRowCount(
            len(
                self.results
            )
        )

        for table_row, row in enumerate(
            self.results
        ):
            values = [
                row.get(
                    "stacking_candidate_index",
                    row.get(
                        "index",
                        "",
                    ),
                ),
                row.get(
                    "cs_index",
                    "",
                ),
                row.get(
                    "axis1_cells",
                    "",
                ),
                row.get(
                    "axis2_cells",
                    "",
                ),
                row.get(
                    "number_of_layers",
                    row.get(
                        "axis3_cells",
                        "",
                    ),
                ),
                row.get(
                    "p_alternate",
                    "",
                ),
                row.get(
                    "stacking_distance_grid",
                    "",
                ),
                row.get(
                    "stacking_distance",
                    "",
                ),
                row.get(
                    "stacking_u",
                    "",
                ),
                row.get(
                    "stacking_v",
                    "",
                ),
                row.get(
                    "rwp",
                    "",
                ),
                row.get(
                    "fit_elapsed_s",
                    "",
                ),
                row.get(
                    "detail_file",
                    "",
                ),
                row.get(
                    "candidate_input_file",
                    "",
                ),
            ]

            for column, value in enumerate(values):
                if column <= 11:
                    item = NumericTableItem(
                        value
                    )
                else:
                    item = QtWidgets.QTableWidgetItem(
                        str(value)
                    )

                item.setData(
                    QtCore.Qt.UserRole,
                    row,
                )

                self.results_table.setItem(
                    table_row,
                    column,
                    item,
                )

        self.results_table.setSortingEnabled(
            True
        )

    def _table_selection_changed(
        self,
    ) -> None:
        selected_items = self.results_table.selectedItems()

        if not selected_items:
            return

        row = selected_items[0].data(
            QtCore.Qt.UserRole
        )

        if isinstance(
            row,
            dict,
        ):
            self.select_candidate(
                row
            )

    def _table_item_double_clicked(
        self,
        item,
    ) -> None:
        row = item.data(
            QtCore.Qt.UserRole
        )

        if isinstance(
            row,
            dict,
        ):
            self.select_candidate(
                row
            )
            self.tabs.setCurrentWidget(
                self.candidate_fit
            )

    def _select_result_table_row(
        self,
        row: Dict[str, Any],
    ) -> None:
        candidate_index = row.get(
            "stacking_candidate_index",
            row.get(
                "index"
            ),
        )

        self.results_table.blockSignals(
            True
        )

        try:
            for table_row in range(
                self.results_table.rowCount()
            ):
                item = self.results_table.item(
                    table_row,
                    0,
                )

                if item is None:
                    continue

                candidate_row = item.data(
                    QtCore.Qt.UserRole
                )

                if (
                    isinstance(
                        candidate_row,
                        dict,
                    )
                    and candidate_row.get(
                        "stacking_candidate_index",
                        candidate_row.get(
                            "index"
                        ),
                    )
                    == candidate_index
                ):
                    self.results_table.selectRow(
                        table_row
                    )
                    break

        finally:
            self.results_table.blockSignals(
                False
            )

    def _apply_selected(
        self,
    ) -> None:
        if isinstance(
            self.selected_candidate,
            dict,
        ):
            self.applyCandidateRequested.emit(
                self.selected_candidate
            )

    def _open_selected_input(
        self,
    ) -> None:
        if not isinstance(
            self.selected_candidate,
            dict,
        ):
            return

        path = str(
            self.selected_candidate.get(
                "candidate_input_file",
                "",
            )
            or ""
        )

        if path and os.path.exists(path):
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(
                    path
                )
            )

    def _open_selected_folder(
        self,
    ) -> None:
        if not isinstance(
            self.selected_candidate,
            dict,
        ):
            return

        paths = [
            self.selected_candidate.get(
                "detail_file"
            ),
            self.selected_candidate.get(
                "candidate_input_file"
            ),
        ]

        for path in paths:
            path = str(
                path
                or ""
            )

            if path and os.path.exists(path):
                QtGui.QDesktopServices.openUrl(
                    QtCore.QUrl.fromLocalFile(
                        os.path.dirname(path)
                    )
                )
                return

    def _open_separate_window(
        self,
    ) -> None:
        self.openSeparateWindowRequested.emit(
            {
                "results": self.results,
                "selected": self.selected_candidate,
            }
        )

    def _update_action_buttons(
        self,
    ) -> None:
        row = self.selected_candidate

        has_selection = isinstance(
            row,
            dict,
        )

        self.button_apply.setEnabled(
            has_selection
        )

        if not has_selection:
            self.button_open_input.setEnabled(
                False
            )

            self.button_open_folder.setEnabled(
                False
            )

            return

        candidate_input = str(
            row.get(
                "candidate_input_file",
                "",
            )
            or ""
        )

        detail_file = str(
            row.get(
                "detail_file",
                "",
            )
            or ""
        )

        has_input = bool(
            candidate_input
            and os.path.exists(
                candidate_input
            )
        )

        has_folder = bool(
            (
                candidate_input
                and os.path.exists(
                    candidate_input
                )
            )
            or (
                detail_file
                and os.path.exists(
                    detail_file
                )
            )
        )

        self.button_open_input.setEnabled(
            has_input
        )

        self.button_open_folder.setEnabled(
            has_folder
        )

        if not has_input:
            self.button_open_input.setToolTip(
                "No candidate input was stored.\n\n"
                "In the finite-crystallite stacking dialog, set:\n"
                "Store candidate details = "
                "'Store best fit for every CS' or 'Store every candidate'."
            )
        else:
            self.button_open_input.setToolTip(
                candidate_input
            )

        if not has_folder:
            self.button_open_folder.setToolTip(
                "No result folder exists because candidate details "
                "were not stored."
            )
        else:
            self.button_open_folder.setToolTip(
                "Open the directory containing saved candidate files."
            )