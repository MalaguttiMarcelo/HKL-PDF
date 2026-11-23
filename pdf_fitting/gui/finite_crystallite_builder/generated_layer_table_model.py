from __future__ import annotations

from typing import List

from PySide6 import QtCore, QtGui
from PySide6.QtCore import Qt

from pdf_fitting.models.layer_stacking import (
    RealizedLayer,
)


_PALETTE = (
    (230, 75, 75),
    (65, 125, 225),
    (65, 180, 95),
    (195, 90, 210),
    (240, 150, 50),
    (45, 180, 180),
    (145, 95, 45),
    (125, 125, 125),
)


class GeneratedLayerTableModel(
    QtCore.QAbstractTableModel
):
    """
    Read-only table showing the finite realization used by the preview.
    """

    HEADERS = (
        "Layer",
        "Sequence row",
        "Repeat",
        "Label",
        "u",
        "v",
        "Stack coordinate",
        "Position / Å",
        "Atoms inside",
    )

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(
            parent
        )

        self._layers: List[
            RealizedLayer
        ] = []

    def rowCount(
        self,
        parent=QtCore.QModelIndex(),
    ):
        if parent.isValid():
            return 0

        return len(
            self._layers
        )

    def columnCount(
        self,
        parent=QtCore.QModelIndex(),
    ):
        if parent.isValid():
            return 0

        return len(
            self.HEADERS
        )

    def headerData(
        self,
        section,
        orientation,
        role=Qt.DisplayRole,
    ):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            if (
                0
                <= section
                < len(self.HEADERS)
            ):
                return self.HEADERS[
                    section
                ]

        return str(
            section + 1
        )

    def flags(
        self,
        index,
    ):
        if not index.isValid():
            return Qt.NoItemFlags

        return (
            Qt.ItemIsEnabled
            | Qt.ItemIsSelectable
        )

    def data(
        self,
        index,
        role=Qt.DisplayRole,
    ):
        if not index.isValid():
            return None

        row = index.row()
        column = index.column()

        if not (
            0
            <= row
            < len(self._layers)
        ):
            return None

        layer = self._layers[
            row
        ]

        if role == Qt.DisplayRole:
            values = (
                layer.realized_index,
                layer.sequence_index,
                layer.repeat_index,
                layer.label,
                f"{layer.shift_u:.8g}",
                f"{layer.shift_v:.8g}",
                f"{layer.stack_coordinate:.8g}",
                f"{layer.normal_offset:.8g}",
                layer.atom_count,
            )

            return values[
                column
            ]

        if role == Qt.TextAlignmentRole:
            if column == 3:
                return int(
                    Qt.AlignLeft
                    | Qt.AlignVCenter
                )

            return int(
                Qt.AlignRight
                | Qt.AlignVCenter
            )

        if role == Qt.BackgroundRole:
            red, green, blue = _PALETTE[
                layer.sequence_index
                % len(_PALETTE)
            ]

            alpha = (
                115
                if layer.atom_count > 0
                else 28
            )

            return QtGui.QBrush(
                QtGui.QColor(
                    red,
                    green,
                    blue,
                    alpha,
                )
            )

        if (
            role == Qt.FontRole
            and layer.atom_count > 0
        ):
            font = QtGui.QFont()
            font.setBold(
                True
            )
            return font

        return None

    def set_layers(
        self,
        layers: List[RealizedLayer],
    ) -> None:
        self.beginResetModel()

        self._layers = list(
            layers
            or []
        )

        self.endResetModel()