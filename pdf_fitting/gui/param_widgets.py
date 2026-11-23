from __future__ import annotations

import re
from typing import Optional, Dict, Any, List, Tuple

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from .defaults import DEFAULT_PARAMS
from .tooltips import param_tooltip


class ParamTableWidget(QtWidgets.QTableWidget):
    """Small helper table for parameter value/bounds/refine flag."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(0, 5, parent)
        self._block = False

        self.setHorizontalHeaderLabels(["Parameter", "Value", "Min", "Max", "Refine?"])

        hdr = self.horizontalHeader()
        hdr.setStretchLastSection(False)

        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.Interactive)
        hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.Interactive)
        hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)

        self.setColumnWidth(1, 170)
        self.setColumnWidth(2, 90)
        self.setColumnWidth(3, 90)

        self.setMinimumHeight(115)
        self.verticalHeader().setDefaultSectionSize(26)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        hdr.setMinimumSectionSize(60)

        self.setAlternatingRowColors(True)
        self.setToolTip(
            "Edit values/bounds and tick parameters to refine. Hover parameter names for help."
        )

        self.itemChanged.connect(self._on_item_changed)

    def _on_item_changed(self, _item: QtWidgets.QTableWidgetItem):
        if not self._block:
            self.changed.emit()

    def clear_params(self):
        self._block = True
        try:
            self.setRowCount(0)
        finally:
            self._block = False

    def set_params(
        self,
        keys: List[str],
        initial: Dict[str, Any],
        refinable: Dict[str, Any],
        bounds: Dict[str, Any],
        *,
        tooltips: Optional[Dict[str, str]] = None,
        label_overrides: Optional[Dict[str, str]] = None,
        enabled_keys: Optional[set[str]] = None,
    ) -> None:
        self._block = True
        try:
            self.setRowCount(len(keys))
            tooltips = tooltips or {}
            label_overrides = label_overrides or {}

            for i, k in enumerate(keys):
                k_str = str(k)
                shown = label_overrides.get(k_str, k_str)

                it_name = QtWidgets.QTableWidgetItem(shown)
                it_name.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                it_name.setToolTip(tooltips.get(k_str, param_tooltip(k_str)))
                it_name.setData(Qt.UserRole, k_str)
                self.setItem(i, 0, it_name)

                val = initial.get(k_str, DEFAULT_PARAMS.get(k_str, ""))
                it_val = QtWidgets.QTableWidgetItem("" if val is None else str(val))
                it_val.setToolTip(param_tooltip(k_str))
                self.setItem(i, 1, it_val)

                lo, hi = ("", "")
                if k_str in bounds:
                    try:
                        lo, hi = bounds[k_str]
                    except Exception:
                        lo, hi = ("", "")

                it_lo = QtWidgets.QTableWidgetItem("" if lo in (None, "") else str(lo))
                it_hi = QtWidgets.QTableWidgetItem("" if hi in (None, "") else str(hi))
                it_lo.setToolTip("Lower bound")
                it_hi.setToolTip("Upper bound")
                self.setItem(i, 2, it_lo)
                self.setItem(i, 3, it_hi)

                chk = QtWidgets.QTableWidgetItem("")
                chk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
                chk.setCheckState(Qt.Checked if bool(refinable.get(k_str, False)) else Qt.Unchecked)
                chk.setToolTip("Tick to refine this parameter.")
                self.setItem(i, 4, chk)

                if enabled_keys is not None and k_str not in enabled_keys:
                    for c in range(5):
                        it = self.item(i, c)
                        if it is None:
                            continue
                        it.setFlags(it.flags() & ~Qt.ItemIsEnabled)

        finally:
            self._block = False

    def extract(self) -> Tuple[Dict[str, Any], Dict[str, bool], Dict[str, str]]:
        initial_out: Dict[str, Any] = {}
        refinable_out: Dict[str, bool] = {}
        bounds_out: Dict[str, str] = {}

        for row in range(self.rowCount()):
            it_name = self.item(row, 0)
            if it_name is None:
                continue

            key = it_name.data(Qt.UserRole) or it_name.text().strip()
            key = str(key).strip()
            if not key:
                continue

            v_item = self.item(row, 1)
            lo_item = self.item(row, 2)
            hi_item = self.item(row, 3)
            chk_item = self.item(row, 4)

            v_txt = (v_item.text().strip() if v_item else "")
            lo_txt = (lo_item.text().strip() if lo_item else "")
            hi_txt = (hi_item.text().strip() if hi_item else "")

            if v_txt != "":
                try:
                    initial_out[key] = float(v_txt)
                except Exception:
                    initial_out[key] = v_txt

            refinable_out[key] = bool(chk_item and chk_item.checkState() == Qt.Checked)

            if lo_txt != "" or hi_txt != "":
                bounds_out[key] = f"{lo_txt}, {hi_txt}"

        return initial_out, refinable_out, bounds_out


class DynParamTableWidget(QtWidgets.QTableWidget):
    """Local-dynamics parameter table with per-parameter Use? flag."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(0, 6, parent)
        self._block = False

        self.setHorizontalHeaderLabels(["Parameter", "Value", "Min", "Max", "Use?", "Refine?"])

        hdr = self.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)

        for c in (1, 2, 3):
            hdr.setSectionResizeMode(c, QtWidgets.QHeaderView.Interactive)

        hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)

        self.setColumnWidth(1, 120)
        self.setColumnWidth(2, 90)
        self.setColumnWidth(3, 90)
        hdr.setMinimumSectionSize(60)

        self.setAlternatingRowColors(True)
        self.setToolTip(
            "Edit values/bounds. Use? controls whether the parameter is written to the input. "
            "Refine? controls whether it is refined."
        )

        self.itemChanged.connect(self._on_item_changed)

    def _on_item_changed(self, _item: QtWidgets.QTableWidgetItem):
        if not self._block:
            self.changed.emit()

    def clear_params(self):
        self._block = True
        try:
            self.setRowCount(0)
        finally:
            self._block = False

    def set_params(
        self,
        keys: List[str],
        initial: Dict[str, Any],
        refinable: Dict[str, Any],
        bounds: Dict[str, Any],
        *,
        tooltips: Optional[Dict[str, str]] = None,
        label_overrides: Optional[Dict[str, str]] = None,
        use_flags: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._block = True
        try:
            self.setRowCount(len(keys))
            tooltips = tooltips or {}
            label_overrides = label_overrides or {}
            use_flags = use_flags or {}

            for i, k in enumerate(keys):
                k_str = str(k)
                shown = label_overrides.get(k_str, k_str)

                it_name = QtWidgets.QTableWidgetItem(shown)
                it_name.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                it_name.setToolTip(tooltips.get(k_str, param_tooltip(k_str)))
                it_name.setData(Qt.UserRole, k_str)
                self.setItem(i, 0, it_name)

                val = initial.get(k_str, DEFAULT_PARAMS.get(k_str, ""))
                it_val = QtWidgets.QTableWidgetItem("" if val is None else str(val))
                it_val.setToolTip(param_tooltip(k_str))
                self.setItem(i, 1, it_val)

                lo, hi = ("", "")
                if k_str in bounds:
                    try:
                        lo, hi = bounds[k_str]
                    except Exception:
                        lo, hi = ("", "")

                it_lo = QtWidgets.QTableWidgetItem("" if lo in (None, "") else str(lo))
                it_hi = QtWidgets.QTableWidgetItem("" if hi in (None, "") else str(hi))
                it_lo.setToolTip("Lower bound")
                it_hi.setToolTip("Upper bound")
                self.setItem(i, 2, it_lo)
                self.setItem(i, 3, it_hi)

                chk_use = QtWidgets.QTableWidgetItem("")
                chk_use.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
                
                k_low = k_str.lower()
                
                # A parameter should be ON by default only if:
                #   1. an explicit use flag says so, OR
                #   2. it exists explicitly in the loaded input file.
                #
                # This prevents DEFAULT_PARAMS delta1/delta2 from being automatically activated.
                explicit_in_input = (
                    k_str in initial
                    or k_str in refinable
                    or k_str in bounds
                )
                
                if k_str in use_flags:
                    use_value = bool(use_flags[k_str])
                else:
                    if k_low in ("delta1", "delta2"):
                        # Global delta1/delta2 should NOT be auto-enabled just because DEFAULT_PARAMS has them.
                        use_value = bool(explicit_in_input)
                    elif k_low.startswith("delta1_") or k_low.startswith("delta2_"):
                        # Pair-specific deltas should be enabled only if explicitly present.
                        use_value = bool(explicit_in_input)
                    else:
                        # Biso and other local-dynamics parameters remain enabled by default.
                        use_value = True
                
                chk_use.setCheckState(Qt.Checked if use_value else Qt.Unchecked)
                chk_use.setToolTip("Tick to include this parameter in the model and export it.")
                self.setItem(i, 4, chk_use)

                chk_ref = QtWidgets.QTableWidgetItem("")
                chk_ref.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
                chk_ref.setCheckState(Qt.Checked if bool(refinable.get(k_str, False)) else Qt.Unchecked)
                chk_ref.setToolTip("Tick to refine this parameter.")
                self.setItem(i, 5, chk_ref)

        finally:
            self._block = False

    def extract_all(self) -> Tuple[Dict[str, Any], Dict[str, bool], Dict[str, str], Dict[str, bool]]:
        initial_out: Dict[str, Any] = {}
        refinable_out: Dict[str, bool] = {}
        bounds_out: Dict[str, str] = {}
        use_out: Dict[str, bool] = {}

        def _coerce(txt):
            try:
                return float(txt)
            except Exception:
                return txt

        for row in range(self.rowCount()):
            it_name = self.item(row, 0)
            if it_name is None:
                continue

            key = it_name.data(Qt.UserRole) or it_name.text().strip()
            key = str(key).strip()
            if not key:
                continue

            it_val = self.item(row, 1)
            it_lo = self.item(row, 2)
            it_hi = self.item(row, 3)
            it_use = self.item(row, 4)
            it_ref = self.item(row, 5)

            val_txt = it_val.text().strip() if it_val is not None else ""
            lo_txt = it_lo.text().strip() if it_lo is not None else ""
            hi_txt = it_hi.text().strip() if it_hi is not None else ""

            use_out[key] = bool(it_use and it_use.checkState() == Qt.Checked)
            refinable_out[key] = bool(it_ref and it_ref.checkState() == Qt.Checked)

            if val_txt != "":
                initial_out[key] = _coerce(val_txt)

            if lo_txt != "" or hi_txt != "":
                bounds_out[key] = f"{lo_txt}, {hi_txt}"

        return initial_out, refinable_out, bounds_out, use_out


class SiteParamTableWidget(QtWidgets.QTableWidget):
    """
    Atomic-site table for x/y/z, occupancy, and Biso.

    It exports flat parameter names:
        x_li1, y_li1, z_li1, occ_li1, biso_li1

    Biso has Use? and Refine? controls directly in this table.
    """

    changed = QtCore.Signal()

    # Emitted as (row element, kind) when the user toggles a refinement
    # checkbox; kind is "occ" (Rocc) or "xyz" (Rx/Ry/Rz).
    refineFlagToggled = QtCore.Signal(str, str)

    COL_EL = 2
    COL_OCC = 9
    COL_ROCC = 10
    COL_OCC_MIN = 11
    COL_OCC_MAX = 12
    COL_BISO = 13
    COL_BISO_MIN = 14
    COL_BISO_MAX = 15
    COL_BISO_USE = 16
    COL_BISO_REF = 17
    BISO_COLUMNS = (13, 14, 15, 16, 17)

    COL_RX = 4
    COL_RY = 6
    COL_RZ = 8

    REFINE_COLUMNS = {
        "occ": (10,),
        "xyz": (4, 6, 8),
    }

    def __init__(self, parent=None):
        super().__init__(0, 18, parent)
        self._block = False

        self.setHorizontalHeaderLabels([
            "#",
            "Label",
            "El",
            "x",
            "Rx",
            "y",
            "Ry",
            "z",
            "Rz",
            "occ",
            "Rocc",
            "Occ Min",
            "Occ Max",
            "Biso",
            "Biso Min",
            "Biso Max",
            "Use Biso?",
            "Rbiso",
        ])

        hdr = self.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setMinimumSectionSize(35)

        for column in range(self.columnCount()):
            hdr.setSectionResizeMode(
                column,
                QtWidgets.QHeaderView.Interactive,
            )

        self.setColumnWidth(0, 42)
        self.setColumnWidth(1, 90)
        self.setColumnWidth(2, 45)

        self.setColumnWidth(3, 95)
        self.setColumnWidth(4, 42)

        self.setColumnWidth(5, 95)
        self.setColumnWidth(6, 42)

        self.setColumnWidth(7, 95)
        self.setColumnWidth(8, 42)

        self.setColumnWidth(9, 75)
        self.setColumnWidth(10, 50)

        self.setColumnWidth(self.COL_OCC_MIN, 65)
        self.setColumnWidth(self.COL_OCC_MAX, 65)

        self.setColumnWidth(self.COL_BISO, 90)
        self.setColumnWidth(self.COL_BISO_MIN, 75)
        self.setColumnWidth(self.COL_BISO_MAX, 75)
        self.setColumnWidth(self.COL_BISO_USE, 70)
        self.setColumnWidth(self.COL_BISO_REF, 55)

        hdr.setSectionResizeMode(
            1,
            QtWidgets.QHeaderView.Stretch,
        )
        # Make the whole widget use available space.
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )

        self.setMinimumHeight(260)

        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.itemChanged.connect(self._on_item_changed)

        self.setToolTip(
            "Atomic-site parameters. Rx/Ry/Rz/Rocc/Rbiso are refinement flags. "
            "Use Biso? controls whether this site Biso is written to the input."
        )

    def _on_item_changed(self, item):
        if self._block:
            return

        if item is not None:
            for kind, columns in self.REFINE_COLUMNS.items():
                if item.column() not in columns:
                    continue

                el_item = self.item(item.row(), self.COL_EL)

                if el_item is not None:
                    self.refineFlagToggled.emit(
                        str(el_item.text()).strip(),
                        kind,
                    )

        self.changed.emit()

    @staticmethod
    def _split_bounds(value) -> Tuple[str, str]:
        """(lo, hi) tuple/list or "lo, hi" string -> ("lo", "hi") text."""
        if value is None:
            return "", ""

        try:
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                lo, hi = value[0], value[1]
            else:
                parts = [p.strip() for p in str(value).split(",", 1)]

                if len(parts) != 2:
                    return "", ""

                lo, hi = parts

            lo = "" if lo in (None, "") else str(lo)
            hi = "" if hi in (None, "") else str(hi)
            return lo, hi

        except Exception:
            return "", ""

    @staticmethod
    def _safe_key(label: str) -> str:
        s = str(label).strip()
        s = re.sub(r"[^A-Za-z0-9]+", "", s)
        return s.lower()

    def clear_params(self):
        self._block = True
        try:
            self.setRowCount(0)
        finally:
            self._block = False

    def set_sites(
        self,
        sites: List[Dict[str, Any]],
        initial: Optional[Dict[str, Any]] = None,
        refinable: Optional[Dict[str, Any]] = None,
        bounds: Optional[Dict[str, Any]] = None,
        use_flags: Optional[Dict[str, Any]] = None,
    ) -> None:
        initial = initial or {}
        refinable = refinable or {}
        bounds = bounds or {}
        use_flags = use_flags or {}

        sites = list(
            sites
            or []
        )

        old_sorting_enabled = self.isSortingEnabled()

        self._block = True
        self.blockSignals(True)
        self.setUpdatesEnabled(False)
        self.setSortingEnabled(False)

        try:
            self.clearContents()
            self.setRowCount(
                len(sites)
            )

            enabled_selectable_flags = (
                Qt.ItemIsEnabled
                | Qt.ItemIsSelectable
            )

            editable_flags = (
                Qt.ItemIsEnabled
                | Qt.ItemIsSelectable
                | Qt.ItemIsEditable
            )

            checkbox_flags = (
                Qt.ItemIsEnabled
                | Qt.ItemIsSelectable
                | Qt.ItemIsUserCheckable
            )

            def make_text_item(
                value,
                *,
                editable: bool,
                parameter_key: str | None = None,
                tooltip: str = "",
            ):
                item = QtWidgets.QTableWidgetItem(
                    ""
                    if value is None
                    else str(value)
                )

                item.setFlags(
                    editable_flags
                    if editable
                    else enabled_selectable_flags
                )

                if parameter_key:
                    item.setData(
                        Qt.UserRole,
                        str(parameter_key),
                    )

                if tooltip:
                    item.setToolTip(
                        tooltip
                    )

                return item

            def make_checkbox_item(
                checked: bool,
                *,
                tooltip: str = "",
            ):
                item = QtWidgets.QTableWidgetItem("")
                item.setFlags(
                    checkbox_flags
                )

                item.setCheckState(
                    Qt.Checked
                    if bool(checked)
                    else Qt.Unchecked
                )

                if tooltip:
                    item.setToolTip(
                        tooltip
                    )

                return item

            for row, record in enumerate(sites):
                label = str(
                    record.get(
                        "label",
                        f"site{row + 1}",
                    )
                    or f"site{row + 1}"
                ).strip()

                element = str(
                    record.get(
                        "el",
                        "",
                    )
                    or ""
                ).strip()

                fractional = record.get(
                    "frac",
                    (
                        0.0,
                        0.0,
                        0.0,
                    ),
                )

                try:
                    x_value = fractional[0]
                    y_value = fractional[1]
                    z_value = fractional[2]

                except Exception:
                    x_value = 0.0
                    y_value = 0.0
                    z_value = 0.0

                occupancy_value = record.get(
                    "occ",
                    1.0,
                )

                biso_value = record.get(
                    "biso",
                    "",
                )

                safe_label = self._safe_key(
                    label
                )

                x_key = f"x_{safe_label}"
                y_key = f"y_{safe_label}"
                z_key = f"z_{safe_label}"
                occupancy_key = f"occ_{safe_label}"
                biso_key = f"biso_{safe_label}"

                x_value = initial.get(
                    x_key,
                    x_value,
                )

                y_value = initial.get(
                    y_key,
                    y_value,
                )

                z_value = initial.get(
                    z_key,
                    z_value,
                )

                occupancy_value = initial.get(
                    occupancy_key,
                    occupancy_value,
                )

                biso_value = initial.get(
                    biso_key,
                    biso_value,
                )

                self.setItem(
                    row,
                    0,
                    make_text_item(
                        row,
                        editable=False,
                    ),
                )

                label_item = make_text_item(
                    label,
                    editable=False,
                )

                label_item.setData(
                    Qt.UserRole,
                    safe_label,
                )

                self.setItem(
                    row,
                    1,
                    label_item,
                )

                self.setItem(
                    row,
                    2,
                    make_text_item(
                        element,
                        editable=False,
                    ),
                )

                self.setItem(
                    row,
                    3,
                    make_text_item(
                        x_value,
                        editable=True,
                        parameter_key=x_key,
                    ),
                )

                self.setItem(
                    row,
                    4,
                    make_checkbox_item(
                        refinable.get(
                            x_key,
                            False,
                        ),
                        tooltip="Refine this x coordinate.",
                    ),
                )

                self.setItem(
                    row,
                    5,
                    make_text_item(
                        y_value,
                        editable=True,
                        parameter_key=y_key,
                    ),
                )

                self.setItem(
                    row,
                    6,
                    make_checkbox_item(
                        refinable.get(
                            y_key,
                            False,
                        ),
                        tooltip="Refine this y coordinate.",
                    ),
                )

                self.setItem(
                    row,
                    7,
                    make_text_item(
                        z_value,
                        editable=True,
                        parameter_key=z_key,
                    ),
                )

                self.setItem(
                    row,
                    8,
                    make_checkbox_item(
                        refinable.get(
                            z_key,
                            False,
                        ),
                        tooltip="Refine this z coordinate.",
                    ),
                )

                self.setItem(
                    row,
                    9,
                    make_text_item(
                        occupancy_value,
                        editable=True,
                        parameter_key=occupancy_key,
                    ),
                )

                self.setItem(
                    row,
                    10,
                    make_checkbox_item(
                        refinable.get(
                            occupancy_key,
                            False,
                        ),
                        tooltip="Refine this occupancy.",
                    ),
                )

                occ_minimum, occ_maximum = self._split_bounds(
                    bounds.get(occupancy_key)
                )

                self.setItem(
                    row,
                    self.COL_OCC_MIN,
                    make_text_item(
                        occ_minimum,
                        editable=True,
                        tooltip=(
                            "Lower occupancy limit for this site. "
                            "Blank: use the global occupancy limits."
                        ),
                    ),
                )

                self.setItem(
                    row,
                    self.COL_OCC_MAX,
                    make_text_item(
                        occ_maximum,
                        editable=True,
                        tooltip=(
                            "Upper occupancy limit for this site. "
                            "Blank: use the global occupancy limits."
                        ),
                    ),
                )

                biso_minimum = "0"
                biso_maximum = "10"

                if biso_key in bounds:
                    bound_value = bounds.get(
                        biso_key
                    )

                    try:
                        if (
                            isinstance(
                                bound_value,
                                (
                                    tuple,
                                    list,
                                ),
                            )
                            and len(bound_value) >= 2
                        ):
                            lower_value = bound_value[0]
                            upper_value = bound_value[1]

                            if lower_value not in (
                                None,
                                "",
                            ):
                                biso_minimum = str(
                                    lower_value
                                )

                            if upper_value not in (
                                None,
                                "",
                            ):
                                biso_maximum = str(
                                    upper_value
                                )

                        else:
                            parts = [
                                part.strip()
                                for part in str(
                                    bound_value
                                ).split(
                                    ",",
                                    1,
                                )
                            ]

                            if len(parts) == 2:
                                if parts[0]:
                                    biso_minimum = parts[0]

                                if parts[1]:
                                    biso_maximum = parts[1]

                    except Exception:
                        biso_minimum = "0"
                        biso_maximum = "10"

                self.setItem(
                    row,
                    self.COL_BISO,
                    make_text_item(
                        biso_value,
                        editable=True,
                        parameter_key=biso_key,
                    ),
                )

                self.setItem(
                    row,
                    self.COL_BISO_MIN,
                    make_text_item(
                        biso_minimum,
                        editable=True,
                        tooltip="Lower bound for this site Biso.",
                    ),
                )

                self.setItem(
                    row,
                    self.COL_BISO_MAX,
                    make_text_item(
                        biso_maximum,
                        editable=True,
                        tooltip="Upper bound for this site Biso.",
                    ),
                )

                self.setItem(
                    row,
                    self.COL_BISO_USE,
                    make_checkbox_item(
                        use_flags.get(
                            biso_key,
                            True,
                        ),
                        tooltip=(
                            "Include this site Biso parameter "
                            "in the input."
                        ),
                    ),
                )

                self.setItem(
                    row,
                    self.COL_BISO_REF,
                    make_checkbox_item(
                        refinable.get(
                            biso_key,
                            False,
                        ),
                        tooltip="Refine this site Biso.",
                    ),
                )

        finally:
            self.setSortingEnabled(
                old_sorting_enabled
            )

            self.blockSignals(False)
            self.setUpdatesEnabled(True)
            self._block = False

            self.viewport().update()

            
    def extract_all(self) -> Tuple[Dict[str, Any], Dict[str, bool], Dict[str, str], Dict[str, bool]]:
        initial_out: Dict[str, Any] = {}
        refinable_out: Dict[str, bool] = {}
        bounds_out: Dict[str, str] = {}
        use_out: Dict[str, bool] = {}

        def _coerce(txt):
            try:
                return float(txt)
            except Exception:
                return txt

        def _value_key(row, col):
            it = self.item(row, col)
            if it is None:
                return "", ""
            return str(it.data(Qt.UserRole) or ""), str(it.text()).strip()

        for row in range(self.rowCount()):
            # x/y/z/occ are always used if present.
            for value_col, refine_col in (
                (3, 4),   # x
                (5, 6),   # y
                (7, 8),   # z
                (9, 10),  # occ
            ):
                key, txt = _value_key(row, value_col)
                if not key:
                    continue

                if txt != "":
                    initial_out[key] = _coerce(txt)

                chk_ref = self.item(row, refine_col)
                refinable_out[key] = bool(
                    chk_ref and chk_ref.checkState() == Qt.Checked
                )
                use_out[key] = True

                if value_col == self.COL_OCC:
                    lo_item = self.item(row, self.COL_OCC_MIN)
                    hi_item = self.item(row, self.COL_OCC_MAX)
                    lo_txt = lo_item.text().strip() if lo_item is not None else ""
                    hi_txt = hi_item.text().strip() if hi_item is not None else ""

                    if lo_txt != "" or hi_txt != "":
                        bounds_out[key] = f"{lo_txt}, {hi_txt}"

            # Biso has bounds, Use?, and Refine?.
            key, txt = _value_key(row, self.COL_BISO)
            if key:
                lo_item = self.item(row, self.COL_BISO_MIN)
                hi_item = self.item(row, self.COL_BISO_MAX)
                chk_use = self.item(row, self.COL_BISO_USE)
                chk_ref = self.item(row, self.COL_BISO_REF)

                lo_txt = lo_item.text().strip() if lo_item is not None else ""
                hi_txt = hi_item.text().strip() if hi_item is not None else ""

                use_biso = bool(chk_use and chk_use.checkState() == Qt.Checked)
                use_out[key] = use_biso

                if use_biso:
                    refinable_out[key] = bool(
                        chk_ref and chk_ref.checkState() == Qt.Checked
                    )

                    if txt != "":
                        initial_out[key] = _coerce(txt)

                    if lo_txt != "" or hi_txt != "":
                        bounds_out[key] = f"{lo_txt}, {hi_txt}"

        return initial_out, refinable_out, bounds_out, use_out

    def set_refine_flags_for_element(
        self,
        element: str,
        checked: bool,
        kind: str = "occ",
    ) -> int:
        """
        Set the refine flags of ``kind`` ("occ": Rocc, "xyz": Rx/Ry/Rz) on
        every row whose element matches. Returns the number of checkboxes
        changed. Emits ``changed`` once.
        """
        target = str(element).strip().lower()
        state = Qt.Checked if bool(checked) else Qt.Unchecked
        columns = self.REFINE_COLUMNS.get(kind, ())
        n_changed = 0

        self._block = True

        try:
            for row in range(self.rowCount()):
                el_item = self.item(row, self.COL_EL)

                if el_item is None:
                    continue

                if str(el_item.text()).strip().lower() != target:
                    continue

                for col in columns:
                    chk = self.item(row, col)

                    if chk is None or chk.checkState() == state:
                        continue

                    chk.setCheckState(state)
                    n_changed += 1

        finally:
            self._block = False

        if n_changed:
            self.changed.emit()

        return n_changed

    def element_refine_state(self, element: str, kind: str = "occ"):
        """
        True if every ``kind`` flag of every row of ``element`` is checked,
        False if not, None if the element has no rows.
        """
        target = str(element).strip().lower()
        columns = self.REFINE_COLUMNS.get(kind, ())
        flags = []

        for row in range(self.rowCount()):
            el_item = self.item(row, self.COL_EL)

            if el_item is None or str(el_item.text()).strip().lower() != target:
                continue

            for col in columns:
                chk = self.item(row, col)
                flags.append(chk is not None and chk.checkState() == Qt.Checked)

        if not flags:
            return None

        return all(flags)

    def set_biso_controls_enabled(self, enabled: bool) -> None:
        """
        Enable/disable site-specific Biso editing/refinement columns.
    
        Columns:
            11 Biso
            12 Biso Min
            13 Biso Max
            14 Use Biso?
            15 Rbiso
        """
        enabled = bool(enabled)
    
        for row in range(self.rowCount()):
            for col in self.BISO_COLUMNS:
                it = self.item(row, col)
                if it is None:
                    continue
                
                flags = it.flags()
    
                if enabled:
                    flags |= Qt.ItemIsEnabled
    
                    if col in (self.COL_BISO, self.COL_BISO_MIN, self.COL_BISO_MAX):
                        flags |= Qt.ItemIsEditable
                else:
                    flags &= ~Qt.ItemIsEnabled
                    flags &= ~Qt.ItemIsEditable
    
                it.setFlags(flags)

class LambdaListWidget(QtWidgets.QWidget):
    """Per-element-pair lambda coefficients.

    Preferred parameter names:
        lambda_{A}-{B}_{k}

    Example:
        lambda_fe-fe_0

    Also accepts this on load:
        lambda_fe_fe_0

    Legacy global lambdas are also supported:
        lambda_0
        lambda_1
    """

    changed = QtCore.Signal()

    _PAIR_LEGACY = "Global (legacy)"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._block = False

        self._pairs: List[str] = []
        self._data: Dict[str, Dict[int, Dict[str, Any]]] = {}
        
        # Track which pair the table currently represents.
        # This prevents committing old table values into the newly selected pair.
        self._current_pair: str = ""

        self.combo_pair = QtWidgets.QComboBox()
        self.combo_pair.currentIndexChanged.connect(self._on_pair_changed)

        self.tbl = QtWidgets.QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(["k", "Value", "Min", "Max", "Refine?"])

        hdr = self.tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        for c in (1, 2, 3):
            hdr.setSectionResizeMode(c, QtWidgets.QHeaderView.Interactive)
        hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)

        self.tbl.setColumnWidth(1, 120)
        self.tbl.setColumnWidth(2, 90)
        self.tbl.setColumnWidth(3, 90)

        self.tbl.setAlternatingRowColors(True)
        self.tbl.itemChanged.connect(self._on_item_changed)

        btn_add = QtWidgets.QPushButton("Add λ")
        btn_del = QtWidgets.QPushButton("Remove selected")
        btn_save = QtWidgets.QPushButton("Save λ .txt")

        btn_add.setToolTip("Add a lambda coefficient for the selected pair.")
        btn_del.setToolTip("Remove selected lambda coefficient rows.")
        btn_save.setToolTip(
            "Save all lambda coefficients for all pair types as a tab-separated text file."
        )

        btn_add.clicked.connect(self.add_lambda)
        btn_del.clicked.connect(self.remove_selected)
        btn_save.clicked.connect(self.save_lambdas_txt)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Pair:"))
        top.addWidget(self.combo_pair, 1)
        top.addWidget(btn_add)
        top.addWidget(btn_del)
        top.addWidget(btn_save)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self.tbl, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_pairs(self, pairs: List[str], *, commit_current: bool = True) -> None:
        """Set available unordered element pairs."""

        def _norm_pair(p: str) -> str:
            p = str(p).strip()
            if not p:
                return ""
            if p == self._PAIR_LEGACY:
                return p

            p = p.replace("_", "-")

            if "-" not in p:
                return p.lower()

            a, b = [x.strip().lower() for x in p.split("-", 1)]
            aa, bb = sorted((a, b), key=lambda z: z.lower())
            return f"{aa}-{bb}"

        # Commit the currently displayed table into the OLD pair before rebuilding the combo.
        if commit_current and not self._block:
            self._commit_current_table(self._current_pair or self.combo_pair.currentText().strip())

        pairs_norm = [_norm_pair(p) for p in (pairs or []) if str(p).strip()]
        pairs_norm = sorted(set(pairs_norm), key=lambda s: s.lower())

        if self._PAIR_LEGACY in self._data and self._PAIR_LEGACY not in pairs_norm:
            pairs_norm = [self._PAIR_LEGACY] + pairs_norm

        previous_pair = self._current_pair or self.combo_pair.currentText().strip()

        self._pairs = pairs_norm

        self._block = True
        try:
            self.combo_pair.clear()
            self.combo_pair.addItems(self._pairs)

            if previous_pair and previous_pair in self._pairs:
                self.combo_pair.setCurrentText(previous_pair)
            elif self._pairs:
                self.combo_pair.setCurrentIndex(0)

            self._current_pair = self.combo_pair.currentText().strip()

        finally:
            self._block = False

        self._rebuild_table_for_current_pair()
    def set_lambdas(
        self,
        initial: Dict[str, Any],
        refinable: Dict[str, Any],
        bounds: Dict[str, Any],
    ) -> None:
        """Populate from flat dictionaries keyed by parameter name."""
        self._data = {}

        initial = initial or {}
        refinable = refinable or {}
        bounds = bounds or {}

        all_keys = set(initial.keys()) | set(refinable.keys()) | set(bounds.keys())

        for key in sorted(all_keys, key=str):
            name = str(key).strip()

            pair, k = self._parse_pair_lambda_name(name)
            if pair is not None and k is not None:
                self._set_entry(
                    pair,
                    k,
                    initial.get(key, 0.0),
                    refinable.get(key, False),
                    bounds.get(key, ("", "")),
                )
                continue

            lk = self._parse_legacy_lambda_name(name)
            if lk is not None:
                self._set_entry(
                    self._PAIR_LEGACY,
                    lk,
                    initial.get(key, 0.0),
                    refinable.get(key, False),
                    bounds.get(key, ("", "")),
                )

        found_pairs = sorted(self._data.keys(), key=lambda s: s.lower())

        if not self._pairs:
            self.set_pairs(found_pairs, commit_current=False)
        else:
            merged = list(self._pairs)
            for p in found_pairs:
                if p not in merged:
                    merged.append(p)
            self.set_pairs(merged, commit_current=False)

        self._rebuild_table_for_current_pair()

    def extract_all(self) -> Tuple[Dict[str, Any], Dict[str, bool], Dict[str, Tuple[str, str]]]:
        """Return flat dicts: initial, refinable, bounds."""
        self._commit_current_table()

        ini: Dict[str, Any] = {}
        ref: Dict[str, bool] = {}
        bnd: Dict[str, Tuple[str, str]] = {}

        for pair, kmap in (self._data or {}).items():
            for k, data in (kmap or {}).items():
                name = self._format_name(pair, k)
                ini[name] = self._coerce_num(data.get("value", 0.0))
                ref[name] = bool(data.get("refine", False))

                lo = str(data.get("min", "")).strip()
                hi = str(data.get("max", "")).strip()
                bnd[name] = (lo, hi)

        return ini, ref, bnd

    # ------------------------------------------------------------------
    # UI actions
    # ------------------------------------------------------------------
    def add_lambda(self) -> None:
        pair = self.combo_pair.currentText().strip()
        if not pair:
            return

        self._commit_current_table()

        kmap = self._data.setdefault(pair, {})
        next_k = 0 if not kmap else max(kmap.keys()) + 1

        self._set_entry(pair, next_k, 0.0, False, ("-1", "1"))
        self._rebuild_table_for_current_pair()
        self.changed.emit()

    def remove_selected(self) -> None:
        pair = self.combo_pair.currentText().strip()
        if not pair:
            return

        self._commit_current_table()

        rows = sorted({i.row() for i in self.tbl.selectedItems()}, reverse=True)
        if not rows:
            return

        kmap = self._data.get(pair, {})
        for r in rows:
            k = self._row_k(r)
            if k is not None:
                kmap.pop(k, None)

        self._data[pair] = kmap
        self._rebuild_table_for_current_pair()
        self.changed.emit()

    def save_lambdas_txt(self) -> None:
        """
        Save all lambda coefficients for all pair types as a tab-separated file.
        """
        try:
            # Store any edits currently visible in the table before exporting.
            self._commit_current_table()

            if not self._data:
                QtWidgets.QMessageBox.information(
                    self,
                    "No lambda coefficients",
                    "There are no lambda coefficients to save.",
                )
                return

            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save lambda coefficients",
                "lambda_coefficients.txt",
                "Text files (*.txt);;CSV files (*.csv);;All files (*)",
            )

            if not path:
                return

            with open(path, "w", encoding="utf-8") as file:
                file.write("# Lambda coefficients\n")
                file.write(
                    "# Columns: parameter\tpair\tk\tvalue\tmin\tmax\trefine\n"
                )

                for pair in sorted(
                    self._data.keys(),
                    key=lambda value: str(value).lower(),
                ):
                    kmap = self._data.get(pair, {}) or {}

                    for k in sorted(kmap.keys()):
                        data = kmap.get(k, {}) or {}

                        parameter_name = self._format_name(
                            pair,
                            int(k),
                        )

                        value = data.get(
                            "value",
                            "",
                        )

                        minimum = data.get(
                            "min",
                            "",
                        )

                        maximum = data.get(
                            "max",
                            "",
                        )

                        refine = bool(
                            data.get(
                                "refine",
                                False,
                            )
                        )

                        pair_text = (
                            "global"
                            if pair == self._PAIR_LEGACY
                            else str(pair)
                        )

                        file.write(
                            f"{parameter_name}\t"
                            f"{pair_text}\t"
                            f"{int(k)}\t"
                            f"{value}\t"
                            f"{minimum}\t"
                            f"{maximum}\t"
                            f"{'true' if refine else 'false'}\n"
                        )

            QtWidgets.QMessageBox.information(
                self,
                "Lambda coefficients saved",
                f"Saved lambda coefficients to:\n{path}",
            )

        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Save lambda coefficients failed",
                str(exc),
            )
            
    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _on_pair_changed(self, idx: int) -> None:
        if self._block:
            return

        # Commit visible table into the OLD pair, not the newly selected one.
        old_pair = self._current_pair
        if old_pair:
            self._commit_current_table(old_pair)

        # Now switch active pair.
        new_pair = self.combo_pair.itemText(idx).strip()
        self._current_pair = new_pair

        # Load the new pair's own lambda values.
        self._rebuild_table_for_current_pair()

    def _on_item_changed(self, _it=None) -> None:
        if not self._block:
            self.changed.emit()

    def _commit_current_table(self, pair: Optional[str] = None) -> None:
        if pair is None:
            pair = self._current_pair or self.combo_pair.currentText().strip()

        pair = str(pair).strip()
        if not pair:
            return

        kmap: Dict[int, Dict[str, Any]] = {}

        for r in range(self.tbl.rowCount()):
            k = self._row_k(r)
            if k is None:
                continue
            kmap[k] = self._row_to_data(r)

        self._data[pair] = kmap

    def _rebuild_table_for_current_pair(self) -> None:
        pair = self.combo_pair.currentText().strip()
        self._current_pair = pair

        self._block = True
        try:
            self.tbl.setRowCount(0)

            if not pair:
                return

            rows = sorted(
                (self._data.get(pair, {}) or {}).items(),
                key=lambda kv: kv[0],
            )

            self.tbl.setRowCount(len(rows))

            for i, (k, data) in enumerate(rows):
                param_name = self._format_name(pair, k)

                it_k = QtWidgets.QTableWidgetItem(str(int(k)))
                it_k.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                it_k.setToolTip(f"This row writes parameter: {param_name}")
                self.tbl.setItem(i, 0, it_k)

                self.tbl.setItem(i, 1, QtWidgets.QTableWidgetItem(self._fmt_num(data.get("value", ""))))
                self.tbl.setItem(i, 2, QtWidgets.QTableWidgetItem(self._fmt_num(data.get("min", ""))))
                self.tbl.setItem(i, 3, QtWidgets.QTableWidgetItem(self._fmt_num(data.get("max", ""))))

                chk = QtWidgets.QTableWidgetItem("")
                chk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                chk.setCheckState(Qt.Checked if bool(data.get("refine", False)) else Qt.Unchecked)
                self.tbl.setItem(i, 4, chk)

        finally:
            self._block = False

    def _row_k(self, row: int) -> Optional[int]:
        it = self.tbl.item(row, 0)
        if it is None:
            return None

        try:
            return int(str(it.text()).strip())
        except Exception:
            return None

    def _row_to_data(self, row: int) -> Dict[str, Any]:
        def _txt(c: int) -> str:
            it = self.tbl.item(row, c)
            return it.text().strip() if it is not None else ""

        chk = self.tbl.item(row, 4)
        refine = bool(chk and chk.checkState() == Qt.Checked)

        return {
            "value": _txt(1),
            "min": _txt(2),
            "max": _txt(3),
            "refine": refine,
        }

    def _set_entry(self, pair: str, k: int, value: Any, refine: bool, bound: Any) -> None:
        pair = self._normalize_pair_name(pair)
        if not pair:
            return

        k = int(k)

        lo, hi = "", ""
        if isinstance(bound, str):
            if "," in bound:
                parts = [x.strip() for x in bound.split(",", 1)]
                if len(parts) == 2:
                    lo, hi = parts[0], parts[1]
        elif isinstance(bound, (tuple, list)) and len(bound) >= 2:
            lo = "" if bound[0] in (None, "") else str(bound[0])
            hi = "" if bound[1] in (None, "") else str(bound[1])
        
        # Default lambda bounds if none were provided.

        if str(lo).strip() == "" and str(hi).strip() == "":
            lo, hi = "-1", "1"
            
        self._data.setdefault(pair, {})[k] = {
            "value": value,
            "min": lo,
            "max": hi,
            "refine": bool(refine),
        }

    @staticmethod
    def _fmt_num(x: Any, ndigits: int = 3) -> str:
        """Format lambda table numbers with fixed decimals when possible."""
        if x is None:
            return ""
        s = str(x).strip()
        if s == "":
            return ""
        try:
            return f"{float(s):.{ndigits}f}"
        except Exception:
            return s
        
    @staticmethod
    def _coerce_num(x: Any) -> Any:
        try:
            if x is None:
                return 0.0
            return float(x)
        except Exception:
            return x

    @classmethod
    def _normalize_pair_name(cls, pair: str) -> str:
        pair = str(pair).strip()
        if not pair:
            return ""

        if pair == cls._PAIR_LEGACY:
            return pair

        pair = pair.replace("_", "-")

        if "-" not in pair:
            return pair.lower()

        a, b = [x.strip().lower() for x in pair.split("-", 1)]
        aa, bb = sorted((a, b), key=lambda z: z.lower())
        return f"{aa}-{bb}"

    @classmethod
    def _parse_pair_lambda_name(cls, name: str) -> Tuple[Optional[str], Optional[int]]:
        """Parse pair lambda names.

        Accepted:
            lambda_fe-fe_0
            lambda_Fe-Fe_0
            lambda_fe_fe_0
            lambda_Fe_Fe_0

        Returned pair is normalized:
            fe-fe
        """
        s = (name or "").strip()

        # Preferred format: lambda_A-B_k
        m = re.match(r"(?i)^lambda_([A-Za-z0-9]+)-([A-Za-z0-9]+)_(\d+)$", s)
        if m:
            a = m.group(1).strip().lower()
            b = m.group(2).strip().lower()
            k = int(m.group(3))
            return cls._normalize_pair_name(f"{a}-{b}"), k

        # Tolerant format: lambda_A_B_k
        m = re.match(r"(?i)^lambda_([A-Za-z0-9]+)_([A-Za-z0-9]+)_(\d+)$", s)
        if m:
            a = m.group(1).strip().lower()
            b = m.group(2).strip().lower()
            k = int(m.group(3))
            return cls._normalize_pair_name(f"{a}-{b}"), k

        return None, None

    @staticmethod
    def _parse_legacy_lambda_name(name: str) -> Optional[int]:
        s = (name or "").strip()
        m = re.match(r"(?i)^lambda_?(\d+)$", s)
        if not m:
            return None
        return int(m.group(1))

    def _format_name(self, pair: str, k: int) -> str:
        if pair == self._PAIR_LEGACY:
            return f"lambda_{int(k)}"

        pair = self._normalize_pair_name(pair)
        return f"lambda_{pair}_{int(k)}"