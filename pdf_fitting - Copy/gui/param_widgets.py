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

    def __init__(self, parent=None):
        super().__init__(0, 16, parent)
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
            "Biso",
            "Biso Min",
            "Biso Max",
            "Use Biso?",
            "Rbiso",
        ])

        hdr = self.horizontalHeader()
        hdr.setStretchLastSection(False)

        # Compact columns by default.
        for c in range(self.columnCount()):
            hdr.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)

        # Keep Label compact.
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)

        # Let Biso column absorb extra horizontal space instead of leaving a blank area.
        # This keeps the label compact but makes the table occupy the available width.
        hdr.setSectionResizeMode(11, QtWidgets.QHeaderView.Stretch)

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

    def _on_item_changed(self, _item):
        if not self._block:
            self.changed.emit()

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
        use_flags = use_flags or {}
        bounds = bounds or {}
        
        if not sites:
            self._block = True
            try:
                self.setRowCount(0)
            finally:
                self._block = False

            return
        
        self._block = True
        try:
            self.setRowCount(len(sites or []))

            for row, rec in enumerate(sites or []):
                label = str(rec.get("label", f"site{row + 1}")).strip()
                el = str(rec.get("el", "")).strip()

                x, y, z = rec.get("frac", (0.0, 0.0, 0.0))
                occ = rec.get("occ", 1.0)
                biso = rec.get("biso", "")

                key = self._safe_key(label)

                keys = {
                    "x": f"x_{key}",
                    "y": f"y_{key}",
                    "z": f"z_{key}",
                    "occ": f"occ_{key}",
                    "biso": f"biso_{key}",
                }

                values = {
                    "x": initial.get(keys["x"], x),
                    "y": initial.get(keys["y"], y),
                    "z": initial.get(keys["z"], z),
                    "occ": initial.get(keys["occ"], occ),
                    "biso": initial.get(keys["biso"], biso),
                }

                def item(text, editable=True):
                    it = QtWidgets.QTableWidgetItem("" if text is None else str(text))
                    flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
                    if editable:
                        flags |= Qt.ItemIsEditable
                    it.setFlags(flags)
                    return it

                def chk(param_key, default=False):
                    it = QtWidgets.QTableWidgetItem("")
                    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Checked if bool(refinable.get(param_key, default)) else Qt.Unchecked)
                    return it

                def use_chk(param_key, default=True):
                    it = QtWidgets.QTableWidgetItem("")
                    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Checked if bool(use_flags.get(param_key, default)) else Qt.Unchecked)
                    return it

                # #
                self.setItem(row, 0, item(row, editable=False))

                # Label
                it_label = item(label, editable=False)
                it_label.setData(Qt.UserRole, key)
                self.setItem(row, 1, it_label)

                # Element
                self.setItem(row, 2, item(el, editable=False))

                # x/Rx
                it_x = item(values["x"], editable=True)
                it_x.setData(Qt.UserRole, keys["x"])
                self.setItem(row, 3, it_x)
                self.setItem(row, 4, chk(keys["x"]))

                # y/Ry
                it_y = item(values["y"], editable=True)
                it_y.setData(Qt.UserRole, keys["y"])
                self.setItem(row, 5, it_y)
                self.setItem(row, 6, chk(keys["y"]))

                # z/Rz
                it_z = item(values["z"], editable=True)
                it_z.setData(Qt.UserRole, keys["z"])
                self.setItem(row, 7, it_z)
                self.setItem(row, 8, chk(keys["z"]))

                # occ/Rocc
                it_occ = item(values["occ"], editable=True)
                it_occ.setData(Qt.UserRole, keys["occ"])
                self.setItem(row, 9, it_occ)
                self.setItem(row, 10, chk(keys["occ"]))

                # Biso / Biso Min / Biso Max / Use Biso / Rbiso
                it_biso = item(values["biso"], editable=True)
                it_biso.setData(Qt.UserRole, keys["biso"])
                self.setItem(row, 11, it_biso)

                # Bounds for Biso.
                # Default Biso bounds are physically conservative:
                #   0 <= Biso <= 10 Å²
                # User can edit these in the GUI.
                biso_lo, biso_hi = "0", "10"
                
                if keys["biso"] in bounds:
                    bv = bounds.get(keys["biso"])
                
                    try:
                        if isinstance(bv, (tuple, list)) and len(bv) >= 2:
                            if bv[0] not in (None, ""):
                                biso_lo = str(bv[0])
                            if bv[1] not in (None, ""):
                                biso_hi = str(bv[1])
                        else:
                            parts = [x.strip() for x in str(bv).split(",", 1)]
                            if len(parts) == 2:
                                if parts[0] != "":
                                    biso_lo = parts[0]
                                if parts[1] != "":
                                    biso_hi = parts[1]
                    except Exception:
                        biso_lo, biso_hi = "0", "10"

                it_biso_lo = item(biso_lo, editable=True)
                it_biso_lo.setToolTip("Lower bound for this site Biso.")
                self.setItem(row, 12, it_biso_lo)

                it_biso_hi = item(biso_hi, editable=True)
                it_biso_hi.setToolTip("Upper bound for this site Biso.")
                self.setItem(row, 13, it_biso_hi)

                self.setItem(row, 14, use_chk(keys["biso"], default=True))
                self.setItem(row, 15, chk(keys["biso"]))

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

            # Biso has bounds, Use?, and Refine?.
            key, txt = _value_key(row, 11)
            if key:
                lo_item = self.item(row, 12)
                hi_item = self.item(row, 13)
                chk_use = self.item(row, 14)
                chk_ref = self.item(row, 15)

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
            for col in (11, 12, 13, 14, 15):
                it = self.item(row, col)
                if it is None:
                    continue
                
                flags = it.flags()
    
                if enabled:
                    flags |= Qt.ItemIsEnabled
    
                    if col in (11, 12, 13):
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

        btn_add.clicked.connect(self.add_lambda)
        btn_del.clicked.connect(self.remove_selected)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Pair:"))
        top.addWidget(self.combo_pair, 1)
        top.addWidget(btn_add)
        top.addWidget(btn_del)

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