from __future__ import annotations

import contextlib
import faulthandler
import functools
import inspect
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from PySide6 import QtCore
from PySide6 import QtWidgets


TIMING_ENABLED = (
    os.environ.get(
        "PDF_FITTING_TIMING",
        "1",
    ).strip().lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)

TIMING_PRINT_START = (
    os.environ.get(
        "PDF_FITTING_TIMING_PRINT_START",
        "1",
    ).strip().lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)

TIMING_MIN_SECONDS = float(
    os.environ.get(
        "PDF_FITTING_TIMING_MIN_SECONDS",
        "0.001",
    )
)

SLOW_QT_EVENT_SECONDS = float(
    os.environ.get(
        "PDF_FITTING_SLOW_QT_EVENT_SECONDS",
        "0.050",
    )
)

EVENT_LOOP_INTERVAL_MS = int(
    os.environ.get(
        "PDF_FITTING_EVENT_LOOP_INTERVAL_MS",
        "250",
    )
)

EVENT_LOOP_WARNING_SECONDS = float(
    os.environ.get(
        "PDF_FITTING_EVENT_LOOP_WARNING_SECONDS",
        "0.100",
    )
)

TRACEBACK_DELAY_SECONDS = float(
    os.environ.get(
        "PDF_FITTING_TRACEBACK_DELAY_SECONDS",
        "10.0",
    )
)

TIMING_LOG_PATH = Path(
    os.environ.get(
        "PDF_FITTING_TIMING_LOG",
        str(
            Path.cwd()
            / "gui_timing.jsonl"
        ),
    )
)


_thread_state = threading.local()
_log_lock = threading.RLock()


def _thread_depth() -> int:
    return int(
        getattr(
            _thread_state,
            "depth",
            0,
        )
    )


def _set_thread_depth(
    value: int,
) -> None:
    _thread_state.depth = max(
        0,
        int(value),
    )


def _safe_summary(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        text = str(
            value
        )

        if len(text) > 300:
            return (
                text[:300]
                + "..."
            )

        return value

    try:
        if hasattr(
            value,
            "shape",
        ):
            return {
                "type": type(value).__name__,
                "shape": tuple(
                    value.shape
                ),
                "dtype": str(
                    getattr(
                        value,
                        "dtype",
                        "",
                    )
                ),
            }
    except Exception:
        pass

    if isinstance(
        value,
        dict,
    ):
        return {
            "type": "dict",
            "length": len(value),
            "keys": [
                str(key)
                for key in list(
                    value.keys()
                )[:20]
            ],
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return {
            "type": type(value).__name__,
            "length": len(value),
        }

    return {
        "type": type(value).__name__,
        "text": str(value)[:300],
    }


def _object_details(
    instance: Any,
) -> dict:
    details = {
        "object_type": type(
            instance
        ).__name__,
    }

    try:
        if hasattr(
            instance,
            "rowCount",
        ):
            details["row_count"] = int(
                instance.rowCount()
            )
    except Exception:
        pass

    try:
        if hasattr(
            instance,
            "columnCount",
        ):
            details["column_count"] = int(
                instance.columnCount()
            )
    except Exception:
        pass

    try:
        if hasattr(
            instance,
            "structure",
        ):
            structure = getattr(
                instance,
                "structure",
                None,
            )

            if structure is not None:
                details["structure_sites"] = len(
                    structure.sites
                )
    except Exception:
        pass

    try:
        if hasattr(
            instance,
            "structure_handler",
        ):
            handler = getattr(
                instance,
                "structure_handler",
                None,
            )

            if handler is not None:
                structure = getattr(
                    handler,
                    "structure",
                    None,
                )

                if structure is not None:
                    details["structure_sites"] = len(
                        structure.sites
                    )
    except Exception:
        pass

    try:
        if hasattr(
            instance,
            "_shell_frac",
        ):
            shell_fractional = getattr(
                instance,
                "_shell_frac",
                None,
            )

            if shell_fractional is not None:
                details["shell_count"] = int(
                    shell_fractional.shape[0]
                )
    except Exception:
        pass

    return details


def _call_details(
    function,
    args,
    kwargs,
) -> dict:
    details = {}

    if args:
        details.update(
            _object_details(
                args[0]
            )
        )

    try:
        signature = inspect.signature(
            function
        )

        bound = signature.bind_partial(
            *args,
            **kwargs,
        )

        for parameter_name, value in bound.arguments.items():
            if parameter_name in {
                "self",
                "parent",
            }:
                continue

            if parameter_name in {
                "initial",
                "refinable",
                "bounds",
                "sites",
                "keys",
                "pairs",
                "params",
                "config",
                "cfg",
                "shell_table",
                "unique_site_indices",
            }:
                details[
                    parameter_name
                ] = _safe_summary(
                    value
                )

            elif parameter_name in {
                "path",
                "filename",
                "cif_path",
                "structure_file",
                "pair_cutoff",
                "r_max",
                "overwrite_lattice",
                "commit_edits",
            }:
                details[
                    parameter_name
                ] = _safe_summary(
                    value
                )

    except Exception:
        pass

    return details

@contextlib.contextmanager
def timed_block(
    name: str,
    **details,
):
    """
    Time a block of code.

    Example
    -------
    with timed_block(
        "Read structure metadata",
        path=path,
    ):
        metadata = read_structure_metadata(path)
    """
    if not TIMING_ENABLED:
        yield
        return

    old_depth = _thread_depth()

    thread = threading.current_thread()

    marker = (
        "GUI"
        if thread is threading.main_thread()
        else "WORKER"
    )

    indent = (
        "  "
        * old_depth
    )

    if TIMING_PRINT_START:
        print(
            f"[START {marker}] "
            f"{indent}"
            f"{name} "
            f"thread={thread.name} "
            f"details={details}",
            flush=True,
        )

    _set_thread_depth(
        old_depth + 1
    )

    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    status = "ok"

    try:
        yield

    except Exception:
        status = "error"

        details = dict(
            details
        )

        details[
            "traceback"
        ] = traceback.format_exc()

        raise

    finally:
        wall_elapsed = (
            time.perf_counter()
            - wall_start
        )

        cpu_elapsed = (
            time.process_time()
            - cpu_start
        )

        _set_thread_depth(
            old_depth
        )

        details = dict(
            details
        )

        details[
            "cpu_seconds"
        ] = float(
            cpu_elapsed
        )

        log_timing(
            name,
            wall_elapsed,
            status=status,
            details=details,
        )


def timed(
    name: str | None = None,
    *,
    details_factory=None,
):
    """
    Decorator for timing a function.

    Example
    -------
    @timed("Builder: update structure")
    def update_structure(self):
        ...
    """
    def decorator(
        function,
    ):
        display_name = (
            str(name)
            if name
            else (
                f"{function.__module__}."
                f"{function.__qualname__}"
            )
        )

        if getattr(
            function,
            "_pdf_fitting_timing_wrapped",
            False,
        ):
            return function

        @functools.wraps(
            function
        )
        def wrapped(
            *args,
            **kwargs,
        ):
            details = _call_details(
                function,
                args,
                kwargs,
            )

            if details_factory is not None:
                try:
                    extra_details = details_factory(
                        *args,
                        **kwargs,
                    )

                    if isinstance(
                        extra_details,
                        dict,
                    ):
                        details.update(
                            extra_details
                        )

                except Exception as exc:
                    details[
                        "details_factory_error"
                    ] = str(
                        exc
                    )

            with timed_block(
                display_name,
                **details,
            ):
                return function(
                    *args,
                    **kwargs,
                )

        wrapped._pdf_fitting_timing_wrapped = True

        return wrapped

    return decorator


def profiled(
    name: str | None = None,
    *,
    details_factory=None,
):
    """
    Compatibility decorator.

    At this debugging stage it records wall-clock and CPU time in the same
    way as @timed. It can later be extended to generate cProfile reports.
    """
    return timed(
        name,
        details_factory=details_factory,
    )


def _write_record(
    record: dict,
) -> None:
    try:
        TIMING_LOG_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with _log_lock:
            with TIMING_LOG_PATH.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    json.dumps(
                        record,
                        default=str,
                    )
                )

                file.write(
                    "\n"
                )

    except Exception as exc:
        print(
            "[TIMING] Could not write timing log: "
            f"{exc}",
            flush=True,
        )


def log_timing(
    name: str,
    elapsed_seconds: float,
    *,
    status: str = "ok",
    details: dict | None = None,
) -> None:
    if not TIMING_ENABLED:
        return

    elapsed_seconds = float(
        elapsed_seconds
    )

    if elapsed_seconds < TIMING_MIN_SECONDS:
        return

    thread = threading.current_thread()

    record = {
        "timestamp": time.time(),
        "name": str(
            name
        ),
        "elapsed_seconds": elapsed_seconds,
        "status": str(
            status
        ),
        "thread_name": str(
            thread.name
        ),
        "thread_id": int(
            threading.get_ident()
        ),
        "main_thread": bool(
            thread
            is threading.main_thread()
        ),
        "depth": int(
            _thread_depth()
        ),
        "details": details
        or {},
    }

    marker = (
        "GUI"
        if record[
            "main_thread"
        ]
        else "WORKER"
    )

    indent = (
        "  "
        * int(
            record[
                "depth"
            ]
        )
    )

    print(
        f"[TIMING {marker}] "
        f"{indent}"
        f"{record['name']}: "
        f"{elapsed_seconds:.6f} s "
        f"status={status} "
        f"thread={thread.name}",
        flush=True,
    )

    _write_record(
        record
    )


def _wrap_function(
    owner,
    attribute_name: str,
    display_name: str,
) -> None:
    try:
        original = getattr(
            owner,
            attribute_name,
        )
    except Exception:
        return

    if getattr(
        original,
        "_pdf_fitting_timing_wrapped",
        False,
    ):
        return

    @functools.wraps(
        original
    )
    def wrapped(
        *args,
        **kwargs,
    ):
        old_depth = _thread_depth()

        details = _call_details(
            original,
            args,
            kwargs,
        )

        thread = threading.current_thread()

        marker = (
            "GUI"
            if thread
            is threading.main_thread()
            else "WORKER"
        )

        indent = (
            "  "
            * old_depth
        )

        if TIMING_PRINT_START:
            print(
                f"[START {marker}] "
                f"{indent}"
                f"{display_name} "
                f"thread={thread.name} "
                f"details={details}",
                flush=True,
            )

        _set_thread_depth(
            old_depth + 1
        )

        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        status = "ok"

        try:
            return original(
                *args,
                **kwargs,
            )

        except Exception:
            status = "error"

            details[
                "traceback"
            ] = traceback.format_exc()

            raise

        finally:
            wall_elapsed = (
                time.perf_counter()
                - wall_start
            )

            cpu_elapsed = (
                time.process_time()
                - cpu_start
            )

            _set_thread_depth(
                old_depth
            )

            details[
                "cpu_seconds"
            ] = float(
                cpu_elapsed
            )

            log_timing(
                display_name,
                wall_elapsed,
                status=status,
                details=details,
            )

    wrapped._pdf_fitting_timing_wrapped = True

    setattr(
        owner,
        attribute_name,
        wrapped,
    )


def _patch_module_function(
    module,
    function_name: str,
    display_name: str,
) -> None:
    _wrap_function(
        module,
        function_name,
        display_name,
    )


def install_timing_instrumentation() -> None:
    if not TIMING_ENABLED:
        return

    print(
        "[TIMING] Installing timing instrumentation.",
        flush=True,
    )

    print(
        "[TIMING] Log file: "
        f"{TIMING_LOG_PATH}",
        flush=True,
    )

    from pdf_fitting.gui.builder import InputBuilder
    from pdf_fitting.gui.mainwindow import MainWindow

    from pdf_fitting.gui.param_widgets import (
        DynParamTableWidget,
        LambdaListWidget,
        ParamTableWidget,
        SiteParamTableWidget,
    )

    from pdf_fitting.gui.model_prepare_worker import (
        ModelPrepareWorker,
    )

    from pdf_fitting.gui.cs_grid_worker import (
        CSGridSearchWorker,
    )

    from pdf_fitting.gui.worker import (
        FitWorker,
    )

    from pdf_fitting.structure_handler import (
        StructureHandler,
    )

    from pdf_fitting.models.gr_model import (
        PDFCalculator,
    )

    from pdf_fitting.fit_engine import (
        FitEngine,
    )

    from pdf_fitting.gui import (
        structure_metadata,
    )

    from pdf_fitting.gui import (
        cif_utils,
    )

    builder_methods = {
        "_on_structure_path_changed": (
            "Builder: structure path changed"
        ),
        "_update_symmetry_from_structure": (
            "Builder: update structure metadata"
        ),
        "_apply_params_to_tables": (
            "Builder: apply parameters to tables"
        ),
        "_apply_params_to_tables_from_current": (
            "Builder: collect and rebuild tables"
        ),
        "_set_struct_from_structure": (
            "Builder: set structure lattice"
        ),
        "_update_shape_summary_label": (
            "Builder: update shape summary"
        ),
        "_current_structure_for_shape_dialog": (
            "Builder: load structure for shape dialog"
        ),
        "set_from_config": (
            "Builder: load configuration into GUI"
        ),
        "to_updates": (
            "Builder: serialize GUI parameters"
        ),
    }

    for method_name, display_name in builder_methods.items():
        _wrap_function(
            InputBuilder,
            method_name,
            display_name,
        )

    main_window_methods = {
        "_on_builder_changed": (
            "MainWindow: Builder changed"
        ),
        "_on_raw_changed": (
            "MainWindow: raw input changed"
        ),
        "_apply_builder_into_raw": (
            "MainWindow: apply Builder to raw input"
        ),
        "_compose_input_text_for_run": (
            "MainWindow: compose input text"
        ),
        "_parse_raw_to_dict": (
            "MainWindow: parse raw input"
        ),
        "_refresh_structure_view": (
            "MainWindow: refresh structure view"
        ),
        "_start_background_model_preparation": (
            "MainWindow: start background model preparation"
        ),
        "_on_model_prepared": (
            "MainWindow: receive prepared model"
        ),
        "load_input": (
            "MainWindow: load input file"
        ),
        "calculate_once": (
            "MainWindow: calculate once"
        ),
        "run_fit": (
            "MainWindow: start refinement"
        ),
        "run_cs_grid_search": (
            "MainWindow: start finite-shape run"
        ),
        "open_structure_studio": (
            "MainWindow: open Structure Studio"
        ),
        "_current_structure_view_configuration": (
            "MainWindow: compose Structure Studio configuration"
        ),
    }

    for method_name, display_name in main_window_methods.items():
        _wrap_function(
            MainWindow,
            method_name,
            display_name,
        )

    parameter_table_methods = {
        "set_params": (
            "ParamTableWidget: set parameters"
        ),
        "extract": (
            "ParamTableWidget: extract parameters"
        ),
    }

    for method_name, display_name in parameter_table_methods.items():
        _wrap_function(
            ParamTableWidget,
            method_name,
            display_name,
        )

    dynamic_table_methods = {
        "set_params": (
            "DynParamTableWidget: set parameters"
        ),
        "extract_all": (
            "DynParamTableWidget: extract parameters"
        ),
    }

    for method_name, display_name in dynamic_table_methods.items():
        _wrap_function(
            DynParamTableWidget,
            method_name,
            display_name,
        )

    site_table_methods = {
        "set_sites": (
            "SiteParamTableWidget: set atomic sites"
        ),
        "extract_all": (
            "SiteParamTableWidget: extract atomic sites"
        ),
        "set_biso_controls_enabled": (
            "SiteParamTableWidget: update Biso controls"
        ),
    }

    for method_name, display_name in site_table_methods.items():
        _wrap_function(
            SiteParamTableWidget,
            method_name,
            display_name,
        )

    lambda_methods = {
        "set_pairs": (
            "LambdaListWidget: set pairs"
        ),
        "set_lambdas": (
            "LambdaListWidget: set lambdas"
        ),
        "extract_all": (
            "LambdaListWidget: extract lambdas"
        ),
        "_rebuild_table_for_current_pair": (
            "LambdaListWidget: rebuild table"
        ),
    }

    for method_name, display_name in lambda_methods.items():
        _wrap_function(
            LambdaListWidget,
            method_name,
            display_name,
        )

    structure_handler_methods = {
        "__init__": (
            "StructureHandler: complete initialization"
        ),
        "_generate_shells_fast_cached": (
            "StructureHandler: load or build shell cache"
        ),
        "_generate_shells_fast": (
            "StructureHandler: generate shells"
        ),
        "_get_neighbor_list_for_unique_sites_parallel": (
            "StructureHandler: neighbor list"
        ),
        "_get_neighbor_rows_by_images_numba": (
            "StructureHandler: image-based neighbor rows"
        ),
    }

    for method_name, display_name in structure_handler_methods.items():
        _wrap_function(
            StructureHandler,
            method_name,
            display_name,
        )

    pdf_methods = {
        "__init__": (
            "PDFCalculator: complete initialization"
        ),
        "_load_shell_table_arrays": (
            "PDFCalculator: load shell arrays"
        ),
        "_prepare_shell_runtime_arrays": (
            "PDFCalculator: prepare runtime arrays"
        ),
        "replace_shell_table": (
            "PDFCalculator: replace shell table"
        ),
        "build_pdf_tick_data": (
            "PDFCalculator: build PDF ticks"
        ),
        "build_local_trends": (
            "PDFCalculator: build local trends"
        ),
        "build_shape_factor_directions": (
            "PDFCalculator: build shape factors"
        ),
    }

    for method_name, display_name in pdf_methods.items():
        _wrap_function(
            PDFCalculator,
            method_name,
            display_name,
        )

    fit_engine_methods = {
        "__init__": (
            "FitEngine: initialization"
        ),
        "run_refinement": (
            "FitEngine: complete refinement"
        ),
    }

    for method_name, display_name in fit_engine_methods.items():
        _wrap_function(
            FitEngine,
            method_name,
            display_name,
        )

    worker_methods = {
        ModelPrepareWorker: (
            "ModelPrepareWorker: complete run"
        ),
        FitWorker: (
            "FitWorker: complete run"
        ),
        CSGridSearchWorker: (
            "CSGridSearchWorker: complete run"
        ),
    }

    for worker_class, display_name in worker_methods.items():
        _wrap_function(
            worker_class,
            "run",
            display_name,
        )

    metadata_functions = {
        "read_structure_metadata": (
            "Structure metadata: complete read"
        ),
        "read_text": (
            "Structure metadata: read CIF text"
        ),
        "parse_atom_site_loop": (
            "Structure metadata: parse atom-site loop"
        ),
    }

    for function_name, display_name in metadata_functions.items():
        _patch_module_function(
            structure_metadata,
            function_name,
            display_name,
        )

    cif_functions = {
        "read_cif_asu_sites": (
            "CIF utilities: read CIF sites"
        ),
        "_read_cif_asu_sites_cached": (
            "CIF utilities: parse CIF"
        ),
        "_parse_atom_site_loop_fast": (
            "CIF utilities: fast atom-site parse"
        ),
        "_make_structure_from_fast_cif": (
            "CIF utilities: build pymatgen Structure"
        ),
    }

    for function_name, display_name in cif_functions.items():
        _patch_module_function(
            cif_utils,
            function_name,
            display_name,
        )

    print(
        "[TIMING] Timing instrumentation installed.",
        flush=True,
    )


def instrument_structure_studio_if_loaded() -> None:
    module_name = (
        "pdf_fitting.gui.structure_studio.studio"
    )

    module = sys.modules.get(
        module_name
    )

    if module is None:
        return

    studio_class = getattr(
        module,
        "StructureStudioWindow",
        None,
    )

    if studio_class is None:
        return

    methods = {
        "load_structure": (
            "Structure Studio: load structure"
        ),
        "_rebuild_display_model": (
            "Structure Studio: rebuild display model"
        ),
        "_rebuild_scene": (
            "Structure Studio: rebuild scene"
        ),
        "_add_atoms": (
            "Structure Studio: add atoms"
        ),
        "_add_bonds": (
            "Structure Studio: add bonds"
        ),
    }

    for method_name, display_name in methods.items():
        _wrap_function(
            studio_class,
            method_name,
            display_name,
        )


class TimedApplication(
    QtWidgets.QApplication
):
    def notify(
        self,
        receiver,
        event,
    ):
        if not TIMING_ENABLED:
            return super().notify(
                receiver,
                event,
            )

        start = time.perf_counter()
        status = "ok"

        try:
            return super().notify(
                receiver,
                event,
            )

        except Exception:
            status = "error"
            raise

        finally:
            elapsed = (
                time.perf_counter()
                - start
            )

            if elapsed >= SLOW_QT_EVENT_SECONDS:
                try:
                    receiver_type = type(
                        receiver
                    ).__name__
                except Exception:
                    receiver_type = "unknown"

                try:
                    receiver_name = str(
                        receiver.objectName()
                    )
                except Exception:
                    receiver_name = ""

                try:
                    event_type = (
                        QtCore.QEvent.Type(
                            event.type()
                        ).name
                    )
                except Exception:
                    event_type = str(
                        event.type()
                    )

                log_timing(
                    (
                        "Qt event: "
                        f"{event_type} -> "
                        f"{receiver_type}"
                    ),
                    elapsed,
                    status=status,
                    details={
                        "receiver_type": (
                            receiver_type
                        ),
                        "receiver_name": (
                            receiver_name
                        ),
                        "event_type": (
                            event_type
                        ),
                    },
                )


class EventLoopLagMonitor(
    QtCore.QObject
):
    def __init__(
        self,
        parent=None,
    ):
        super().__init__(
            parent
        )

        self.interval_seconds = (
            float(
                EVENT_LOOP_INTERVAL_MS
            )
            / 1000.0
        )

        self.last_time = time.perf_counter()

        self.timer = QtCore.QTimer(
            self
        )

        self.timer.setInterval(
            EVENT_LOOP_INTERVAL_MS
        )

        self.timer.timeout.connect(
            self._tick
        )

        self.timer.start()

    def _tick(
        self,
    ) -> None:
        current_time = time.perf_counter()

        elapsed = (
            current_time
            - self.last_time
        )

        self.last_time = current_time

        lag = max(
            0.0,
            elapsed
            - self.interval_seconds
        )

        if lag >= EVENT_LOOP_WARNING_SECONDS:
            log_timing(
                "Qt event-loop lag",
                lag,
                status="warning",
                details={
                    "timer_interval_seconds": (
                        self.interval_seconds
                    ),
                },
            )


def enable_periodic_traceback_dump() -> None:
    if TRACEBACK_DELAY_SECONDS <= 0.0:
        return

    try:
        faulthandler.enable()

        faulthandler.dump_traceback_later(
            TRACEBACK_DELAY_SECONDS,
            repeat=True,
        )

        print(
            "[TIMING] Python thread traceback dump enabled every "
            f"{TRACEBACK_DELAY_SECONDS:.1f} seconds.",
            flush=True,
        )

    except Exception as exc:
        print(
            "[TIMING] Could not enable traceback dump: "
            f"{exc}",
            flush=True,
        )