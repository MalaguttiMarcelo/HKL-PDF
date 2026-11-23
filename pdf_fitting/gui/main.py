from __future__ import annotations

import os
import sys

from PySide6 import QtCore
from PySide6 import QtWidgets

from .timing import (
    TimedApplication,
    enable_periodic_traceback_dump,
    install_timing_instrumentation,
)


def _log_to_file_without_console() -> None:
    """
    The installed program has no console (sys.stdout is None). Send the
    messages to %LOCALAPPDATA%\\HKL-PDF\\hkl-pdf.log instead — useful for bug
    reports, and some libraries fail when stdout is missing.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        from pdf_fitting.user_paths import user_data_dir

        path = os.path.join(user_data_dir(), "hkl-pdf.log")
        if os.path.exists(path) and os.path.getsize(path) > 5_000_000:
            os.replace(path, path + ".old")
        log = open(path, "a", encoding="utf-8", errors="replace", buffering=1)
        if sys.stdout is None:
            sys.stdout = log
        if sys.stderr is None:
            sys.stderr = log
    except OSError:
        pass


def _version() -> str:
    try:
        from pdf_fitting import __version__

        return __version__
    except ImportError:
        return ""


def main() -> None:
    """
    hkl-pdf                    project window
    hkl-pdf project.json       ... and open a project / .inp file
    hkl-pdf --classic          previous single-phase window
    hkl-pdf --background       load the libraries and wait in the tray (login start)
    hkl-pdf --warmup           compile the numerical kernels once and exit (installer)
    hkl-pdf --install-gpu      download the optional NVIDIA GPU support (installer)
    hkl-pdf --gpu-check        test the GPU support and exit (exit code 0 = works)

    A second start hands its request to the running copy, which opens the
    window without loading anything again (see resident.py).
    """
    _log_to_file_without_console()
    # Log output redirected to a file / pipe uses the ANSI code page on
    # Windows: replace characters it cannot encode (Δ, σ, …) instead of
    # raising inside a calculation.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = [a for a in sys.argv[1:]]
    classic = "--classic" in args
    background = "--background" in args
    paths = [a for a in args if not a.startswith("-")]

    # Downloaded NVIDIA GPU support (if any) must be on sys.path before
    # anything imports CuPy.
    from pdf_fitting import gpu_install

    if not gpu_install.cupy_available_natively():
        gpu_install.activate()

    if "--warmup" in args:
        from .warmup import run_warmup

        sys.exit(run_warmup())

    if "--gpu-check" in args:
        res = gpu_install.check()
        print(f"[GPU CHECK] {'OK' if res['ok'] else 'FAILED'}: {res['message']}", flush=True)
        sys.exit(0 if res["ok"] else 1)

    app = QtWidgets.QApplication.instance()

    if app is None:
        app = TimedApplication(
            []
        )

    if "--install-gpu" in args:
        from .gpu_dialog import run_gpu_install_dialog
        from .splash import app_icon

        app.setWindowIcon(app_icon())
        run_gpu_install_dialog(autostart=True)
        return

    from . import resident

    if not background and resident.send_to_running(
        {"open": os.path.abspath(paths[0]) if paths else None, "classic": classic}
    ):
        return  # the running copy opens the window

    enable_periodic_traceback_dump()

    from .splash import app_icon, load_with_splash

    app.setApplicationName("HKL-PDF")
    app.setWindowIcon(app_icon())

    # Listen first, so a start during loading is queued, not loaded twice.
    controller = resident.AppController(app)
    app._hklpdf_controller = controller

    # Import every heavy library now (not on first use); with the loading
    # screen, or silently for a login start.
    splash = load_with_splash(
        app, _version(), show=not background, show_when=lambda: bool(controller._queue),
    )

    if classic:
        install_timing_instrumentation()
    controller.set_ready()
    if not background:
        splash.set_message("Opening the main window…")
        controller.open_window(paths[0] if paths else None, classic)
    splash.close()

    app.exec()
