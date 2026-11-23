from __future__ import annotations

import os
from pathlib import Path


def configure_runtime() -> None:
    """
    Configure writable cache locations before importing Numba, Matplotlib,
    PySide6, or the HKL-PDF package.
    """
    local_app_data = os.environ.get(
        "LOCALAPPDATA",
        str(Path.home()),
    )

    app_dir = Path(local_app_data) / "HKL-PDF"
    numba_dir = app_dir / "numba_cache"
    matplotlib_dir = app_dir / "matplotlib_cache"

    app_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    numba_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matplotlib_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.environ.setdefault(
        "QT_API",
        "pyside6",
    )

    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        str(numba_dir),
    )

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(matplotlib_dir),
    )


def _run_python_module(argv) -> None:
    """
    ``HKL-PDF.exe -m <module> [args]`` behaves like ``python -m``. Libraries
    that probe in a child process start ``[sys.executable, "-m", ...]``; in
    the installed program sys.executable is HKL-PDF.exe, which would
    otherwise open a second HKL-PDF instead.
    """
    import runpy
    import sys

    # A GUI program may have no stdout even when the parent reads a pipe.
    for fd, name in ((1, "stdout"), (2, "stderr")):
        if getattr(sys, name) is None:
            try:
                setattr(sys, name, open(fd, "w", encoding="utf-8", closefd=False))
            except OSError:
                pass
    from pdf_fitting import gpu_install

    if not gpu_install.cupy_available_natively():
        gpu_install.activate()
    sys.argv = [argv[2]] + list(argv[3:])
    runpy.run_module(argv[2], run_name="__main__", alter_sys=True)


def main() -> None:
    import sys

    configure_runtime()

    if len(sys.argv) > 2 and sys.argv[1] == "-m":
        _run_python_module(sys.argv)
        return

    from pdf_fitting.gui.main import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()