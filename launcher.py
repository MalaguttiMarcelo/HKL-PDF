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


def main() -> None:
    configure_runtime()

    from pdf_fitting.gui.main import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()