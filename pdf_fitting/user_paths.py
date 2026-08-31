from __future__ import annotations

import os
from pathlib import Path


def user_data_dir() -> str:
    """
    Return the writable per-user HKL-PDF data directory.
    """
    base = os.environ.get(
        "LOCALAPPDATA",
        str(Path.home()),
    )

    path = Path(base) / "HKL-PDF"

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return str(path)


def user_cache_dir(name: str = "cache") -> str:
    """
    Return a writable per-user cache subdirectory.
    """
    path = Path(user_data_dir()) / str(name)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return str(path)