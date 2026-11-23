"""
Optional NVIDIA GPU support, downloaded on request instead of shipped with
the installer.

``gpu_packages.json`` lists the exact PyPI wheels (CuPy, the CUDA runtime,
NVRTC and cuBLAS) that match this build, with their SHA-256 checksums. The
installed program downloads them into
``%LOCALAPPDATA%\\HKL-PDF\\gpu\\<python tag>``, checks every file, unpacks
the wheels there and adds the folder to ``sys.path`` at start-up
(``activate``). Nothing needs administrator rights.

When CuPy is already importable (a development environment), nothing is
activated or offered.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

MANIFEST = Path(__file__).with_name("gpu_packages.json")
MARKER = "installed.json"


class InstallStopped(Exception):
    pass


def manifest() -> Dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def gpu_dir() -> Path:
    from pdf_fitting.user_paths import user_data_dir

    return Path(user_data_dir()) / "gpu" / manifest().get("python", "py")


def download_size() -> int:
    try:
        return int(sum(w["size"] for w in manifest()["wheels"]))
    except (OSError, ValueError, KeyError):
        return 0


def installed() -> bool:
    """The downloaded files match this program's manifest."""
    try:
        mark = json.loads((gpu_dir() / MARKER).read_text(encoding="utf-8"))
        want = sorted(w["sha256"] for w in manifest()["wheels"])
        return sorted(mark.get("sha256", [])) == want
    except (OSError, ValueError, KeyError):
        return False


def cupy_available_natively() -> bool:
    """CuPy importable without the downloaded folder (development setup)."""
    folder = str(gpu_dir())
    if folder in sys.path:
        return False
    try:
        return importlib.util.find_spec("cupy") is not None
    except (ImportError, ValueError):
        return False


def nvidia_driver_present() -> bool:
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        ctypes.WinDLL("nvcuda.dll")
        return True
    except OSError:
        return False


def activate() -> bool:
    """Make the downloaded CuPy importable (call before importing cupy)."""
    if not installed():
        return False
    folder = gpu_dir()
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    # CuPy's library finder (cuda.pathfinder) looks for the NVIDIA wheels in
    # site.getsitepackages(); list the GPU folder there too. Otherwise it
    # falls back to probing in a child "python -m ..." process.
    import site

    original = getattr(site, "getsitepackages", None)
    if not getattr(original, "_hklpdf", False):
        def getsitepackages(prefixes=None, _orig=original, _dir=str(folder)):
            try:
                base = list(_orig(prefixes)) if _orig is not None else []
            except Exception:  # noqa: BLE001
                base = []
            return [_dir] + [p for p in base if p != _dir]

        getsitepackages._hklpdf = True
        site.getsitepackages = getsitepackages
    # DLL folders of the NVIDIA wheels (CUDA runtime, NVRTC, cuBLAS).
    if hasattr(os, "add_dll_directory"):
        for bin_dir in folder.glob("nvidia/*/bin"):
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                pass
    os.environ["PATH"] = os.pathsep.join(
        [str(p) for p in folder.glob("nvidia/*/bin")] + [os.environ.get("PATH", "")]
    )
    return True


def install(progress: Optional[Callable[[str, int, int], None]] = None,
            stop: Optional[threading.Event] = None) -> Path:
    """
    Download, check and unpack the wheels. ``progress(text, done, total)``
    reports bytes. Raises on any failure; a previous installation stays
    usable until the new one is complete.
    """
    wheels: List[Dict] = manifest()["wheels"]
    total = sum(int(w["size"]) for w in wheels)
    target = gpu_dir()
    cache = target.parent / "_downloads"
    staging = target.parent / (target.name + ".new")
    cache.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    done = 0
    for w in wheels:
        path = cache / w["filename"]
        if not (path.exists() and _sha256(path) == w["sha256"]):
            _download(w, path, lambda n, base=done: progress and progress(f"Downloading {w['name']} {w['version']}", base + n, total), stop)
            if _sha256(path) != w["sha256"]:
                path.unlink(missing_ok=True)
                raise RuntimeError(f"Checksum mismatch for {w['filename']} — download damaged, please retry.")
        done += int(w["size"])
        if progress:
            progress(f"Unpacking {w['name']}", done, total)
        with zipfile.ZipFile(path) as z:
            z.extractall(staging)
        if stop is not None and stop.is_set():
            raise InstallStopped()

    (staging / MARKER).write_text(
        json.dumps({"sha256": [w["sha256"] for w in wheels], "wheels": [w["filename"] for w in wheels]}, indent=1),
        encoding="utf-8",
    )
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
    shutil.rmtree(cache, ignore_errors=True)
    if progress:
        progress("Installed", total, total)
    return target


def uninstall() -> None:
    shutil.rmtree(gpu_dir().parent, ignore_errors=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(w: Dict, path: Path, report: Callable[[int], None], stop) -> None:
    tmp = path.with_suffix(".part")
    req = urllib.request.Request(w["url"], headers={"User-Agent": "HKL-PDF-gpu-installer"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as fh:
        n = 0
        while True:
            if stop is not None and stop.is_set():
                fh.close()
                tmp.unlink(missing_ok=True)
                raise InstallStopped()
            block = r.read(1 << 20)
            if not block:
                break
            fh.write(block)
            n += len(block)
            report(n)
    os.replace(tmp, path)


def check() -> Dict[str, object]:
    """
    Run a small GPU calculation: device query, a compiled kernel (NVRTC) and
    a matrix product (cuBLAS). Returns {"ok": bool, "message": str, ...}.
    """
    try:
        import cupy as cp
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": f"CuPy cannot be loaded: {exc}"}
    try:
        n = cp.cuda.runtime.getDeviceCount()
        if n < 1:
            return {"ok": False, "message": "No NVIDIA GPU found."}
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        kernel = cp.ElementwiseKernel("float64 x", "float64 y", "y = x * x + 1.0", "hklpdf_check")
        a = cp.arange(1000, dtype=cp.float64)
        s1 = float(kernel(a).sum())
        m = cp.ones((64, 64), dtype=cp.float64)
        s2 = float((m @ m).sum())
        ok = abs(s1 - (332833500.0 + 1000.0)) < 1e-3 and abs(s2 - 64.0 ** 3) < 1e-6
        mem = props.get("totalGlobalMem", 0) / 2**30
        return {"ok": ok, "message": f"{name} ({mem:.1f} GB): CUDA kernels and cuBLAS work." if ok else "GPU results are wrong.", "device": name}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": f"GPU test failed: {exc}"}
