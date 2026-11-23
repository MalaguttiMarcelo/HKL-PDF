"""
Write pdf_fitting/gpu_packages.json: the exact PyPI wheel files (URL, size,
SHA-256) of the NVIDIA GPU support that ``HKL-PDF --install-gpu`` downloads.

The versions are those pinned in installer/requirements-build.txt, so the
downloaded CuPy is the one the program was built and tested with. Run by
installer/build.ps1 (needs internet); commit the result.

    python installer/make_gpu_manifest.py
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "installer" / "requirements-build.txt"
OUT = ROOT / "pdf_fitting" / "gpu_packages.json"

# Only what HKL-PDF uses: CuPy, its library finder, the CUDA
# runtime, the kernel compiler (NVRTC) and cuBLAS (matrix products). The
# other CUDA libraries (cuSPARSE, cuSOLVER, cuFFT, cuRAND, ~1.3 GB) are not
# needed.
PACKAGES = (
    "cupy-cuda12x",
    "cuda-pathfinder",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cublas-cu12",
)

PY_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"


def pinned_versions():
    pins = {}
    for line in REQ.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", line.strip())
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def pick(files, name):
    """The Windows x64 wheel for this Python (or a universal one)."""
    ok = []
    for f in files:
        fn = f["filename"]
        if not fn.endswith(".whl"):
            continue
        if f"-{PY_TAG}-{PY_TAG}-win_amd64.whl" in fn or fn.endswith("-py3-none-win_amd64.whl") or fn.endswith("-py3-none-any.whl") or fn.endswith("-abi3-win_amd64.whl"):
            ok.append(f)
    if not ok:
        raise SystemExit(f"No Windows wheel of {name} for {PY_TAG}.")
    return ok[0]


def main():
    pins = pinned_versions()
    wheels = []
    for name in PACKAGES:
        version = pins.get(name)
        if not version:
            raise SystemExit(f"{name} is not pinned in {REQ.name}.")
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60) as r:
            data = json.load(r)
        f = pick(data["urls"], name)
        wheels.append({
            "name": name,
            "version": version,
            "filename": f["filename"],
            "url": f["url"],
            "size": f["size"],
            "sha256": f["digests"]["sha256"],
        })
        print(f"{f['size'] / 1e6:8.1f} MB  {f['filename']}")
    manifest = {"python": PY_TAG, "wheels": wheels}
    OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"total {sum(w['size'] for w in wheels) / 1e6:.0f} MB -> {OUT}")


if __name__ == "__main__":
    main()
