from __future__ import annotations

from typing import Any, Dict


_CUPY = None
_CUPY_IMPORT_ERROR = None


def get_cupy():
    """
    Import CuPy lazily and verify that CUDA array operations and kernel
    compilation work.

    Detecting a CUDA device alone is insufficient. CuPy may see the GPU while
    still being unable to compile kernels because CUDA headers or NVRTC are
    missing.
    """
    global _CUPY
    global _CUPY_IMPORT_ERROR

    if _CUPY is not None:
        return _CUPY

    if _CUPY_IMPORT_ERROR is not None:
        raise RuntimeError(
            "CuPy initialization previously failed in this process."
        ) from _CUPY_IMPORT_ERROR

    try:
        import cupy as cp

        device_count = int(
            cp.cuda.runtime.getDeviceCount()
        )

        if device_count <= 0:
            raise RuntimeError(
                "CuPy was imported, but no CUDA devices were found."
            )

        # ---------------------------------------------------------
        # Compilation probe
        #
        # getDeviceCount() only verifies driver/device access. A real CuPy
        # operation verifies that NVRTC and the required CUDA headers are
        # available as well.
        # ---------------------------------------------------------
        with cp.cuda.Device(0):
            probe = cp.asarray(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ],
                dtype=cp.float32,
            )

            probe_result = float(
                cp.sum(
                    probe
                ).get()
            )

            if abs(
                probe_result
                - 10.0
            ) > 1.0e-5:
                raise RuntimeError(
                    "CuPy CUDA compilation probe returned an invalid result."
                )

            del probe

        _CUPY = cp

        return _CUPY

    except Exception as exc:
        _CUPY_IMPORT_ERROR = exc

        raise RuntimeError(
            "CUDA/CuPy initialization failed. The NVIDIA GPU may be visible, "
            "but CuPy cannot compile CUDA kernels. Install the CUDA 12.x "
            "toolkit components with:\n\n"
            "    python -m pip install --upgrade \"cupy-cuda12x[ctk]\"\n\n"
            "Alternatively, install the NVIDIA CUDA 12.x Toolkit and set "
            "CUDA_PATH to its installation directory."
        ) from exc

def cuda_available() -> bool:
    """
    Return True when CuPy can access at least one CUDA device.
    """
    try:
        get_cupy()
        return True
    except Exception:
        return False


def resolve_backend(
    requested: Any,
    *,
    default: str = "cpu",
) -> str:
    """
    Resolve cpu/cuda/auto into cpu or cuda.
    """
    value = str(
        requested
        if requested is not None
        else default
    ).strip().lower()

    aliases = {
        "gpu": "cuda",
        "cupy": "cuda",
        "nvidia": "cuda",
        "automatic": "auto",
    }

    value = aliases.get(
        value,
        value,
    )

    if value not in (
        "cpu",
        "cuda",
        "auto",
    ):
        value = str(
            default
        ).strip().lower()

    if value == "cpu":
        return "cpu"

    if value == "cuda":
        get_cupy()
        return "cuda"

    return (
        "cuda"
        if cuda_available()
        else "cpu"
    )


def select_device(
    device_id: int = 0,
) -> None:
    cp = get_cupy()

    device_id = int(
        device_id
    )

    device_count = int(
        cp.cuda.runtime.getDeviceCount()
    )

    if (
        device_id < 0
        or device_id >= device_count
    ):
        raise RuntimeError(
            f"Invalid CUDA device index {device_id}. "
            f"Available CUDA devices: 0 to {device_count - 1}."
        )

    cp.cuda.Device(
        device_id
    ).use()

def gpu_memory_info(
    device_id: int = 0,
) -> Dict[str, int]:
    cp = get_cupy()

    with cp.cuda.Device(
        int(device_id)
    ):
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()

    return {
        "free": int(free_bytes),
        "total": int(total_bytes),
    }


def gpu_summary(
    device_id: int = 0,
) -> str:
    cp = get_cupy()

    with cp.cuda.Device(
        int(device_id)
    ):
        properties = cp.cuda.runtime.getDeviceProperties(
            int(device_id)
        )

        name = properties.get(
            "name",
            b"Unknown CUDA device",
        )

        if isinstance(name, bytes):
            name = name.decode(
                "utf-8",
                errors="replace",
            )

        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()

    return (
        f"{name}, device={int(device_id)}, "
        f"free={free_bytes / 1024 ** 3:.2f} GiB, "
        f"total={total_bytes / 1024 ** 3:.2f} GiB"
    )


def clear_gpu_memory_pool() -> None:
    """
    Release unused CuPy memory-pool blocks.
    """
    try:
        cp = get_cupy()

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    except Exception:
        pass