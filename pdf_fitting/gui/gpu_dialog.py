"""Window that downloads and tests the optional NVIDIA GPU support."""

from __future__ import annotations

import threading

from PySide6 import QtCore, QtWidgets

from pdf_fitting import gpu_install


class _Bridge(QtCore.QObject):
    progress = QtCore.Signal(str, float, float)
    done = QtCore.Signal(str)      # "" = success, otherwise the error


class GpuInstallDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NVIDIA GPU support — HKL-PDF")
        self.setMinimumWidth(520)
        lay = QtWidgets.QVBoxLayout(self)
        mb = gpu_install.download_size() / 1e6
        self.lbl = QtWidgets.QLabel(
            "HKL-PDF can run large calculations on an NVIDIA graphics card. This downloads "
            f"CuPy and the NVIDIA CUDA libraries it needs (about {mb:.0f} MB) from the Python "
            "Package Index into your user folder. No administrator rights are needed."
        )
        self.lbl.setWordWrap(True)
        lay.addWidget(self.lbl)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        self.buttons = QtWidgets.QDialogButtonBox()
        self.btn_go = self.buttons.addButton("Download and install", QtWidgets.QDialogButtonBox.AcceptRole)
        self.btn_close = self.buttons.addButton(QtWidgets.QDialogButtonBox.Close)
        self.btn_go.clicked.connect(self._start)
        self.btn_close.clicked.connect(self._close)
        lay.addWidget(self.buttons)
        self._stop = threading.Event()
        self._running = False
        self.success = False
        self._bridge = _Bridge()
        self._bridge.progress.connect(self._on_progress, QtCore.Qt.QueuedConnection)
        self._bridge.done.connect(self._on_done, QtCore.Qt.QueuedConnection)
        if gpu_install.installed():
            self.status.setText("GPU support is already installed. Download again to repair it.")
            self.btn_go.setText("Download again")
        if not gpu_install.nvidia_driver_present():
            self.status.setText(
                "<b>No NVIDIA driver was found on this computer.</b> GPU support only works with an "
                "NVIDIA graphics card and its driver. You can still download it."
            )

    def _start(self):
        self._running = True
        self._stop.clear()
        self.btn_go.setEnabled(False)
        self.btn_close.setText("Cancel")
        self.bar.setVisible(True)
        threading.Thread(target=self._work, name="hklpdf-gpu-install", daemon=True).start()

    def _work(self):
        try:
            gpu_install.install(lambda t, d, n: self._bridge.progress.emit(t, float(d), float(n)), self._stop)
            self._bridge.done.emit("")
        except gpu_install.InstallStopped:
            self._bridge.done.emit("Cancelled.")
        except Exception as exc:  # noqa: BLE001
            self._bridge.done.emit(f"{type(exc).__name__}: {exc}")

    def _on_progress(self, text, done, total):
        self.bar.setValue(int(1000 * done / max(total, 1.0)))
        self.status.setText(f"{text} · {done / 1e6:.0f} / {total / 1e6:.0f} MB")

    def _on_done(self, error: str):
        self._running = False
        self.btn_close.setText("Close")
        if error:
            self.status.setText(f"<b>Not installed.</b> {error}")
            self.btn_go.setEnabled(True)
            self.btn_go.setText("Retry")
            return
        gpu_install.activate()
        self.status.setText("Testing the GPU…")
        QtWidgets.QApplication.processEvents()
        res = gpu_install.check()
        self.success = True
        if res["ok"]:
            self.status.setText(
                f"<b>GPU support installed.</b> {res['message']}<br>"
                "Select <b>CUDA</b> in Refinement settings › Backends to use it."
            )
        else:
            self.status.setText(f"<b>Installed, but the GPU test failed:</b> {res['message']}<br>HKL-PDF will use the CPU.")

    def _close(self):
        if self._running:
            self._stop.set()
            self.status.setText("Cancelling…")
            return
        self.accept()

    def closeEvent(self, event):
        if self._running:
            self._stop.set()
            event.ignore()
            return
        super().closeEvent(event)


def run_gpu_install_dialog(parent=None, autostart: bool = False) -> bool:
    """``autostart``: begin the download at once (the installer's checkbox was the consent)."""
    dlg = GpuInstallDialog(parent)
    if autostart:
        QtCore.QTimer.singleShot(200, dlg._start)
    dlg.exec()
    return dlg.success
