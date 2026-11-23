"""
Single running copy of HKL-PDF and the "keep loaded in the background" mode.

* The first HKL-PDF process listens on a local socket (per user). Starting
  HKL-PDF again — shortcut, double-clicked project — does not load the
  libraries again: the new process hands its request (file to open, classic
  window) to the running copy and exits; the running copy opens the window
  in about a second.
* With "Keep HKL-PDF loaded after closing" (on by default) closing the last
  window leaves the process running with a tray icon until "Quit HKL-PDF" or
  the computer restarts. Turned off, closing the last window quits.
* "Start HKL-PDF at Windows login" (off by default) loads the libraries in
  the background at login (``--background``), so even the first window after
  a restart opens quickly.
"""

from __future__ import annotations

import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtNetwork import QLocalServer, QLocalSocket

ORG = APP = "HKL-PDF"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def server_name() -> str:
    user = "".join(ch for ch in getpass.getuser() if ch.isalnum()) or "user"
    return f"hkl-pdf-{user}"


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------


def _settings() -> QtCore.QSettings:
    return QtCore.QSettings(ORG, APP)


def keep_loaded() -> bool:
    return _settings().value("keep_loaded_after_close", True, type=bool)


def set_keep_loaded(on: bool) -> None:
    _settings().setValue("keep_loaded_after_close", bool(on))


def login_command() -> str:
    """Command line that starts HKL-PDF in the background."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" --background'
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    exe = pythonw if pythonw.exists() else exe
    root = str(Path(__file__).resolve().parents[2])
    code = (
        f"import sys; sys.path.insert(0, r'{root}'); "
        "sys.argv=['hkl-pdf','--background']; "
        "from pdf_fitting.gui.main import main; main()"
    )
    return f'"{exe}" -c "{code}"'


def start_at_login() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            winreg.QueryValueEx(key, APP)
        return True
    except OSError:
        return False


def set_start_at_login(on: bool) -> None:
    if sys.platform != "win32":
        return
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        if on:
            winreg.SetValueEx(key, APP, 0, winreg.REG_SZ, login_command())
        else:
            try:
                winreg.DeleteValue(key, APP)
            except OSError:
                pass


# -----------------------------------------------------------------------------
# Client side: hand the request to a running copy
# -----------------------------------------------------------------------------


def send_to_running(request: Dict[str, Any], timeout_ms: int = 400) -> bool:
    """True when a running HKL-PDF accepted the request (this process may exit)."""
    sock = QLocalSocket()
    sock.connectToServer(server_name())
    if not sock.waitForConnected(timeout_ms):
        return False
    if sys.platform == "win32":
        # Let the running copy bring its window to the front.
        try:
            import ctypes

            ctypes.windll.user32.AllowSetForegroundWindow(-1)
        except Exception:  # noqa: BLE001
            pass
    sock.write((json.dumps(request) + "\n").encode("utf-8"))
    sock.flush()
    sock.waitForBytesWritten(1000)
    ok = sock.waitForReadyRead(5000) and bytes(sock.readAll()).strip() == b"ok"
    sock.disconnectFromServer()
    return ok


# -----------------------------------------------------------------------------
# Server side
# -----------------------------------------------------------------------------


class AppController(QtCore.QObject):
    """Owns the windows of this process, the local server and the tray icon."""

    def __init__(self, app: QtWidgets.QApplication):
        super().__init__(app)
        self.app = app
        app.setQuitOnLastWindowClosed(False)
        self.windows: List[QtWidgets.QWidget] = []
        self._ready = False
        self._queue: List[Dict[str, Any]] = []
        self._told_background = False

        self.server = QLocalServer(self)
        self.server.setSocketOptions(QLocalServer.UserAccessOption)
        QLocalServer.removeServer(server_name())  # stale socket of a crashed copy
        self.server.newConnection.connect(self._on_connection)
        if not self.server.listen(server_name()):
            print(f"[HKL-PDF] single-instance server unavailable: {self.server.errorString()}")

        self.tray: Optional[QtWidgets.QSystemTrayIcon] = None
        if QtWidgets.QSystemTrayIcon.isSystemTrayAvailable():
            self._build_tray()

    # ---- tray -------------------------------------------------------------

    def _build_tray(self):
        from .splash import app_icon

        self.tray = QtWidgets.QSystemTrayIcon(app_icon(), self)
        self.tray.setToolTip("HKL-PDF (loading…)")
        menu = QtWidgets.QMenu()
        menu.addAction("Open HKL-PDF", lambda: self.request({}))
        menu.addAction("Open classic window", lambda: self.request({"classic": True}))
        menu.addSeparator()
        self.act_keep = menu.addAction("Keep loaded after closing (faster start)")
        self.act_keep.setCheckable(True)
        self.act_keep.setChecked(keep_loaded())
        self.act_keep.toggled.connect(self._toggle_keep)
        self.act_login = menu.addAction("Start at Windows login")
        self.act_login.setCheckable(True)
        self.act_login.setChecked(start_at_login())
        self.act_login.setEnabled(sys.platform == "win32")
        self.act_login.toggled.connect(self._toggle_login)
        menu.addSeparator()
        menu.addAction("Quit HKL-PDF", self.quit_all)
        self._menu = menu
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()

    def sync_menu(self):
        """Refresh the check marks (settings changed from a window menu)."""
        if self.tray is None:
            return
        for act, val in ((self.act_keep, keep_loaded()), (self.act_login, start_at_login())):
            act.blockSignals(True)
            act.setChecked(val)
            act.blockSignals(False)

    def _toggle_keep(self, on: bool):
        set_keep_loaded(on)
        if not on and not self._visible_windows():
            self.app.quit()

    def _toggle_login(self, on: bool):
        try:
            set_start_at_login(on)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(None, "HKL-PDF", f"Could not change the login setting:\n{exc}")
        self.sync_menu()

    def _tray_activated(self, reason):
        if reason in (QtWidgets.QSystemTrayIcon.Trigger, QtWidgets.QSystemTrayIcon.DoubleClick):
            vis = self._visible_windows()
            if vis:
                self._raise(vis[-1])
            else:
                self.request({})

    # ---- requests -----------------------------------------------------------

    def _on_connection(self):
        while self.server.hasPendingConnections():
            sock = self.server.nextPendingConnection()
            sock.readyRead.connect(lambda s=sock: self._on_data(s))

    def _on_data(self, sock: QLocalSocket):
        if not sock.canReadLine():
            return
        try:
            request = json.loads(bytes(sock.readLine()).decode("utf-8") or "{}")
        except ValueError:
            request = {}
        sock.write(b"ok\n")
        sock.flush()
        sock.disconnectFromServer()
        self.request(request if isinstance(request, dict) else {})

    def request(self, request: Dict[str, Any]):
        if not self._ready:
            self._queue.append(request)
            return
        self.open_window(request.get("open"), bool(request.get("classic")))

    def set_ready(self):
        self._ready = True
        if self.tray is not None:
            self.tray.setToolTip("HKL-PDF")
        queue, self._queue = self._queue, []
        for req in queue:
            self.request(req)

    # ---- windows --------------------------------------------------------------

    def open_window(self, path: Optional[str] = None, classic: bool = False):
        if classic:
            from .mainwindow import MainWindow

            win = MainWindow()
        else:
            from .project_window import ProjectWindow

            win = ProjectWindow()
            win.controller = self
        win.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        win.destroyed.connect(self._window_gone)
        self.windows.append(win)
        win.showMaximized()
        self._raise(win)
        if path and not classic:
            QtCore.QTimer.singleShot(0, lambda: win.open_project(os.path.abspath(path)))
        return win

    def _raise(self, win):
        if win.isMinimized():
            win.showMaximized()
        win.raise_()
        win.activateWindow()

    def _visible_windows(self):
        alive = []
        for w in self.windows:
            try:
                if w.isVisible():
                    alive.append(w)
            except RuntimeError:  # already deleted
                pass
        return alive

    def _window_gone(self, obj=None):
        self.windows = [w for w in self.windows if w is not obj]
        QtCore.QTimer.singleShot(0, self._check_last)

    def _check_last(self):
        if self._visible_windows():
            return
        if not keep_loaded() or self.tray is None:  # no tray: nothing to reopen from
            self.app.quit()
            return
        if self.tray is not None and not self._told_background:
            self._told_background = True
            self.tray.showMessage(
                "HKL-PDF is still loaded",
                "It keeps running in the background so it opens instantly next time. "
                "Right-click this icon to quit.",
                QtWidgets.QSystemTrayIcon.Information,
                6000,
            )

    def quit_all(self):
        for w in list(self._visible_windows()):
            w.close()
            if w in self._visible_windows():  # the user cancelled (unsaved work)
                return
        self.app.quit()
