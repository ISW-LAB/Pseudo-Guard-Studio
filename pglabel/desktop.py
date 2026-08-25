#!/usr/bin/env python3
r"""The double-click entry point — what PG-Label.exe and ``python -m pglabel`` run.

``cli.main`` is the researcher's path: a terminal, explicit flags, a checkout. This is the
desktop path, and it has to cope with what a double-click gives you instead: no arguments, no
console to read errors from, no writable install folder, and a user who expects a second
double-click to focus the running app rather than start a second server.

    pglabel                              open the last dataset (or the start screen)
    pglabel --where                      print every resolved path, then exit (support tool)
    pglabel --set-train-python PATH       register the torch environment, then exit
    pglabel --port 8123 --no-browser --images D:\imgs --classes cat,dog

Every ``cli`` flag still works; this only adds the desktop-shaped behaviour on top: settings
defaults, a log file, single-instance focus, and a free-port search.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from . import cli, console, dataset_setup, fsutil, paths, state, training

BANNER = "Pseudo-Guard Studio (PG-Label) — Collaborative Auto-Labeling"
LOG_ROTATE_BYTES = 2_000_000
PORT_SCAN_RANGE = 40


# ------------------------------------------------------------------ console / logging
class _Tee:
    """Write to the console *and* the log file.

    A windowed PyInstaller build has no stdout at all (``sys.stdout is None``) and every
    ``print()`` in the app would raise. Routing output through here means the app behaves the
    same whether it was started from a shortcut, from a terminal, or by the installer.
    """

    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, text):
        if self._stream is not None:
            try:
                self._stream.write(text)
            except Exception:
                pass
        try:
            self._fh.write(text)
            self._fh.flush()
        except Exception:
            pass
        return len(text)

    def flush(self):
        for s in (self._stream, self._fh):
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return bool(self._stream and getattr(self._stream, "isatty", lambda: False)())


def start_logging() -> Path:
    """Tee stdout/stderr into the per-user log file, rotating it at 2 MB."""
    log = paths.log_path()
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        if log.exists() and log.stat().st_size > LOG_ROTATE_BYTES:
            # os.replace, not Path.rename: renaming onto an existing file fails on Windows, and
            # the previous rotation is exactly what is sitting there.
            fsutil.replace_file(log, log.with_suffix(".1.log"))
        fh = open(log, "a", encoding="utf-8", errors="replace", buffering=1)
    except OSError:
        return log                              # unwritable profile: console-only, still runs
    fh.write(f"\n===== {BANNER} — started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    sys.stdout, sys.stderr = _Tee(sys.stdout, fh), _Tee(sys.stderr, fh)
    return log


# ----------------------------------------------------------------------- port handling
def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # No SO_REUSEADDR here: on Windows it lets two sockets bind the SAME port, so the probe
        # would call an occupied port free and the second instance would silently steal traffic.
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def pick_port(host: str, preferred: int) -> int:
    if port_is_free(host, preferred):
        return preferred
    for p in range(preferred + 1, preferred + PORT_SCAN_RANGE):
        if port_is_free(host, p):
            return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:     # last resort: let the OS pick
        s.bind((host, 0))
        return s.getsockname()[1]


def already_running(host: str, port: int) -> bool:
    """True when THIS app is already serving on that port.

    /api/config is the cheap fingerprint: any PG-Label answers it with a ``needs_setup`` key.
    Some unrelated service on port 8000 answers something else (or nothing) and we move on.
    """
    view_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    try:
        with urllib.request.urlopen(f"http://{view_host}:{port}/api/config", timeout=1.5) as r:
            return "needs_setup" in json.loads(r.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return False


# ---------------------------------------------------------------------------- settings
def apply_settings(args, argv: list) -> None:
    """Fill in flags the user did NOT pass from settings.json.

    Anything typed on the command line wins, so a shortcut can override a stored preference
    without having to clear it first.
    """
    st = paths.load_settings()
    given = {a.split("=", 1)[0] for a in argv if a.startswith("--")}

    def unset(flag: str) -> bool:
        return f"--{flag.replace('_', '-')}" not in given

    if unset("port") and st.get("port"):
        args.port = int(st["port"])
    if unset("host") and st.get("host"):
        args.host = str(st["host"])
    if unset("train_device") and st.get("train_device"):
        args.train_device = str(st["train_device"])
    if unset("train_env") and st.get("train_env"):
        args.train_env = str(st["train_env"])
    if unset("train_python") and paths.train_python():
        args.train_python = paths.train_python()
    # Re-open the last dataset as the PREFILLED start screen, not as a silent auto-load: the
    # user still presses Start, so a moved or deleted folder is visible rather than fatal.
    last = st.get("last_dataset") or {}
    if unset("default-images") and args.default_images is None and last.get("images"):
        if Path(last["images"]).is_dir():
            args.default_images = Path(last["images"])
            args.default_labels = Path(last["labels"]) if last.get("labels") else None
            args.default_classes = last.get("classes") or None


def remember_dataset_on_setup() -> None:
    """Persist the dataset the user picks so the next launch prefills it.

    Wraps ``dataset_setup.run_setup`` instead of editing it: the research launcher has no
    settings.json and must keep behaving exactly as before.
    """
    original = dataset_setup.run_setup

    def wrapper(body):
        result = original(body)
        try:
            _payload, code = result
            if code == 200:
                paths.save_settings({"last_dataset": {
                    "images": str(state.CFG["images"]), "labels": str(state.CFG["labels"]),
                    "classes": ",".join(state.CFG["classes"])}})
        except Exception:
            pass                                # persistence is a convenience, never fatal
        return result

    dataset_setup.run_setup = wrapper


# ------------------------------------------------------------------- training pack
def _load_gpu_pack():
    """Import the training-pack installer, which lives in ``packaging/``, not in this package.

    It is deliberately outside ``pglabel``: it is a one-time setup step, not part of the app, and
    keeping it out means the annotator never imports pip/venv machinery. Frozen builds name it as
    a hidden import; a source checkout finds it next to the repository root.
    """
    import importlib
    for cand in (paths.bundle_root() / "packaging", paths.install_root() / "packaging"):
        if (cand / "gpu_pack.py").exists() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
    return importlib.import_module("gpu_pack")


# -------------------------------------------------------------------------------- main
def _print_banner(url: str, log: Path) -> None:
    line = "=" * 70
    print(line, flush=True)
    print(f"  {BANNER}", flush=True)
    print(f"  Open in your browser:   {url}", flush=True)
    print(f"  Your labels are in:     {paths.workspace_dir()}", flush=True)
    print(f"  Log file:               {log}", flush=True)
    print("  Keep this window open while you work — closing it stops the app.", flush=True)
    print(line, flush=True)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    console.enable()            # before ANY print: a redirected stdout is code-page limited
    log = start_logging()

    if "--where" in argv:                       # support tool: what did the app resolve?
        print(json.dumps(paths.describe(), indent=2, ensure_ascii=False))
        return 0
    if "--install-gpu-pack" in argv or "--find-python" in argv:
        return _load_gpu_pack().main(argv, paths)   # only the setup path pays this import
    if "--set-train-python" in argv:            # register a torch environment
        i = argv.index("--set-train-python")
        exe = argv[i + 1] if i + 1 < len(argv) else ""
        if not exe or not Path(exe).exists():
            print(f"[pg-label] not a file: {exe!r}", file=sys.stderr)
            return 2
        paths.save_settings({"train_python": str(Path(exe).resolve())})
        print(f"[pg-label] training interpreter registered: {exe}")
        print(f"[pg-label] saved to {paths.settings_path()}")
        return 0

    parser = cli.build_parser()
    # The three flags above are answered BEFORE argparse, because each one exits instead of
    # starting a server. They are declared here anyway so ``--help`` lists them — a flag the
    # README documents but the program will not admit to having is worse than no flag at all.
    desktop_group = parser.add_argument_group("desktop")
    desktop_group.add_argument("--no-browser", action="store_true",
                               help="do not open a web browser")
    desktop_group.add_argument("--where", action="store_true",
                               help="print every resolved path as JSON, then exit")
    desktop_group.add_argument("--set-train-python", metavar="PATH",
                               help="register an interpreter that has torch, then exit")
    desktop_group.add_argument("--install-gpu-pack", action="store_true",
                               help="install the training pack (torch + ultralytics), then exit")
    desktop_group.add_argument("--find-python", action="store_true",
                               help="list the interpreters the training pack could be built from")
    args = parser.parse_args(argv)
    apply_settings(args, argv)

    if args.images is not None and not Path(args.images).is_dir():
        print(f"[pg-label] --images folder not found: {args.images}", file=sys.stderr)
        return 2

    view_host = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    # A second double-click should focus the running app, not fight it for the port.
    if already_running(args.host, args.port):
        url = f"http://{view_host}:{args.port}"
        print(f"[pg-label] already running — opening {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    args.port = pick_port(args.host, args.port)
    url = f"http://{view_host}:{args.port}"
    remember_dataset_on_setup()
    _print_banner(url, log)
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        cli.serve(args)
    except KeyboardInterrupt:
        pass                                    # Ctrl+C is how a console window says "stop"
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"\n[pg-label] the app stopped with an error. Details are in: {log}", file=sys.stderr)
        if os.name == "nt" and sys.stdin and sys.stdin.isatty():
            input("Press Enter to close…")      # keep the crash readable in a shortcut window
        return 1
    finally:
        # A training run is a separate process holding the GPU. Quitting must take it with us,
        # or Ctrl+C leaves an invisible job occupying the card until the next reboot.
        if state.job_running():
            print("[pg-label] stopping the training job that is still running…")
            training.stop_training()
    print("[pg-label] stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
