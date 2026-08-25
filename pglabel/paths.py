#!/usr/bin/env python3
r"""Where PG-Label reads its code and assets from, and where it writes the user's data.

One module, so the app behaves identically in the four ways it gets started:

  1. source checkout   ``python run_app.py …``                      (development / research)
  2. installed package ``python -m pglabel …``                      (pip install)
  3. frozen build      ``dist\PG-Label\PG-Label.exe``               (PyInstaller onedir)
  4. installed app     ``%LOCALAPPDATA%\Programs\PG-Label\PG-Label.exe``  (Inno Setup)

The difference that matters: in a checkout everything (code, demo data, workspace) lives under
one writable project folder. Once packaged, the payload sits in a read-only install folder that
a standard user must not write to, so labels/logs/settings move to a per-user data folder while
assets keep coming from the bundle.

Every configurable root can be overridden — environment variable wins over settings.json wins
over the built-in default — so a lab machine can point one install at a shared dataset drive:

    PGLABEL_DATA_DIR       user data root (workspace, settings, logs)
    PGLABEL_WORKSPACE_DIR  label workspace only
    PGLABEL_DATASETS_DIR   folder scanned for the start-screen dataset presets
    PGLABEL_RESEARCH_DIR   the algorithm library the Train button imports (``pseudoguard``)
    PGLABEL_TRAIN_PYTHON   interpreter that has torch + ultralytics (the "training pack")
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

FROZEN = bool(getattr(sys, "frozen", False))
APP_NAME = "PG-Label"
APP_PUBLISHER = "Pseudo-Guard Studio"
# Single source of truth for the version: the .exe resource, the installer and the About line
# all read it from here (build.py generates version_info.txt from this string).
APP_VERSION = "1.0.0"

_SETTINGS_CACHE: Optional[dict] = None


# --------------------------------------------------------------------------- roots
def bundle_root() -> Path:
    r"""Read-only payload root: holds pglabel/, pgcount/, pseudoguard/, tools/, demo/.

    Frozen: PyInstaller's extraction dir (``sys._MEIPASS``) — ``_internal\`` for a onedir build.
    Source: the repository root (the folder that contains the ``pglabel`` package).
    """
    if FROZEN:
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)).resolve()
    return Path(__file__).resolve().parent.parent


def install_root() -> Path:
    r"""Folder that holds the executable (frozen); the repository root in a checkout.

    Used for things a user may drop NEXT TO the app rather than inside the sealed bundle — a
    ``datasets\`` folder placed beside PG-Label.exe is picked up automatically.
    """
    if FROZEN:
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def packaged() -> bool:
    """True when this is a DISTRIBUTED copy of the app rather than a source checkout.

    Two shapes qualify: a PyInstaller build (``sys.frozen``), and the portable build, which is
    ordinary .py files run by a bundled interpreter and so cannot be detected that way — the
    release assembler drops a ``.pglabel-packaged`` marker in its root for exactly this reason.

    What hangs on it: a distributed copy must never write the user's labels, settings or logs
    into its own folder. That folder gets replaced on upgrade and deleted on uninstall.
    """
    return FROZEN or (bundle_root() / ".pglabel-packaged").exists()


def data_root() -> Path:
    """Per-user writable root: settings, logs and (when packaged) the label workspace.

    A source checkout keeps writing inside the project folder, so a researcher's ``workspace/``
    stays exactly where the scripts expect it.
    """
    env = os.environ.get("PGLABEL_DATA_DIR")
    if env:
        return _ensure(Path(env).expanduser())
    if not packaged():
        return bundle_root()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return _ensure(Path(base) / APP_NAME)
    return _ensure(Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "pg-label")


def _ensure(p: Path) -> Path:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return p


# ----------------------------------------------------------------------- settings
def settings_path() -> Path:
    return data_root() / "settings.json"


def load_settings(refresh: bool = False) -> dict:
    """Read settings.json (never raises — a corrupt file must not stop the app from starting)."""
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is not None and not refresh:
        return _SETTINGS_CACHE
    data = {}
    p = settings_path()
    if p.exists():
        try:
            loaded = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}
    _SETTINGS_CACHE = data
    return data


def save_settings(values: dict) -> None:
    """Merge ``values`` into settings.json (best effort — a read-only profile must not crash us)."""
    global _SETTINGS_CACHE
    data = dict(load_settings())
    data.update(values)
    try:
        settings_path().write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        _SETTINGS_CACHE = data
    except OSError:
        pass


def setting(key: str, default=None):
    return load_settings().get(key, default)


def _configured_dir(env_var: str, settings_key: str) -> Optional[Path]:
    """A root the user pinned explicitly, if it still exists.

    A stale pin falls through to the auto-detected default instead of leaving the feature dead.
    """
    for raw in (os.environ.get(env_var), setting(settings_key)):
        if raw:
            p = Path(str(raw)).expanduser()
            if p.is_dir():
                return p.resolve()
    return None


# ------------------------------------------------------------------- app payload
def package_dir() -> Path:
    """The ``pglabel`` package itself (where static/ lives)."""
    return bundle_root() / "pglabel"


def static_dir() -> Path:
    """The single-page UI. ``pglabel/static`` normally; a bare ``static/`` is tolerated so a
    stripped-down bundle still finds index.html."""
    d = package_dir() / "static"
    return d if d.is_dir() else bundle_root() / "static"


def tools_dir() -> Path:
    """``tools/`` — the training entry points (train_and_predict.py, gen_noise_crops.py).

    These are NOT imported by the app; they are handed as file paths to a SEPARATE
    torch-enabled interpreter, so they must stay readable .py files and are never frozen.
    """
    return bundle_root() / "tools"


def demo_dir() -> Path:
    """Small bundled sample dataset (``demo/images`` + ``demo/gt_labels``); may be absent."""
    return bundle_root() / "demo"


# ------------------------------------------------------------ user-facing folders
def workspace_dir() -> Path:
    """Where the app writes labels when the user has not chosen a labels folder."""
    p = _configured_dir("PGLABEL_WORKSPACE_DIR", "workspace_dir")
    if p is not None:
        return p
    raw = os.environ.get("PGLABEL_WORKSPACE_DIR") or setting("workspace_dir")
    if raw:                                        # pinned but missing → create it, don't ignore it
        return _ensure(Path(str(raw)).expanduser())
    return _ensure(data_root() / "workspace")


def datasets_root() -> Optional[Path]:
    r"""Folder scanned for the start-screen dataset presets.

    A preset is any ``<name>/images/train`` + ``<name>/labels/train`` + a ``*.yaml`` naming the
    classes. Search order: env / settings pin → ``datasets\`` next to the app → ``datasets/`` in
    the user's data folder. None when nothing is found, in which case the start screen just
    shows the manual folder picker.
    """
    p = _configured_dir("PGLABEL_DATASETS_DIR", "datasets_dir")
    if p is not None:
        return p
    for cand in (install_root() / "datasets", data_root() / "datasets"):
        if cand.is_dir():
            return cand.resolve()
    return None


def research_root() -> Optional[Path]:
    """Root that must be importable by the TRAINING interpreter for ``import pseudoguard``.

    In this repository the algorithm library ships in-tree, so this is normally the bundle root
    itself. The env/settings override stays supported for the one case it exists to serve:
    pointing a packaged install at a separate research checkout.
    """
    p = _configured_dir("PGLABEL_RESEARCH_DIR", "research_dir")
    if p is not None:
        return p
    for cand in (bundle_root(), install_root()):
        if (cand / "pseudoguard" / "__init__.py").exists():
            return cand.resolve()
    return None


def train_python() -> Optional[str]:
    """Interpreter of the environment that has torch + ultralytics, if one is registered.

    Set by the GPU-pack installer, or by hand in settings.json. Falls back to the sidecar venv
    the installer creates, so reinstalling the app keeps training working without re-registering.
    """
    for raw in (os.environ.get("PGLABEL_TRAIN_PYTHON"), setting("train_python")):
        if raw and Path(str(raw)).exists():
            return str(Path(str(raw)).resolve())
    exe = "python.exe" if os.name == "nt" else "python"
    sub = "Scripts" if os.name == "nt" else "bin"
    for cand in (data_root() / "gpu-env" / sub / exe, install_root() / "gpu-env" / sub / exe):
        if cand.exists():
            return str(cand.resolve())
    return None


def log_path() -> Path:
    """Launcher log file.

    The caller creates the folder — asking where the log lives must not have the side effect of
    making directories, because ``--where`` is a read-only question.
    """
    return data_root() / "logs" / "pglabel.log"


def describe() -> dict:
    """One-glance diagnostic (printed by ``--where`` and shown in the About panel)."""
    return {"version": APP_VERSION, "python": sys.version.split()[0],
            "frozen": FROZEN, "packaged": packaged(),
            "bundle_root": str(bundle_root()), "install_root": str(install_root()),
            "data_root": str(data_root()), "settings": str(settings_path()),
            "static": str(static_dir()), "tools": str(tools_dir()),
            "workspace": str(workspace_dir()),
            "datasets_root": str(datasets_root() or ""),
            "research_root": str(research_root() or ""),
            "train_python": train_python() or "", "log": str(log_path())}


if __name__ == "__main__":
    print(json.dumps(describe(), indent=2, ensure_ascii=False))
