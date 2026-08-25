# -*- mode: python ; coding: utf-8 -*-
r"""PyInstaller recipe for PG-Label.exe — drive it through ``packaging/build.py``, not by hand.

    pyinstaller --noconfirm packaging/PG-Label.spec

WHAT GOES INSIDE THE EXE
    The annotator: Python + Pillow + the ``pglabel`` package + ``pgcount``'s selection logic.
    That is the whole manual labeling tool, YOLO/COCO I/O, and count-guided Automate Label over
    a precomputed overlay — a 60-90 MB install folder depending on the Python it is frozen with.

WHAT DELIBERATELY STAYS OUT
    torch / torchvision / ultralytics / numpy / cv2. Freezing them would add ~2.5 GB and still
    not give the user a working CUDA stack, because the right wheel depends on their driver. The
    Train button therefore shells out to a SEPARATE interpreter (the training pack), so a machine
    without torch still runs everything else.

WHAT SHIPS AS PLAIN FILES NEXT TO THE EXE (datas, not frozen bytecode)
    pglabel/static/     the UI, read at request time
    tools/*.py          handed to the training interpreter as file paths — must stay .py
    pgcount/*.py        imported by those tools under that other interpreter, which cannot see
    pseudoguard/*.py    the frozen copies inside the exe
    demo/               optional sample dataset (PGLABEL_WITH_DEMO=0 to omit)

BUILD SWITCHES (set by build.py as environment variables)
    PGLABEL_WITH_DEMO=1      bundle demo/images + demo/gt_labels (~2 MB) — first-run sample data
    PGLABEL_WITH_TRAINING=1  bundle tools/ + pseudoguard/, which is what makes Train available
    PGLABEL_CONSOLE=0        windowed build (no console; output goes to the log file only)
"""

import os
from pathlib import Path

# SPECPATH is injected by PyInstaller and points at this file's folder.
PKG = Path(SPECPATH).resolve()                 # noqa: F821  — packaging/
ROOT = PKG.parent                              # repository root

WITH_DEMO = os.environ.get("PGLABEL_WITH_DEMO", "1") not in ("0", "", "false", "False")
WITH_TRAINING = os.environ.get("PGLABEL_WITH_TRAINING", "1") not in ("0", "", "false", "False")
CONSOLE = os.environ.get("PGLABEL_CONSOLE", "1") not in ("0", "", "false", "False")


def tree(src: Path, dest: str, patterns=("*",), skip_dirs=()) -> list:
    """(file, target-folder) pairs for everything under ``src``, keeping the folder shape.

    PyInstaller's own Tree() would sweep in __pycache__ and previous training artefacts; this
    filters them out so the bundle holds only what the app actually reads.
    """
    out = []
    if not src.is_dir():
        return out
    skip = {"__pycache__", ".git", ".pgtrain", "runs", "cache", "build", "dist"} | set(skip_dirs)
    for pat in patterns:
        for f in src.rglob(pat):
            if not f.is_file() or any(part in skip for part in f.relative_to(src).parts[:-1]):
                continue
            if f.name.startswith(".") or f.suffix in (".pyc", ".pyo"):
                continue
            out.append((str(f), str(Path(dest) / f.relative_to(src).parent)))
    return out


def pillow_sidecar_libs() -> list:
    """Pillow's out-of-package native libraries, when the wheel ships them that way.

    manylinux wheels put libjpeg/libtiff/… in a SIBLING ``pillow.libs/`` folder that the stock
    hook does not walk, so a build without this dies at ``import PIL.Image``. Windows wheels keep
    their DLLs inside the package, so this returns nothing there — a portability guard, not a
    platform hack.
    """
    import importlib.util
    spec = importlib.util.find_spec("PIL")
    if not spec or not spec.origin:
        return []
    site = Path(spec.origin).parent.parent
    out = []
    for folder in ("pillow.libs", "PIL.libs", "Pillow.libs"):
        d = site / folder
        if d.is_dir():
            out += [(str(f), folder) for f in d.iterdir() if f.is_file()]
    return out


datas = []
datas += tree(ROOT / "pglabel" / "static", "pglabel/static")
datas += tree(ROOT / "pgcount", "pgcount", patterns=("*.py",))
if WITH_TRAINING:
    # The trainer runs under ANOTHER interpreter, so these must be readable .py files beside the
    # exe. Freezing them would make them invisible to that interpreter.
    datas += tree(ROOT / "tools", "tools", patterns=("*.py",))
    datas += tree(ROOT / "pseudoguard", "pseudoguard", patterns=("*.py",))
if WITH_DEMO:
    # Images only — a .txt sitting beside them would be one past session's output, and shipping
    # it would make the first run look like the user had already labeled everything.
    datas += tree(ROOT / "demo" / "images", "demo/images", patterns=("*.jpg", "*.png"))
    datas += tree(ROOT / "demo" / "gt_labels", "demo/gt_labels", patterns=("*.txt",))

hiddenimports = [
    # The training-pack installer is imported by name only when --install-gpu-pack is passed.
    "gpu_pack",
    # pglabel imports pgcount lazily, inside the functions that need it — the analyser cannot
    # see through that, so name them here or Automate Label breaks only at runtime.
    "pgcount", "pgcount.backend", "pgcount.config", "pgcount.count_guided_labeler",
    "pgcount.crops", "pgcount.metrics", "pgcount.operating_point", "pgcount.seed_density",
    "pgcount.telemetry",
]

excludes = [
    # The heavy ML stack: reached only through the separate training interpreter (see above).
    "torch", "torchvision", "ultralytics", "numpy", "cv2", "scipy", "pandas", "sklearn",
    "matplotlib", "yaml", "tqdm",
    # GUI toolkits Pillow can pull in; this app's only UI is the browser.
    "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6", "wx",
    # Dev-only machinery.
    "pytest", "IPython", "jupyter", "notebook", "setuptools._distutils",
]

a = Analysis(                                   # noqa: F821
    [str(ROOT / "run_app.py")],
    pathex=[str(ROOT), str(PKG)],       # PKG so `import gpu_pack` resolves
    binaries=pillow_sidecar_libs(),
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)                               # noqa: F821

exe = EXE(                                      # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,                      # onedir: faster start, and far fewer antivirus
    name="PG-Label",                            # false positives than a self-extracting onefile
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                                  # UPX-packed exes are a classic AV trigger
    console=CONSOLE,
    disable_windowed_traceback=False,
    icon=str(PKG / "assets" / "pglabel.ico"),
    version=str(PKG / "version_info.txt") if (PKG / "version_info.txt").exists() else None,
)

coll = COLLECT(                                 # noqa: F821
    exe, a.binaries, a.datas,
    strip=False, upx=False, name="PG-Label",
)
