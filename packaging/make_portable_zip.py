#!/usr/bin/env python3
r"""Assemble pseudo-guard-studio.zip — the "unzip on Windows and run it" release.

WHY THIS EXISTS ALONGSIDE build.py
    build.py freezes the app with PyInstaller, which only works ON Windows. This builds the same
    product from ANY machine (Linux included) by shipping the pieces instead of fusing them:
    Python's official Windows embeddable runtime, the Pillow wheel for that runtime, this app's
    sources, and three small .exe launchers.

    Same result for the user — unzip, install, run, no prerequisites — reached without a Windows
    build machine. It costs transparency (the app ships as readable .py) and about 20 MB.

HOW THE .EXE FILES WORK
    They are the launcher stub pip itself uses for Windows console scripts (distlib's "simple
    launcher"): stub + a shebang line + a zip holding __main__.py. Running the .exe makes the
    stub start the interpreter named in the shebang and hand it the .exe, whose trailing zip
    Python imports. The shebang uses distlib's ``<launcher_dir>\`` token, which the stub replaces
    with the folder the .exe sits in — that is what makes the unzipped folder runnable from
    wherever the user dropped it, and movable afterwards.

    Every .exe has a .cmd twin doing the same thing, as an escape hatch for machines whose
    endpoint protection quarantines unsigned launchers.

USAGE
    python packaging/make_portable_zip.py                 # build the zip
    python packaging/make_portable_zip.py --no-demo       # smaller: no sample images
    python packaging/make_portable_zip.py --out DIR       # where to write the .zip
    python packaging/make_portable_zip.py --keep-tree     # leave the staged tree for inspection

Downloads (cached under packaging/build/cache) come from python.org and PyPI.
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
CACHE = PKG / "build" / "cache"
STAGE = PKG / "build" / "release"

sys.path.insert(0, str(ROOT))
from pglabel import console, fsutil, paths  # noqa: E402
console.enable()

# The runtime the app runs on. 3.12 keeps the widest wheel coverage while being new enough for
# everything the app uses; the Pillow wheel below must match this exact ABI.
PY_VERSION = "3.12.10"
PY_ABI = "cp312"
PY_EMBED_URL = f"https://www.python.org/ftp/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip"
RELEASE_NAME = "pseudo-guard-studio"


def say(msg: str) -> None:
    print(f"[release] {msg}", flush=True)


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        say(f"cached  {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    say(f"fetching {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)
    return dest


def get_pillow_wheel() -> Path:
    """The Windows Pillow wheel matching PY_ABI (pip resolves the current release for us)."""
    hits = sorted(CACHE.glob(f"pillow-*-{PY_ABI}-{PY_ABI}-win_amd64.whl"))
    if hits:
        say(f"cached  {hits[-1].name}")
        return hits[-1]
    CACHE.mkdir(parents=True, exist_ok=True)
    say("fetching the Windows Pillow wheel")
    rc = subprocess.call([sys.executable, "-m", "pip", "download", "--only-binary=:all:",
                          "--platform", "win_amd64",
                          "--python-version", PY_VERSION.rsplit(".", 1)[0],
                          "--implementation", "cp", "--abi", PY_ABI,
                          "pillow>=10", "-d", str(CACHE)])
    if rc != 0:
        sys.exit("[release] could not download the Pillow wheel (network/proxy?)")
    hits = sorted(CACHE.glob(f"pillow-*-{PY_ABI}-{PY_ABI}-win_amd64.whl"))
    if not hits:
        sys.exit("[release] pip reported success but no matching wheel appeared")
    return hits[-1]


def copy_tree(src: Path, dest: Path, patterns=("*",), skip=()) -> int:
    """Copy the sources the app needs, skipping caches and previous training artefacts."""
    skip = {"__pycache__", ".git", ".pgtrain", "runs", "cache", "build", "dist"} | set(skip)
    n = 0
    if not src.is_dir():
        return 0
    for pat in patterns:
        for f in src.rglob(pat):
            rel = f.relative_to(src)
            if not f.is_file() or any(p in skip for p in rel.parts[:-1]):
                continue
            if f.name.startswith(".") or f.suffix in (".pyc", ".pyo"):
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            n += 1
    return n


# ------------------------------------------------------------------- exe launchers
def launcher_stub() -> bytes:
    """distlib's console launcher — the same bytes pip writes for every Windows console script."""
    try:
        import pip._vendor.distlib as distlib
    except ImportError:
        sys.exit("[release] pip's vendored distlib is unavailable; cannot build the .exe stubs")
    stub = Path(distlib.__file__).parent / "t64.exe"
    if not stub.exists():
        sys.exit(f"[release] launcher stub not found: {stub}")
    return stub.read_bytes()


def make_exe(dest: Path, main_py: str,
             interpreter: str = r"<launcher_dir>\runtime\python.exe") -> None:
    r"""Write a Windows .exe = stub + shebang + zip(__main__.py).

    ``interpreter`` is left unquoted on purpose. The stub splits the shebang on whitespace to
    separate the executable from any interpreter flags, then quotes the executable itself when it
    builds the command line — so an unquoted, space-free ``<launcher_dir>\…`` token survives an
    install path like ``C:\Users\Kim Min Su\…``, while a pre-quoted token may not be parsed at all.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("__main__.py", main_py)
    dest.write_bytes(launcher_stub() + f"#!{interpreter}\n".encode("utf-8") + buf.getvalue())


BOOT = r'''# PG-Label — {what}
# Generated by packaging/make_portable_zip.py. Runs inside the .exe's appended zip: sys.argv[0]
# is the .exe, so its folder is the app root.
import sys
from pathlib import Path

def _root():
    for cand in (sys.argv[0], globals().get("__file__", "")):
        if cand:
            p = Path(cand).resolve()
            # __file__ is "<app root>/PG-Label.exe/__main__.py" — one level deeper than argv[0]
            p = p.parent.parent if p.name == "__main__.py" else p.parent
            if (p / "runtime").is_dir() or (p / "pglabel").is_dir():
                return p
    return Path.cwd()

ROOT = _root()
sys.path.insert(0, str(ROOT / "packaging"))
sys.path.insert(0, str(ROOT))

{body}
'''

APP_BODY = '''from pglabel.desktop import main
sys.exit(main(sys.argv[1:]))'''

INSTALL_BODY = '''import winsetup
sys.exit(winsetup.main(["--install"] + sys.argv[1:]))'''

UNINSTALL_BODY = '''import winsetup
sys.exit(winsetup.main(["--uninstall"] + sys.argv[1:]))'''

CMD = '''@echo off
REM {what} — the .exe next to this file does exactly the same thing.
REM This .cmd is the fallback for machines whose security software blocks unsigned launchers.
setlocal
cd /d "%~dp0"
"%~dp0runtime\\python.exe" "%~dp0{script}" {args} %*
if errorlevel 1 pause
'''


def build(args) -> Path:
    stage = STAGE / RELEASE_NAME
    if stage.exists() and not fsutil.remove_tree(stage):
        sys.exit(f'[release] could not clear the staging folder: {stage}')
    stage.mkdir(parents=True)

    # 1 — the Windows Python runtime
    say(f"unpacking the Python {PY_VERSION} embeddable runtime")
    runtime = stage / "runtime"
    with zipfile.ZipFile(fetch(PY_EMBED_URL,
                               CACHE / f"python-{PY_VERSION}-embed-amd64.zip")) as z:
        z.extractall(runtime)
    site = runtime / "Lib" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(get_pillow_wheel()) as z:
        z.extractall(site)
    # The embeddable runtime reads sys.path from this file and NOTHING else (no site, no env).
    # Without the site-packages line, `import PIL` fails; the app adds its own folders at runtime.
    pth = next(runtime.glob("python*._pth"))
    pth.write_text("\n".join([
        f"python{PY_ABI[2:]}.zip",           # the stdlib
        ".",                                  # the runtime folder itself (.pyd extension modules)
        r"Lib\site-packages",                 # Pillow — without this line `import PIL` fails
        "",
        "# No 'import site' line: nothing here needs site.py, and leaving it out keeps a Python",
        "# installed elsewhere on the PC from leaking its packages into this runtime.",
        "# A ._pth also implies -E -s and safe_path=1. That is fine for the .exe launchers:",
        "# CPython prepends a zipapp to sys.path regardless of safe_path (Modules/main.c), and",
        "# the launchers add their own folders explicitly.",
        ""]), encoding="utf-8")

    # 2 — the app, as readable source
    say("copying the app")
    n = copy_tree(ROOT / "pglabel", stage / "pglabel", patterns=("*.py",))
    # static/ is copied wholesale rather than by pattern: a pattern like "static/*" matches only
    # files directly inside it, which silently dropped css/ and js/ and shipped a UI with no
    # styles and no behaviour. verify_stage() below now fails the build if that ever recurs.
    n += copy_tree(ROOT / "pglabel" / "static", stage / "pglabel" / "static")
    n += copy_tree(ROOT / "pgcount", stage / "pgcount", patterns=("*.py",))
    shutil.copy2(ROOT / "run_app.py", stage / "run_app.py")
    pack = stage / "packaging"
    pack.mkdir(parents=True, exist_ok=True)
    for f in ("winsetup.py", "gpu_pack.py"):
        shutil.copy2(PKG / f, pack / f)
    if (PKG / "assets" / "pglabel.ico").exists():
        (stage / "assets").mkdir(exist_ok=True)
        shutil.copy2(PKG / "assets" / "pglabel.ico", stage / "assets" / "pglabel.ico")
    say(f"copied {n} application files")

    # 3 — the algorithm library + trainers, so the Train button exists after install
    if not args.no_training:
        say("copying the algorithm library and training tools")
        copy_tree(ROOT / "pseudoguard", stage / "pseudoguard", patterns=("*.py",))
        copy_tree(ROOT / "tools", stage / "tools", patterns=("*.py",))

    # 4 — the sample dataset
    if not args.no_demo:
        n = copy_tree(ROOT / "demo" / "images", stage / "demo" / "images",
                      patterns=("*.jpg", "*.png"))
        n += copy_tree(ROOT / "demo" / "gt_labels", stage / "demo" / "gt_labels",
                       patterns=("*.txt",))
        say(f"bundled the demo dataset ({n} files)")

    # 5 — the marker that tells the app it is a DISTRIBUTED copy, so the user's labels go to
    #     %LOCALAPPDATA%\PG-Label instead of into this folder (which upgrades replace)
    (stage / ".pglabel-packaged").write_text(
        f"{paths.APP_NAME} {paths.APP_VERSION} portable build\n", encoding="utf-8")

    # 6 — the launchers
    say("writing the .exe launchers")
    make_exe(stage / "Install PG-Label.exe", BOOT.format(what="installer", body=INSTALL_BODY))
    make_exe(stage / "PG-Label.exe", BOOT.format(what="app launcher", body=APP_BODY))
    make_exe(stage / "Uninstall PG-Label.exe", BOOT.format(what="uninstaller", body=UNINSTALL_BODY))
    (stage / "Install PG-Label.cmd").write_text(
        CMD.format(what="Install PG-Label", script="packaging\\winsetup.py", args="--install"),
        encoding="utf-8")
    (stage / "PG-Label.cmd").write_text(
        CMD.format(what="PG-Label", script="run_app.py", args=""), encoding="utf-8")
    (stage / "README_FIRST.txt").write_text(readme_first(), encoding="utf-8")

    verify_stage(stage, with_training=not args.no_training)

    # 7 — zip it
    out_dir = Path(args.out).expanduser().resolve() if args.out else ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{RELEASE_NAME}.zip"
    if zip_path.exists():
        zip_path.unlink()
    say(f"compressing → {zip_path}")
    total = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                zf.write(f, Path(RELEASE_NAME) / f.relative_to(stage))
                total += 1
    say(f"{total} files · {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
    if not args.keep_tree:
        fsutil.remove_tree(stage)
    return zip_path


# Everything the unzipped release must contain to actually run. Checked before zipping, because
# a missing asset here is invisible until a user opens the app and gets a blank page.
REQUIRED_IN_RELEASE = [
    "run_app.py", ".pglabel-packaged", "PG-Label.exe", "PG-Label.cmd",
    "Install PG-Label.exe", "packaging/winsetup.py", "packaging/gpu_pack.py",
    "pglabel/__init__.py", "pglabel/desktop.py", "pglabel/api.py",
    "pglabel/static/index.html", "pglabel/static/css/app.css", "pglabel/static/js/app.js",
    "pgcount/__init__.py", "pgcount/backend.py", "pgcount/operating_point.py",
    "runtime/python.exe",
]
REQUIRED_FOR_TRAINING = [
    "pseudoguard/__init__.py", "pseudoguard/config.py", "pseudoguard/data/noise_generator.py",
    "pseudoguard/models/detection/yolov8_wrapper.py",
    "tools/train_and_predict.py", "tools/gen_noise_crops.py", "tools/common.py",
]


def verify_stage(stage: Path, with_training: bool) -> None:
    """Fail the build if the staged tree is missing anything the app needs at runtime."""
    required = list(REQUIRED_IN_RELEASE) + (REQUIRED_FOR_TRAINING if with_training else [])
    missing = [rel for rel in required if not (stage / rel).is_file()]
    if missing:
        for rel in missing:
            say(f"MISSING from the release: {rel}")
        sys.exit(f"[release] {len(missing)} required file(s) missing — not shipping this build")
    say(f"staged tree verified ({len(required)} required files present)")


def readme_first() -> str:
    return f"""PG-Label (Pseudo-Guard Studio) {paths.APP_VERSION}
==================================================================

1. Keep this folder unzipped and run "Install PG-Label.exe".
   Installs to %LOCALAPPDATA%\\Programs\\PG-Label — no administrator rights needed.
   SmartScreen may warn: More info -> Run anyway (expected for unsigned software).
2. To try it without installing, run "PG-Label.exe".
3. If an .exe is blocked, run the .cmd file of the same name instead — identical behaviour.

The app opens in your browser (http://127.0.0.1:8000). Keep the console window open while you
work; closing it stops the app. Labels are written to %LOCALAPPDATA%\\PG-Label\\workspace.

Training (the Train and Run-cycle buttons) is optional and needs PyTorch:
run "packaging\\gpu_pack.py" through the Start-menu entry, or point the app at any Python that
already has torch + ultralytics (Settings -> training interpreter). Training runs on the CPU
when no GPU is present — slower, but it works.

What is inside
--------------
runtime\\                Python {PY_VERSION} (official python.org embeddable build) + Pillow
pglabel\\                the annotation application
pgcount\\                count-guided acceptance
pseudoguard\\, tools\\    the algorithm library and the training entry points
demo\\                   sample images with ground truth
*.exe                   launcher stubs from distlib (the same ones pip uses)

Nothing here phones home; the server listens on 127.0.0.1 only.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="folder to write the .zip into (default: repo root)")
    ap.add_argument("--no-demo", action="store_true", help="omit the sample dataset")
    ap.add_argument("--no-training", action="store_true",
                    help="omit pseudoguard/ and tools/ (label-only release)")
    ap.add_argument("--keep-tree", action="store_true", help="keep the staged tree for inspection")
    args = ap.parse_args(argv)
    t0 = time.time()
    say(f"{paths.APP_NAME} {paths.APP_VERSION} — assembling the Windows portable release")
    zip_path = build(args)
    say(f"done in {time.time() - t0:.0f}s → {zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
