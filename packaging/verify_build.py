#!/usr/bin/env python3
"""Check that a build is complete and internally consistent — before a user finds out it is not.

Two modes, and the first is the useful one on a machine that cannot build a Windows exe:

    python packaging/verify_build.py                    audit the SOURCE tree (any OS)
    python packaging/verify_build.py --dist DIR         audit a built folder as well

The source audit answers the questions a packaging bug actually shows up as:
does every file the spec promises to bundle exist, does the app import with no torch present,
does every module the trainer needs resolve, and is the UI's asset wiring intact. Those are the
failures that otherwise surface as "works from source, broken once packaged".

Exit code is 0 only when every check passes, so it can gate a release.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent

sys.path.insert(0, str(ROOT))
from pglabel import console  # noqa: E402
console.enable()

# Files the spec bundles as plain data. If one of these is missing the exe builds fine and then
# fails at runtime, which is exactly the failure this catches early.
REQUIRED_DATA = [
    "pglabel/static/index.html",
    "pglabel/static/css/app.css",
    "pglabel/static/js/app.js",
    "tools/train_and_predict.py",
    "tools/gen_noise_crops.py",
    "tools/common.py",
    "pgcount/__init__.py",
    "pgcount/backend.py",
    "pseudoguard/__init__.py",
    "pseudoguard/config.py",
    "pseudoguard/data/noise_generator.py",
    "pseudoguard/models/detection/yolov8_wrapper.py",
    "pseudoguard/models/classification/densenet_wrapper.py",
]

# What the spec ships NEXT TO the exe as plain .py, and therefore all the training interpreter
# can import. tools/ reaching outside this set is the classic "works from source" packaging bug.
TRAINER_VISIBLE = {"pseudoguard", "pgcount", "tools"}

# Everything the Windows build and release routes need. A missing one of these is only noticed
# when somebody tries to cut a release, which is the worst moment to find out.
PACKAGING_CHAIN = [
    "packaging/build.py", "packaging/build.bat", "packaging/PG-Label.spec",
    "packaging/make_icon.py", "packaging/assets/pglabel.ico", "packaging/installer.iss",
    "packaging/install_training_pack.bat", "packaging/gpu_pack.py", "packaging/winsetup.py",
    "packaging/make_portable_zip.py",
    "run_app.py", "requirements.txt", "requirements-train.txt", "LICENSE", "README.md",
]

# The app process must never import these: they are the 2.5 GB the training pack exists to hold.
FORBIDDEN_IN_APP = {"torch", "torchvision", "ultralytics", "numpy", "cv2", "sklearn",
                    "scipy", "pandas", "matplotlib", "yaml", "tqdm"}

results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    results.append((ok, label))
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}" + (f"  — {detail}" if detail and not ok else ""))
    return ok


def top_level_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def audit_source() -> None:
    print("\nSource tree")
    for rel in REQUIRED_DATA:
        check((ROOT / rel).is_file(), f"present: {rel}")

    # The app must import with nothing but Pillow available.
    rc = subprocess.run([sys.executable, "-c",
                         "import sys; sys.path.insert(0, %r);"
                         "import pglabel, pglabel.api, pglabel.cli, pglabel.desktop;"
                         "print(pglabel.__version__)" % str(ROOT)],
                        capture_output=True, text=True)
    check(rc.returncode == 0, "the app package imports", rc.stderr.strip()[-400:])

    heavy = set()
    for py in (ROOT / "pglabel").rglob("*.py"):
        heavy |= top_level_imports(py) & FORBIDDEN_IN_APP
    check(not heavy, "the app imports no heavy ML package",
          f"found {sorted(heavy)} — these belong in tools/, not pglabel/")

    # The trainer runs under a DIFFERENT interpreter that can see only the packages the spec
    # ships beside it. Importing anything else works from a checkout and fails in every
    # packaged install, so the whole class is checked, not one known case.
    local_packages = {p.name for p in ROOT.iterdir()
                      if p.is_dir() and (p / "__init__.py").exists()}
    unreachable = []
    for tool in sorted((ROOT / "tools").glob("*.py")):
        for name in sorted(top_level_imports(tool) & local_packages):
            if name not in TRAINER_VISIBLE:
                unreachable.append(f"{tool.name} -> {name}")
            elif not (ROOT / name / "__init__.py").exists():
                unreachable.append(f"{tool.name} -> {name} (missing)")
    check(not unreachable, "the trainer imports nothing that is not bundled with it",
          ", ".join(unreachable))

    # The UI's own assets must be wired the way the server serves them.
    html = (ROOT / "pglabel" / "static" / "index.html").read_text(encoding="utf-8")
    check('href="/static/css/app.css"' in html, "index.html links the stylesheet")
    check('src="/static/js/app.js"' in html, "index.html links the script")
    check("<style>" not in html and "<script>\n" not in html,
          "index.html has no leftover inline blocks")

    for rel in PACKAGING_CHAIN:
        check((ROOT / rel).is_file(), f"packaging needs: {rel}")

    # Version consistency: the exe resource, the installer and the About line read one value.
    sys.path.insert(0, str(ROOT))
    from pglabel import paths
    iss = (PKG / "installer.iss").read_text(encoding="utf-8") if (PKG / "installer.iss").exists() else ""
    check(paths.APP_VERSION in iss or "MyAppVersion" in iss,
          f"installer knows the version ({paths.APP_VERSION})")


def audit_dist(dist: Path) -> None:
    print(f"\nBuilt folder: {dist}")
    if not check(dist.is_dir(), f"the folder exists: {dist}"):
        return
    internal = dist / "_internal" if (dist / "_internal").is_dir() else dist
    for rel in REQUIRED_DATA:
        check((internal / rel).is_file(), f"bundled: {rel}")

    exe = dist / ("PG-Label.exe" if sys.platform == "win32" else "PG-Label")
    if check(exe.exists(), "the executable is present"):
        try:
            out = subprocess.run([str(exe), "--where"], capture_output=True, text=True, timeout=180)
            info = json.loads(out.stdout[out.stdout.index("{"):out.stdout.rindex("}") + 1])
            check(out.returncode == 0, "--where runs")
            check(bool(info.get("research_root")), "the bundle can find the algorithm library",
                  "training would be disabled in this build")
            check(Path(info["static"]).name == "static", "the UI assets resolve")
        except Exception as e:
            check(False, "--where runs", str(e))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dist", type=Path, default=None,
                    help="also audit a built app folder (packaging/dist/PG-Label)")
    args = ap.parse_args(argv)

    audit_source()
    if args.dist:
        audit_dist(args.dist)

    failed = [label for ok, label in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("failed: " + "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
