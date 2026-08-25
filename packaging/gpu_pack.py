#!/usr/bin/env python3
r"""Build the "GPU pack": the separate Python environment PG-Label trains through.

PG-Label.exe is deliberately torch-free (see PG-Label.spec). The Train / Run-cycle buttons shell
out to another interpreter, and this module creates one — a plain venv with torch + ultralytics —
then registers its python.exe in settings.json so the app finds it on the next launch.

    PG-Label.exe --install-gpu-pack              auto-detect CUDA, install, register
    PG-Label.exe --install-gpu-pack --cuda cpu   force CPU-only wheels (no NVIDIA GPU)
    PG-Label.exe --install-gpu-pack --python "C:\Python311\python.exe"
    PG-Label.exe --find-python                   list the interpreters it can see, then stop

It needs ONE thing the app cannot provide: a real Python installation to build the venv from
(the frozen interpreter inside the .exe cannot create environments). If none is found it prints
the winget one-liner and stops — the app keeps working, only training stays unavailable.

Expect a ~2.5 GB download for the CUDA build. Everything lands in ONE folder, so uninstalling
the pack is deleting it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

# torch publishes one wheel index per CUDA build; "cpu" is the fallback that always works.
CUDA_INDEX = {
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cpu": "https://download.pytorch.org/whl/cpu",
}
# What the trainer needs beyond torch: the detector framework, plus the handful of helpers
# `pseudoguard` and `tools/` import directly. Kept in step with requirements-train.txt.
EXTRA_PACKAGES = ["ultralytics", "opencv-python-headless", "pyyaml", "tqdm", "pillow",
                  "numpy", "scikit-learn"]
MIN_PY, MAX_PY = (3, 9), (3, 12)                 # the range with torch wheels for every CUDA build


def _run(cmd, **kw) -> int:
    """Run a command, streaming its output; return the exit code (never raises on failure)."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    try:
        return subprocess.call(cmd, **kw)
    except FileNotFoundError:
        print(f"[gpu-pack] not found: {cmd[0]}", file=sys.stderr)
        return 127


def _probe(exe: Path | str):
    """(version_tuple, is_64bit) for an interpreter, or None when it will not run."""
    try:
        out = subprocess.run(
            [str(exe), "-c", "import sys,struct;print(sys.version_info[:3], struct.calcsize('P')*8)"],
            capture_output=True, text=True, timeout=25,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    m = re.search(r"\((\d+),\s*(\d+),\s*(\d+)\)\s+(\d+)", out.stdout)
    if not m:
        return None
    return (int(m[1]), int(m[2]), int(m[3])), int(m[4]) == 64


def find_interpreters() -> list[dict]:
    """Every usable Python on this machine, best first.

    Looks where Windows actually puts them: the `py` launcher's registry list, PATH, the per-user
    python.org location, and conda/miniforge env folders — in that order, because the py launcher
    is the only source that is authoritative about what is installed.
    """
    seen, found = set(), []

    def add(exe, note):
        try:
            p = Path(exe).resolve()
        except OSError:
            return
        if not p.exists() or str(p).lower() in seen:
            return
        info = _probe(p)
        if not info:
            return
        ver, is64 = info
        seen.add(str(p).lower())
        found.append({"exe": str(p), "version": ver, "x64": is64, "source": note,
                      "usable": is64 and MIN_PY <= ver[:2] <= MAX_PY})

    if os.name == "nt":
        try:                                     # `py -0p` lists every registered install + path
            out = subprocess.run(["py", "-0p"], capture_output=True, text=True, timeout=25,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in out.stdout.splitlines():
                m = re.search(r"([A-Za-z]:\\[^\r\n]*python\.exe)", line)
                if m:
                    add(m.group(1), "py launcher")
        except Exception:
            pass
        local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python"
        for d in sorted(local.glob("Python3*"), reverse=True) if local.is_dir() else []:
            add(d / "python.exe", "python.org (user)")
        for base in (Path("C:/"), Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))):
            for d in sorted(base.glob("Python3*"), reverse=True):
                add(d / "python.exe", "python.org (machine)")

    import shutil as _sh
    for name in ("python3", "python"):
        w = _sh.which(name)
        if w:
            add(w, "PATH")

    home = Path.home()
    for root in ("miniconda3", "anaconda3", "miniforge3", "mambaforge"):
        base = home / root
        if not base.is_dir():
            continue
        exe = "python.exe" if os.name == "nt" else "bin/python"
        add(base / exe, f"{root} (base)")
        for env in sorted((base / "envs").glob("*")) if (base / "envs").is_dir() else []:
            add(env / exe, f"{root} env '{env.name}'")

    found.sort(key=lambda c: (not c["usable"], c["source"] != "py launcher", [-x for x in c["version"]]))
    return found


def detect_cuda() -> str:
    """Pick a wheel index from the installed NVIDIA driver.

    nvidia-smi reports the newest CUDA runtime the DRIVER supports, which is the real constraint —
    a cu126 wheel on a driver that tops out at 12.1 fails at the first kernel launch, not at
    install time, so guessing high is the expensive mistake here.
    """
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=25,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                             if os.name == "nt" else 0)
    except Exception:
        print("[gpu-pack] no nvidia-smi → installing CPU-only wheels.")
        return "cpu"
    if out.returncode != 0:
        return "cpu"
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out.stdout)
    if not m:
        return "cu121"
    major, minor = int(m[1]), int(m[2])
    print(f"[gpu-pack] driver supports CUDA {major}.{minor}")
    if (major, minor) >= (12, 6):
        return "cu126"
    if (major, minor) >= (12, 4):
        return "cu124"
    if (major, minor) >= (12, 1):
        return "cu121"
    if major >= 11:
        return "cu118"
    return "cpu"


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def install(base_python: str, venv: Path, cuda: str, upgrade: bool = False) -> int:
    """Create `venv` from `base_python` and fill it with torch + the trainer's dependencies."""
    py = venv_python(venv)
    if not py.exists():
        venv.parent.mkdir(parents=True, exist_ok=True)
        if _run([base_python, "-m", "venv", str(venv)]) != 0:
            print("[gpu-pack] could not create the environment "
                  "(is the 'venv' module available in that Python?)", file=sys.stderr)
            return 1
    if not py.exists():
        print(f"[gpu-pack] environment missing after creation: {py}", file=sys.stderr)
        return 1

    _run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel"])
    index = CUDA_INDEX.get(cuda, CUDA_INDEX["cpu"])
    args = ["--upgrade"] if upgrade else []
    print(f"\n[gpu-pack] installing torch ({cuda}) — this downloads ~"
          f"{'2.5 GB' if cuda != 'cpu' else '250 MB'} and takes a while.")
    if _run([str(py), "-m", "pip", "install", *args, "torch", "torchvision",
             "--index-url", index]) != 0:
        print("[gpu-pack] torch install failed — check the network/proxy, then re-run.",
              file=sys.stderr)
        return 1
    # Separate call, default index: these are not on the pytorch index, and pinning them there
    # would silently resolve to nothing.
    if _run([str(py), "-m", "pip", "install", *args, *EXTRA_PACKAGES]) != 0:
        print("[gpu-pack] the extra packages failed to install.", file=sys.stderr)
        return 1
    return 0


def verify(py: Path) -> bool:
    """Report what the finished environment can actually do (CUDA present? which GPU?)."""
    code = ("import torch, ultralytics;"
            "print('torch', torch.__version__, '| ultralytics', ultralytics.__version__);"
            "print('cuda available:', torch.cuda.is_available());"
            "print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')")
    return _run([str(py), "-c", code]) == 0


def main(argv: list[str], paths_mod) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="PG-Label --install-gpu-pack", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install-gpu-pack", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--find-python", action="store_true", help="list detected interpreters and exit")
    ap.add_argument("--python", default=None, help="base interpreter to build the env from")
    ap.add_argument("--cuda", default="auto", choices=["auto", *CUDA_INDEX],
                    help="wheel flavour (default: auto-detect from the NVIDIA driver)")
    ap.add_argument("--dest", default=None, help="where to create the env "
                                                 "(default: <data folder>\\gpu-env)")
    ap.add_argument("--upgrade", action="store_true", help="upgrade an existing pack in place")
    args = ap.parse_args(argv)

    cands = find_interpreters()
    if args.find_python:
        if not cands:
            print("[gpu-pack] no Python installation found on this machine.")
        for c in cands:
            v = ".".join(str(x) for x in c["version"])
            flag = "usable" if c["usable"] else f"skip (needs 64-bit {MIN_PY[0]}.{MIN_PY[1]}–{MAX_PY[0]}.{MAX_PY[1]})"
            print(f"  {c['exe']}\n      python {v} · {c['source']} · {flag}")
        return 0

    base = args.python
    if base and not Path(base).exists():
        print(f"[gpu-pack] --python is not a file: {base}", file=sys.stderr)
        return 2
    if not base:
        usable = [c for c in cands if c["usable"]]
        if not usable:
            print("[gpu-pack] No suitable Python found (need 64-bit "
                  f"{MIN_PY[0]}.{MIN_PY[1]}–{MAX_PY[0]}.{MAX_PY[1]}).")
            print("[gpu-pack] Install one, then run this again:")
            print("             winget install -e --id Python.Python.3.11")
            print("           …or point at an existing one:  --python \"C:\\path\\to\\python.exe\"")
            return 3
        base = usable[0]["exe"]
        print(f"[gpu-pack] using {base}  ({usable[0]['source']})")

    cuda = detect_cuda() if args.cuda == "auto" else args.cuda
    venv = Path(args.dest).expanduser() if args.dest else (paths_mod.data_root() / "gpu-env")
    print(f"[gpu-pack] target environment: {venv}")

    rc = install(base, venv, cuda, upgrade=args.upgrade)
    if rc != 0:
        return rc

    py = venv_python(venv)
    print("\n[gpu-pack] verifying …")
    verify(py)
    paths_mod.save_settings({"train_python": str(py.resolve()), "gpu_pack_cuda": cuda})
    print(f"\n[gpu-pack] done. Registered in {paths_mod.settings_path()}")
    print("[gpu-pack] Restart PG-Label — the Train button is now enabled.")
    return 0
