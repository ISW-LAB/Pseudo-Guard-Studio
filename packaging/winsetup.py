#!/usr/bin/env python3
r"""Install / uninstall the portable PG-Label release on Windows.

This is what "Install PG-Label.exe" runs. It takes the unzipped folder the user downloaded and
turns it into an installed application:

    %LOCALAPPDATA%\Programs\PG-Label\        the app (this folder, copied)
    Start menu\PG-Label\                     shortcuts (run · GPU pack · uninstall)
    HKCU\…\Uninstall\PG-Label                so it appears in Settings ▸ Apps

No administrator rights, nothing outside the user's profile, and no registry beyond that one
uninstall entry.

    Install PG-Label.exe                     install, then start the app
    Install PG-Label.exe --dest "D:\PG"      install somewhere else
    Install PG-Label.exe --no-launch         install quietly
    Uninstall PG-Label.exe                   remove the app (keeps your labels)

The user's labels, settings and GPU pack live in %LOCALAPPDATA%\PG-Label and are deliberately
left alone by both operations — an uninstall must never be able to destroy annotation work.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

IS_WINDOWS = os.name == "nt"
APP_NAME = "PG-Label"
DISPLAY_NAME = "PG-Label (Pseudo-Guard Studio)"
REG_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\PG-Label"

HERE = Path(__file__).resolve().parent          # …\packaging, inside the unzipped release
SRC_ROOT = HERE.parent                          # the unzipped release root


def say(msg: str) -> None:
    print(msg, flush=True)


def _app_paths():
    """The app's own paths module — the authority on where its data lives.

    Asking it (rather than recomputing %LOCALAPPDATA% here) means the installer can never report
    a different folder than the one the app actually writes to.
    """
    sys.path.insert(0, str(SRC_ROOT))
    try:
        from pglabel import console, paths
        console.enable()          # this installer's output is routinely piped to a log file
        return paths
    except Exception:
        return None


def _fsutil():
    """The app's filesystem helpers — Windows-safe delete, used by the uninstaller."""
    sys.path.insert(0, str(SRC_ROOT))
    from pglabel import fsutil
    return fsutil


def app_version() -> str:
    p = _app_paths()
    return p.APP_VERSION if p else "1.0.0"


def default_dest() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "Programs" / APP_NAME


def user_data_dir() -> Path:
    p = _app_paths()
    if p is not None:
        try:
            return p.data_root()
        except Exception:
            pass
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / APP_NAME


# ------------------------------------------------------------------ .exe launchers
def rebuild_exe(exe: Path, interpreter: str) -> bool:
    r"""Re-point a launcher .exe at `interpreter`, keeping its payload.

    The launcher is `stub + "#!<interpreter>\n" + zip(__main__.py)`. The shipped .exe files use
    distlib's relocatable `<launcher_dir>\` token; once installed we replace it with the absolute
    path of the installed runtime. That is the form pip writes for every console script it
    installs, so the installed app does not depend on the token being supported at all.
    """
    try:
        data = exe.read_bytes()
        with zipfile.ZipFile(exe) as zf:
            payload = zf.read("__main__.py")
            zip_start = min(i.header_offset for i in zf.infolist())
        stub_end = data.rindex(b"#!", 0, zip_start)          # start of the current shebang line
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("__main__.py", payload)
        exe.write_bytes(data[:stub_end] + f"#!{interpreter}\n".encode("utf-8") + buf.getvalue())
        return True
    except Exception as e:
        say(f"  ! could not re-point {exe.name}: {e}")
        return False


# ----------------------------------------------------------------------- shortcuts
_PS_SHORTCUT = (
    "$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}');"
    "$s.TargetPath = '{target}';"
    "$s.Arguments = '{args}';"
    "$s.WorkingDirectory = '{workdir}';"
    "$s.IconLocation = '{icon}';"
    "$s.Description = '{desc}';"
    "$s.Save()"
)

_VBS_SHORTCUT = """Set sh = CreateObject("WScript.Shell")
Set lnk = sh.CreateShortcut("{lnk}")
lnk.TargetPath = "{target}"
lnk.Arguments = "{args}"
lnk.WorkingDirectory = "{workdir}"
lnk.IconLocation = "{icon}"
lnk.Description = "{desc}"
lnk.Save
"""


def make_shortcut(lnk: Path, target: Path, args: str = "", icon: Path | None = None,
                  desc: str = "") -> bool:
    """Create a .lnk. PowerShell first, VBScript second — no third-party modules either way."""
    if not IS_WINDOWS:
        return False
    lnk.parent.mkdir(parents=True, exist_ok=True)
    fields = {"lnk": str(lnk), "target": str(target), "args": args,
              "workdir": str(target.parent), "icon": str(icon or target), "desc": desc}
    try:
        rc = subprocess.call(["powershell", "-NoProfile", "-NonInteractive",
                              "-ExecutionPolicy", "Bypass", "-Command", _PS_SHORTCUT.format(**fields)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if rc == 0 and lnk.exists():
            return True
    except Exception:
        pass
    try:                                          # PowerShell missing or locked down by policy
        vbs = Path(os.environ.get("TEMP", ".")) / "pglabel_shortcut.vbs"
        # UTF-16 with a BOM: cscript reads a .vbs as ANSI unless it finds one, so a UTF-8
        # file whose paths contain non-ASCII (C:\\Users\\<Korean name>\\...) produces a
        # shortcut pointing at mojibake. utf-16 is what the script host actually documents.
        vbs.write_text(_VBS_SHORTCUT.format(**fields), encoding="utf-16")
        subprocess.call(["cscript", "//nologo", str(vbs)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        vbs.unlink(missing_ok=True)
        return lnk.exists()
    except Exception:
        return False


def start_menu_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / APP_NAME


def desktop_dir() -> Path:
    d = Path.home() / "Desktop"
    return d if d.is_dir() else Path(os.environ.get("USERPROFILE", Path.home())) / "Desktop"


# ------------------------------------------------------------------------ registry
def register_uninstall(dest: Path, version: str) -> bool:
    """Add the Settings ▸ Apps entry (HKCU only — no admin rights, no machine-wide state)."""
    if not IS_WINDOWS:
        return False
    try:
        import winreg
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, REG_KEY) as k:
            size_kb = int(sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1024)
            for name, value in (("DisplayName", DISPLAY_NAME), ("DisplayVersion", version),
                                ("Publisher", "Pseudo-Guard Studio"),
                                ("InstallLocation", str(dest)),
                                ("DisplayIcon", str(dest / "assets" / "pglabel.ico")),
                                ("UninstallString", f'"{dest / "Uninstall PG-Label.exe"}"')):
                winreg.SetValueEx(k, name, 0, winreg.REG_SZ, value)
            winreg.SetValueEx(k, "EstimatedSize", 0, winreg.REG_DWORD, size_kb)
            winreg.SetValueEx(k, "NoModify", 0, winreg.REG_DWORD, 1)
            winreg.SetValueEx(k, "NoRepair", 0, winreg.REG_DWORD, 1)
        return True
    except Exception as e:
        say(f"  ! could not write the uninstall entry: {e}")
        return False


def unregister_uninstall() -> None:
    if not IS_WINDOWS:
        return
    try:
        import winreg
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, REG_KEY)
    except Exception:
        pass


# -------------------------------------------------------------------------- install
def copy_release(src: Path, dest: Path) -> int:
    """Copy the release tree. An in-use file (the app is running) is reported, not swallowed."""
    n, busy = 0, []
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        if "__pycache__" in rel.parts:
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(f, target)
            n += 1
        except PermissionError:
            busy.append(rel)
    if busy:
        say("\n  ! These files are in use — close PG-Label and run the installer again:")
        for b in busy[:5]:
            say(f"      {b}")
        raise SystemExit(1)
    return n


def install(args) -> int:
    version = app_version()
    dest = Path(args.dest).expanduser().resolve() if args.dest else default_dest()
    src = SRC_ROOT

    say("=" * 68)
    say(f"  {DISPLAY_NAME} {version} — installing")
    say("=" * 68)
    if not (src / "runtime").is_dir() and IS_WINDOWS:
        say(f"  ! this does not look like the unzipped release folder: {src}")
        say("    Unzip pseudo-guard-studio.zip first, then run the installer from inside it.")
        return 2
    if src == dest:
        say(f"  already installed in {dest}")
    else:
        say(f"  from : {src}")
        say(f"  to   : {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        n = copy_release(src, dest)
        say(f"  copied {n} files")

    # Absolute interpreter path → the installed launchers stop depending on the relocatable token.
    runtime = dest / "runtime" / "python.exe"
    interp = f'"{runtime}"' if " " in str(runtime) else str(runtime)
    for name in ("PG-Label.exe", "Install PG-Label.exe", "Uninstall PG-Label.exe"):
        exe = dest / name
        if exe.exists():
            rebuild_exe(exe, interp)

    app_exe = dest / "PG-Label.exe"
    icon = dest / "assets" / "pglabel.ico"
    made = []
    if IS_WINDOWS:
        sm = start_menu_dir()
        if make_shortcut(sm / f"{APP_NAME}.lnk", app_exe, icon=icon,
                         desc="Collaborative auto-labeling for object detection"):
            made.append(str(sm / f"{APP_NAME}.lnk"))
        make_shortcut(sm / "Install training pack.lnk", app_exe, args="--install-gpu-pack",
                      icon=icon, desc="Add PyTorch so the Train button works (large download)")
        make_shortcut(sm / f"Uninstall {APP_NAME}.lnk", dest / "Uninstall PG-Label.exe", icon=icon,
                      desc=f"Remove {APP_NAME} (your labels are kept)")
        if not args.no_desktop_icon:
            make_shortcut(desktop_dir() / f"{APP_NAME}.lnk", app_exe, icon=icon)
        register_uninstall(dest, version)

    say("")
    say("  Installed.")
    say(f"    app       : {app_exe}")
    say(f"    shortcuts : {start_menu_dir() if IS_WINDOWS else '(skipped — not Windows)'}")
    say(f"    your data : {user_data_dir()}   (labels, settings, logs — never removed)")
    say("")
    if args.no_launch:
        say("  Start it from the Start menu when you are ready.")
        return 0
    say("  Starting PG-Label — it opens in your browser …")
    if IS_WINDOWS:
        # Detached, with its own console: the app keeps running after this installer window closes.
        subprocess.Popen([str(app_exe)], cwd=str(dest),
                         creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
    return 0


# ------------------------------------------------------------------------ uninstall
def _write_cmd(path: Path, text: str) -> None:
    """Write a .cmd the way cmd.exe reads one: the machine's ANSI code page, not UTF-8.

    The install path can contain non-ASCII characters, and a UTF-8 .cmd would hand ``rmdir`` a
    mojibake target that silently removes nothing.
    """
    for encoding in ("mbcs", "utf-8"):
        try:
            path.write_text(text, encoding=encoding)
            return
        except (LookupError, UnicodeEncodeError):
            continue
    path.write_text(text, encoding="utf-8", errors="replace")


_SELF_DELETE = """@echo off
rem Wait for the uninstaller to exit, then remove the install folder and this script.
ping 127.0.0.1 -n 4 >nul
rmdir /s /q "{dest}"
del "%~f0"
"""


def uninstall(args) -> int:
    dest = Path(args.dest).expanduser().resolve() if args.dest else \
        (SRC_ROOT if (SRC_ROOT / "runtime").is_dir() else default_dest())
    say("=" * 68)
    say(f"  {DISPLAY_NAME} — uninstalling")
    say("=" * 68)
    say(f"  folder: {dest}")

    if IS_WINDOWS:
        for lnk in (start_menu_dir(), desktop_dir() / f"{APP_NAME}.lnk"):
            try:
                if lnk.is_dir():
                    shutil.rmtree(lnk, ignore_errors=True)
                elif lnk.exists():
                    lnk.unlink()
            except OSError:
                pass
        unregister_uninstall()
        say("  removed shortcuts and the Settings ▸ Apps entry")

    data = user_data_dir()
    # On Windows this ALWAYS leaves something behind: the uninstaller is running from inside the
    # folder it is deleting, and an open .exe cannot be removed. So this is not an error path —
    # it is the normal path, and it must be driven by the RESULT, not by an exception that
    # remove_tree deliberately does not raise.
    removed = _fsutil().remove_tree(dest)
    if removed:
        say("  removed the program folder")
    elif IS_WINDOWS:
        # Hand the rest to a detached script that starts once this process is gone.
        bat = Path(os.environ.get("TEMP", ".")) / "pglabel_cleanup.cmd"
        _write_cmd(bat, _SELF_DELETE.format(dest=dest))
        subprocess.Popen(["cmd", "/c", "start", "/min", "", str(bat)],
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        say("  the program folder will be removed in a few seconds")
    else:
        say("  could not remove the program folder (in use)")

    say("")
    say(f"  Your labels and settings were KEPT in: {data}")
    say("  Delete that folder by hand if you want them gone too.")
    if IS_WINDOWS and sys.stdin and sys.stdin.isatty():
        input("\n  Press Enter to close…")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true", help="install (the default)")
    ap.add_argument("--uninstall", action="store_true", help="remove the installed app")
    ap.add_argument("--dest", default=None, help="install folder (default: %%LOCALAPPDATA%%\\Programs\\PG-Label)")
    ap.add_argument("--no-launch", action="store_true", help="do not start the app afterwards")
    ap.add_argument("--no-desktop-icon", action="store_true", help="skip the desktop shortcut")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return uninstall(args) if args.uninstall else install(args)
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        if IS_WINDOWS and sys.stdin and sys.stdin.isatty():
            input("\n  Something went wrong. Press Enter to close…")
        return 1


if __name__ == "__main__":
    sys.exit(main())
