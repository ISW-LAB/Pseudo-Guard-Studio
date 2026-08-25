"""Behaviour that only misbehaves on Windows, pinned from any OS.

Windows breaks three POSIX assumptions this app relies on, and each one fails in a way that is
invisible during development on Linux or macOS:

    paths      joining an ABSOLUTE path discards the base, so ``images / "C:\\Windows\\x"``
               escapes the dataset entirely — a traversal that does not exist on POSIX.
    encoding   a redirected stdout falls back to the machine's ANSI code page, which cannot
               represent the characters the logs actually contain.
    deletion   read-only files refuse to be removed, and renaming onto an existing file fails.

These tests exercise the guards rather than the platform, so they are meaningful everywhere.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import PureWindowsPath
from pathlib import Path

from support import ROOT, ServerCase, make_dataset, write_boxes    # noqa: E402
from pglabel import api, console, fsutil, state                    # noqa: E402


class TestPathTraversal(ServerCase):
    """Every endpoint that takes a file name from the URL must stay inside the dataset."""

    overlay = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.secret = cls.tmp / "SECRET.txt"
        cls.secret.write_text("private key material", encoding="utf-8")

    def test_relative_escape_is_refused(self):
        for attack in ("/api/file/../SECRET.txt", "/api/file/..%2FSECRET.txt",
                       "/api/file/..%5CSECRET.txt", "/api/file/%2e%2e%2fSECRET.txt"):
            code, body = self.get(attack)
            self.assertEqual(code, 404, attack)
            self.assertNotIn(b"private key", body if isinstance(body, bytes) else b"")

    def test_absolute_paths_are_refused(self):
        # The Windows case: PureWindowsPath("D:/imgs") / "C:/Windows/win.ini" == C:/Windows/win.ini
        self.assertEqual(str(PureWindowsPath("D:/imgs") / "C:/Windows/win.ini"),
                         str(PureWindowsPath("C:/Windows/win.ini")),
                         "this is the platform behaviour safe_image() exists to defeat")
        for attack in ("/api/file/%2Fetc%2Fhostname", "/api/file/C%3A%5CWindows%5Cwin.ini"):
            self.assertEqual(self.get(attack)[0], 404, attack)

    def test_label_and_candidate_routes_are_guarded_too(self):
        for route in ("/api/labels/..%2FSECRET.txt", "/api/candidates/..%2FSECRET.txt"):
            self.assertEqual(self.get(route)[0], 404, route)

    def test_writes_cannot_escape_either(self):
        code, _ = self.post("/api/labels/..%2Fpwned.txt", {"boxes": []})
        self.assertEqual(code, 404)
        self.assertFalse((self.tmp / "pwned.txt").exists())

    def test_legitimate_names_still_work(self):
        name = self.names[0]
        self.assertEqual(self.get(f"/api/file/{name}")[0], 200)
        self.assertEqual(self.get(f"/api/labels/{name}")[0], 200)
        self.assertEqual(self.get(f"/api/candidates/{name}")[0], 200)

    def test_safe_image_accepts_only_real_files_in_the_dataset(self):
        self.assertIsNotNone(api.safe_image(self.names[0]))
        self.assertIsNone(api.safe_image("../SECRET.txt"))
        self.assertIsNone(api.safe_image("does-not-exist.jpg"))
        self.assertIsNone(api.safe_image(""))            # the folder itself is not a file


class TestConsoleEncoding(unittest.TestCase):
    """A Windows code page must never be able to kill a run."""

    def test_enable_is_none_safe(self):
        # A windowed PyInstaller build has sys.stdout is None; this must not raise.
        saved = sys.stdout, sys.stderr
        try:
            sys.stdout = sys.stderr = None
            self.assertFalse(console.enable())
        finally:
            sys.stdout, sys.stderr = saved

    def test_safe_downgrades_every_character_a_code_page_rejects(self):
        text = console.safe("round 1 — done · ⏸ paused → ✅ ⏹ ▸ … ≥ – ─")
        for codec in ("cp949", "cp1252", "cp437", "ascii"):
            text.encode(codec)                            # must not raise on any of them

    def test_safe_keeps_the_message_readable(self):
        self.assertEqual(console.safe("a — b"), "a - b")
        self.assertEqual(console.safe("x → y"), "x -> y")
        self.assertEqual(console.safe("⏸ wait"), "[paused] wait")

    def test_a_redirected_run_survives_the_characters_the_logs_contain(self):
        # Simulates `PG-Label.exe > log.txt` on a machine whose ANSI code page is cp949: the
        # child's stdout is a PIPE, so Python would fall back to that encoding without console.
        script = (
            "import sys;"
            "sys.path.insert(0, %r);"
            "from pglabel import console; console.enable();"
            "print('round 1 \\u2014 done \\u00b7 \\u23f8 \\u2705 \\u2192')" % str(ROOT)
        )
        env = dict(os.environ)
        env.pop("PYTHONIOENCODING", None)
        env.pop("PYTHONUTF8", None)
        env["PYTHONLEGACYWINDOWSSTDIO"] = "1"             # ignored off Windows, harmless
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr.decode("utf-8", "replace"))
        self.assertIn(b"round 1", out.stdout)

    def test_every_tool_still_starts_after_the_utf8_wiring(self):
        # A module-level call added to the wrong file is a NameError at import time, which the
        # --help check catches immediately and a source grep would not.
        for tool in ("train_and_predict", "gen_noise_crops", "train_validator",
                     "precompute_overlays"):
            out = subprocess.run([sys.executable, str(ROOT / "tools" / f"{tool}.py"), "--help"],
                                 capture_output=True, text=True, timeout=120)
            self.assertEqual(out.returncode, 0, f"{tool}: {out.stderr[-300:]}")

    def test_every_entry_point_enables_utf8_before_printing(self):
        entry_points = {
            "pglabel/desktop.py": "console.enable()",
            "packaging/build.py": "console.enable()",
            "packaging/make_portable_zip.py": "console.enable()",
            "packaging/verify_build.py": "console.enable()",
            "packaging/winsetup.py": "console.enable()",
            "tools/train_and_predict.py": "common.enable_utf8_output()",
            "tools/gen_noise_crops.py": "common.enable_utf8_output()",
            "tools/train_validator.py": "common.enable_utf8_output()",
            "tools/precompute_overlays.py": "enable_utf8_output()",
        }
        for rel, call in entry_points.items():
            self.assertIn(call, (ROOT / rel).read_text(encoding="utf-8"),
                          f"{rel} prints without forcing UTF-8 first")


class TestWindowsFilesystem(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="winfs-"))

    def tearDown(self):
        for p in self.tmp.rglob("*"):
            try:
                os.chmod(p, stat.S_IWRITE)
            except OSError:
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_read_only_files_do_not_block_removal(self):
        # Virtual environments and extracted runtimes are full of these; plain rmtree raises
        # PermissionError on Windows and leaves the build half-cleaned.
        tree = self.tmp / "venv" / "Lib"
        tree.mkdir(parents=True)
        target = tree / "locked.pyd"
        target.write_text("binary")
        os.chmod(target, stat.S_IREAD)
        self.assertTrue(fsutil.remove_tree(self.tmp / "venv"))
        self.assertFalse((self.tmp / "venv").exists())

    def test_removing_a_missing_tree_is_success_not_an_error(self):
        self.assertTrue(fsutil.remove_tree(self.tmp / "never-existed"))

    def test_remove_file_clears_the_read_only_bit(self):
        f = self.tmp / "ro.txt"
        f.write_text("x")
        os.chmod(f, stat.S_IREAD)
        self.assertTrue(fsutil.remove_file(f))
        self.assertFalse(f.exists())

    def test_replace_file_overwrites_an_existing_destination(self):
        # Path.rename raises FileExistsError on Windows here; os.replace is the portable form.
        src, dst = self.tmp / "new.log", self.tmp / "old.log"
        src.write_text("fresh")
        dst.write_text("stale")
        self.assertTrue(fsutil.replace_file(src, dst))
        self.assertEqual(dst.read_text(), "fresh")
        self.assertFalse(src.exists())

    def test_no_packaging_script_deletes_a_tree_unguarded(self):
        offenders = []
        for rel in ("packaging/build.py", "packaging/make_portable_zip.py",
                    "packaging/winsetup.py"):
            for n, line in enumerate((ROOT / rel).read_text(encoding="utf-8").splitlines(), 1):
                if "shutil.rmtree(" in line and "ignore_errors" not in line:
                    offenders.append(f"{rel}:{n}")
        self.assertEqual(offenders, [],
                         "these would raise PermissionError on a Windows read-only file")


class TestWindowsShellIntegration(unittest.TestCase):
    """The generated launchers and installer fragments, checked as text."""

    def test_the_vbs_shortcut_is_written_in_an_encoding_cscript_reads(self):
        # cscript treats a .vbs as ANSI unless it finds a UTF-16 BOM, so a UTF-8 file with a
        # non-ASCII install path (C:\Users\<Korean name>\...) yields a broken shortcut.
        src = (ROOT / "packaging" / "winsetup.py").read_text(encoding="utf-8")
        self.assertIn('encoding="utf-16"', src)
        self.assertNotIn('_VBS_SHORTCUT.format(**fields), encoding="utf-8"', src)

    def test_batch_files_are_ascii_only(self):
        # .bat/.cmd are read in the console's OEM code page; a non-ASCII byte becomes mojibake
        # or, in a label, a broken command.
        for bat in sorted((ROOT / "packaging").glob("*.bat")):
            try:
                bat.read_text(encoding="ascii")
            except UnicodeDecodeError as e:
                self.fail(f"{bat.name} contains non-ASCII: {e}")

    def test_the_installer_script_is_ascii_safe_where_it_matters(self):
        iss = (ROOT / "packaging" / "installer.iss").read_text(encoding="utf-8")
        self.assertIn("PrivilegesRequired=lowest", iss)   # per-user: no UAC on lab machines
        self.assertIn("_internal", iss)                   # the onedir payload is cleaned up

    def test_the_uninstaller_expects_removal_to_fail_and_schedules_a_cleanup(self):
        # On Windows the uninstaller runs from inside the folder it deletes, so a partial
        # removal is the NORMAL outcome. It must branch on the result, not on an exception
        # that the removal helper deliberately never raises.
        src = (ROOT / "packaging" / "winsetup.py").read_text(encoding="utf-8")
        self.assertIn("removed = _fsutil().remove_tree(dest)", src)
        self.assertIn("elif IS_WINDOWS:", src)
        self.assertIn("pglabel_cleanup.cmd", src)

    def test_generated_cmd_files_use_the_ansi_code_page(self):
        # A .cmd written as UTF-8 hands rmdir a mojibake path when the install folder contains
        # non-ASCII characters, and then silently deletes nothing.
        src = (ROOT / "packaging" / "winsetup.py").read_text(encoding="utf-8")
        self.assertIn("def _write_cmd(", src)
        self.assertIn('"mbcs"', src)
        self.assertNotIn('_SELF_DELETE.format(dest=dest), encoding="utf-8"', src)

    def test_windows_drive_listing_is_callable_off_windows(self):
        from pglabel import dataset_setup
        self.assertIsInstance(dataset_setup._windows_drives(), list)

    def test_the_process_group_flags_exist_on_windows(self):
        # Referenced only inside `if IS_WINDOWS:`, so a typo would surface on a user's PC.
        src = (ROOT / "pglabel" / "training.py").read_text(encoding="utf-8")
        self.assertIn("CREATE_NEW_PROCESS_GROUP", src)
        self.assertIn("CREATE_NO_WINDOW", src)
        self.assertIn("taskkill", src)                    # Windows has no killpg


class TestWindowsDataLocations(unittest.TestCase):
    """A packaged install must write user data outside its own folder on Windows."""

    def test_a_packaged_install_asks_windows_for_a_per_user_folder(self):
        # os.name cannot be faked here — pathlib refuses to build a WindowsPath on POSIX — so
        # this pins the branch itself: packaged + nt must resolve through %LOCALAPPDATA%.
        src = (ROOT / "pglabel" / "paths.py").read_text(encoding="utf-8")
        self.assertIn('if os.name == "nt":', src)
        self.assertIn('os.environ.get("LOCALAPPDATA")', src)
        self.assertIn("XDG_DATA_HOME", src)               # and a POSIX equivalent

    def test_the_data_root_override_is_honoured_on_every_platform(self):
        from pglabel import paths
        saved = os.environ.get("PGLABEL_DATA_DIR")
        tmp = Path(tempfile.mkdtemp(prefix="appdata-"))
        try:
            os.environ["PGLABEL_DATA_DIR"] = str(tmp)
            paths._SETTINGS_CACHE = None
            self.assertEqual(paths.data_root(), tmp)
            self.assertTrue(str(paths.workspace_dir()).startswith(str(tmp)))
        finally:
            if saved is None:
                os.environ.pop("PGLABEL_DATA_DIR", None)
            else:
                os.environ["PGLABEL_DATA_DIR"] = saved
            paths._SETTINGS_CACHE = None
            shutil.rmtree(tmp, ignore_errors=True)

    def test_label_files_never_carry_windows_line_endings(self):
        from pglabel.labelio import save_yolo
        tmp = Path(tempfile.mkdtemp(prefix="crlf-"))
        try:
            save_yolo(tmp, "a", [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}])
            self.assertNotIn(b"\r\n", (tmp / "a.txt").read_bytes())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
