"""Packaging: the audit that gates a release, and the layout the spec depends on."""

import subprocess
import sys
import unittest
from pathlib import Path

from support import ROOT                                             # noqa: E402

PKG = ROOT / "packaging"


class TestVerifyBuild(unittest.TestCase):
    def test_the_source_audit_passes(self):
        # This is the same audit build.py runs; if it fails, a packaged build would be broken.
        r = subprocess.run([sys.executable, str(PKG / "verify_build.py")],
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:])
        self.assertIn("checks passed", r.stdout)


class TestSpec(unittest.TestCase):
    def setUp(self):
        self.spec = (PKG / "PG-Label.spec").read_text(encoding="utf-8")

    def test_the_heavy_stack_is_excluded_from_the_app(self):
        for name in ("torch", "ultralytics", "numpy", "cv2"):
            self.assertIn(f'"{name}"', self.spec)

    def test_lazy_imports_are_named_as_hidden_imports(self):
        for name in ("pgcount.backend", "pgcount.operating_point", "gpu_pack"):
            self.assertIn(name, self.spec)

    def test_the_trainer_ships_as_readable_py_files(self):
        # tools/ and pseudoguard/ must be datas, not frozen: another interpreter reads them.
        self.assertIn('tree(ROOT / "tools", "tools", patterns=("*.py",))', self.spec)
        self.assertIn('tree(ROOT / "pseudoguard", "pseudoguard", patterns=("*.py",))', self.spec)

    def test_the_entry_point_exists(self):
        self.assertIn('ROOT / "run_app.py"', self.spec)
        self.assertTrue((ROOT / "run_app.py").exists())


class TestPackagingScripts(unittest.TestCase):
    def test_every_packaging_script_parses(self):
        for py in sorted(PKG.glob("*.py")):
            r = subprocess.run([sys.executable, "-m", "py_compile", str(py)],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, f"{py.name}: {r.stderr[-300:]}")

    def test_build_and_release_scripts_expose_a_help(self):
        for name in ("build.py", "make_portable_zip.py", "verify_build.py"):
            r = subprocess.run([sys.executable, str(PKG / name), "--help"],
                               capture_output=True, text=True, timeout=180)
            self.assertEqual(r.returncode, 0, f"{name}: {r.stderr[-300:]}")

    def test_the_windows_installer_names_the_current_version(self):
        from pglabel import paths
        iss = (PKG / "installer.iss").read_text(encoding="utf-8")
        self.assertIn(paths.APP_VERSION, iss)


class TestPortableRelease(unittest.TestCase):
    """The staging step of the portable release, without touching the network."""

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("mpz", PKG / "make_portable_zip.py")
        self.mpz = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mpz)

    def test_copying_the_app_includes_the_nested_ui_assets(self):
        # Regression: a "static/*" pattern matches only files DIRECTLY inside static/, so css/
        # and js/ were dropped and the release shipped a page with no styles and no behaviour.
        import shutil, tempfile
        tmp = Path(tempfile.mkdtemp(prefix="release-stage-"))
        try:
            self.mpz.copy_tree(ROOT / "pglabel", tmp / "pglabel", patterns=("*.py",))
            self.mpz.copy_tree(ROOT / "pglabel" / "static", tmp / "pglabel" / "static")
            for rel in ("pglabel/static/index.html", "pglabel/static/css/app.css",
                        "pglabel/static/js/app.js", "pglabel/desktop.py"):
                self.assertTrue((tmp / rel).is_file(), rel)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_release_manifest_names_the_ui_assets(self):
        required = self.mpz.REQUIRED_IN_RELEASE + self.mpz.REQUIRED_FOR_TRAINING
        for rel in ("pglabel/static/css/app.css", "pglabel/static/js/app.js",
                    "tools/train_and_predict.py", "pseudoguard/config.py"):
            self.assertIn(rel, required)

    def test_verify_stage_refuses_an_incomplete_tree(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                self.mpz.verify_stage(Path(tmp), with_training=True)

    def test_the_cmd_fallback_keeps_arguments_separated(self):
        # "--install%*" would glue the user's own flags onto ours ("--install--dest C:\\x").
        cmd = self.mpz.CMD.format(what="Install", script="packaging\\winsetup.py",
                                  args="--install")
        self.assertIn("--install %*", cmd)
        self.assertNotIn("\\\\winsetup", cmd)          # no doubled path separator
        app = self.mpz.CMD.format(what="App", script="run_app.py", args="")
        self.assertIn("%*", app)
        self.assertIn("run_app.py", app)

    def test_the_launcher_boots_the_desktop_entry_point(self):
        body = self.mpz.BOOT.format(what="app", body=self.mpz.APP_BODY)
        self.assertIn("from pglabel.desktop import main", body)
        self.assertIn("packaging", body)          # winsetup/gpu_pack must be importable


class TestRepoLayout(unittest.TestCase):
    def test_the_repository_is_self_contained(self):
        for pkg in ("pglabel", "pgcount", "pseudoguard", "tools"):
            self.assertTrue((ROOT / pkg / "__init__.py").exists(), pkg)

    def test_no_module_points_at_the_old_research_checkout(self):
        # Assembled at runtime so this test file is not itself a match.
        needle = "1." + " experiment code"
        stale = []
        for py in ROOT.rglob("*.py"):
            if "build" in py.parts or "__pycache__" in py.parts or py.name == Path(__file__).name:
                continue
            if needle in py.read_text(encoding="utf-8", errors="replace"):
                stale.append(str(py.relative_to(ROOT)))
        self.assertEqual(stale, [], "these still reference the original research folder")

    def test_the_readme_and_requirements_exist(self):
        for name in ("README.md", "requirements.txt", "requirements-train.txt", "LICENSE"):
            self.assertTrue((ROOT / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
