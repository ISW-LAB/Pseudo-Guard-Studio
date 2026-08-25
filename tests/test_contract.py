"""Contracts that hold the pieces together, and that nothing else would notice breaking.

Two of them:

    the UI ↔ server route contract — the browser calls paths by string. Rename a route and the
    Python tests still pass while the app is dead in the browser, so the two lists are compared
    directly here.

    the packaged path contract — a frozen build resolves every root differently (a read-only
    bundle, a per-user data folder). Getting that wrong is invisible from a checkout and fatal
    once installed, so the frozen layout is simulated rather than trusted.
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from support import ROOT                                            # noqa: E402
from pglabel import api, paths                                      # noqa: E402


def ui_api_paths() -> set:
    """Every /api/... string the shipped UI can call."""
    text = (ROOT / "pglabel" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    text += (ROOT / "pglabel" / "static" / "index.html").read_text(encoding="utf-8")
    return {m.group(0) for m in re.finditer(r"/api/[A-Za-z0-9_/]*", text)}


def served_routes() -> tuple[set, list]:
    exact = set(api.GET_PUBLIC) | set(api.GET_DATASET) | set(api.POST_PUBLIC) | set(api.POST_DATASET)
    prefixes = [p for p, _ in api.GET_DATASET_PREFIX] + [p for p, _ in api.POST_DATASET_PREFIX]
    return exact, prefixes


class TestUIContract(unittest.TestCase):
    def test_every_endpoint_the_ui_calls_is_served(self):
        exact, prefixes = served_routes()
        missing = [p for p in sorted(ui_api_paths())
                   if p not in exact and not any(p.startswith(pre) for pre in prefixes)]
        self.assertEqual(missing, [], "the UI calls routes the server does not serve")

    def test_no_route_is_dead(self):
        # A route nothing calls is either a leftover or a missing UI feature; both are worth
        # noticing at the moment it happens.
        called = ui_api_paths()
        exact, _prefixes = served_routes()
        dead = [r for r in sorted(exact) if not any(c.startswith(r) for c in called)]
        self.assertEqual(dead, [], "these routes are served but never called")

    def test_the_ui_loads_its_assets_from_the_served_prefix(self):
        html = (ROOT / "pglabel" / "static" / "index.html").read_text(encoding="utf-8")
        for asset in ("/static/css/app.css", "/static/js/app.js"):
            self.assertIn(asset, html)
            self.assertTrue((ROOT / "pglabel" / asset.lstrip("/")).is_file(), asset)


class TestFrozenPaths(unittest.TestCase):
    """Simulate a PyInstaller onedir install and check every root lands where the spec puts it."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="frozen-"))
        self.install = self.tmp / "PG-Label"           # the folder holding the executable
        self.bundle = self.install / "_internal"       # PyInstaller's _MEIPASS for onedir
        for rel in ("pglabel/static", "tools", "pgcount", "pseudoguard", "demo/images"):
            (self.bundle / rel).mkdir(parents=True, exist_ok=True)
        (self.bundle / "pseudoguard" / "__init__.py").write_text("")
        (self.bundle / "pglabel" / "static" / "index.html").write_text("<html></html>")
        (self.install / "PG-Label.exe").write_text("")
        self.data = self.tmp / "userdata"

        self._saved = (paths.FROZEN, paths.bundle_root, paths.install_root)
        paths.FROZEN = True
        paths.bundle_root = lambda: self.bundle
        paths.install_root = lambda: self.install
        paths._SETTINGS_CACHE = None
        import os
        self._env = os.environ.get("PGLABEL_DATA_DIR")
        os.environ["PGLABEL_DATA_DIR"] = str(self.data)

    def tearDown(self):
        import os
        import shutil
        paths.FROZEN, paths.bundle_root, paths.install_root = self._saved
        paths._SETTINGS_CACHE = None
        if self._env is None:
            os.environ.pop("PGLABEL_DATA_DIR", None)
        else:
            os.environ["PGLABEL_DATA_DIR"] = self._env
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_assets_come_from_the_sealed_bundle(self):
        self.assertEqual(paths.static_dir(), self.bundle / "pglabel" / "static")
        self.assertEqual(paths.tools_dir(), self.bundle / "tools")
        self.assertEqual(paths.demo_dir(), self.bundle / "demo")

    def test_the_algorithm_library_is_found_inside_the_bundle(self):
        # If this returns None the packaged app silently ships without a Train button.
        self.assertEqual(paths.research_root(), self.bundle.resolve())

    def test_user_data_never_lands_inside_the_bundle(self):
        # The install folder is replaced on upgrade and deleted on uninstall — writing labels
        # there would lose a user's annotation work.
        for p in (paths.workspace_dir(), paths.settings_path(), paths.log_path()):
            self.assertFalse(str(p).startswith(str(self.bundle)), p)
            self.assertTrue(str(p).startswith(str(self.data)), p)

    def test_packaged_is_true_when_frozen(self):
        self.assertTrue(paths.packaged())

    def test_a_dataset_folder_beside_the_executable_is_picked_up(self):
        self.assertIsNone(paths.datasets_root())
        (self.install / "datasets").mkdir()
        self.assertEqual(paths.datasets_root(), (self.install / "datasets").resolve())

    def test_describe_reports_a_complete_picture(self):
        info = paths.describe()
        for key in ("version", "frozen", "packaged", "bundle_root", "install_root",
                    "data_root", "static", "tools", "workspace", "research_root"):
            self.assertIn(key, info)
        self.assertTrue(info["frozen"])
        self.assertTrue(json.dumps(info))          # --where must always be printable JSON


if __name__ == "__main__":
    unittest.main()
