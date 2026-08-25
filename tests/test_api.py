"""The HTTP surface, exercised against a real server on a throwaway dataset."""

import unittest

from support import ServerCase                                       # noqa: E402
from pglabel import state                                            # noqa: E402


class TestManualMode(ServerCase):
    """No AI attached: the annotator must be fully usable on its own."""

    overlay = False

    def test_index_and_static_assets_are_served(self):
        code, body = self.get("/")
        self.assertEqual(code, 200)
        self.assertIn(b"<html", body.lower())
        for asset, needle in (("/static/css/app.css", b"{"), ("/static/js/app.js", b"function")):
            code, body = self.get(asset)
            self.assertEqual(code, 200, asset)
            self.assertIn(needle, body)

    def test_static_path_traversal_is_refused(self):
        code, _ = self.get("/static/../../pglabel/cli.py")
        self.assertEqual(code, 404)

    def test_config_reports_the_open_dataset(self):
        code, cfg = self.get("/api/config")
        self.assertEqual(code, 200)
        self.assertFalse(cfg["needs_setup"])
        # Tests in this class share one server, and one of them adds a class — assert the
        # dataset's own classes are there rather than that nothing was ever appended.
        self.assertEqual(cfg["classes"][:2], ["cat", "dog"])
        self.assertEqual(len(cfg["images"]), len(self.names))
        self.assertFalse(cfg["has_ai"])

    def test_status_separates_human_from_auto_and_unlabeled(self):
        code, st = self.get("/api/status")
        self.assertEqual(code, 200)
        self.assertGreaterEqual(st["human"], self.seed_images)   # another test may add one
        self.assertEqual(st["total"], len(self.names))
        owners = set(st["owner"].values())
        self.assertIn("human", owners)
        self.assertIn("none", owners)

    def test_saving_boxes_makes_an_image_human_owned(self):
        target = self.names[-1]
        code, out = self.post(f"/api/labels/{target}",
                              {"boxes": [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]})
        self.assertEqual(code, 200)
        self.assertEqual(out["n"], 1)
        _code, st = self.get("/api/status")
        self.assertEqual(st["owner"][target], "human")
        _code, labels = self.get(f"/api/labels/{target}")
        self.assertEqual(len(labels["boxes"]), 1)
        self.assertEqual(labels["width"], 120)

    def test_adding_a_class_and_rejecting_duplicates(self):
        code, out = self.post("/api/classes", {"name": "bird"})
        self.assertEqual(code, 200)
        self.assertIn("bird", out["classes"])
        code, _ = self.post("/api/classes", {"name": "bird"})
        self.assertEqual(code, 400)
        code, _ = self.post("/api/classes", {"name": "   "})
        self.assertEqual(code, 400)

    def test_automate_without_a_backend_is_refused_clearly(self):
        code, out = self.post("/api/automate_all", {})
        self.assertEqual(code, 400)
        self.assertIn("manual mode", out["detail"])

    def test_coco_export_matches_the_saved_labels(self):
        code, out = self.get("/api/export?fmt=coco")
        self.assertEqual(code, 200)
        self.assertEqual(out["images"], len(self.names))
        self.assertGreaterEqual(out["annotations"], self.seed_images * 2)
        code, out = self.get("/api/export?fmt=yolo")
        self.assertEqual(out["format"], "yolo")

    def test_unknown_routes_are_404_not_500(self):
        self.assertEqual(self.get("/api/nope")[0], 404)
        self.assertEqual(self.post("/api/nope", {})[0], 404)

    def test_filenames_with_spaces_round_trip(self):
        # The bundled sample dataset has names like "1 (115).jpg"; the client percent-encodes
        # them and the server unquotes the suffix. Regression cover for that seam.
        import shutil, urllib.parse
        spaced = "a photo (2).jpg"
        shutil.copy2(self.images / self.names[0], self.images / spaced)
        try:
            quoted = urllib.parse.quote(spaced)
            code, out = self.post(f"/api/labels/{quoted}",
                                  {"boxes": [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]})
            self.assertEqual(code, 200)
            code, labels = self.get(f"/api/labels/{quoted}")
            self.assertEqual(code, 200)
            self.assertEqual(len(labels["boxes"]), 1)
            self.assertEqual(self.get(f"/api/file/{quoted}")[0], 200)
        finally:
            (self.images / spaced).unlink(missing_ok=True)
            (self.labels / "a photo (2).txt").unlink(missing_ok=True)

    def test_missing_image_is_404(self):
        self.assertEqual(self.get("/api/labels/not-here.jpg")[0], 404)
        self.assertEqual(self.get("/api/file/not-here.jpg")[0], 404)

    def test_training_endpoints_refuse_when_training_is_disabled(self):
        code, out = self.post("/api/train", {})
        self.assertEqual(code, 400)
        self.assertIn("unavailable", out["detail"])
        code, _ = self.post("/api/train/stop", {})
        self.assertEqual(code, 400)
        code, _ = self.post("/api/train/confirm", {})
        self.assertEqual(code, 400)


class TestWithOverlay(ServerCase):
    """A precomputed overlay is a full AI backend — no torch, no GPU, deterministic."""

    overlay = True

    def test_config_reports_an_ai(self):
        _code, cfg = self.get("/api/config")
        self.assertTrue(cfg["has_ai"])

    def test_candidates_are_normalised_and_carry_both_scores(self):
        code, out = self.get(f"/api/candidates/{self.names[0]}")
        self.assertEqual(code, 200)
        self.assertTrue(out["candidates"])
        for c in out["candidates"]:
            self.assertTrue(0.0 <= c["cx"] <= 1.0 and 0.0 <= c["cy"] <= 1.0)
            self.assertIn("p_good", c)
            self.assertIn("det_conf", c)

    def test_score_summary_covers_only_the_auto_targets(self):
        code, out = self.get("/api/score_summary")
        self.assertEqual(code, 200)
        self.assertEqual(out["n_images"], len(self.names) - self.seed_images)
        self.assertEqual(sum(out["p_good_hist"]), out["n_candidates"])

    def test_automate_one_image_returns_scored_boxes(self):
        target = self.names[-1]
        code, out = self.post(f"/api/automate/{target}", {"method": "pseudoguard"})
        self.assertEqual(code, 200)
        self.assertTrue(out["boxes"])
        for b in out["boxes"]:
            self.assertTrue(b["ai"])
            self.assertIn(b["band"], ("green", "amber", "red"))

    def test_automate_all_writes_labels_for_every_non_seed_image(self):
        code, out = self.post("/api/automate_all", {"method": "pseudoguard"})
        self.assertEqual(code, 200)
        self.assertEqual(out["auto_labeled"], len(self.names) - self.seed_images)
        _code, st = self.get("/api/status")
        for name in self.names[:self.seed_images]:
            self.assertEqual(st["owner"][name], "human")      # the seed is never overwritten

    def test_this_image_agrees_with_auto_label_all(self):
        # The whole point of fitting one global operating point: the two buttons must agree.
        target = self.names[-1]
        _code, single = self.post(f"/api/automate/{target}", {"method": "pseudoguard"})
        self.post("/api/automate_all", {"method": "pseudoguard"})
        _code, saved = self.get(f"/api/labels/{target}")
        self.assertEqual(len(single["boxes"]), len(saved["boxes"]))

    def test_manual_threshold_accepts_monotonically(self):
        _c, low = self.post("/api/automate_all", {"method": "manual", "thr": 0.1})
        _c, high = self.post("/api/automate_all", {"method": "manual", "thr": 0.9})
        self.assertGreaterEqual(low["total_boxes"], high["total_boxes"])

    def test_confidence_baselines_are_available_and_stricter_at_higher_cuts(self):
        _c, methods_out = self.get("/api/methods")
        ids = {m["id"] for m in methods_out["methods"]}
        self.assertTrue({"conf25", "conf50", "conf90"} <= ids)
        _c, loose = self.post("/api/automate_all", {"method": "conf25"})
        _c, strict = self.post("/api/automate_all", {"method": "conf90"})
        self.assertGreaterEqual(loose["total_boxes"], strict["total_boxes"])


class TestSetupMode(unittest.TestCase):
    """With no dataset open, only the setup-screen routes may answer."""

    @classmethod
    def setUpClass(cls):
        import tempfile, threading, shutil
        from pathlib import Path
        from pglabel import cli
        cls.tmp = Path(tempfile.mkdtemp(prefix="pglabel-setup-"))
        state.reset_for_tests()
        args = cli.build_parser().parse_args(["--no-train", "--port", "0"])
        cli.configure(args)
        state.CFG["images"] = None                      # setup mode
        cls.httpd = cli.make_server("127.0.0.1", 0)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls._shutil, cls._Path = shutil, Path

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)
        state.reset_for_tests()
        cls._shutil.rmtree(cls.tmp, ignore_errors=True)

    def _get(self, path):
        return ServerCase.get(self, path)

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def test_config_asks_for_setup(self):
        code, cfg = self._get("/api/config")
        self.assertEqual(code, 200)
        self.assertTrue(cfg["needs_setup"])

    def test_browse_and_datasets_answer_before_a_dataset_is_open(self):
        self.assertEqual(self._get("/api/browse")[0], 200)
        self.assertEqual(self._get("/api/datasets")[0], 200)

    def test_dataset_routes_are_guarded(self):
        code, out = self._get("/api/status")
        self.assertEqual(code, 400)
        self.assertIn("no dataset", out["detail"])


if __name__ == "__main__":
    unittest.main()
