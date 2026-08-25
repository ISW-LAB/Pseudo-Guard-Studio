"""The training entry points: their command lines, and the helpers they share."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from support import ROOT, make_dataset, write_boxes, requires_torch  # noqa: E402
from tools import common                                             # noqa: E402

TOOLS = ["train_and_predict", "gen_noise_crops", "train_validator", "precompute_overlays"]


class TestToolCLIs(unittest.TestCase):
    def test_every_tool_has_a_working_help(self):
        for name in TOOLS:
            r = subprocess.run([sys.executable, str(ROOT / "tools" / f"{name}.py"), "--help"],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, f"{name}: {r.stderr[-300:]}")
            self.assertIn("--device", r.stdout + r.stderr, name)

    def test_device_defaults_to_auto_everywhere(self):
        for name in TOOLS:
            r = subprocess.run([sys.executable, str(ROOT / "tools" / f"{name}.py"), "--help"],
                               capture_output=True, text=True, timeout=120)
            self.assertIn("auto", r.stdout, f"{name} should default --device to auto")

    def test_missing_required_arguments_fail_fast(self):
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "train_and_predict.py")],
                           capture_output=True, text=True, timeout=120)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("required", (r.stderr + r.stdout).lower())


class TestCommonHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tools-"))
        self.images, self.labels = make_dataset(self.tmp, n_images=4)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_repo_root_is_importable_for_the_library(self):
        root = common.repo_root()
        self.assertTrue((root / "pseudoguard" / "__init__.py").exists())
        self.assertTrue((root / "pgcount" / "__init__.py").exists())

    def test_read_boxes_parses_yolo_rows(self):
        write_boxes(self.labels, "img00", [(1, 0.5, 0.5, 0.2, 0.3)])
        boxes = common.read_boxes(self.labels / "img00.txt")
        self.assertEqual(boxes[0][0], 1)
        self.assertAlmostEqual(boxes[0][3], 0.2)

    def test_read_boxes_on_a_missing_file_is_empty(self):
        self.assertEqual(common.read_boxes(self.labels / "nope.txt"), [])

    def test_scope_human_honours_the_ownership_manifest(self):
        for i in range(3):
            write_boxes(self.labels, f"img{i:02d}", [(0, 0.5, 0.5, 0.2, 0.2)])
        (self.labels / ".pglabel_human.json").write_text(json.dumps(["img00.jpg"]))
        human, note = common.seed_images(self.images, self.labels, "human")
        every, _ = common.seed_images(self.images, self.labels, "all")
        self.assertEqual([n for n, _ in human], ["img00.jpg"])
        self.assertEqual(len(every), 3)
        self.assertIn("human", note)

    def test_no_manifest_means_every_labeled_image_counts_as_human(self):
        write_boxes(self.labels, "img01", [(0, 0.5, 0.5, 0.2, 0.2)])
        seed, note = common.seed_images(self.images, self.labels, "human")
        self.assertEqual(len(seed), 1)
        self.assertIn("all labeled", note)

    def test_stage_split_writes_an_ultralytics_layout(self):
        write_boxes(self.labels, "img00", [(0, 0.5, 0.5, 0.2, 0.2)])
        seed, _ = common.seed_images(self.images, self.labels, "all")
        root = self.tmp / "ds"
        common.stage_split(root, "train", [(self.images / n, b) for n, b in seed])
        self.assertTrue((root / "images" / "train" / "img00.jpg").exists())
        self.assertTrue((root / "labels" / "train" / "img00.txt").exists())

    def test_stage_split_can_collapse_to_one_class(self):
        write_boxes(self.labels, "img00", [(3, 0.5, 0.5, 0.2, 0.2)])
        seed, _ = common.seed_images(self.images, self.labels, "all")
        root = self.tmp / "ds1"
        common.stage_split(root, "train", [(self.images / n, b) for n, b in seed],
                           class_agnostic=True)
        self.assertTrue((root / "labels" / "train" / "img00.txt")
                        .read_text().startswith("0 "))

    def test_preprocess_none_is_a_no_op_returning_the_source_dir(self):
        out = common.ensure_preprocessed(self.images, ["img00.jpg"], self.tmp / "prep", "none")
        self.assertEqual(out, self.images)

    def test_build_noise_config_drops_ui_only_keys(self):
        cfg = common.build_noise_config({"deviation_per_image": 12, "negative_rule": "refined",
                                         "deviation_shift": 0.5})
        self.assertEqual(cfg.negative_rule, "refined")
        self.assertAlmostEqual(cfg.deviation_shift, 0.5)
        self.assertFalse(hasattr(cfg, "deviation_per_image"))

    def test_build_noise_config_coerces_json_lists_to_tuples(self):
        cfg = common.build_noise_config({"empty_size_jitter": [0.6, 1.4]})
        self.assertEqual(cfg.empty_size_jitter, (0.6, 1.4))

    def test_build_noise_config_rejects_impossible_ratios(self):
        with self.assertRaises(ValueError):
            common.build_noise_config({"good_crop_ratio": 0.5, "empty_crop_ratio": 0.5,
                                       "deviation_ratio": 0.5})

    def test_missing_rule_file_falls_back_to_defaults(self):
        cfg = common.build_noise_config(self.tmp / "not-there.json")
        self.assertAlmostEqual(cfg.deviation_shift, 0.80)

    def test_resolve_device_never_returns_an_absent_gpu(self):
        from pseudoguard import device
        resolved = common.resolve_device("cuda:0")
        self.assertEqual(resolved.startswith("cuda"), device.cuda_available())


class TestSyntheticOverlay(unittest.TestCase):
    """precompute_overlays --synthetic runs the whole selection path with no model at all."""

    def test_synthetic_overlay_is_produced_and_loadable(self):
        tmp = Path(tempfile.mkdtemp(prefix="overlay-tool-"))
        out = tmp / "overlay.json"
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "precompute_overlays.py"),
                            "--synthetic", "--n-images", "4", "--out", str(out)],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-300:])
        data = json.loads(out.read_text())
        self.assertEqual(len(data), 4)
        first = next(iter(data.values()))
        self.assertTrue(all({"box_id", "box_xyxy", "det_conf", "p_good"} <= set(c) for c in first))
        shutil.rmtree(tmp, ignore_errors=True)

    def test_real_mode_without_models_errors_instead_of_crashing(self):
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "precompute_overlays.py"),
                            "--out", "/tmp/x.json"], capture_output=True, text=True, timeout=120)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--detector", r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main()
