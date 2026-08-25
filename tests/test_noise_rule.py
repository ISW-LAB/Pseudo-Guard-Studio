"""The negative-crop rule as the review screen exposes it."""

import math
import shutil
import tempfile
import unittest
from pathlib import Path

from support import make_dataset, write_boxes                       # noqa: E402
from pglabel import noise_rule, state                                # noqa: E402


class TestValidation(unittest.TestCase):
    def test_defaults_are_the_published_setting(self):
        d = noise_rule.defaults()
        self.assertEqual(d["negative_rule"], "refined")
        self.assertAlmostEqual(d["deviation_shift"], 0.80)
        self.assertAlmostEqual(d["negative_iou_reject"], 0.75)

    def test_ratios_are_renormalised_not_rejected(self):
        # A half-typed slider value must never crash the generator downstream.
        cfg = noise_rule.validate_noise_config(
            {"good_crop_ratio": 2, "empty_crop_ratio": 1, "deviation_ratio": 1})
        total = cfg["good_crop_ratio"] + cfg["empty_crop_ratio"] + cfg["deviation_ratio"]
        self.assertAlmostEqual(total, 1.0)
        self.assertAlmostEqual(cfg["good_crop_ratio"], 0.5)

    def test_unknown_rule_falls_back_to_baseline(self):
        self.assertEqual(noise_rule.validate_noise_config({"negative_rule": "xyz"})["negative_rule"],
                         "baseline")

    def test_shift_is_clamped_to_the_supported_range(self):
        self.assertEqual(noise_rule.validate_noise_config({"deviation_shift": 99})["deviation_shift"],
                         noise_rule.SHIFT_MAX)
        self.assertEqual(noise_rule.validate_noise_config({"deviation_shift": -5})["deviation_shift"],
                         noise_rule.SHIFT_MIN)

    def test_garbage_values_fall_back_to_defaults(self):
        cfg = noise_rule.validate_noise_config({"deviation_shift": "abc",
                                                "deviation_per_image": None})
        self.assertAlmostEqual(cfg["deviation_shift"], 0.80)
        self.assertEqual(cfg["deviation_per_image"], noise_rule.NOISE_DEFAULTS["deviation_per_image"])

    def test_unknown_keys_are_dropped(self):
        self.assertNotIn("nonsense", noise_rule.validate_noise_config({"nonsense": 1}))

    def test_shift_band_matches_the_library(self):
        from pseudoguard.config import NoiseGenerationConfig
        lo, hi = noise_rule.shift_band(0.8)
        self.assertEqual((round(lo, 6), round(hi, 6)),
                         tuple(round(v, 6) for v in NoiseGenerationConfig().deviation_shift_range))


class TestPreview(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="preview-"))
        self.images, self.labels = make_dataset(self.tmp, n_images=1, size=(400, 400))
        write_boxes(self.labels, "img00", [(0, 0.5, 0.5, 0.25, 0.25)])
        state.reset_for_tests()
        state.CFG["images"], state.CFG["labels"] = self.images, self.labels

    def tearDown(self):
        state.reset_for_tests()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_preview_returns_gt_and_deviated_boxes(self):
        out = noise_rule.preview_deviated("img00.jpg", noise_rule.defaults())
        self.assertEqual(out["width"], 400)
        self.assertEqual(len(out["gt"]), 1)
        self.assertTrue(out["deviated"])
        for b in out["deviated"]:
            self.assertTrue(0.0 <= b["cx"] <= 1.0 and 0.0 <= b["cy"] <= 1.0)

    def test_displacement_stays_inside_the_requested_band(self):
        shift = 0.8
        cfg = noise_rule.validate_noise_config({"deviation_shift": shift,
                                                "deviation_per_image": 30})
        out = noise_rule.preview_deviated("img00.jpg", cfg)
        lo, hi = noise_rule.shift_band(shift)
        for b in out["deviated"]:
            # Boxes clipped by the image edge move less than requested, so this bounds the top.
            frac = math.hypot((b["cx"] - 0.5) * 400 / 100, (b["cy"] - 0.5) * 400 / 100)
            self.assertLessEqual(frac, hi + 0.4)

    def test_a_bigger_shift_moves_boxes_further(self):
        def mean_shift(value):
            cfg = noise_rule.validate_noise_config({"deviation_shift": value,
                                                    "deviation_per_image": 30})
            out = noise_rule.preview_deviated("img00.jpg", cfg)
            if not out["deviated"]:
                return 0.0
            return sum(math.hypot(b["cx"] - 0.5, b["cy"] - 0.5) for b in out["deviated"]) \
                / len(out["deviated"])
        self.assertLess(mean_shift(0.4), mean_shift(1.2))

    def test_image_with_no_labels_yields_no_deviated_boxes(self):
        out = noise_rule.preview_deviated("img00.jpg", noise_rule.defaults())
        (self.labels / "img00.txt").write_text("")
        empty = noise_rule.preview_deviated("img00.jpg", noise_rule.defaults())
        self.assertTrue(out["deviated"])
        self.assertEqual(empty["deviated"], [])


if __name__ == "__main__":
    unittest.main()
