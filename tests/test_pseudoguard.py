"""The algorithm library: settings validation, device fallback, and the negative-crop rule."""

import math
import random
import unittest

from support import requires_torch                                  # noqa: E402
from pseudoguard import device                                       # noqa: E402
from pseudoguard.config import SHIFT_BAND, NoiseGenerationConfig     # noqa: E402


class TestConfig(unittest.TestCase):
    def test_published_default_is_a_0_80_centre_shift(self):
        cfg = NoiseGenerationConfig()
        self.assertAlmostEqual(cfg.deviation_shift, 0.80)
        self.assertEqual(tuple(round(v, 6) for v in cfg.deviation_shift_range),
                         (round(0.80 - SHIFT_BAND, 6), round(0.80 + SHIFT_BAND, 6)))

    def test_scalar_shift_expands_into_a_band(self):
        self.assertEqual(tuple(round(v, 6) for v in
                               NoiseGenerationConfig(deviation_shift=0.4).deviation_shift_range),
                         (0.28, 0.52))

    def test_an_explicit_band_wins_and_updates_the_scalar(self):
        cfg = NoiseGenerationConfig(deviation_shift_range=(0.73, 0.97))
        self.assertEqual(cfg.deviation_shift_range, (0.73, 0.97))
        self.assertAlmostEqual(cfg.deviation_shift, 0.85)

    def test_ratios_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            NoiseGenerationConfig(good_crop_ratio=0.5, empty_crop_ratio=0.5, deviation_ratio=0.5)

    def test_unknown_negative_rule_is_rejected(self):
        with self.assertRaises(ValueError):
            NoiseGenerationConfig(negative_rule="sideways")

    def test_describe_records_the_whole_rule(self):
        d = NoiseGenerationConfig(negative_rule="refined").describe()
        for key in ("negative_rule", "deviation_shift", "deviation_shift_range",
                    "negative_iou_reject", "good_crop_jitter"):
            self.assertIn(key, d)


class TestDevice(unittest.TestCase):
    def test_auto_never_returns_cuda_when_it_is_unavailable(self):
        resolved = device.resolve("auto")
        if not device.cuda_available():
            self.assertEqual(resolved, "cpu")
        else:
            self.assertTrue(resolved.startswith("cuda"))

    def test_explicit_cuda_degrades_instead_of_crashing(self):
        if not device.cuda_available():
            self.assertEqual(device.resolve("cuda:0", log=None), "cpu")

    def test_cpu_and_other_devices_pass_through(self):
        self.assertEqual(device.resolve("cpu"), "cpu")
        self.assertEqual(device.resolve("mps"), "mps")

    def test_none_is_treated_as_auto(self):
        self.assertIn(device.resolve(None), ("cpu", "cuda:0"))


@requires_torch
class TestNoiseGenerator(unittest.TestCase):
    """The geometry the paper describes, checked numerically."""

    def setUp(self):
        import torch
        from PIL import Image
        self.Image = Image
        self.img = Image.new("RGB", (640, 480))
        self.gt = torch.tensor([[200.0, 150.0, 300.0, 250.0]])       # one 100x100 box

    def _generator(self, **kw):
        from pseudoguard.data.noise_generator import NoiseGenerator
        return NoiseGenerator(NoiseGenerationConfig(**kw), device="cpu")

    def _shift_fractions(self, generator, n=200):
        random.seed(0)
        out = []
        for b in generator._collect_deviated_boxes(self.img, self.gt, n):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            out.append(math.hypot(float(cx) - 250.0, float(cy) - 200.0) / 100.0)
        return out

    def test_refined_deviation_lands_in_the_configured_band(self):
        fracs = self._shift_fractions(self._generator(negative_rule="refined"))
        self.assertTrue(fracs)
        self.assertGreaterEqual(min(fracs), 0.80 - SHIFT_BAND - 0.01)
        self.assertLessEqual(max(fracs), 0.80 + SHIFT_BAND + 0.01)
        self.assertAlmostEqual(sum(fracs) / len(fracs), 0.80, delta=0.05)

    def test_a_smaller_shift_produces_harder_negatives(self):
        near = self._shift_fractions(self._generator(negative_rule="refined", deviation_shift=0.4))
        far = self._shift_fractions(self._generator(negative_rule="refined", deviation_shift=1.2))
        self.assertLess(sum(near) / len(near), sum(far) / len(far))

    def test_baseline_rule_is_unchanged_and_more_aggressive(self):
        baseline = self._shift_fractions(self._generator(negative_rule="baseline"))
        refined = self._shift_fractions(self._generator(negative_rule="refined"))
        self.assertGreater(sum(baseline) / len(baseline), sum(refined) / len(refined))

    def test_empty_boxes_avoid_the_annotated_object(self):
        from pseudoguard.utils.box_ops import box_iou
        import torch
        generator = self._generator(negative_rule="refined")
        boxes = generator._collect_empty_boxes(self.img, self.gt, 50)
        self.assertTrue(boxes)
        for b in boxes:
            iou = float(box_iou(torch.as_tensor(b).unsqueeze(0), self.gt).max())
            self.assertLess(iou, 0.75)

    def test_crops_come_back_at_the_validator_input_size(self):
        generator = self._generator()
        for crops in (generator.good_crops(self.img, self.gt),
                      generator.empty_crops(self.img, self.gt, 3),
                      generator.deviated_crops(self.img, self.gt, 3)):
            self.assertTrue(crops)
            self.assertEqual(crops[0].size, (256, 256))

    def test_generator_falls_back_to_cpu_when_cuda_is_absent(self):
        generator = self._generator()
        if not device.cuda_available():
            self.assertEqual(generator.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
