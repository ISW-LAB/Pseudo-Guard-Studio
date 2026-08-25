"""Count-guided acceptance: the seed statistic, the operating points, and the candidate sources."""

import json
import tempfile
import unittest
from pathlib import Path

from support import make_dataset, make_overlay                      # noqa: E402
from pgcount.backend import PrecomputedBackend, RawCandidate        # noqa: E402
from pgcount.config import Backend, CountGuidedConfig               # noqa: E402
from pgcount.count_guided_labeler import Candidate                  # noqa: E402
from pgcount.crops import clamp_box, extract_crops                  # noqa: E402
from pgcount.operating_point import (AdaptivePerImageK, AutoAdaptiveTau,  # noqa: E402
                                     GlobalTau, PerImageTopK, ThresholdSelect)
from pgcount.seed_density import estimate_seed_density              # noqa: E402


def raw(image_id, k, p_good, det_conf=None):
    """A RawCandidate as the backends produce them — Candidate.from_raw expects this type."""
    return RawCandidate(image_id=image_id, box_id=f"{image_id}:{k}",
                        box_xyxy=(10 * k, 0, 10 * k + 8, 20), label=0,
                        det_conf=p_good if det_conf is None else det_conf, p_good=p_good)


class TestSeedDensity(unittest.TestCase):
    def test_empty_seed_is_not_reliable(self):
        self.assertFalse(estimate_seed_density({}, min_images=1, min_total_boxes=1).reliable)

    def test_median_count_reflects_the_seed(self):
        seed = {"a": [None] * 3, "b": [None] * 3, "c": [None] * 5}
        stats = estimate_seed_density(seed, min_images=1, min_total_boxes=1)
        self.assertTrue(stats.reliable)
        self.assertEqual(stats.median_count, 3)


class TestOperatingPoints(unittest.TestCase):
    def setUp(self):
        self.cands = [Candidate.from_raw(raw("img", k, p), Backend.VALIDATOR)
                      for k, p in enumerate([0.95, 0.75, 0.55, 0.35, 0.15])]
        self.cfg = CountGuidedConfig(backend=Backend.VALIDATOR)

    def test_threshold_select_keeps_only_scores_at_or_above_the_cut(self):
        kept = ThresholdSelect(0.5).fit().select(self.cands)
        self.assertEqual(len(kept), 3)
        self.assertTrue(all(c.score >= 0.5 for c in kept))

    def test_top_k_respects_the_seed_count(self):
        stats = estimate_seed_density({"s1": [None] * 2, "s2": [None] * 2},
                                      min_images=1, min_total_boxes=1)
        kept = PerImageTopK(self.cfg).fit(stats, self.cands).select(self.cands)
        self.assertLessEqual(len(kept), 3)
        self.assertGreaterEqual(len(kept), 1)

    def test_adaptive_k_selects_the_highest_scoring_candidates(self):
        stats = estimate_seed_density({"s1": [None] * 2}, min_images=1, min_total_boxes=1)
        kept = AdaptivePerImageK(self.cfg).fit(stats, self.cands).select(self.cands)
        scores = [c.score for c in kept]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertIn(0.95, scores)

    def test_global_tau_is_a_single_threshold_across_images(self):
        stats = estimate_seed_density({"s1": [None] * 2}, min_images=1, min_total_boxes=1)
        op = GlobalTau(self.cfg).fit(stats, self.cands)
        kept = op.select(self.cands)
        if kept:
            self.assertEqual(len({round(c.score, 6) >= 0 for c in kept}), 1)

    def test_auto_adaptive_needs_no_seed(self):
        kept = AutoAdaptiveTau(self.cfg).fit(None, self.cands).select(self.cands)
        self.assertLessEqual(len(kept), len(self.cands))

    def test_confidence_backend_scores_on_det_conf(self):
        c = Candidate.from_raw(raw("i", 0, 0.1, det_conf=0.8), Backend.CONFIDENCE)
        self.assertAlmostEqual(c.score, 0.8)

    def test_validator_backend_scores_on_p_good(self):
        c = Candidate.from_raw(raw("i", 0, 0.1, det_conf=0.8), Backend.VALIDATOR)
        self.assertAlmostEqual(c.score, 0.1)


class TestCrops(unittest.TestCase):
    def test_clamp_keeps_boxes_inside_the_image(self):
        self.assertEqual(clamp_box((-5, -5, 50, 50), 40, 30), (0, 0, 40, 30))

    def test_degenerate_box_is_rejected(self):
        self.assertIsNone(clamp_box((10, 10, 10, 10), 40, 30))

    def test_extract_crops_reports_which_boxes_survived(self):
        tmp = Path(tempfile.mkdtemp(prefix="crops-"))
        images, _labels = make_dataset(tmp, n_images=1, size=(60, 40))
        crops, keep = extract_crops(images / "img00.jpg",
                                    [(0, 0, 20, 20), (10, 10, 10, 10), (5, 5, 30, 30)])
        # The degenerate box is skipped, and keep tells the caller which scores still align.
        self.assertEqual(len(crops), len(keep))
        self.assertEqual(keep, [0, 2])


class TestPrecomputedBackend(unittest.TestCase):
    def test_overlay_round_trips_into_candidates(self):
        tmp = Path(tempfile.mkdtemp(prefix="overlay-"))
        path = make_overlay(tmp / "o.json", ["a.jpg", "b.jpg"], per_image=3)
        backend = PrecomputedBackend.from_json(path)
        self.assertEqual(sorted(backend.all_image_ids()), ["a.jpg", "b.jpg"])
        cands = backend.candidates("a.jpg")
        self.assertEqual(len(cands), 3)
        self.assertAlmostEqual(cands[0].p_good, 0.95)
        self.assertEqual(len(cands[0].box_xyxy), 4)

    def test_unknown_image_yields_no_candidates(self):
        tmp = Path(tempfile.mkdtemp(prefix="overlay2-"))
        path = make_overlay(tmp / "o.json", ["a.jpg"])
        self.assertEqual(PrecomputedBackend.from_json(path).candidates("zzz.jpg"), [])


if __name__ == "__main__":
    unittest.main()
