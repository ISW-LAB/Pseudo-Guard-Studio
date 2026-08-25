"""The acceptance registry, and the decisions derived from the human seed."""

import shutil
import tempfile
import unittest
from pathlib import Path

from support import make_dataset, write_boxes                       # noqa: E402
from pglabel import methods, state                                   # noqa: E402


class TestRegistry(unittest.TestCase):
    def test_default_method_exists_and_is_seed_driven(self):
        self.assertIn(methods.DEFAULT_METHOD, methods.METHOD_MAP)
        self.assertTrue(methods.METHOD_MAP[methods.DEFAULT_METHOD]["seed"])

    def test_every_method_declares_a_backend_and_an_operating_point(self):
        for m in methods.METHODS:
            self.assertIn(m["backend"], ("validator", "confidence"), m["id"])
            self.assertTrue(m["op"], m["id"])

    def test_public_view_hides_internal_fields(self):
        for m in methods.public_methods():
            self.assertEqual(set(m), {"id", "label", "group", "seed"})

    def test_comparison_baselines_are_confidence_only(self):
        # The comparison is fair only if the baselines differ ONLY in the acceptance rule.
        baselines = [m for m in methods.METHODS if m["group"] == methods.GROUP_COMPARISON]
        self.assertTrue(baselines)
        self.assertTrue(all(m["backend"] == "confidence" for m in baselines))


class TestAdaptive(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="adaptive-"))
        self.images, self.labels = make_dataset(self.tmp, n_images=8)
        state.reset_for_tests()
        state.CFG["images"], state.CFG["labels"] = self.images, self.labels

    def tearDown(self):
        state.reset_for_tests()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _seed(self, boxes_per_image):
        for i, boxes in enumerate(boxes_per_image):
            name = f"img{i:02d}.jpg"
            write_boxes(self.labels, Path(name).stem, boxes)
            state.HUMAN_SET.add(name)

    def test_no_seed_leaves_containment_on_and_no_cap(self):
        methods.compute_adaptive()
        self.assertTrue(methods.containment_on())
        self.assertIsNone(state.ADAPT["max_topk"])

    def test_dense_non_nested_seed_turns_containment_on(self):
        # Six well-separated boxes per image: nested detections here are duplicates to remove.
        boxes = [[(0, 0.1 + 0.15 * k, 0.5, 0.08, 0.2) for k in range(6)] for _ in range(4)]
        self._seed(boxes)
        methods.compute_adaptive()
        self.assertTrue(methods.containment_on())
        self.assertEqual(state.ADAPT["seed_mean"], 6.0)
        self.assertIsNotNone(state.ADAPT["max_topk"])

    def test_sparse_seed_turns_containment_off(self):
        # One or two objects per image: a nested detection is a multi-scale duplicate of one
        # object, and removing the inner box would delete a real detection.
        self._seed([[(0, 0.5, 0.5, 0.3, 0.3)] for _ in range(4)])
        methods.compute_adaptive()
        self.assertFalse(methods.containment_on())

    def test_nested_seed_turns_containment_off(self):
        # A big box with a small box inside it — genuine nesting, so inner objects must survive.
        nested = [(0, 0.5, 0.5, 0.8, 0.8), (1, 0.5, 0.5, 0.15, 0.15),
                  (0, 0.2, 0.2, 0.05, 0.05), (0, 0.8, 0.8, 0.05, 0.05)]
        self._seed([nested for _ in range(4)])
        methods.compute_adaptive()
        self.assertFalse(methods.containment_on())
        self.assertGreater(state.ADAPT["nest_frac"], 0)

    def test_max_topk_tracks_observed_density(self):
        self._seed([[(0, 0.1 + 0.09 * k, 0.5, 0.05, 0.2) for k in range(10)] for _ in range(5)])
        methods.compute_adaptive()
        self.assertGreaterEqual(state.ADAPT["max_topk"], 10)

    def test_ai_written_labels_do_not_feed_the_prior(self):
        # An image with labels but NOT in HUMAN_SET is AI-owned: excluded from the count prior.
        write_boxes(self.labels, "img07", [(0, 0.5, 0.5, 0.2, 0.2)] * 3)
        seed, stats = methods.seed_stats()
        self.assertEqual(seed, {})
        self.assertFalse(stats.reliable)


if __name__ == "__main__":
    unittest.main()
