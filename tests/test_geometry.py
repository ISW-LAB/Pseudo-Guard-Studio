"""Overlap arithmetic, and the two duplicate-removal passes that depend on it."""

import unittest

from support import Candidate                                    # noqa: E402
from pglabel.geometry import (CONTAINMENT_THRESHOLD, area_xyxy, containment,  # noqa: E402
                              dedup, iou_xyxy, nested, nms_candidates)


class TestOverlap(unittest.TestCase):
    def test_identical_boxes_have_iou_one(self):
        self.assertAlmostEqual(iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_disjoint_boxes_have_iou_zero(self):
        self.assertEqual(iou_xyxy((0, 0, 10, 10), (50, 50, 60, 60)), 0.0)

    def test_half_overlap(self):
        self.assertAlmostEqual(iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)), 50 / 150)

    def test_degenerate_box_never_divides_by_zero(self):
        self.assertEqual(iou_xyxy((5, 5, 5, 5), (0, 0, 10, 10)), 0.0)
        self.assertEqual(area_xyxy((5, 5, 5, 5)), 0.0)

    def test_containment_is_directional(self):
        inner, outer = (2, 2, 4, 4), (0, 0, 10, 10)
        self.assertAlmostEqual(containment(inner, outer), 1.0)
        self.assertAlmostEqual(containment(outer, inner), 4 / 100)
        self.assertTrue(nested(inner, outer))

    def test_nested_uses_the_documented_threshold(self):
        # 80% of the small box inside the large one is the boundary case.
        self.assertGreaterEqual(CONTAINMENT_THRESHOLD, 0.5)
        self.assertFalse(nested((0, 0, 10, 10), (9, 9, 19, 19)))


class TestNMS(unittest.TestCase):
    def test_duplicate_collapses_to_the_higher_p_good_box(self):
        a = Candidate((0, 0, 10, 10), p_good=0.6)
        b = Candidate((1, 0, 11, 10), p_good=0.9)
        kept = nms_candidates([a, b], use_containment=True)
        self.assertEqual(len(kept), 1)
        self.assertIs(kept[0], b)

    def test_distinct_objects_are_both_kept(self):
        a = Candidate((0, 0, 10, 10), p_good=0.9)
        b = Candidate((50, 50, 60, 60), p_good=0.8)
        self.assertEqual(len(nms_candidates([a, b], use_containment=True)), 2)

    def test_containment_on_keeps_the_larger_box(self):
        whole = Candidate((0, 0, 100, 100), p_good=0.5)
        part = Candidate((10, 10, 30, 30), p_good=0.99)     # higher score, but nested
        kept = nms_candidates([whole, part], use_containment=True)
        self.assertEqual([c.box_xyxy for c in kept], [(0, 0, 100, 100)])

    def test_containment_off_keeps_both(self):
        # A dataset with genuine nesting (helmet inside person) must not lose the inner object.
        whole = Candidate((0, 0, 100, 100), p_good=0.5)
        part = Candidate((10, 10, 30, 30), p_good=0.99)
        kept = nms_candidates([whole, part], use_containment=False)
        self.assertEqual(len(kept), 2)


class TestDedup(unittest.TestCase):
    def test_near_duplicates_are_dropped(self):
        a = Candidate((0, 0, 10, 10))
        b = Candidate((0, 0, 10, 9))
        self.assertEqual(len(dedup([a, b], use_containment=True)), 1)

    def test_dedup_follows_the_same_containment_decision_as_nms(self):
        whole = Candidate((0, 0, 100, 100))
        part = Candidate((10, 10, 30, 30))
        self.assertEqual(len(dedup([whole, part], use_containment=True)), 1)
        self.assertEqual(len(dedup([whole, part], use_containment=False)), 2)

    def test_empty_input(self):
        self.assertEqual(dedup([], use_containment=True), [])
        self.assertEqual(nms_candidates([], use_containment=True), [])


if __name__ == "__main__":
    unittest.main()
