"""Label files on disk: what gets written, what gets refused, what survives a round trip."""

import tempfile
import unittest
from pathlib import Path

from support import make_dataset, write_boxes            # noqa: E402
from pglabel.labelio import (IMG_EXTS, clip_norm_box, image_size, list_images,  # noqa: E402
                             load_yolo, save_yolo)


class TestLabelIO(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="labelio-"))
        self.images, self.labels = make_dataset(self.tmp, n_images=3)

    def test_list_images_is_sorted_and_filtered(self):
        (self.images / "notes.txt").write_text("not an image")
        names = list_images(self.images)
        self.assertEqual(names, sorted(names))
        self.assertTrue(all(Path(n).suffix in IMG_EXTS for n in names))
        self.assertNotIn("notes.txt", names)

    def test_round_trip_preserves_values(self):
        boxes = [{"cls": 1, "cx": 0.25, "cy": 0.5, "w": 0.1, "h": 0.2}]
        save_yolo(self.labels, "img00", boxes)
        back = load_yolo(self.labels, "img00")
        self.assertEqual(len(back), 1)
        self.assertEqual(back[0]["cls"], 1)
        self.assertAlmostEqual(back[0]["cx"], 0.25, places=5)
        self.assertAlmostEqual(back[0]["h"], 0.2, places=5)

    def test_missing_file_is_no_boxes_not_an_error(self):
        self.assertEqual(load_yolo(self.labels, "does-not-exist"), [])

    def test_bom_prefixed_file_still_parses(self):
        # A label file edited in a Windows editor can carry a BOM; utf-8-sig is why this works.
        (self.labels / "bom.txt").write_bytes("﻿0 0.5 0.5 0.2 0.2\n".encode("utf-8"))
        self.assertEqual(len(load_yolo(self.labels, "bom")), 1)

    def test_off_canvas_box_is_clipped_to_the_image(self):
        clipped = clip_norm_box({"cls": 0, "cx": 0.95, "cy": 0.5, "w": 0.4, "h": 0.2})
        self.assertIsNotNone(clipped)
        self.assertLessEqual(clipped["cx"] + clipped["w"] / 2, 1.0 + 1e-9)
        self.assertGreater(clipped["w"], 0)

    def test_fully_off_canvas_box_is_dropped(self):
        self.assertIsNone(clip_norm_box({"cls": 0, "cx": 1.6, "cy": 0.5, "w": 0.2, "h": 0.2}))
        save_yolo(self.labels, "img01", [{"cls": 0, "cx": 1.6, "cy": 0.5, "w": 0.2, "h": 0.2}])
        self.assertEqual(load_yolo(self.labels, "img01"), [])

    def test_line_endings_are_lf_on_every_platform(self):
        # Byte-identical label files regardless of the OS the session ran on.
        save_yolo(self.labels, "img02", [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}])
        self.assertNotIn(b"\r\n", (self.labels / "img02.txt").read_bytes())

    def test_empty_box_list_writes_an_empty_file(self):
        save_yolo(self.labels, "img00", [])
        self.assertEqual((self.labels / "img00.txt").read_text(), "")

    def test_image_size(self):
        write_boxes(self.labels, "img00", [(0, 0.5, 0.5, 0.2, 0.2)])
        self.assertEqual(image_size(self.images / "img00.jpg"), (120, 90))


if __name__ == "__main__":
    unittest.main()
