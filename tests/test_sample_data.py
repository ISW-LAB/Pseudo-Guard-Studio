"""The bundled sample dataset, and the memory ceiling of the crop pipeline.

``data/`` exists so every command in the README runs against a real dataset the moment the
repository is cloned. That only holds if the folder keeps its shape — and if it stays PARTLY
labeled, because a fully labeled folder gives Automate Label nothing to do and the first thing
a new user tries appears to be broken.
"""

from __future__ import annotations

import gc
import shutil
import tempfile
import unittest
from pathlib import Path

from support import ROOT, requires_torch                            # noqa: E402
from pglabel.labelio import list_images, load_yolo                  # noqa: E402


class TestSampleDataset(unittest.TestCase):
    def setUp(self):
        self.data = ROOT / "data"

    def test_the_folder_has_the_layout_the_app_expects(self):
        for rel in ("images", "labels", "classes.txt", "README.md"):
            self.assertTrue((self.data / rel).exists(), rel)

    def test_it_is_partly_labeled_on_purpose(self):
        images = list_images(self.data / "images")
        labeled = sorted((self.data / "labels").glob("*.txt"))
        self.assertGreaterEqual(len(images), 8, "too few images to demonstrate the loop")
        self.assertGreater(len(labeled), 0, "no seed: the count prior would have nothing to fit")
        self.assertLess(len(labeled), len(images),
                        "every image is labeled: Automate Label would have no target")

    def test_every_label_file_belongs_to_an_image(self):
        stems = {Path(n).stem for n in list_images(self.data / "images")}
        for label in (self.data / "labels").glob("*.txt"):
            self.assertIn(label.stem, stems, f"{label.name} labels an image that is not here")

    def test_the_boxes_are_valid_normalised_yolo_rows(self):
        total = 0
        for label in (self.data / "labels").glob("*.txt"):
            boxes = load_yolo(self.data / "labels", label.stem)
            self.assertTrue(boxes, f"{label.name} is empty")
            for b in boxes:
                self.assertGreaterEqual(b["cls"], 0)
                for key in ("cx", "cy", "w", "h"):
                    self.assertGreater(b[key], 0.0, f"{label.name}: {key}")
                    self.assertLessEqual(b[key], 1.0, f"{label.name}: {key}")
                total += 1
        self.assertGreater(total, 0)

    def test_the_class_names_cover_every_class_id_used(self):
        names = [c for c in (self.data / "classes.txt").read_text(encoding="utf-8").split()
                 if c.strip()]
        used = set()
        for label in (self.data / "labels").glob("*.txt"):
            used |= {b["cls"] for b in load_yolo(self.data / "labels", label.stem)}
        self.assertTrue(used)
        self.assertLess(max(used), len(names),
                        f"class id {max(used)} has no name in classes.txt ({names})")

    def test_the_readme_points_at_this_folder(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("./data/images", readme)
        self.assertIn("./data/labels", readme)

    def test_it_stays_small_enough_to_clone_comfortably(self):
        size = sum(f.stat().st_size for f in self.data.rglob("*") if f.is_file())
        self.assertLess(size, 5 * 1024 * 1024, "the sample dataset should stay a few MB")


@requires_torch
class TestCropMemoryCeiling(unittest.TestCase):
    """Crop generation must not scale its memory with the crop budget."""

    def _run(self, cap, use_sink):
        import resource
        import torch
        from PIL import Image
        from pseudoguard.config import NoiseGenerationConfig
        from pseudoguard.data.noise_generator import NoiseGenerator
        from tools.common import CropWriter

        class Fake:
            def __len__(self): return 10_000
            def __getitem__(self, i):
                return (Image.new("RGB", (640, 480)),
                        {"boxes": torch.tensor([[50., 50., 200., 200.],
                                                [250., 60., 400., 210.]])})

        gc.collect()
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out = Path(tempfile.mkdtemp(prefix="cropmem-"))
        try:
            generator = NoiseGenerator(NoiseGenerationConfig(negative_rule="refined"),
                                       device="cpu")
            sink = CropWriter(out) if use_sink else None
            crops, labels = generator.generate_training_crops(
                Fake(), mode="rule_based", max_samples=cap, sink=sink)
            produced = sink.total if use_sink else len(crops)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            del crops, labels
            return produced, max(0, peak - before) / 1024        # MB
        finally:
            shutil.rmtree(out, ignore_errors=True)
            gc.collect()

    def test_streaming_keeps_memory_flat_as_the_budget_grows(self):
        # Buffering costs ~200 KB per crop; the sink exists so a bigger budget is free.
        small, small_mb = self._run(1000, use_sink=True)
        large, large_mb = self._run(6000, use_sink=True)
        self.assertGreaterEqual(small, 1000)
        self.assertGreaterEqual(large, 6000)
        self.assertLess(large_mb, 200,
                        f"6,000 streamed crops added {large_mb:.0f} MB — the sink is not working")

    def test_the_sink_reports_the_counts_the_caller_needs(self):
        from tools.common import CropWriter
        produced, _mb = self._run(500, use_sink=True)
        self.assertGreaterEqual(produced, 500)

    def test_labels_still_describe_what_was_produced(self):
        import torch
        from PIL import Image
        from pseudoguard.config import NoiseGenerationConfig
        from pseudoguard.data.noise_generator import NoiseGenerator
        from tools.common import CropWriter

        class Fake:
            def __len__(self): return 40
            def __getitem__(self, i):
                return (Image.new("RGB", (640, 480)),
                        {"boxes": torch.tensor([[50., 50., 200., 200.]])})

        out = Path(tempfile.mkdtemp(prefix="cropcount-"))
        try:
            sink = CropWriter(out)
            _crops, labels = NoiseGenerator(NoiseGenerationConfig(), device="cpu") \
                .generate_training_crops(Fake(), mode="rule_based", max_samples=60, sink=sink)
            self.assertEqual(len(labels), sink.total)
            self.assertEqual(sum(labels), sink.n_good)
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
