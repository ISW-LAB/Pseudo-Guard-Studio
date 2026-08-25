#!/usr/bin/env python3
"""Fixtures shared by the test modules: a throwaway dataset, an overlay, and a live server.

The suite runs on the standard library plus Pillow — the same footprint the application itself
needs — so ``python -m unittest discover tests`` works in a fresh checkout with nothing
installed. Tests that need torch skip themselves rather than fail.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


requires_torch = unittest.skipUnless(have_torch(), "torch is not installed")


def make_dataset(root: Path, n_images: int = 6, size=(120, 90)) -> tuple[Path, Path]:
    """A tiny images+labels dataset on disk. Returns (images_dir, labels_dir).

    Images are solid colours: nothing here tests pixels, only geometry and bookkeeping, and a
    generated dataset keeps the suite free of binary fixtures.
    """
    from PIL import Image
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for i in range(n_images):
        Image.new("RGB", size, (20 * i % 255, 90, 160)).save(images / f"img{i:02d}.jpg", quality=90)
    return images, labels


def write_boxes(labels: Path, stem: str, boxes) -> None:
    """boxes = [(cls, cx, cy, w, h)] in normalised coordinates."""
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes]
    (labels / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_overlay(path: Path, image_names, per_image: int = 4) -> Path:
    """A candidate overlay with a predictable score gradient.

    Each image gets ``per_image`` boxes whose p_good descends from 0.95, so a test can assert
    exactly which ones an operating point should accept.
    """
    overlay = {}
    for name in image_names:
        cands = []
        for k in range(per_image):
            x = 10 + 20 * k
            cands.append({"box_id": f"{name}:{k}", "box_xyxy": [x, 10, x + 15, 40],
                          "label": k % 2, "det_conf": round(0.9 - 0.1 * k, 3),
                          "p_good": round(0.95 - 0.2 * k, 3)})
        overlay[name] = cands
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overlay), encoding="utf-8")
    return path


class Candidate:
    """Minimal stand-in for pgcount's RawCandidate — geometry tests need nothing more."""

    def __init__(self, box_xyxy, p_good=0.9, det_conf=0.9, label=0, image_id="img"):
        self.box_xyxy = tuple(box_xyxy)
        self.p_good = p_good
        self.det_conf = det_conf
        self.label = label
        self.image_id = image_id

    def __repr__(self):
        return f"Candidate({self.box_xyxy}, p_good={self.p_good})"


class ServerCase(unittest.TestCase):
    """Base class that boots a REAL server on a throwaway dataset and talks HTTP to it.

    Testing through HTTP rather than by calling handlers keeps the route table, the JSON shapes
    and the dataset guard under test — the parts that break when routing is refactored.
    """

    overlay = False              # subclasses set True to attach a precomputed AI backend
    seed_images = 2              # how many images start out human-labeled

    @classmethod
    def setUpClass(cls):
        from pglabel import cli, state
        cls.tmp = Path(tempfile.mkdtemp(prefix="pglabel-test-"))
        cls.images, cls.labels = make_dataset(cls.tmp)
        cls.names = sorted(p.name for p in cls.images.iterdir())
        state.reset_for_tests()

        argv = ["--images", str(cls.images), "--labels", str(cls.labels),
                "--classes", "cat,dog", "--no-train", "--port", "0"]
        if cls.overlay:
            overlay = make_overlay(cls.tmp / "overlay.json", cls.names)
            argv += ["--overlay", str(overlay)]
        args = cli.build_parser().parse_args(argv)
        cli.configure(args)

        # Seed a couple of human-labeled images so the count prior has something to work with.
        for name in cls.names[:cls.seed_images]:
            write_boxes(cls.labels, Path(name).stem, [(0, 0.3, 0.5, 0.2, 0.3),
                                                      (1, 0.7, 0.5, 0.2, 0.3)])
            state.HUMAN_SET.add(name)
        state.save_human_set()

        cls.httpd = cli.make_server("127.0.0.1", 0)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        from pglabel import state
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)
        state.reset_for_tests()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---------------------------------------------------------------- HTTP helpers
    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def get(self, path: str):
        """(status, parsed-json-or-bytes)."""
        req = urllib.request.Request(self.url(path))
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                body = r.read()
                ctype = r.headers.get("Content-Type", "")
                return r.status, (json.loads(body) if "json" in ctype else body)
        except urllib.error.HTTPError as e:
            body = e.read()
            try:
                return e.code, json.loads(body)
            except Exception:
                return e.code, body

    def post(self, path: str, payload: dict):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(self.url(path), data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read()
            try:
                return e.code, json.loads(body)
            except Exception:
                return e.code, body
