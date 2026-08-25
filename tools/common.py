#!/usr/bin/env python3
"""Helpers shared by the training tools.

Everything in ``tools/`` runs under the TRAINING interpreter (torch + ultralytics), launched as
a subprocess by the app or by hand from a terminal. These are the pieces more than one of them
needs: locating the repository, reading the app's label files, optional image enhancement, and
turning a JSON rule from the review screen into a validated ``NoiseGenerationConfig``.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")

PREPROCESS_MODES = ("none", "clahe", "histeq", "gray", "denoise", "sharpen")


def weight_search_paths() -> list:
    """Folders scanned for pretrained base weights, most specific first.

    A packaged install reads its code from a sealed bundle, so the place a user can actually
    drop a ``yolov8n.pt`` is the install folder beside the executable — named by the app in
    ``PGLABEL_INSTALL_DIR``. Without this, an offline machine fails at the first training step
    with a download error instead of using the weights sitting right there.
    """
    roots = [repo_root(), repo_root() / "weights"]
    install = os.environ.get("PGLABEL_INSTALL_DIR")
    if install:
        roots += [Path(install), Path(install) / "weights"]
    return [p for p in roots if p.is_dir()]


def repo_root() -> Path:
    """The folder that must be importable for ``import pseudoguard`` / ``import pgcount``.

    The app passes ``PGLABEL_ROOT`` explicitly, because in a packaged install the tools live
    inside the bundle and "one folder up" is no longer the checkout.
    """
    return Path(os.environ.get("PGLABEL_ROOT") or Path(__file__).resolve().parent.parent)


def bootstrap_path() -> Path:
    """Put the repository root on ``sys.path`` and return it. Call this before any local import."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def enable_utf8_output() -> None:
    """Force UTF-8 on our own streams.

    The app sets PYTHONUTF8/PYTHONIOENCODING for us, but these tools are also run by hand and
    their output is routinely redirected to a file — where a Windows code page would refuse a
    character the training log happens to contain.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def log(msg: str) -> None:
    """Print for a caller that is tailing our stdout — unbuffered, one line at a time."""
    print(msg, flush=True)


# ------------------------------------------------------------------------ labels on disk
def list_images(images_dir: Path) -> List[str]:
    return sorted(p.name for p in Path(images_dir).iterdir()
                  if p.is_file() and p.suffix in IMG_EXTS)


def read_boxes(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """YOLO txt → [(cls, cx, cy, w, h)]; missing or empty file → []."""
    out = []
    if Path(label_path).exists():
        for line in Path(label_path).read_text(encoding="utf-8-sig").splitlines():
            p = line.split()
            if len(p) >= 5:
                out.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out


def seed_images(images_dir: Path, labels_dir: Path, scope: str = "human"):
    """The labeled images to train from, as [(name, boxes)].

    ``scope="human"`` honours the app's ownership manifest, so training uses only what a person
    actually labeled or approved. ``scope="all"`` also takes AI-written labels, which is what
    the self-training cycle folds back in on later rounds. With no manifest present, every
    labeled image counts as human — that is the safe reading of a folder the app did not create.
    """
    human = None
    if scope != "all":
        manifest = Path(labels_dir) / ".pglabel_human.json"
        if manifest.exists():
            try:
                human = set(json.loads(manifest.read_text(encoding="utf-8")))
            except Exception:
                human = None
    out = []
    for name in list_images(images_dir):
        if human is not None and name not in human:
            continue
        boxes = read_boxes(Path(labels_dir) / f"{Path(name).stem}.txt")
        if boxes:
            out.append((name, boxes))
    return out, ("human + AI labels" if scope == "all"
                 else ("human only" if human is not None else "all labeled"))


def clear_tree(path) -> None:
    """Remove a staged directory, including files Windows marks read-only.

    ``ignore_errors=True`` would leave those behind, and a stale image in a staged dataset is
    silently trained on — a wrong result rather than an error.

    Deliberately NOT imported from ``pglabel.fsutil``, even though that module does the same
    thing: this file runs under the TRAINING interpreter, whose import roots are only
    ``pseudoguard``, ``pgcount`` and ``tools``. Reaching into the app package works from a
    checkout and fails in every packaged install — the exact bug class this separation exists
    to prevent.
    """
    target = Path(path)
    if not target.exists():
        return

    def clear_readonly(func, name, _exc):
        try:
            os.chmod(name, stat.S_IWRITE)
            func(name)
        except OSError:
            pass

    if sys.version_info >= (3, 12):
        shutil.rmtree(target, onexc=clear_readonly)
    else:
        shutil.rmtree(target, onerror=clear_readonly)
    if target.exists():
        log(f"[warn] could not fully remove {target} — a previous run may still hold it")


def stage_split(root: Path, split: str, items, class_agnostic: bool = False) -> None:
    """Materialise (image, boxes) pairs as an ultralytics dataset under ``root``.

    Images are symlinked when the filesystem allows it and copied otherwise, so staging a large
    dataset costs no disk. ``class_agnostic`` collapses every box to class 0, which is how the
    localisation-only detector modes are trained.
    """
    img_dir = Path(root) / "images" / split
    lab_dir = Path(root) / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for img_path, boxes in items:
        dst = img_dir / img_path.name
        if not dst.exists():
            try:
                dst.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, dst)
        lines = [f"{0 if class_agnostic else int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                 for (c, cx, cy, w, h) in boxes]
        (lab_dir / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def ensure_pretrained_in_cwd(model_type: str, size: str, search: Sequence[Path]) -> None:
    """Copy a locally present pretrained weight into the cwd so ultralytics loads it OFFLINE.

    Without this, a machine with no internet (or a lab proxy) fails at the first training step
    with a download error instead of using the weights already sitting in the repository.
    """
    names = {"yolov8": f"yolov8{size}.pt", "yolov11": f"yolo11{size}.pt",
             "yolo26": f"yolo26{size}.pt", "rtdetr": f"rtdetr-{size}.pt"}
    name = names.get(model_type, f"yolov8{size}.pt")
    if Path(name).exists():
        return
    for base in search:
        src = Path(base) / name
        if src.exists():
            shutil.copy2(src, Path.cwd() / name)
            log(f"[weights] using local {name} (from {Path(base).name}) — offline")
            return
    log(f"[weights] local {name} not found; ultralytics may attempt a download")


class CropWriter:
    """Writes validator crops straight to disk as they are produced.

    The generator can either return every crop or hand each one to a sink. Returning them all
    costs about 200 KB of RAM per crop — ~1.6 GB at the default budget of 8,000, and unbounded
    when the budget is disabled — for images that are written to disk moments later anyway.
    This is that sink: peak memory stays flat regardless of how many crops are asked for.

        yes/  clf_train_yes/*.jpg   good crops (label 1)
        no/   clf_train_no/*.jpg    empty + deviated crops (label 0)
    """

    def __init__(self, root: Path, quality: int = 85):
        self.yes_dir = Path(root) / "clf_train_yes"
        self.no_dir = Path(root) / "clf_train_no"
        for d in (self.yes_dir, self.no_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.n_good = 0
        self.n_noise = 0
        self._quality = quality

    def __call__(self, crop, label: int) -> None:
        if int(label) == 1:
            crop.convert("RGB").save(self.yes_dir / f"{self.n_good:07d}.jpg", quality=self._quality)
            self.n_good += 1
        else:
            crop.convert("RGB").save(self.no_dir / f"{self.n_noise:07d}.jpg", quality=self._quality)
            self.n_noise += 1

    @property
    def total(self) -> int:
        return self.n_good + self.n_noise


# ------------------------------------------------------------------------- preprocessing
def preprocess_image(pil, mode: str):
    """Return a GEOMETRY-PRESERVING enhanced copy (same W×H, so normalised boxes stay valid).

    Geometry preservation is the whole point: the models see enhanced pixels while the app keeps
    drawing boxes on the ORIGINAL image, and the two agree because nothing moved. Falls back to
    the original image on any error or unknown mode.
    """
    from PIL import Image
    mode = (mode or "none").lower()
    if mode in ("none", ""):
        return pil
    try:
        import cv2
        import numpy as np
        rgb = np.array(pil.convert("RGB"))
        if mode == "clahe":                                # adaptive local contrast on L*
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
        elif mode == "histeq":                             # global equalisation on luma
            ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
            y, cr, cb = cv2.split(ycc)
            y = cv2.equalizeHist(y)
            out = cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2RGB)
        elif mode == "gray":                               # grayscale replicated to 3 channels
            g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            out = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        elif mode == "denoise":                            # edge-preserving smoothing
            out = cv2.bilateralFilter(rgb, 7, 50, 50)
        elif mode == "sharpen":                            # unsharp mask
            blur = cv2.GaussianBlur(rgb, (0, 0), 3)
            out = cv2.addWeighted(rgb, 1.5, blur, -0.5, 0)
        else:
            return pil
        return Image.fromarray(out)
    except Exception:
        return pil


def ensure_preprocessed(src_dir: Path, names: Iterable[str], prep_dir: Path, mode: str) -> Path:
    """Materialise preprocessed copies of ``names`` (idempotent) and return the dir to read from.

    Namespaced per mode, so changing the mode regenerates rather than silently reusing the
    previous enhancement.
    """
    from PIL import Image
    src_dir = Path(src_dir)
    mode = (mode or "none").lower()
    if mode in ("none", ""):
        return src_dir
    prep_dir = Path(prep_dir) / mode
    prep_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        dst = prep_dir / name
        if dst.exists():
            continue
        src = src_dir / name
        try:
            with Image.open(src) as im:
                out = preprocess_image(im, mode).convert("RGB")
            out.save(dst, quality=92) if dst.suffix.lower() in (".jpg", ".jpeg") else out.save(dst)
        except Exception:
            try:
                shutil.copy2(src, dst)                     # never leave a hole downstream
            except Exception:
                pass
    return prep_dir


# --------------------------------------------------------------------- the negative rule
def load_overrides(path) -> dict:
    """Read a rule JSON written by the review screen; anything unreadable means "use defaults"."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return dict(data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_noise_config(overrides):
    """Build a ``NoiseGenerationConfig`` from a dict of overrides (or a path to a JSON).

    Unknown keys are dropped rather than raising: the review screen also carries UI-only fields
    such as ``deviation_per_image`` that the algorithm config does not have. List values coming
    back from JSON are coerced to the tuples the dataclass expects. A rule whose ratios do not
    sum to 1.0 raises ValueError, which the caller reports to the user.
    """
    import dataclasses
    bootstrap_path()
    from pseudoguard.config import NoiseGenerationConfig

    if isinstance(overrides, (str, Path)):
        overrides = load_overrides(overrides)
    overrides = dict(overrides or {})
    tuple_fields = ("empty_crop_min_size", "empty_size_jitter", "deviation_shift_range")
    valid = {f.name for f in dataclasses.fields(NoiseGenerationConfig)}
    clean = {}
    for k, v in overrides.items():
        if k not in valid:
            continue
        clean[k] = tuple(v) if (k in tuple_fields and isinstance(v, list)) else v
    return NoiseGenerationConfig(**clean)


def resolve_device(preferred: Optional[str]) -> str:
    """``auto`` / an absent GPU → cpu. Thin re-export so tools need one import, not two."""
    bootstrap_path()
    from pseudoguard.device import resolve
    return resolve(preferred, log=log)
