#!/usr/bin/env python3
"""Choosing a dataset, and everything that has to be true once one is open.

Three jobs live here:

    the folder picker      ``browse`` — a server-side directory listing, because the browser
                           cannot see the filesystem and typing paths by hand is how people
                           end up labeling into the wrong folder.
    dataset presets        ``list_datasets`` — any ``<name>/images/train`` + ``labels/train``
                           + a ``*.yaml`` naming the classes becomes a one-click start card.
    opening a dataset      ``apply_dataset`` and the seeding it triggers.

The seeding deserves the emphasis it gets below: the count prior that the proposed method needs
comes from labeled images, so a dataset that opens with zero labels cannot demonstrate the
method at all. Pre-seeding a few ground-truth images per class is what makes the first run show
the mechanism instead of a fallback.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from . import candidates, paths, state
from .backend import AIBackend
from .labelio import list_images, load_yolo, save_yolo

IS_WINDOWS = os.name == "nt"
MAX_CLASSES = 9              # the UI binds classes to the digit keys 1–9


# ------------------------------------------------------------------- folder picker
def _windows_drives() -> list:
    """Drive letters that currently exist — the picker's top level on Windows, which (unlike
    POSIX) has no single filesystem root to walk up into."""
    out = []
    for letter in "CDEFGHIJKLMNOPQRSTUVWXYZAB":
        d = f"{letter}:\\"
        try:
            if Path(d).is_dir():
                out.append({"name": d, "path": d})
        except OSError:
            pass
    return out


def browse(path: str) -> dict:
    """List sub-folders of ``path`` for the setup screen.

    An empty path starts somewhere sensible (the datasets folder if there is one, else home).
    Each child is returned with its FULL path: the client must never do path arithmetic, or it
    would join a Windows path with a POSIX separator it cannot know is wrong.
    """
    if path == "\\" and IS_WINDOWS:              # the synthetic "above C:\" level
        return {"path": "\\", "parent": None, "dirs": _windows_drives(), "images": 0}
    if path:
        p = Path(path).expanduser()
    else:
        cand = paths.datasets_root()
        p = cand if cand else Path.home()
    try:
        p = p.resolve()
        if not p.is_dir():
            p = Path.home()
    except Exception:
        p = Path.home()
    dirs = []
    try:
        for child in sorted(p.iterdir(), key=lambda c: c.name.lower()):
            if child.is_dir() and not child.name.startswith("."):
                dirs.append({"name": child.name, "path": str(child)})
    except (PermissionError, OSError):
        pass
    try:
        n_img = len(list_images(p))
    except Exception:
        n_img = 0
    parent = str(p.parent) if str(p.parent) != str(p) else ("\\" if IS_WINDOWS else None)
    return {"path": str(p), "parent": parent, "dirs": dirs, "images": n_img}


# ------------------------------------------------------------------------- classes
def _classes_path() -> Path:
    return state.CFG["labels"] / ".pglabel_classes.json"


def save_classes() -> None:
    try:
        _classes_path().write_text(
            json.dumps({"images": str(state.CFG["images"]), "classes": state.CFG["classes"]},
                       ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def load_classes(base: list) -> list:
    """Restore a class list EXTENDED by the +Add-class button.

    The persisted list is adopted only when it was saved for THIS images folder and it extends
    the same base names — so added classes survive a restart, but a different dataset reusing
    the same labels folder cannot inherit stale extras.
    """
    p = _classes_path()
    if not p.exists():
        return base
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return base
    if isinstance(data, dict):
        if data.get("images") != str(state.CFG["images"]):
            return base
        saved = [str(c) for c in data.get("classes", []) if str(c).strip()]
    else:                                        # legacy bare-list format
        saved = [str(c) for c in data if str(c).strip()]
    if len(saved) > len(base) and saved[:len(base)] == base:
        return saved
    return base


def add_class(name: str):
    """POST /api/classes — append a class.

    Training and inference pick it up automatically: the trainer gets ``--classes`` from CFG,
    and the detector's class count grows once the new class is actually used in labels.
    """
    name = str(name or "").strip().replace(",", " ").strip()    # a comma breaks the CSV
    if not name:
        return {"detail": "the class name is empty"}, 400
    if name in state.CFG["classes"]:
        return {"detail": f"class already exists: {name}"}, 400
    if len(state.CFG["classes"]) >= MAX_CLASSES:
        return {"detail": f"at most {MAX_CLASSES} classes are supported (digit keys 1–9)"}, 400
    state.CFG["classes"].append(name)
    save_classes()
    return {"ok": True, "classes": state.CFG["classes"]}, 200


# ------------------------------------------------------------------ opening a dataset
def default_labels_dir(images: Path) -> Path:
    """Where labels go when the user named no labels folder.

    Beside the images is right for a checkout — that IS the dataset layout every script here
    expects. It is wrong for a packaged install, where the sample images sit inside a read-only
    bundle: labels written there are invisible to the user and are deleted by the next upgrade.
    So when the images live inside a distributed bundle (or the folder simply is not writable),
    fall back to the per-user workspace, one subfolder per dataset.
    """
    try:
        in_bundle = paths.packaged() and images.is_relative_to(paths.bundle_root())
    except (ValueError, OSError):
        in_bundle = False
    if not in_bundle and os.access(images, os.W_OK):
        return images
    name = images.name
    if name.lower() in ("images", "image", "img", "imgs", "train", "jpegimages"):
        name = images.parent.name                # ".../african-wildlife/images" → the dataset
    return paths.workspace_dir() / (name or "dataset") / "labels"


def apply_dataset(images, labels, classes_str, ai=None) -> None:
    """Point the tool at a dataset, from CLI arguments or from the setup screen.

    Paths are resolved to absolutes here because the training subprocess runs with a DIFFERENT
    working directory, where a relative path would resolve somewhere else entirely.
    """
    state.CFG["images"] = Path(images).expanduser().resolve()
    state.CFG["labels"] = (Path(labels).expanduser().resolve() if labels
                           else default_labels_dir(state.CFG["images"]))
    state.CFG["labels"].mkdir(parents=True, exist_ok=True)
    state.CFG["classes"] = [c.strip() for c in str(classes_str).split(",") if c.strip()] or ["object"]
    state.CFG["classes"] = load_classes(state.CFG["classes"])
    save_classes()
    if ai is not None:
        state.CFG["ai"] = ai
    state.load_human_set()      # restore ownership (human seed vs AI) across restarts
    seed_from_ground_truth()    # pre-load a few GT images as the seed, if configured


def seed_quota(names: list, gt: Path, n_classes: int) -> dict:
    """How many seed images each class needs.

    ``seed_percent`` is taken over the number of images that ACTUALLY contain that class, so a
    rare class still gets at least one image and the draw stays class-balanced on skewed
    datasets. Otherwise the flat ``seed_count`` applies to every class.
    """
    pct = float(state.CFG.get("seed_percent", 0) or 0)
    if pct <= 0:
        n = int(state.CFG.get("seed_count", 0) or 0)
        return {c: n for c in range(n_classes)}
    have = {c: 0 for c in range(n_classes)}
    for name in names:
        for c in {int(b["cls"]) for b in load_yolo(gt, Path(name).stem)
                  if b.get("w", 0) > 0 and b.get("h", 0) > 0}:
            if c in have:
                have[c] += 1
    return {c: (max(1, round(pct / 100.0 * have[c])) if have[c] else 0) for c in range(n_classes)}


def seed_from_ground_truth() -> None:
    """Pre-seed a few images PER CLASS from a ground-truth folder as the human seed.

    Per class, not first-N-by-filename: every class has to be represented or the detector never
    learns some of them. Images are drawn RANDOMLY but reproducibly (``seed_sample``), and each
    seeded image is added to HUMAN_SET with its ground-truth boxes written to the workspace — so
    the user SEES the starting point on the canvas and can verify it rather than trusting it.

    Runs only when no REAL seed exists yet. "Real" means at least one human-owned image that
    actually has boxes: an image the user merely checked off (human, zero boxes) must not
    permanently block seeding, which is exactly what a plain ``if HUMAN_SET`` guard would do.
    """
    gt = state.CFG.get("seed_gt")
    if not gt:
        return
    gt = Path(gt).expanduser()
    if not gt.is_dir():
        return
    if any(load_yolo(state.CFG["labels"], Path(img).stem) for img in state.HUMAN_SET):
        return                                   # a genuine seed exists → never seed over it
    guard = state.CFG.get("seed_gt_guard")       # a demo seed must not bleed onto another dataset
    if guard is not None and state.CFG["images"] != Path(guard):
        return

    n_classes = len(state.CFG["classes"])
    names = list_images(state.CFG["images"])
    quota = seed_quota(names, gt, n_classes)
    if not any(quota.values()):
        return
    per_class = {c: 0 for c in range(n_classes)}
    random.Random(int(state.CFG.get("seed_sample", 42))).shuffle(names)
    seeded = 0
    for name in names:
        stem = Path(name).stem
        boxes = [b for b in load_yolo(gt, stem)
                 if b.get("w", 0) > 0 and b.get("h", 0) > 0]   # drop boxes save_yolo would skip
        if not boxes:
            continue
        img_classes = {int(b["cls"]) for b in boxes}
        if any(per_class.get(c, 0) < quota.get(c, 0) for c in img_classes):
            save_yolo(state.CFG["labels"], stem, boxes)        # keep the original class ids
            state.HUMAN_SET.add(name)
            seeded += 1
            for c in img_classes:
                if c in per_class:
                    per_class[c] += 1
        if all(per_class[c] >= quota.get(c, 0) for c in range(n_classes)):
            break
    if seeded:
        state.save_human_set()
        pct = float(state.CFG.get("seed_percent", 0) or 0)
        how = f"{pct:g}%/class" if pct > 0 else f"{int(state.CFG.get('seed_count', 0) or 0)}/class"
        coverage = ", ".join(f"{state.CFG['classes'][c]}:{per_class[c]}/{quota.get(c, 0)}"
                             for c in range(n_classes))
        print(f"[seed] pre-loaded {seeded} image(s) as the human seed — {how}, "
              f"random seed={state.CFG.get('seed_sample', 42)} ({coverage})")


# -------------------------------------------------------------------------- presets
def parse_yaml_class_names(yaml_path) -> list:
    """Pull class names out of an ultralytics data.yaml.

    Handles both shapes seen in the wild: the block form (``names:\\n  0: car``) and the inline
    list (``names: [car, van]``). Parsed with a regex rather than a YAML dependency — this is
    the app process, which deliberately has no third-party imports beyond Pillow.
    """
    try:
        text = Path(yaml_path).read_text(encoding="utf-8-sig")
    except Exception:
        return []
    m = re.search(r'^\s*names:\s*\[(.*?)\]', text, re.M | re.S)      # inline list
    if m:
        return [x.strip().strip('\'"') for x in m.group(1).split(',') if x.strip()]
    out, in_names = {}, False                                        # block map
    for line in text.splitlines():
        if re.match(r'^\s*names:\s*$', line):
            in_names = True
            continue
        if in_names:
            mm = re.match(r'^\s+(\d+)\s*:\s*(.+?)\s*$', line)
            if mm:
                out[int(mm.group(1))] = mm.group(2).strip().strip('\'"')
            elif line.strip() and not line[0].isspace():
                break
    return [out[i] for i in sorted(out)] if out else []


def list_datasets() -> list:
    """Preset dataset cards for the start screen.

    Each preset gets its OWN workspace labels folder, so switching datasets can never mix one
    dataset's labels into another's.
    """
    root = paths.datasets_root()
    workspace = paths.workspace_dir()
    out = []
    if root is None or not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        img, lab = d / "images" / "train", d / "labels" / "train"
        if not (img.is_dir() and lab.is_dir()):
            continue
        yamls = sorted(d.glob("*.yaml"))
        classes = parse_yaml_class_names(yamls[0]) if yamls else []
        if not classes:
            continue
        val = d / "images" / "val"
        out.append({
            "name": d.name, "images": str(img.resolve()), "seed_labels": str(lab.resolve()),
            "labels": str((workspace / d.name / "labels").resolve()),
            "classes": ",".join(classes), "num_classes": len(classes), "seed_percent": 5,
            "train_images": len(list_images(img)),
            "val_images": len(list_images(val)) if val.is_dir() else 0})
    return out


def run_setup(body):
    """POST /api/setup — open the dataset the user typed or picked.

    A preset card also posts ``seed_labels`` + ``seed_percent``, which is what makes the
    proposed method demonstrable on the first run instead of falling back for lack of a seed.
    """
    images = str(body.get("images", "")).strip()
    if not images:
        return {"error": "enter the path to an images folder"}, 400
    p = Path(images).expanduser()
    if not p.is_dir():
        return {"error": f"folder not found: {images}"}, 400
    n = len(list_images(p))
    if n == 0:
        return {"error": f"no images found (.jpg / .png / …): {images}"}, 400

    seed_labels = body.get("seed_labels")
    seed_count = int(body.get("seed_count") or 0)
    seed_percent = float(body.get("seed_percent") or 0)
    cli = state.CFG.get("cli_seed")
    demo = state.CFG.get("demo_seed")
    if seed_labels and Path(seed_labels).expanduser().is_dir() and (seed_count > 0 or seed_percent > 0):
        state.CFG["seed_gt"] = Path(seed_labels).expanduser()
        state.CFG["seed_count"], state.CFG["seed_percent"] = seed_count, seed_percent
        state.CFG["seed_gt_guard"] = None
    elif cli and cli["gt"].is_dir() and (cli["images"] is None or cli["images"] == p.resolve()):
        # No preset clicked, but the launcher passed --seed-labels for exactly this dataset.
        state.CFG["seed_gt"], state.CFG["seed_count"] = cli["gt"], cli["count"]
        state.CFG["seed_percent"], state.CFG["seed_gt_guard"] = cli.get("percent", 0.0), None
    elif demo and p.resolve() == demo["images"] and demo["gt"].is_dir():
        # Pressing Start on the prefilled bundled DEMO — the one path a first-run user takes.
        # Without this the demo opens with zero labels, the count prior is empty, and Automate
        # Label can only fall back to auto-adaptive: the proposed method cannot show what it
        # does. Pinned to the demo images folder so a real dataset never inherits these labels.
        state.CFG["seed_gt"], state.CFG["seed_count"] = demo["gt"], demo["count"]
        state.CFG["seed_percent"], state.CFG["seed_gt_guard"] = 0.0, demo["images"]
    else:
        state.CFG["seed_gt"], state.CFG["seed_count"], state.CFG["seed_percent"] = None, 0, 0.0
        state.CFG["seed_gt_guard"] = None

    apply_dataset(p, body.get("labels") or None, body.get("classes") or "object")
    state.CFG["ai"] = AIBackend()                # switching dataset drops the stale model…
    candidates.invalidate()                      # …and its cached predictions with it
    return {"ok": True, "images": n, "classes": state.CFG["classes"],
            "labels": str(state.CFG["labels"])}, 200


def setup_defaults(args) -> dict:
    """Values that pre-fill the setup screen.

    Explicit ``--default-*`` wins; otherwise the bundled demo dataset is auto-detected so that
    ``python run_app.py`` with no arguments opens ready to press Start.
    """
    demo = paths.demo_dir() / "images"
    auto = args.default_images is None and demo.is_dir()
    imgs = args.default_images or (demo if auto else None)
    # Show the real destination — for a packaged install that is the user's workspace, not the
    # demo folder inside the read-only bundle.
    labels = args.default_labels or (default_labels_dir(demo.resolve()) if auto else None)
    classes = args.default_classes
    if classes is None:
        classes = "buffalo, elephant, rhino, zebra" if auto else ""
    return {"images": str(Path(imgs).expanduser().resolve()) if imgs else "",
            "labels": str(Path(labels).expanduser().resolve()) if labels else "",
            "classes": classes}
