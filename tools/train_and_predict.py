#!/usr/bin/env python3
"""Few-label seed → TRAIN (detector + validator) → PREDICT over every image → report.

This is what the app's Train button runs, and it is also usable on its own from a terminal.
Nothing is re-implemented here: it wires up the ``pseudoguard`` library in the order the method
prescribes.

    seed images + boxes
      → YOLOWrapper.train                          detector (the proposal generator), mAP@50
      → NoiseGenerator.generate_training_crops     good / empty / deviated crops (the rule)
      → TorchvisionClassifierWrapper.train         validator (the proposal quality model)
      → FrozenBackend.candidates over ALL images   the candidate overlay the app consumes

Outputs
    --out-overlay   { image_id: [ {box_id, box_xyxy, label, det_conf, p_good}, … ], … }
    --out-report    { ok, device, seconds, detector{…}, validator{…}, prediction{…} }

Staged runs (``--stage``) exist for the human review: ``detector`` stops after the detector so
the crops can be inspected, then ``validator`` resumes from the approved crops.

Runs on CPU when there is no GPU (``--device auto``), which is slow but correct — useful for a
smoke test, a laptop demo, or CI.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root, before local imports
from tools import common                                          # noqa: E402
from tools.common import IMG_EXTS, log                            # noqa: E402

common.enable_utf8_output()   # a redirected stdout must not be code-page limited


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path, help="folder of ALL images")
    ap.add_argument("--labels", required=True, type=Path, help="folder of YOLO .txt labels")
    ap.add_argument("--classes", default="object", help="comma-separated class names")
    ap.add_argument("--work-dir", required=True, type=Path, help="staging + runs + checkpoints")
    ap.add_argument("--out-overlay", required=True, type=Path)
    ap.add_argument("--out-report", required=True, type=Path)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:0")
    ap.add_argument("--det-epochs", type=int, default=100)
    ap.add_argument("--det-size", default="n", help="n|s|m|l|x")
    ap.add_argument("--det-img-size", type=int, default=640)
    ap.add_argument("--det-model-type", default="yolov8")
    ap.add_argument("--val-epochs", type=int, default=6)
    ap.add_argument("--val-batch", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.2, help="detector val holdout fraction")
    ap.add_argument("--train-scope", default="human", choices=["human", "all"],
                    help="human = only your labels; all = your labels + AI-labeled images")
    ap.add_argument("--max-crops", type=int, default=8000)
    ap.add_argument("--max-boxes", type=int, default=100, help="cap candidates per image at predict")
    ap.add_argument("--pseudo-conf", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", default="full", choices=["full", "detector", "validator"],
                    help="full = detector+validator+predict; detector = stop after the detector; "
                         "validator = train the filter (from --crops-dir if given) + predict")
    ap.add_argument("--noise-config", type=Path, default=None,
                    help="JSON of NoiseGenerationConfig overrides (the human-tuned rule)")
    ap.add_argument("--crops-dir", type=Path, default=None,
                    help="(validator stage) pre-generated, human-approved crops")
    ap.add_argument("--preprocess", default="none", choices=common.PREPROCESS_MODES,
                    help="geometry-preserving image enhancement for train+predict")
    ap.add_argument("--det-mode", default="multi", choices=["multi", "only", "classify"],
                    help="multi = class-aware detector (detect+classify jointly); "
                         "only = class-agnostic detector (localisation only); "
                         "classify = class-agnostic detector + a SEPARATE crop classifier for "
                         "the class, which is far stronger than joint few-label classification")
    return ap


def split_seed(seed, val_frac: float, rng_seed: int, class_aware: bool):
    """Hold out a validation slice so the reported detector mAP is honest.

    For the CLASS-AWARE detector every seed class must appear in TRAIN: the class count is
    inferred from the train split, so a class the random holdout isolated into val would go
    untrained or crash the run. Class-agnostic modes skip that rebalancing, where per-class
    coverage is irrelevant and moving images would needlessly shrink the honest holdout.
    """
    import torch
    generator = torch.Generator().manual_seed(rng_seed)
    order = torch.randperm(len(seed), generator=generator).tolist()
    n_val = max(1, int(round(len(seed) * val_frac))) if len(seed) >= 2 else 0
    val_idx = set(order[:n_val])

    if class_aware:
        def classes_of(i):
            return {int(c) for c, *_ in seed[i][1]}
        all_cls = set().union(*(classes_of(i) for i in range(len(seed)))) if seed else set()
        train_cls = set().union(*(classes_of(i) for i in range(len(seed)) if i not in val_idx))
        for c in all_cls - train_cls:
            for i in list(val_idx):
                if c in classes_of(i):
                    val_idx.discard(i)
                    train_cls |= classes_of(i)
                    break

    train_items = [seed[i] for i in range(len(seed)) if i not in val_idx]
    val_items = [seed[i] for i in range(len(seed)) if i in val_idx] or list(train_items)
    return train_items, val_items, n_val


def train_detector(args, report, all_imgs):
    """Stage 1 — the proposal generator. Returns (seed items, validator dataset root)."""
    import torch
    from pseudoguard.data.det_loader import DetDatasetSpec, build_dataset
    from pseudoguard.models.detection.yolov8_wrapper import YOLOWrapper

    seed, scope_note = common.seed_images(args.images, args.labels, args.train_scope)
    if not seed:
        raise SystemExit("no labeled images with boxes found — label a few first")
    log(f"[seed] {len(seed)} images / {len(all_imgs)} total  (scope: {scope_note})")

    # Optional enhancement: the detector and the validator crops see enhanced pixels, while the
    # app keeps drawing boxes on the ORIGINAL image — the geometry is identical either way.
    seed_src = common.ensure_preprocessed(args.images, [n for n, _ in seed],
                                          args.work_dir / "prep", args.preprocess)
    seed = [(seed_src / n, b) for n, b in seed]

    class_aware = args.det_mode == "multi"
    train_items, val_items, n_val = split_seed(seed, args.val_frac, args.seed, class_aware)
    log(f"[split] detector train={len(train_items)}  val={len(val_items)}"
        + ("  (val=train; only 1 labeled image)" if n_val == 0 else ""))

    det_root = args.work_dir / "det_ds"
    val_root = args.work_dir / "val_ds"
    common.clear_tree(det_root)
    common.stage_split(det_root, "train", train_items, class_agnostic=not class_aware)
    common.stage_split(det_root, "val", val_items, class_agnostic=not class_aware)
    common.clear_tree(val_root)
    common.stage_split(val_root, "train", seed)     # keeps real classes for the crop classifier
    log(f"[detector] mode={args.det_mode} (class-{'aware nc=N' if class_aware else 'agnostic nc=1'})")

    os.chdir(args.work_dir)                         # ultralytics writes runs/ and resolves weights here
    # A previously STOPPED run can leave a truncated .pt from an interrupted download, or a
    # half-written runs/ folder. Drop both so this run is not broken by the last one's debris.
    shutil.rmtree(args.work_dir / "runs", ignore_errors=True)
    import zipfile
    for pt in args.work_dir.glob("*.pt"):
        try:
            if not zipfile.is_zipfile(pt):
                pt.unlink()
        except Exception:
            pass
    common.ensure_pretrained_in_cwd(args.det_model_type, args.det_size,
                                    common.weight_search_paths())

    log(f"[detector] training {args.det_model_type}{args.det_size} "
        f"{args.det_epochs} epochs on {args.device} …")
    spec = DetDatasetSpec(name="fewlabel_det", root=det_root, yaml_path=None, data={})
    train_ds = build_dataset(spec, split="train", img_size=args.det_img_size)
    val_ds = build_dataset(spec, split="val", img_size=args.det_img_size)
    det = YOLOWrapper(model_type=args.det_model_type, size=args.det_size,
                      img_size=args.det_img_size, device=args.device)
    metrics = det.train(train_ds, val_ds, epochs=args.det_epochs,
                        batch_size=min(16, max(1, len(train_items))), lr=0.001,
                        patience=max(5, args.det_epochs // 4),
                        project=str(args.work_dir / "runs"), name="detector", num_workers=2)
    det.save_checkpoint(str(args.work_dir / "detection_model.pt"))
    report["detector"] = {
        "model": f"{args.det_model_type}{args.det_size}", "epochs": args.det_epochs,
        "map50": float(metrics.get("map50", 0) or 0),
        "map50_95": float(metrics.get("map50_95", 0) or 0),
        "precision": float(metrics.get("precision", 0) or 0),
        "recall": float(metrics.get("recall", 0) or 0),
        "n_train": len(train_items), "n_val": len(val_items)}
    log(f"[detector] mAP@50={report['detector']['map50']:.4f} "
        f"P={report['detector']['precision']:.3f} R={report['detector']['recall']:.3f}")
    return seed, val_root


def train_validator(args, report, val_root):
    """Stage 2 — the proposal validator, from approved crops or freshly generated ones."""
    from pseudoguard.data.classification_dataset import create_train_val_split_from_folder
    from pseudoguard.data.det_loader import DetDatasetSpec, build_dataset
    from pseudoguard.models.classification.densenet_wrapper import TorchvisionClassifierWrapper

    samples = args.work_dir / "clf_samples"
    owns_samples = True
    if args.crops_dir and (args.crops_dir / "clf_train_yes").is_dir():
        samples = args.crops_dir                    # human-APPROVED crops from the review gate
        owns_samples = False
        n_pos = len(list((samples / "clf_train_yes").glob("*.jpg")))
        n_neg = len(list((samples / "clf_train_no").glob("*.jpg")))
        log(f"[validator] using approved crops from {samples.name} (good={n_pos} noise={n_neg})")
    else:
        from pseudoguard.data.noise_generator import NoiseGenerator
        log("[validator] generating rule-based crops (GT=good, empty+deviated=noise) …")
        noise_cfg = common.build_noise_config(args.noise_config)
        spec = DetDatasetSpec(name="fewlabel_val", root=val_root, yaml_path=None, data={})
        crop_ds = build_dataset(spec, split="train", img_size=args.det_img_size)
        generator = NoiseGenerator(noise_cfg, device=args.device)
        max_crops = None if args.max_crops <= 0 else args.max_crops
        # Streamed to disk rather than returned: the crops are only ever written out, and
        # holding 8,000 decoded 256x256 images costs ~1.6 GB of RAM for no benefit.
        common.clear_tree(samples)
        writer = common.CropWriter(samples)
        _crops, labels = generator.generate_training_crops(
            crop_ds, mode="rule_based", max_samples=max_crops, sink=writer)
        n_pos, n_neg = writer.n_good, writer.n_noise
        log(f"[validator] {writer.total} crops (good={n_pos} noise={n_neg})")
        if n_pos == 0 or n_neg == 0:
            raise SystemExit("validator needs both good and noise crops; seed too small")
    if n_pos == 0 or n_neg == 0:
        raise SystemExit(f"validator needs both good and noise crops (good={n_pos} noise={n_neg})")

    val_ckpt = args.work_dir / "classification_model.pt"
    try:
        tr, va = create_train_val_split_from_folder(samples, val_ratio=0.2, img_size=256,
                                                    seed=args.seed)
        log(f"[validator] training densenet121 {args.val_epochs} epochs "
            f"({len(tr)} train / {len(va)} val crops) …")
        clf = TorchvisionClassifierWrapper(model_type="densenet121", img_size=256,
                                           num_classes=2, pretrained=True, device=args.device)
        metrics = clf.train(tr, va, epochs=args.val_epochs, batch_size=args.val_batch, lr=1e-4,
                            patience=3, use_weighted_loss=True, good_weight=1.0,
                            noise_weight=2.0, num_workers=2)
        clf.save_checkpoint(str(val_ckpt))
        report["validator"] = {
            "model": "densenet121", "epochs": args.val_epochs,
            "best_val_acc": float(metrics.get("best_val_acc", 0) or 0),
            "val_precision": metrics.get("val_precision"),
            "val_recall": metrics.get("val_recall"),
            "val_f1": metrics.get("val_f1"),
            "n_good": n_pos, "n_noise": n_neg}
        log(f"[validator] best_val_acc={report['validator']['best_val_acc']:.4f}")
    finally:
        if owns_samples:
            shutil.rmtree(samples, ignore_errors=True)
    return val_ckpt


def train_class_head(args, report, val_root):
    """Optional stage — a DECOUPLED crop→class model used by ``--det-mode classify``.

    The class-agnostic detector localises; this classifier assigns the class from each crop,
    which beats the joint few-label detector's own classification by a wide margin. Trained on
    ground-truth crops, with a STRATIFIED split so every class with two or more crops appears in
    both halves: a plain shuffle can strand a minority class entirely in train (never measured)
    or in val (never trained).
    """
    import random as rnd
    from PIL import Image
    from pseudoguard.data.classification_dataset import CropFolderClassificationDataset
    from pseudoguard.models.classification.densenet_wrapper import TorchvisionClassifierWrapper

    report["species"] = None
    if args.det_mode != "classify":
        return None, 0

    sp_dir = args.work_dir / "class_samples"
    shutil.rmtree(sp_dir, ignore_errors=True)
    sp_dir.mkdir(parents=True)
    vimg, vlab = val_root / "images" / "train", val_root / "labels" / "train"
    paths_, labels_, k = [], [], 0
    for lp in sorted(vlab.glob("*.txt")):
        ip = next((vimg / f"{lp.stem}{e}" for e in IMG_EXTS if (vimg / f"{lp.stem}{e}").exists()), None)
        boxes = common.read_boxes(lp)
        if not ip or not boxes:
            continue
        with Image.open(ip) as im:
            im = im.convert("RGB")
            W, H = im.size
            for (cls, cx, cy, w, h) in boxes:
                x1, y1 = max(0, int((cx - w / 2) * W)), max(0, int((cy - h / 2) * H))
                x2, y2 = min(W, int((cx + w / 2) * W)), min(H, int((cy + h / 2) * H))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                fp = sp_dir / f"{k:07d}.jpg"
                im.crop((x1, y1, x2, y2)).resize((256, 256)).save(fp, quality=85)
                paths_.append(fp)
                labels_.append(int(cls))
                k += 1

    n_named = len([c for c in args.classes.split(",") if c.strip()])
    n_classes = max(n_named, (max(labels_) + 1) if labels_ else 2)
    if len(set(labels_)) < 2 or len(paths_) < 4:
        log(f"[class-head] only {len(set(labels_))} class(es) / {len(paths_)} crops — skipping "
            f"(boxes stay class-agnostic for this seed)")
        return None, n_classes

    rng = rnd.Random(args.seed)
    by_cls = {}
    for i, c in enumerate(labels_):
        by_cls.setdefault(c, []).append(i)
    tri, vai = [], []
    for c, ids in by_cls.items():
        rng.shuffle(ids)
        if len(ids) >= 2:
            kk = min(max(1, int(round(len(ids) * 0.2))), len(ids) - 1)   # ≥1 in val AND in train
            vai.extend(ids[:kk])
            tri.extend(ids[kk:])
        else:
            tri.extend(ids)                                              # singleton → train
    rng.shuffle(tri)
    rng.shuffle(vai)
    if not vai:                                                          # all singletons
        vai.append(tri.pop())
    sp_tr = CropFolderClassificationDataset([paths_[i] for i in tri], [labels_[i] for i in tri],
                                            img_size=256, augment=True)
    sp_va = CropFolderClassificationDataset([paths_[i] for i in vai], [labels_[i] for i in vai],
                                            img_size=256, augment=False)
    per_class = {c: labels_.count(c) for c in sorted(set(labels_))}
    log(f"[class-head] training densenet121 num_classes={n_classes} on {len(paths_)} GT crops "
        f"({len(tri)}/{len(vai)}) — per-class {per_class}")
    sp = TorchvisionClassifierWrapper(model_type="densenet121", img_size=256,
                                      num_classes=n_classes, pretrained=True, device=args.device)
    metrics = sp.train(sp_tr, sp_va, epochs=max(args.val_epochs, 10), batch_size=args.val_batch,
                       lr=1e-4, patience=4, use_weighted_loss=False, num_workers=2)
    ckpt = args.work_dir / "species_model.pt"
    sp.save_checkpoint(str(ckpt))
    report["species"] = {"model": "densenet121", "num_classes": n_classes,
                         "n_crops": len(paths_),
                         "best_val_acc": float(metrics.get("best_val_acc", 0) or 0),
                         "per_class": per_class}
    log(f"[class-head] best_val_acc={report['species']['best_val_acc']:.4f}")
    return ckpt, n_classes


# Collection thresholds tried per image, most confident first.
CONF_LEVELS = (0.02, 0.01, 0.005, 0.002, 0.001, 0.0005)


def predict_overlay(args, report, det_ckpt, val_ckpt, class_ckpt, n_classes, all_imgs):
    """Stage 3 — score every image and write the candidate overlay the app consumes.

    Collection is PER IMAGE and adaptive: each image is collected at the highest threshold that
    yields at least one candidate. A single global stop-at-first-candidate starved coverage — a
    strong detector stopped at 0.05 and left most images empty — while a flat low threshold
    floods confident images with noise. Class-agnostic NMS and count-guided K downstream then
    bound how many of these survive.
    """
    from PIL import Image
    from pgcount.backend import FrozenBackend

    pred_src = common.ensure_preprocessed(args.images, all_imgs, args.work_dir / "prep",
                                          args.preprocess)
    backend = FrozenBackend(det_ckpt, val_ckpt, pseudo_conf_threshold=args.pseudo_conf,
                            det_model_type=args.det_model_type, det_size=args.det_size,
                            det_img_size=args.det_img_size, device=args.device).load()

    class_model = None
    if class_ckpt is not None and Path(class_ckpt).exists():
        from pseudoguard.models.classification.densenet_wrapper import TorchvisionClassifierWrapper
        class_model = TorchvisionClassifierWrapper(model_type="densenet121", img_size=256,
                                                   num_classes=n_classes, pretrained=False,
                                                   device=args.device)
        class_model.load_checkpoint(str(class_ckpt))
        log("[predict] class head active — assigning each box's class from its crop")

    levels, seen = [], set()
    for c in (args.pseudo_conf, *CONF_LEVELS):
        if c not in seen:
            seen.add(c)
            levels.append(c)
    log(f"[predict] per-image adaptive collection (levels {levels}, cap {args.max_boxes}/img) …")

    overlay, n_cand, sum_pgood = {}, 0, 0.0
    for name in all_imgs:
        cands = []
        for conf in levels:
            backend.pseudo_conf_threshold = conf
            cands = backend.candidates(name, pred_src / name)
            if cands:
                break
        cands = sorted(cands, key=lambda c: -c.det_conf)[:args.max_boxes]
        labels_out = [int(c.label) for c in cands]
        if class_model is not None and cands:
            with Image.open(pred_src / name) as im:
                im = im.convert("RGB")
                W, H = im.size
                crops = []
                for c in cands:
                    x1, y1 = max(0, int(c.box_xyxy[0])), max(0, int(c.box_xyxy[1]))
                    x2, y2 = min(W, int(c.box_xyxy[2])), min(H, int(c.box_xyxy[3]))
                    crops.append(im.crop((x1, y1, x2, y2)).resize((256, 256))
                                 if (x2 - x1 >= 2 and y2 - y1 >= 2) else im.resize((256, 256)))
            probs = class_model.predict(crops)
            labels_out = [int(i) for i in probs.argmax(dim=1).tolist()]
        overlay[name] = [{"box_id": c.box_id, "box_xyxy": list(c.box_xyxy), "label": lab,
                          "det_conf": c.det_conf, "p_good": c.p_good}
                         for c, lab in zip(cands, labels_out)]
        n_cand += len(cands)
        sum_pgood += sum(c.p_good for c in cands)

    args.out_overlay.write_text(json.dumps(overlay, ensure_ascii=False))
    report["prediction"] = {"images": len(all_imgs), "candidates": n_cand,
                            "collection_conf": "per-image",
                            "mean_p_good": round(sum_pgood / n_cand, 4) if n_cand else 0.0}
    log(f"[predict] {n_cand} candidates over {len(all_imgs)} images "
        f"(mean P(good)={report['prediction']['mean_p_good']}) → {args.out_overlay.name}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for a in ("images", "labels", "work_dir", "out_overlay", "out_report"):
        setattr(args, a, getattr(args, a).resolve())

    t0 = time.time()
    report = {"ok": False, "device": args.device, "classes": args.classes.split(","),
              "stage": args.stage, "preprocess": args.preprocess, "det_mode": args.det_mode}
    try:
        common.bootstrap_path()
        args.device = common.resolve_device(args.device)
        report["device"] = args.device
        from pseudoguard.device import describe as describe_device
        log(f"[device] {describe_device(args.device)}")

        all_imgs = common.list_images(args.images)
        args.work_dir.mkdir(parents=True, exist_ok=True)
        det_ckpt = args.work_dir / "detection_model.pt"
        val_root = args.work_dir / "val_ds"

        if args.stage in ("full", "detector"):
            seed, val_root = train_detector(args, report, all_imgs)
            if args.stage == "detector":            # pause here for the human crop review
                report.update(ok=True, seed_images=len(seed), n_images=len(all_imgs),
                              seconds=round(time.time() - t0, 1))

        if args.stage in ("full", "validator"):
            os.chdir(args.work_dir)
            common.ensure_pretrained_in_cwd(args.det_model_type, args.det_size,
                                            common.weight_search_paths())
            if not det_ckpt.exists():
                raise SystemExit("detector checkpoint missing — run --stage detector first")
            val_ckpt = train_validator(args, report, val_root)
            class_ckpt, n_classes = train_class_head(args, report, val_root)
            predict_overlay(args, report, det_ckpt, val_ckpt, class_ckpt, n_classes, all_imgs)
            seed_imgs = len(list((val_root / "images" / "train").glob("*"))) if val_root.exists() else 0
            report.update(ok=True, overlay=str(args.out_overlay), seed_images=seed_imgs,
                          n_images=len(all_imgs), seconds=round(time.time() - t0, 1))
    except SystemExit as e:
        report.update(ok=False, error=str(e), seconds=round(time.time() - t0, 1))
        log(f"[fail] {e}")
    except Exception as e:
        report.update(ok=False, error=f"{type(e).__name__}: {e}",
                      seconds=round(time.time() - t0, 1))
        log("[fail] " + traceback.format_exc())

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    log(f"[report] ok={report['ok']} in {report.get('seconds')}s → {args.out_report}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
