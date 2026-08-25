#!/usr/bin/env python3
"""Train ONLY the proposal validator from a labeled seed.

``train_and_predict.py`` trains both models; this trains just the validator, which is the cheap
half — no detector run, no prediction pass. Useful when the rule changed and only the validator
needs to catch up, or to produce a validator for a detector that already exists.

    python tools/train_validator.py --images DIR --labels DIR --out artifacts/validator
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root
from tools import common                                          # noqa: E402
from tools.common import log                                      # noqa: E402

common.enable_utf8_output()   # a redirected stdout must not be code-page limited


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path, help="folder of images")
    ap.add_argument("--labels", required=True, type=Path, help="folder of YOLO .txt labels")
    ap.add_argument("--out", required=True, type=Path, help="output dir for the checkpoint")
    ap.add_argument("--train-scope", default="human", choices=["human", "all"])
    ap.add_argument("--noise-config", type=Path, default=None,
                    help="JSON of NoiseGenerationConfig overrides")
    ap.add_argument("--max-crops", type=int, default=8000, help="0 = unlimited")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--crop-img-size", type=int, default=256)
    ap.add_argument("--det-img-size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:0")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    common.bootstrap_path()
    args.device = common.resolve_device(args.device)

    from pseudoguard.data.classification_dataset import create_train_val_split_from_folder
    from pseudoguard.data.det_loader import DetDatasetSpec, build_dataset
    from pseudoguard.data.noise_generator import NoiseGenerator
    from pseudoguard.models.classification.densenet_wrapper import TorchvisionClassifierWrapper

    seed, scope_note = common.seed_images(args.images, args.labels, args.train_scope)
    if not seed:
        raise SystemExit("no labeled images with boxes found")
    log(f"[seed] {len(seed)} labeled images ({scope_note})")

    # Stage the seed as a dataset so the generator sees exactly what training would.
    args.out.mkdir(parents=True, exist_ok=True)
    ds_root = args.out / "seed_ds"
    common.clear_tree(ds_root)
    common.stage_split(ds_root, "train", [(Path(args.images) / n, b) for n, b in seed])

    cfg = common.build_noise_config(args.noise_config)
    log(f"[rule] {cfg.negative_rule}, shift={cfg.deviation_shift} "
        f"{tuple(round(v, 3) for v in cfg.deviation_shift_range)}, "
        f"ratios good/empty/deviated="
        f"{cfg.good_crop_ratio}/{cfg.empty_crop_ratio}/{cfg.deviation_ratio}")
    spec = DetDatasetSpec(name="seed", root=ds_root, yaml_path=None, data={})
    dataset = build_dataset(spec, split="train", img_size=args.det_img_size)
    generator = NoiseGenerator(cfg, device=args.device)
    # Streamed straight to disk — see tools/common.CropWriter for why buffering them would cost
    # ~1.6 GB of RAM at the default budget.
    samples = args.out / "samples"
    common.clear_tree(samples)
    writer = common.CropWriter(samples)
    generator.generate_training_crops(
        dataset, mode="rule_based",
        max_samples=(None if args.max_crops <= 0 else args.max_crops), sink=writer)
    n_pos, n_neg = writer.n_good, writer.n_noise
    log(f"[crops] {writer.total} total | good={n_pos} noise={n_neg}")
    if n_pos == 0 or n_neg == 0:
        raise SystemExit("need both good and noise crops; the seed may have no boxes")

    try:
        tr, va = create_train_val_split_from_folder(samples, val_ratio=0.2,
                                                    img_size=args.crop_img_size, seed=args.seed)
        log(f"[split] {len(tr)} train / {len(va)} val crops")
        clf = TorchvisionClassifierWrapper(model_type="densenet121", img_size=args.crop_img_size,
                                           num_classes=2, pretrained=True, device=args.device)
        metrics = clf.train(tr, va, epochs=args.epochs, batch_size=args.batch_size, lr=1e-4,
                            patience=args.patience, use_weighted_loss=True, good_weight=1.0,
                            noise_weight=2.0, num_workers=4)
        ckpt = args.out / "classification_model.pt"
        clf.save_checkpoint(str(ckpt))
        log(f"[done] val_acc={metrics.get('best_val_acc', 0):.4f} → {ckpt}")
    finally:
        shutil.rmtree(samples, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
