#!/usr/bin/env python3
"""Command line, start-up configuration, and the serve loop.

``configure`` is the single place that turns parsed arguments into application state, so the
research launcher, the desktop entry point and the tests all reach the same configured server
by different routes without duplicating any of this.

Two start-up shapes:

    with --images     open that dataset immediately (scripted / reproducible runs)
    without           start in SETUP mode and let the user pick a dataset in the browser
"""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path

from . import dataset_setup, paths, state
from .api import Handler
from .backend import AIBackend

DEFAULT_TRAIN_ENV = "pseudoguard"      # conda env assumed to hold torch + ultralytics
DEFAULT_DEMO_SEED_IMAGES = 5           # how many demo images get pre-seeded from ground truth


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pglabel",
        description="PG-Label — collaborative auto-labeling with count-guided acceptance.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    dataset = ap.add_argument_group("dataset")
    dataset.add_argument("--images", type=Path, default=None,
                         help="images folder (omit to pick one on the browser setup screen)")
    dataset.add_argument("--labels", type=Path, default=None,
                         help="labels folder (default: beside the images, or the workspace)")
    dataset.add_argument("--classes", default="object", help="comma-separated class names")
    dataset.add_argument("--default-images", type=Path, default=None,
                         help="pre-fill the setup screen with this images folder")
    dataset.add_argument("--default-labels", type=Path, default=None)
    dataset.add_argument("--default-classes", default=None)

    seed = ap.add_argument_group("few-label seed")
    seed.add_argument("--seed-labels", type=Path, default=None,
                      help="folder of ground-truth YOLO .txt to pre-load as the initial seed")
    seed.add_argument("--seed-count", type=int, default=0,
                      help="how many images to pre-seed PER CLASS from --seed-labels")
    seed.add_argument("--seed-percent", type=float, default=0.0,
                      help="pre-seed this PERCENT of the images per class (e.g. 5 -> 5%%/class, >=1); "
                           "takes precedence over --seed-count")
    seed.add_argument("--seed-sample", type=int, default=42,
                      help="RNG seed for the random few-label draw (reproducible per seed)")

    model = ap.add_argument_group("model")
    model.add_argument("--overlay", type=Path, default=None,
                       help="precomputed candidate overlay JSON (no torch needed)")
    model.add_argument("--detector", type=Path, default=None, help="detector checkpoint")
    model.add_argument("--validator", type=Path, default=None, help="validator checkpoint")
    model.add_argument("--det-model-type", default="yolov8")
    model.add_argument("--device", default="auto",
                       help="inference device: auto | cpu | cuda:0 (auto falls back to CPU)")

    serve_group = ap.add_argument_group("server")
    serve_group.add_argument("--host", default="127.0.0.1")
    serve_group.add_argument("--port", type=int, default=8000)

    train = ap.add_argument_group("training (runs in a separate torch environment)")
    train.add_argument("--train-env", default=DEFAULT_TRAIN_ENV,
                       help="conda env that has torch + ultralytics")
    train.add_argument("--train-python", default=None,
                       help="path to a python that has torch + ultralytics; used instead of "
                            "`conda run -n <env>` (the reliable option on Windows)")
    train.add_argument("--train-device", default="auto",
                       help="training device: auto | cpu | cuda:0 (auto falls back to CPU)")
    train.add_argument("--det-epochs", type=int, default=100)
    train.add_argument("--det-size", default="n", help="n|s|m|l|x")
    train.add_argument("--val-epochs", type=int, default=6)
    train.add_argument("--no-train", action="store_true", help="disable the Train button")
    return ap


def configure(args) -> None:
    """Populate the application state from parsed arguments."""
    state.CFG["setup_defaults"] = dataset_setup.setup_defaults(args)
    state.CFG["ai"] = AIBackend(args.overlay, args.detector, args.validator,
                                args.device, args.det_model_type)
    state.CFG["train_env"] = args.train_env
    # --train-python wins; otherwise fall back to a registered training pack (settings.json,
    # PGLABEL_TRAIN_PYTHON, or the sidecar venv), which is how an installed app finds torch.
    state.CFG["train_python"] = args.train_python or paths.train_python()
    state.CFG["train_device"] = args.train_device
    state.CFG["det_epochs"] = args.det_epochs
    state.CFG["det_size"] = args.det_size
    state.CFG["det_model_type"] = args.det_model_type
    state.CFG["val_epochs"] = args.val_epochs
    state.CFG["seed_sample"] = args.seed_sample

    state.CFG["research_root"] = paths.research_root()
    trainer = paths.tools_dir() / "train_and_predict.py"
    state.CFG["can_train"] = ((not args.no_train) and trainer.exists()
                              and state.CFG["research_root"] is not None)

    _configure_seed(args)

    if args.images:                      # dataset given on the command line
        dataset_setup.apply_dataset(args.images, args.labels, args.classes)
    else:
        state.CFG["images"] = None       # setup mode — the user picks in the browser


def _configure_seed(args) -> None:
    """Decide where the initial human seed comes from.

    Explicit ``--seed-labels`` always wins. Otherwise the bundled demo is auto-seeded, but ONLY
    in setup mode and ONLY onto the demo images themselves — a real dataset whose filenames
    happen to coincide must never inherit the demo's ground truth.
    """
    demo_root = paths.demo_dir()
    demo_auto = (args.images is None and args.default_images is None
                 and (demo_root / "images").is_dir())
    state.CFG["seed_gt_guard"] = None
    state.CFG["cli_seed"] = None
    state.CFG["demo_seed"] = None
    state.CFG["seed_percent"] = float(getattr(args, "seed_percent", 0) or 0)

    if args.seed_labels:
        state.CFG["seed_gt"] = args.seed_labels
        state.CFG["seed_count"] = args.seed_count or (0 if state.CFG["seed_percent"] > 0 else 5)
        # Remembered for the setup screen: launching with --default-images starts in setup mode,
        # and pressing Start there posts no seed_labels unless a preset card was clicked —
        # without this, run_setup would clear the seed the command line explicitly asked for.
        target = args.images or args.default_images
        state.CFG["cli_seed"] = {
            "gt": Path(args.seed_labels).expanduser(),
            "count": state.CFG["seed_count"], "percent": state.CFG["seed_percent"],
            "images": Path(target).expanduser().resolve() if target else None}
    elif demo_auto and (demo_root / "gt_labels").is_dir():
        state.CFG["seed_gt"] = demo_root / "gt_labels"
        state.CFG["seed_count"] = DEFAULT_DEMO_SEED_IMAGES
        state.CFG["seed_gt_guard"] = (demo_root / "images").resolve()
        state.CFG["demo_seed"] = {"gt": demo_root / "gt_labels",
                                  "count": DEFAULT_DEMO_SEED_IMAGES,
                                  "images": (demo_root / "images").resolve()}
    else:
        state.CFG["seed_gt"], state.CFG["seed_count"] = None, 0


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    """The configured HTTP server, not yet serving (the tests drive it directly)."""
    return ThreadingHTTPServer((host, port), Handler)


def serve(args) -> None:
    configure(args)
    how = (state.CFG["train_python"] or f"conda:{args.train_env}") if state.CFG["can_train"] else "off"
    dataset = str(state.CFG["images"]) if state.CFG["images"] else "(none — pick it in the browser)"
    print(f"[pg-label] http://{args.host}:{args.port}  images={dataset} "
          f"AI={'on' if state.ai_available() else 'MANUAL-only'} train={how}")
    httpd = make_server(args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[pg-label] shutting down")
        httpd.shutdown()


def main(argv=None) -> int:
    serve(build_parser().parse_args(argv))
    return 0
