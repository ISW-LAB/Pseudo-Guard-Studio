#!/usr/bin/env python3
"""The server's mutable session state, in one place.

The app is a single-process, single-session tool: one dataset open at a time, one training job
at a time. That makes module-level state the honest representation, but it only stays readable
if every mutable thing lives in ONE module that the rest of the package imports — which is what
this is. Modules do ``from pglabel import state`` and touch ``state.CFG[...]``; nothing here
imports anything else from ``pglabel`` except the pure label I/O, so there are no cycles.

    CFG        the open dataset and the training settings (set by cli.configure / run_setup)
    HUMAN_SET  which images the human owns — the single most consequential piece of state here
    TRAIN      the single training job
    CYCLE      the iterative self-training loop
    GATE       the human-in-the-loop rule-review pause between detector and validator
    ADAPT      per-dataset decisions derived from the seed (see pglabel.methods)
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional, Set

from .labelio import list_images

# ---- open dataset + training settings (populated by cli.configure and run_setup) -----------
CFG = {"images": None, "labels": None, "classes": ["object"], "ai": None,
       "train_env": "pseudoguard", "train_python": None, "train_device": "auto",
       "det_epochs": 100, "det_size": "n", "det_model_type": "yolov8", "val_epochs": 6,
       "research_root": None, "can_train": False}

# ---- the single training job -----------------------------------------------------------
TRAIN = {"state": "idle", "log": [], "report": None, "returncode": None,
         "started_at": None, "overlay": None, "stop": False}

# ---- the iterative cycle (train -> auto-label ALL -> retrain …) --------------------------
CYCLE = {"state": "idle", "log": [], "iters": [], "started_at": None, "total": 0,
         "current": 0, "stop": False}

# The live training subprocess, so the Stop button can terminate it mid-run.
ACTIVE_PROC = {"proc": None}

# ---- human-in-the-loop RULE-REVIEW gate --------------------------------------------------
# Between the detector and the validator, the RULE fabricates 'good' vs 'noise' crops. Training
# PAUSES here so the human can see those crops, tune the rule, and confirm. Shared by the single
# Train button and round 1 of the cycle — ``owner`` says which one is waiting.
GATE = {"active": False, "owner": None, "log": None, "crops": None, "noise_config": None,
        "crops_dir": None, "manifest_path": None, "nc_path": None, "params": None,
        "confirm": threading.Event(), "cancel": False, "busy": False, "sample_files": set()}

# ---- per-dataset adaptive decisions (recomputed from the seed; see methods.compute_adaptive)
ADAPT = {"containment": True, "max_topk": None, "nest_frac": None, "seed_mean": None}

# HUMAN_SET = images the user explicitly labeled or edited. It is the single source of truth for
# ownership and is PERSISTED, so it survives restarts and training runs:
#   • in HUMAN_SET      → seed: feeds the count prior, never overwritten by Auto-label ALL.
#   • not in HUMAN_SET  → AI-owned or unlabeled: re-labelled every time Auto-label ALL runs, so
#                         switching methodology and pressing again updates the whole set.
# A manual Save adds the image; passive navigation auto-save does not.
HUMAN_SET: Set[str] = set()


def human_set_path() -> Path:
    return CFG["labels"] / ".pglabel_human.json"


def save_human_set() -> None:
    """Persist ownership (best effort — losing it costs a re-derive, never data)."""
    try:
        p = human_set_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(sorted(HUMAN_SET)), encoding="utf-8")
    except Exception:
        pass


def load_human_set() -> None:
    """Restore ownership at startup.

    First run (no manifest): treat any PRE-EXISTING label file as human. That is the safe
    default — the app must never auto-overwrite labels it did not create.
    """
    HUMAN_SET.clear()
    imgs = set(list_images(CFG["images"]))
    p = human_set_path()
    if p.exists():
        try:
            HUMAN_SET.update(n for n in json.loads(p.read_text(encoding="utf-8")) if n in imgs)
            return
        except Exception:
            pass
    HUMAN_SET.update(img for img in imgs if (CFG["labels"] / f"{Path(img).stem}.txt").exists())
    save_human_set()


def job_running() -> bool:
    """True while either the single Train job or the cycle is in flight."""
    return TRAIN["state"] == "running" or CYCLE["state"] == "running"


def ai_available() -> bool:
    ai = CFG.get("ai")
    return bool(ai is not None and getattr(ai, "available", False))


def reset_for_tests() -> None:
    """Return every global to its start-up value (used by the test suite, never by the app)."""
    CFG.update(images=None, labels=None, classes=["object"], ai=None, research_root=None,
               can_train=False, train_python=None, train_device="auto")
    TRAIN.update(state="idle", log=[], report=None, returncode=None, started_at=None,
                 overlay=None, stop=False)
    CYCLE.update(state="idle", log=[], iters=[], started_at=None, total=0, current=0, stop=False)
    GATE.update(active=False, owner=None, log=None, crops=None, noise_config=None,
                crops_dir=None, manifest_path=None, nc_path=None, params=None,
                cancel=False, busy=False, sample_files=set())
    GATE["confirm"].clear()
    ADAPT.update(containment=True, max_topk=None, nest_frac=None, seed_mean=None)
    ACTIVE_PROC["proc"] = None
    HUMAN_SET.clear()
