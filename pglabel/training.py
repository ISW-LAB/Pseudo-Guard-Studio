#!/usr/bin/env python3
"""Running the trainer, and the human review that interrupts it.

The app itself never imports torch. Training runs as a SUBPROCESS under a different
interpreter — the one that has torch and ultralytics — and this module is the whole contract
with it: build the command line, stream its stdout into a log the UI polls, hot-swap the
resulting model in, and be able to kill the whole process tree when the user presses Stop.

Keeping the heavy stack in another process is what lets one build serve both audiences: a
label-only install runs everywhere with Pillow alone, and pointing the app at a torch
environment turns the Train button on without changing anything else.

Three entry points, in increasing scope:

    train_once     detector + validator + predict, no interruption
    train_gated    detector → fabricate rule crops → PAUSE for review → validator + predict
    start_cycle    N rounds of (train, auto-label ALL), folding accepted labels back in
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from . import candidates, paths, state
from .backend import AIBackend
from .noise_rule import defaults as noise_defaults, validate_noise_config

IS_WINDOWS = os.name == "nt"

MAX_LOG_LINES = 600          # the UI only ever shows the tail; an unbounded log is a leak
REVIEW_TIMEOUT_S = 1800      # safety net if the confirm event is never set (30 min)


# ------------------------------------------------------------------ paths + launcher
def train_paths():
    """(trainer script, per-dataset work dir). The work dir holds checkpoints, crops, reports."""
    script = paths.tools_dir() / "train_and_predict.py"
    train_dir = state.CFG["labels"] / ".pgtrain"
    train_dir.mkdir(parents=True, exist_ok=True)
    return script, train_dir


def _conda_exe() -> str:
    """Absolute path to the conda launcher.

    On Windows ``conda`` is conda.bat, which CreateProcess cannot run from a bare name —
    resolving it through PATH (which honours PATHEXT) is what makes ``conda run`` work there.
    """
    return shutil.which("conda") or ("conda.bat" if IS_WINDOWS else "conda")


def launcher_for(script: Path):
    """(argv prefix that runs ``script``, human-readable description of the interpreter).

    Either a python that already has torch+ultralytics, or ``conda run -n <env>``. Never
    ``sys.executable``: frozen that is PG-Label.exe, and even from source the app's own
    interpreter is the label-only one.
    """
    if state.CFG.get("train_python"):
        return [state.CFG["train_python"], "-u", str(script)], str(state.CFG["train_python"])
    return ([_conda_exe(), "run", "-n", state.CFG["train_env"], "--no-capture-output",
             "python", "-u", str(script)], f"conda env '{state.CFG['train_env']}'")


def build_train_cmd(params: dict, stage: str = "full", crops_dir=None):
    """Assemble the trainer command line for one stage.

    ``stage`` is what makes the review possible: "detector" stops after the detector, then
    "validator" resumes from the approved crops. "full" does both without pausing.
    """
    script, train_dir = train_paths()
    overlay = train_dir / "overlay_trained.json"
    report = train_dir / "train_report.json"
    launcher, how = launcher_for(script)
    cmd = launcher + [
        "--images", str(state.CFG["images"]),
        "--labels", str(state.CFG["labels"]),
        "--classes", ",".join(state.CFG["classes"]),
        "--work-dir", str(train_dir),
        "--out-overlay", str(overlay),
        "--out-report", str(report),
        "--device", str(params.get("device", state.CFG["train_device"])),
        "--det-epochs", str(params.get("det_epochs", state.CFG["det_epochs"])),
        "--det-size", str(params.get("det_size", state.CFG["det_size"])),
        "--det-model-type", state.CFG["det_model_type"],
        "--val-epochs", str(params.get("val_epochs", state.CFG["val_epochs"])),
        "--det-mode", str(params.get("det_mode", "multi")),
        "--stage", stage,
        "--train-scope", ("all" if str(params.get("scope", "human")) == "all" else "human")]
    if params.get("noise_config_path"):
        cmd += ["--noise-config", str(params["noise_config_path"])]
    if crops_dir:
        cmd += ["--crops-dir", str(crops_dir)]
    return cmd, overlay, report, how


def read_json(path):
    try:
        p = Path(path)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        return None


# ---------------------------------------------------------------- subprocess control
def _group_spawn_kwargs() -> dict:
    """Popen kwargs that put the child in its own killable process group, per platform.

    Both the child (``conda run``) and its grandchild (the trainer that actually holds the GPU)
    must die together: killing only the parent orphans the trainer AND leaves its stdout pipe
    open, which blocks the reader loop below forever.
    """
    if IS_WINDOWS:
        # No console window when the app was started windowed, and a new process group so the
        # whole tree can be taskkill'd by PID (Windows has no killpg).
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


def kill_tree(proc) -> None:
    """Terminate a training subprocess and everything it spawned."""
    if IS_WINDOWS:
        # taskkill /T walks the child list Windows keeps per PID — the only reliable way to
        # reach the grandchild trainer, since terminate() would kill just ``conda run``.
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return
        except Exception:
            proc.kill()
            return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except Exception:
            os.killpg(pgid, signal.SIGKILL)
    except Exception:
        proc.kill()


def train_env_vars() -> dict:
    """Environment for the trainer subprocess.

    It runs under a DIFFERENT interpreter and, in a packaged install, from inside the app
    bundle — so the roots are named explicitly instead of being inferred from the script's
    location. ``PGLABEL_ROOT`` is what the tools put on ``sys.path`` to import ``pseudoguard``
    and ``pgcount``.
    """
    env = dict(os.environ)
    if paths.FROZEN:
        # A frozen app runs with a library search path pointing INTO its own bundle. Handing
        # that to another interpreter makes it load our Python/SSL/JPEG libraries instead of
        # its own — the classic "works from source, mystery crash once packaged" failure.
        # PyInstaller keeps the pre-launch values in <VAR>_ORIG precisely so children can be
        # given them back.
        bundle = str(paths.bundle_root())
        for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "LIBPATH", "PATH"):
            orig = env.pop(f"{var}_ORIG", None)
            if orig is not None:
                env[var] = orig
            elif bundle in env.get(var, ""):       # prepended with nothing saved → strip it
                kept = [x for x in env[var].split(os.pathsep) if x and x != bundle]
                env[var] = os.pathsep.join(kept)
                if not env[var]:
                    env.pop(var)
        for var in ("_MEIPASS2", "_PYI_APPLICATION_HOME_DIR", "_PYI_ARCHIVE_FILE",
                    "_PYI_PARENT_PROCESS_LEVEL", "PYTHONHOME"):
            env.pop(var, None)
    if state.CFG.get("research_root"):
        env["PGLABEL_ROOT"] = str(state.CFG["research_root"])
    env["PGLABEL_BUNDLE_DIR"] = str(paths.bundle_root())
    # Where a user may drop pretrained base weights so training works with no network. In a
    # frozen install the bundle root is the sealed _internal/ folder, so the writable install
    # folder beside the executable has to be named separately.
    env["PGLABEL_INSTALL_DIR"] = str(paths.install_root())
    env["PYTHONUTF8"] = "1"                  # UTF-8 paths/logs regardless of the console codepage
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_stream(cmd, log) -> int:
    """Run a subprocess to completion, streaming stdout into ``log``; return its exit code.

    The live process is published to ``state.ACTIVE_PROC`` so Stop can terminate it mid-run.
    """
    rc = -1
    try:
        # errors="replace": a trainer line can carry a progress-bar glyph or a non-UTF-8 path,
        # and on Windows the child's stdout is whatever code page it chose — never let one odd
        # byte raise UnicodeDecodeError in the middle of a 100-epoch run.
        #
        # Popen as a context manager, so the stdout pipe is closed even when the reader loop is
        # interrupted. A cycle runs this once per round; leaking a descriptor per round would
        # eventually exhaust the process's file limit mid-session.
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1,
                              cwd=str(state.CFG.get("research_root") or paths.bundle_root()),
                              env=train_env_vars(), **_group_spawn_kwargs()) as proc:
            state.ACTIVE_PROC["proc"] = proc
            for line in proc.stdout:
                log.append(line.rstrip("\n"))
                if len(log) > MAX_LOG_LINES:
                    del log[:-MAX_LOG_LINES]
            proc.wait()
            rc = proc.returncode
    except FileNotFoundError as e:
        log.append(f"[server] ERROR: cannot start the trainer ({e}). "
                   f"Set the training interpreter in Settings, or install the training pack.")
    except Exception as e:
        log.append("[server] ERROR: " + str(e))
    finally:
        state.ACTIVE_PROC["proc"] = None
    return rc


def stop_training() -> dict:
    """Stop the running Train / cycle job.

    Sets the stop flags, releases a paused review gate, and terminates the live subprocess. The
    worker then sees a non-zero exit and marks the job 'stopped' rather than 'failed'.
    """
    state.TRAIN["stop"] = True
    state.CYCLE["stop"] = True
    if state.GATE["active"]:
        state.GATE["cancel"] = True
        state.GATE["confirm"].set()
    proc = state.ACTIVE_PROC.get("proc")
    if proc is not None:
        try:
            kill_tree(proc)
        except Exception:
            pass
    return {"ok": True}


# ---------------------------------------------------------------------- status views
def gate_status():
    """UI-facing snapshot of the review gate (None when no gate is open)."""
    if not state.GATE["active"]:
        return None
    return {"active": True, "owner": state.GATE["owner"], "busy": state.GATE["busy"],
            "crops": state.GATE["crops"], "noise_config": state.GATE["noise_config"]}


def train_status() -> dict:
    return {"state": state.TRAIN["state"], "running": state.TRAIN["state"] == "running",
            "log": state.TRAIN["log"][-60:], "report": state.TRAIN["report"],
            "returncode": state.TRAIN["returncode"], "has_ai": state.ai_available(),
            "gate": gate_status(),
            "elapsed": round(time.time() - state.TRAIN["started_at"], 1)
            if state.TRAIN["started_at"] else 0}


def cycle_status() -> dict:
    return {"state": state.CYCLE["state"], "running": state.CYCLE["state"] == "running",
            "log": state.CYCLE["log"][-90:], "iters": state.CYCLE["iters"],
            "current": state.CYCLE["current"], "total": state.CYCLE["total"],
            "has_ai": state.ai_available(), "gate": gate_status(),
            "elapsed": round(time.time() - state.CYCLE["started_at"], 1)
            if state.CYCLE["started_at"] else 0}


# ------------------------------------------------------------------- model hot-swap
def activate_overlay(overlay: Path, log: list) -> None:
    """Make the freshly trained model the active backend, then PRE-WARM its predictions.

    Warming here — in the training thread, where the user is already watching a progress log —
    rather than lazily means the first Automate after training is instant, and no background
    status poll ever lands on a cold cache and stalls the UI mid-session.
    """
    state.CFG["ai"] = AIBackend(overlay=str(overlay))
    candidates.invalidate()                  # the previous model's predictions must not survive
    log.append("[server] trained model is now the active AI backend.")
    try:
        n = sum(len(v) for v in candidates.all_candidates().values())
        log.append(f"[server] cached {n} candidate box(es) from the new model.")
    except Exception as e:
        log.append("[server] candidate pre-warm failed (will build on demand): " + str(e))


def train_once(params: dict, log: list):
    """Full training to completion, no review pause. Returns (ok, report)."""
    cmd, overlay, report, how = build_train_cmd(params)
    log.append(f"$ training via {how} (scope={params.get('scope', 'human')}) …")
    rc = run_stream(cmd, log)
    rep = read_json(report)
    if rc == 0 and rep and rep.get("ok") and overlay.exists():
        activate_overlay(overlay, log)
        return True, rep
    return False, rep


# ------------------------------------------------- rule-review gate (crops + handshake)
def gen_crops_cmd(params, crops_dir, manifest_path, nc_path):
    script, train_dir = train_paths()
    generator = script.parent / "gen_noise_crops.py"
    launcher, _how = launcher_for(generator)
    return launcher + [
        "--images", str(state.CFG["images"]), "--labels", str(state.CFG["labels"]),
        "--work-dir", str(train_dir), "--out-dir", str(crops_dir),
        "--manifest", str(manifest_path),
        "--device", str(params.get("device", state.CFG["train_device"])),
        "--train-scope", ("all" if str(params.get("scope", "human")) == "all" else "human"),
        "--noise-config", str(nc_path)]


def run_gen_crops(params, crops_dir, manifest_path, nc_path, log):
    run_stream(gen_crops_cmd(params, crops_dir, manifest_path, nc_path), log)
    return read_json(manifest_path)


def _sample_whitelist(manifest) -> set:
    """(subfolder, filename) pairs the crop-thumbnail endpoint is allowed to serve.

    An explicit whitelist, not a path check: the endpoint takes a filename from the browser, and
    only names this manifest produced may ever be read off disk.
    """
    whitelist = set()
    for kind, files in (manifest.get("samples") or {}).items():
        sub = "clf_train_yes" if kind == "good" else "clf_train_no"
        for f in files:
            whitelist.add((sub, f))
    return whitelist


def open_gate(owner, log, params, crops_dir, manifest_path, nc_path, cfg, manifest):
    state.GATE.update(active=True, owner=owner, log=log, params=params, crops=manifest,
                      noise_config=cfg, crops_dir=crops_dir, manifest_path=manifest_path,
                      nc_path=nc_path, cancel=False, busy=False,
                      sample_files=_sample_whitelist(manifest))
    state.GATE["confirm"].clear()


def close_gate():
    state.GATE.update(active=False, owner=None, log=None, params=None, crops=None,
                      crops_dir=None, sample_files=set(), busy=False, cancel=False)
    state.GATE["confirm"].clear()


def regen_crops(cfg: dict, log):
    """Regenerate the review crops with ``cfg``. Returns the manifest, or ok=False + error."""
    if not state.GATE["active"]:
        return {"ok": False, "error": "no gate open"}
    if state.GATE["busy"]:
        return {"ok": False, "error": "busy"}
    state.GATE["busy"] = True
    try:
        Path(state.GATE["nc_path"]).write_text(json.dumps(cfg), encoding="utf-8")
        m = run_gen_crops(state.GATE["params"], state.GATE["crops_dir"],
                          state.GATE["manifest_path"], state.GATE["nc_path"], log)
        if m and m.get("ok"):
            state.GATE.update(crops=m, noise_config=cfg, sample_files=_sample_whitelist(m))
        return m or {"ok": False, "error": "crop generation produced no manifest"}
    finally:
        state.GATE["busy"] = False


def train_gated(params: dict, log: list, owner: str):
    """Detector → fabricate rule crops → PAUSE for review → validator + predict.

    Returns (ok, report). The pause is a real block on ``GATE["confirm"]``: the worker thread
    waits while the HTTP thread serves the review screen and eventually sets the event, either
    from Confirm (train the validator) or Cancel (abandon the run).
    """
    script, train_dir = train_paths()
    crops_dir = train_dir / "gate_crops"
    nc_path = train_dir / "noise_config.json"
    manifest_path = train_dir / "crops_manifest.json"

    cmd_a, overlay, report, how = build_train_cmd(params, stage="detector")
    log.append(f"$ [1/3] training the detector (Pseudo-Guard) via {how} "
               f"(scope={params.get('scope', 'human')}) …")
    if run_stream(cmd_a, log) != 0:
        return False, read_json(report)
    det_report = (read_json(report) or {}).get("detector")   # stage B overwrites the file

    cfg = validate_noise_config(params.get("noise_config") or noise_defaults())
    Path(nc_path).write_text(json.dumps(cfg), encoding="utf-8")
    log.append("$ [2/3] fabricating the filter's rule-based 'good vs noise' crops for your review …")
    manifest = run_gen_crops(params, crops_dir, manifest_path, nc_path, log)
    if not manifest or not manifest.get("ok"):
        log.append("[server] crop generation failed: " + str((manifest or {}).get("error", "?")))
        return False, None

    open_gate(owner, log, params, crops_dir, manifest_path, nc_path, cfg, manifest)
    log.append("⏸  awaiting your review — adjust the rule and press Confirm to train the filter …")
    waited = 0
    while not state.GATE["confirm"].wait(timeout=1.0):   # set by BOTH confirm and cancel
        waited += 1
        if waited > REVIEW_TIMEOUT_S:
            log.append("[server] review timed out.")
            close_gate()
            return False, None
    if state.GATE["cancel"]:
        log.append("[server] review canceled.")
        close_gate()
        return False, None
    close_gate()

    params_b = {**params, "noise_config_path": str(nc_path)}
    cmd_b, overlay, report, how = build_train_cmd(params_b, stage="validator", crops_dir=crops_dir)
    log.append("$ [3/3] training the filter on your approved crops, then predicting over all images …")
    if run_stream(cmd_b, log) != 0:
        return False, read_json(report)
    rep = read_json(report)
    if rep and det_report and not rep.get("detector"):   # restore detector mAP for the UI tiles
        rep["detector"] = det_report
    if rep and rep.get("ok") and overlay.exists():
        activate_overlay(overlay, log)
        return True, rep
    return False, rep


# ------------------------------------------------------------------------ job entry points
def _reject_if_not_ready(what: str):
    """Shared preconditions for both jobs; returns (payload, code) or None when ready."""
    if state.job_running():
        return {"detail": "a training / cycle job is already running"}, 409
    if not state.CFG["can_train"]:
        return {"detail": "training unavailable (trainer script or algorithm library not found)"}, 400
    if candidates.label_status()["labeled"] < 1:
        return {"detail": f"hand-label at least 1 image (few-label seed) before {what}"}, 400
    return None


def start_training(params: dict):
    """The Train button: one training run in a daemon thread; the UI polls /api/train/status.

    Training ALWAYS pauses for the rule review — the human sees the fabricated crops and
    confirms or tunes the rule before the validator learns from them.
    """
    rejected = _reject_if_not_ready("training")
    if rejected:
        return rejected
    state.TRAIN.update(state="running", log=[], report=None, returncode=None,
                       started_at=time.time(), stop=False)

    def worker():
        try:
            ok, rep = train_gated(params, state.TRAIN["log"], "train")
        except Exception as e:
            state.TRAIN["log"].append("[server] ERROR: " + str(e))
            ok, rep = False, None
        if state.GATE["active"] and state.GATE["owner"] == "train":
            close_gate()
        state.TRAIN["report"] = rep
        state.TRAIN["state"] = "stopped" if state.TRAIN["stop"] else ("done" if ok else "failed")
        if state.TRAIN["stop"]:
            state.TRAIN["log"].append("[server] ⏹ training stopped by the user.")

    threading.Thread(target=worker, daemon=True).start()
    return {"started": True}, 200


def start_cycle(params: dict):
    """The iterative self-training loop: for N rounds — TRAIN, then AUTO-LABEL ALL.

    Round 1 trains on the human seed; later rounds fold the accepted AI labels back into
    training when ``include_ai``. The rule review runs once, on round 1, and the confirmed rule
    is reused for every later round — asking again each round would make the loop unattendable.
    """
    rejected = _reject_if_not_ready("the cycle")
    if rejected:
        return rejected

    iters = max(1, min(10, int(params.get("iterations", 3))))
    method = params.get("method", "pseudoguard")
    thr, score = params.get("thr"), params.get("score", "p_good")
    include_ai = bool(params.get("include_ai", True))
    # Fold-back quality gate: when folding AI labels into TRAINING, reuse only the top-p_good
    # ones. Applied to intermediate rounds only; the final round keeps FULL density as output.
    try:
        fold_gate = float(params.get("fold_gate", 0.4))
        fold_gate = fold_gate if 0.0 < fold_gate < 1.0 else None
    except (TypeError, ValueError):
        fold_gate = None
    # A self-training loop MUST bound per-image acceptance or pseudo-labels self-amplify each
    # round (box explosion). Pseudo-Guard is count-guided, which supplies that cap; manual mode
    # relies on the user's threshold.
    count_guided = (method == "pseudoguard")
    acceptance = ("count-guided (bounded per image — stable)" if count_guided else
                  (f"manual {score} ≥ {thr}" if method == "manual" else method))
    state.CYCLE.update(state="running", started_at=time.time(), total=iters, current=0,
                       iters=[], stop=False,
                       log=[f"$ self-training cycle — {iters} rounds · accept={acceptance} · "
                            f"fold AI labels into training={include_ai}"])

    def worker():
        try:
            confirmed_nc = None                   # human-approved rule → reused every later round
            for it in range(iters):
                if state.CYCLE["stop"]:           # Stop pressed between rounds
                    state.CYCLE["log"].append("[server] ⏹ cycle stopped by the user.")
                    state.CYCLE["state"] = "stopped"
                    return
                state.CYCLE["current"] = it + 1
                scope = "all" if (it > 0 and include_ai) else "human"
                round_params = {**params, "scope": scope}
                if confirmed_nc:
                    round_params["noise_config_path"] = str(confirmed_nc)
                state.CYCLE["log"].append(f"──── round {it + 1}/{iters} · TRAIN (scope={scope}) ────")
                if it == 0:                       # the rule review runs once, on round 1
                    ok, rep = train_gated(round_params, state.CYCLE["log"], "cycle")
                    confirmed_nc = train_paths()[1] / "noise_config.json"
                else:
                    ok, rep = train_once(round_params, state.CYCLE["log"])
                if not ok:
                    if state.CYCLE["stop"]:       # the subprocess was killed by Stop
                        state.CYCLE["log"].append("[server] ⏹ cycle stopped by the user.")
                        state.CYCLE["state"] = "stopped"
                        return
                    state.CYCLE["log"].append(
                        f"round {it + 1}: training failed / canceled — stopping cycle")
                    state.CYCLE["state"] = "failed"
                    return
                det = (rep or {}).get("detector", {})
                val = (rep or {}).get("validator", {})
                # Gate the fold-back TRAINING labels on non-final rounds; the final round outputs
                # full density because that is the artefact the user keeps.
                fg = None if (it == iters - 1 or not include_ai) else fold_gate
                state.CYCLE["log"].append(
                    f"round {it + 1} · AUTO-LABEL ALL (method={method}"
                    f"{' +count-guided' if count_guided else ''}{f' +fold-gate {fg}' if fg else ''}) …")
                res = candidates.automate_all(method, thr=thr, score=score, fold_gate=fg)
                labeled, boxes = res.get("auto_labeled", 0), res.get("total_boxes", 0)
                state.CYCLE["iters"].append(
                    {"round": it + 1, "scope": scope, "map50": det.get("map50"),
                     "val_acc": val.get("best_val_acc"),
                     "auto_labeled": labeled, "total_boxes": boxes})
                state.CYCLE["log"].append(
                    f"round {it + 1} done · mAP@50={float(det.get('map50', 0) or 0):.3f} · "
                    f"auto-labeled {labeled} imgs / {boxes} boxes")
            state.CYCLE["state"] = "done"
            state.CYCLE["log"].append("[server] ✅ self-training cycle complete.")
        except Exception as e:
            state.CYCLE["log"].append("[server] ERROR: " + str(e))
            state.CYCLE["state"] = "failed"
        finally:
            if state.GATE["active"] and state.GATE["owner"] == "cycle":
                close_gate()

    threading.Thread(target=worker, daemon=True).start()
    return {"started": True}, 200
