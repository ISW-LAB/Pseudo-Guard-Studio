"""The trainer contract: command lines, the subprocess, and the review that interrupts it.

This is the machinery that breaks once the app is packaged — a different interpreter, a
different working directory, a bundle whose library paths must not leak into the child. It is
exercised here against a FAKE trainer: a small script that speaks the same protocol (arguments
in, report/manifest/overlay files out) but returns in milliseconds. That keeps the orchestration
under test without a GPU, a dataset, or torch.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from support import make_dataset, make_overlay, write_boxes          # noqa: E402
from pglabel import paths, state, training                           # noqa: E402

# A stand-in for tools/train_and_predict.py. It honours --stage, writes the report the app reads
# back, and (for the validator stage) an overlay, so activate_overlay has something to load.
FAKE_TRAINER = '''
import argparse, json, sys
from pathlib import Path
ap = argparse.ArgumentParser()
for flag in ("--images", "--labels", "--classes", "--work-dir", "--out-overlay", "--out-report",
             "--device", "--det-epochs", "--det-size", "--det-model-type", "--val-epochs",
             "--det-mode", "--stage", "--train-scope", "--noise-config", "--crops-dir"):
    ap.add_argument(flag)
a, _unknown = ap.parse_known_args()
print(f"[fake] stage={a.stage} scope={a.train_scope} device={a.device}")
report = {"ok": True, "stage": a.stage, "device": a.device, "seconds": 0.1,
          "detector": {"map50": 0.5, "model": "fake"}}
if a.stage in ("full", "validator"):
    report["validator"] = {"best_val_acc": 0.9}
    overlay = {p.name: [{"box_id": p.name + ":0", "box_xyxy": [1, 1, 20, 20], "label": 0,
                         "det_conf": 0.9, "p_good": 0.9}]
               for p in sorted(Path(a.images).iterdir()) if p.suffix == ".jpg"}
    Path(a.out_overlay).write_text(json.dumps(overlay))
    report["overlay"] = a.out_overlay
Path(a.out_report).write_text(json.dumps(report))
sys.exit(0)
'''

# A stand-in for tools/gen_noise_crops.py: writes real thumbnail files and a manifest naming them.
FAKE_CROPPER = '''
import argparse, json
from pathlib import Path
from PIL import Image
ap = argparse.ArgumentParser()
for flag in ("--images", "--labels", "--work-dir", "--out-dir", "--manifest", "--device",
             "--train-scope", "--noise-config"):
    ap.add_argument(flag)
a, _ = ap.parse_known_args()
cfg = json.loads(Path(a.noise_config).read_text()) if a.noise_config else {}
out = Path(a.out_dir)
names = {"good": [], "empty": [], "deviated": []}
for kind, sub in (("good", "clf_train_yes"), ("empty", "clf_train_no"), ("deviated", "clf_train_no")):
    (out / sub).mkdir(parents=True, exist_ok=True)
    for i in range(2):
        fn = f"{kind}_{i}.jpg"
        Image.new("RGB", (32, 32), (10, 20, 30)).save(out / sub / fn)
        names[kind].append(fn)
print("[fake] crops generated with shift=" + str(cfg.get("deviation_shift")))
Path(a.manifest).write_text(json.dumps({
    "ok": True, "seed_images": 2, "counts": {"good": 2, "empty": 2, "deviated": 2, "noise": 4,
                                             "total": 6},
    "config": cfg, "samples": names}))
'''

FAILING_TRAINER = '''
import sys
print("[fake] exploding on purpose")
sys.exit(3)
'''


class TrainingCase(unittest.TestCase):
    """Points the app at a fake tools/ folder and a real (temporary) dataset."""

    trainer = FAKE_TRAINER

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="pgtrain-"))
        self.images, self.labels = make_dataset(self.tmp, n_images=3)
        for name in sorted(p.name for p in self.images.iterdir())[:2]:
            write_boxes(self.labels, Path(name).stem, [(0, 0.5, 0.5, 0.2, 0.2)])

        self.tools = self.tmp / "tools"
        self.tools.mkdir()
        (self.tools / "train_and_predict.py").write_text(self.trainer, encoding="utf-8")
        (self.tools / "gen_noise_crops.py").write_text(FAKE_CROPPER, encoding="utf-8")

        state.reset_for_tests()
        state.CFG.update(images=self.images, labels=self.labels, classes=["cat", "dog"],
                         train_python=sys.executable, train_device="auto", det_epochs=1,
                         det_size="n", det_model_type="yolov8", val_epochs=1,
                         research_root=self.tmp, can_train=True)
        state.load_human_set()
        self._real_tools_dir = paths.tools_dir
        paths.tools_dir = lambda: self.tools           # the app asks paths for the trainer

    def tearDown(self):
        paths.tools_dir = self._real_tools_dir
        state.reset_for_tests()
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestCommandLine(TrainingCase):
    def test_the_command_carries_every_setting_the_trainer_needs(self):
        cmd, overlay, report, how = training.build_train_cmd({}, stage="full")
        self.assertEqual(cmd[0], sys.executable)
        self.assertIn("--images", cmd)
        self.assertIn(str(self.images), cmd)
        self.assertEqual(cmd[cmd.index("--classes") + 1], "cat,dog")
        self.assertEqual(cmd[cmd.index("--stage") + 1], "full")
        self.assertTrue(str(overlay).endswith("overlay_trained.json"))
        self.assertTrue(str(report).endswith("train_report.json"))
        self.assertEqual(how, sys.executable)

    def test_scope_all_is_how_the_cycle_folds_ai_labels_back_in(self):
        cmd, *_ = training.build_train_cmd({"scope": "all"})
        self.assertEqual(cmd[cmd.index("--train-scope") + 1], "all")

    def test_the_approved_crops_are_passed_to_the_validator_stage(self):
        cmd, *_ = training.build_train_cmd({"noise_config_path": "/tmp/nc.json"},
                                           stage="validator", crops_dir="/tmp/crops")
        self.assertEqual(cmd[cmd.index("--stage") + 1], "validator")
        self.assertEqual(cmd[cmd.index("--crops-dir") + 1], "/tmp/crops")
        self.assertEqual(cmd[cmd.index("--noise-config") + 1], "/tmp/nc.json")

    def test_conda_is_the_fallback_when_no_interpreter_is_registered(self):
        state.CFG["train_python"] = None
        state.CFG["train_env"] = "someenv"
        cmd, how = training.launcher_for(Path("x.py"))
        self.assertIn("run", cmd)
        self.assertIn("someenv", cmd)
        self.assertIn("someenv", how)

    def test_the_child_environment_names_the_roots_explicitly(self):
        env = training.train_env_vars()
        self.assertEqual(env["PGLABEL_ROOT"], str(self.tmp))
        self.assertIn("PGLABEL_INSTALL_DIR", env)     # where offline base weights may be dropped
        self.assertEqual(env["PYTHONUTF8"], "1")


class TestTrainOnce(TrainingCase):
    def test_a_successful_run_hot_swaps_the_model_in(self):
        log = []
        ok, report = training.train_once({}, log)
        self.assertTrue(ok, log)
        self.assertTrue(report["ok"])
        self.assertTrue(state.ai_available())          # the overlay became the active backend
        self.assertTrue(any("active AI backend" in line for line in log))

    def test_the_trainers_stdout_is_streamed_into_the_log(self):
        log = []
        training.train_once({}, log)
        self.assertTrue(any("[fake] stage=full" in line for line in log))

    def test_device_auto_is_passed_through_unchanged(self):
        log = []
        training.train_once({}, log)
        self.assertTrue(any("device=auto" in line for line in log))


class TestFailureIsReported(TrainingCase):
    trainer = FAILING_TRAINER

    def test_a_non_zero_exit_is_a_failure_not_a_crash(self):
        log = []
        ok, report = training.train_once({}, log)
        self.assertFalse(ok)
        self.assertFalse(state.ai_available())
        self.assertTrue(any("exploding" in line for line in log))

    def test_a_missing_trainer_reports_a_usable_message(self):
        state.CFG["train_python"] = "/definitely/not/a/python"
        log = []
        ok, _ = training.train_once({}, log)
        self.assertFalse(ok)
        self.assertTrue(any("cannot start the trainer" in line for line in log), log)


class TestReviewGate(TrainingCase):
    """The pause between detector and validator — the human-in-the-loop step."""

    def _run_gated_in_background(self):
        log = []
        result = {}

        def worker():
            result["ok"], result["report"] = training.train_gated({}, log, "train")

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        for _ in range(200):                      # wait for the gate to open (fake steps are fast)
            if state.GATE["active"]:
                break
            time.sleep(0.05)
        return thread, log, result

    def test_the_gate_opens_with_reviewable_crops(self):
        thread, log, result = self._run_gated_in_background()
        try:
            self.assertTrue(state.GATE["active"])
            status = training.gate_status()
            self.assertEqual(status["owner"], "train")
            self.assertEqual(status["crops"]["counts"]["total"], 6)
            # Only the sample files this manifest named may ever be served as thumbnails.
            self.assertIn(("clf_train_yes", "good_0.jpg"), state.GATE["sample_files"])
            self.assertNotIn(("clf_train_no", "../../secret"), state.GATE["sample_files"])
        finally:
            state.GATE["cancel"] = True
            state.GATE["confirm"].set()
            thread.join(timeout=10)

    def test_the_rule_reaches_the_generator(self):
        thread, log, result = self._run_gated_in_background()
        try:
            self.assertTrue(any("shift=0.8" in line for line in log), log)
            self.assertAlmostEqual(state.GATE["noise_config"]["deviation_shift"], 0.80)
        finally:
            state.GATE["cancel"] = True
            state.GATE["confirm"].set()
            thread.join(timeout=10)

    def test_regenerating_with_a_new_shift_re_runs_the_generator(self):
        thread, log, result = self._run_gated_in_background()
        try:
            from pglabel import noise_rule
            cfg = noise_rule.validate_noise_config({"deviation_shift": 1.2})
            manifest = training.regen_crops(cfg, log)
            self.assertTrue(manifest["ok"])
            self.assertTrue(any("shift=1.2" in line for line in log), log)
            self.assertAlmostEqual(state.GATE["noise_config"]["deviation_shift"], 1.2)
        finally:
            state.GATE["cancel"] = True
            state.GATE["confirm"].set()
            thread.join(timeout=10)

    def test_confirming_completes_the_run_and_activates_the_model(self):
        thread, log, result = self._run_gated_in_background()
        state.GATE["confirm"].set()               # what POST /api/train/confirm does
        thread.join(timeout=30)
        self.assertTrue(result.get("ok"), log)
        self.assertFalse(state.GATE["active"])    # the gate is closed either way
        self.assertTrue(state.ai_available())

    def test_cancelling_abandons_the_run_without_activating_anything(self):
        thread, log, result = self._run_gated_in_background()
        state.GATE["cancel"] = True
        state.GATE["confirm"].set()
        thread.join(timeout=30)
        self.assertFalse(result.get("ok"))
        self.assertFalse(state.GATE["active"])
        self.assertFalse(state.ai_available())
        self.assertTrue(any("canceled" in line for line in log))


class TestStop(TrainingCase):
    # A trainer that would run for a long time, so Stop has something to kill.
    trainer = "import time\nprint('[fake] running', flush=True)\ntime.sleep(60)\n"

    def test_stop_kills_the_subprocess_and_marks_the_job_stopped(self):
        out, code = training.start_training({})
        self.assertEqual(code, 200, out)
        for _ in range(200):
            if state.ACTIVE_PROC["proc"] is not None:
                break
            time.sleep(0.05)
        self.assertIsNotNone(state.ACTIVE_PROC["proc"], "the trainer never started")
        training.stop_training()
        for _ in range(200):
            if state.TRAIN["state"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(state.TRAIN["state"], "stopped")
        self.assertIsNone(state.ACTIVE_PROC["proc"])

    def test_a_second_job_is_refused_while_one_is_running(self):
        training.start_training({})
        for _ in range(200):
            if state.TRAIN["state"] == "running":
                break
            time.sleep(0.05)
        out, code = training.start_training({})
        self.assertEqual(code, 409)
        out, code = training.start_cycle({})
        self.assertEqual(code, 409)
        training.stop_training()
        for _ in range(200):
            if state.TRAIN["state"] != "running":
                break
            time.sleep(0.05)


class TestPreconditions(TrainingCase):
    def test_training_is_refused_when_the_library_is_missing(self):
        state.CFG["can_train"] = False
        out, code = training.start_training({})
        self.assertEqual(code, 400)
        self.assertIn("unavailable", out["detail"])

    def test_training_is_refused_with_no_labeled_image(self):
        for f in self.labels.glob("*.txt"):
            f.unlink()
        state.HUMAN_SET.clear()
        out, code = training.start_training({})
        self.assertEqual(code, 400)
        self.assertIn("at least 1 image", out["detail"])


if __name__ == "__main__":
    unittest.main()
