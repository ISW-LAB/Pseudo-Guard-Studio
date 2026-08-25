#!/usr/bin/env python3
"""The HTTP layer: one route table per method, and one request handler that dispatches.

The server is Python's own ``ThreadingHTTPServer`` — no web framework, no third-party
dependency, so the whole app runs on a stock interpreter plus Pillow. That constraint is why
routing is a table here rather than decorators: the table is the complete list of what the UI
can ask for, readable in one screen.

Routes come in two groups, and the split is a real guard rather than bookkeeping:

    PUBLIC    answerable before a dataset is open — the setup screen needs these.
    DATASET   everything else. With no dataset open they return 400 instead of raising
              AttributeError on a None path deep inside a handler.
"""

from __future__ import annotations

import json
import mimetypes
import urllib.parse
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from . import candidates, dataset_setup, export, methods, noise_rule, paths, state, training
from .labelio import image_size, list_images, load_yolo, save_yolo


# ------------------------------------------------------------------- untrusted input
def safe_image(name: str):
    """Resolve an image name that came from the URL, or None if it points outside the dataset.

    Joining the name onto the images folder is NOT enough. ``../secrets`` walks out, and on
    Windows joining an ABSOLUTE path (``C:\\Windows\\win.ini``) discards the base entirely —
    so the same request that is merely awkward on POSIX serves any readable file there. Resolve
    first, then require the result to still be inside the folder.

    Returns the resolved path only when it is a real file, so callers get one check instead of
    three and can answer 404 for "missing" and "not yours" alike — an attacker learns nothing
    about what exists outside the dataset.
    """
    root = state.CFG["images"]
    if root is None:
        return None
    root = Path(root).resolve()
    try:
        candidate = (root / name).resolve()
        candidate.relative_to(root)
    except (ValueError, OSError):
        return None
    return candidate if candidate.is_file() else None


def safe_stem(name: str):
    """The label-file stem for a URL-supplied image name, or None if the name is not ours.

    Writes go through here as well: ``Path(name).stem`` alone happens to strip directories, but
    relying on that would make any future use of the raw name a hole.
    """
    resolved = safe_image(name)
    return resolved.stem if resolved is not None else None


# --------------------------------------------------------------------- GET handlers
def _get_config(req, _suffix=None):
    if state.CFG["images"] is None:                    # setup mode — no dataset yet
        return req.json({"needs_setup": True, "classes": [], "images": [], "has_ai": False,
                         "can_train": state.CFG["can_train"],
                         "defaults": state.CFG.get("setup_defaults", {})})
    methods.compute_adaptive()                         # refresh from the current seed for display
    req.json({"needs_setup": False, "classes": state.CFG["classes"],
              "images": list_images(state.CFG["images"]),
              "has_ai": state.ai_available(), "can_train": state.CFG["can_train"],
              "adaptive": dict(state.ADAPT)})


def _get_browse(req, _suffix=None):
    req.json(dataset_setup.browse(req.query.get("path", [""])[0]))


def _get_datasets(req, _suffix=None):
    req.json({"datasets": dataset_setup.list_datasets()})


def _get_methods(req, _suffix=None):
    req.json({"methods": methods.public_methods()})


def _get_status(req, _suffix=None):
    req.json(candidates.label_status())


def _get_score_summary(req, _suffix=None):
    if not state.ai_available():
        return req.json({"detail": "no AI backend"}, 400)
    req.json(candidates.score_summary())


def _get_train_status(req, _suffix=None):
    req.json(training.train_status())


def _get_cycle_status(req, _suffix=None):
    req.json(training.cycle_status())


def _get_export(req, _suffix=None):
    if req.query.get("fmt", ["coco"])[0] == "yolo":
        req.json(export.export_yolo())
    else:
        req.json(export.export_coco())


def _get_file(req, name):
    p = safe_image(name)
    if p is None:
        return req.json({"detail": "not found"}, 404)
    req.send_bytes(p.read_bytes(), mimetypes.guess_type(str(p))[0] or "image/jpeg")


def _get_labels(req, name):
    p = safe_image(name)
    if p is None:
        return req.json({"detail": "not found"}, 404)
    w, h = image_size(p)
    req.json({"width": w, "height": h, "boxes": load_yolo(state.CFG["labels"], p.stem)})


def _get_candidates(req, name):
    if safe_image(name) is None:
        return req.json({"detail": "not found"}, 404)
    if not state.ai_available():
        return req.json({"detail": "no AI backend"}, 400)
    req.json(candidates.image_candidates(name))


def _get_crop_img(req, _suffix=None):
    """A review-crop thumbnail.

    Sandboxed by WHITELIST, not by path checking: the filename arrives from the browser, and
    only names the current manifest produced may be read off disk.
    """
    kind = req.query.get("type", [""])[0]
    fname = req.query.get("f", [""])[0]
    sub = "clf_train_yes" if kind == "good" else "clf_train_no"
    if not state.GATE["active"] or (sub, fname) not in state.GATE["sample_files"]:
        return req.json({"detail": "not found"}, 404)
    p = Path(state.GATE["crops_dir"]) / sub / fname
    if not p.exists():
        return req.json({"detail": "not found"}, 404)
    # Regeneration reuses filenames, so this one must never be cached.
    req.send_bytes(p.read_bytes(), "image/jpeg", no_store=True)


# -------------------------------------------------------------------- POST handlers
def _post_setup(req, _suffix=None):
    out, code = dataset_setup.run_setup(req.body)
    req.json(out, code)


def _post_labels(req, name):
    stem = safe_stem(name)
    if stem is None:
        return req.json({"detail": "not found"}, 404)
    boxes = req.body.get("boxes", [])
    save_yolo(state.CFG["labels"], stem, boxes)
    if req.body.get("human", True) and name not in state.HUMAN_SET:
        state.HUMAN_SET.add(name)                  # a human edit makes this image seed…
        state.save_human_set()                     # …and that survives restarts
    req.json({"saved": True, "n": len(boxes)})


def _post_classes(req, _suffix=None):
    out, code = dataset_setup.add_class(req.body.get("name"))
    req.json(out, code)


def _post_train(req, _suffix=None):
    out, code = training.start_training(req.body)
    req.json(out, code)


def _post_cycle(req, _suffix=None):
    out, code = training.start_cycle(req.body)
    req.json(out, code)


def _post_train_stop(req, _suffix=None):
    if not state.job_running():
        return req.json({"detail": "no training / cycle job is running"}, 400)
    req.json(training.stop_training())


def _post_train_regen(req, _suffix=None):
    if not state.GATE["active"]:
        return req.json({"detail": "no rule-review gate is open"}, 400)
    cfg = noise_rule.validate_noise_config(req.body.get("config") or {})
    m = training.regen_crops(cfg, state.GATE["log"])
    if not m or not m.get("ok"):
        return req.json({"detail": (m or {}).get("error", "crop generation failed")}, 400)
    req.json({"ok": True, "crops": m})


def _post_train_confirm(req, _suffix=None):
    """Accept the crops and let the paused trainer proceed to the validator stage."""
    if not state.GATE["active"]:
        return req.json({"detail": "no rule-review gate is open"}, 400)
    cfg = noise_rule.validate_noise_config(req.body.get("config") or state.GATE["noise_config"] or {})
    m = training.regen_crops(cfg, state.GATE["log"])   # regenerate with the confirmed rule
    if not m or not m.get("ok"):
        return req.json({"detail": (m or {}).get("error", "crop generation failed")}, 400)
    state.GATE["noise_config"] = cfg
    state.GATE["confirm"].set()                        # unblocks the worker → stage 3
    req.json({"ok": True})


def _post_train_cancel(req, _suffix=None):
    state.GATE["cancel"] = True
    state.GATE["confirm"].set()                        # same event; the worker checks `cancel`
    req.json({"ok": True})


def _post_deviation_preview(req, _suffix=None):
    name = req.body.get("image") or ""
    if not name or safe_image(name) is None:
        return req.json({"detail": "image not found"}, 404)
    cfg = noise_rule.validate_noise_config(req.body.get("config") or {})
    req.json(noise_rule.preview_deviated(name, cfg))


def _post_automate_all(req, _suffix=None):
    if not state.ai_available():
        return req.json({"detail": "no AI backend (manual mode)"}, 400)
    req.json(candidates.automate_all(req.body.get("method", "pseudoguard"),
                                     thr=req.body.get("thr"),
                                     score=req.body.get("score", "p_good")))


def _post_automate(req, name):
    if safe_image(name) is None:
        return req.json({"detail": "not found"}, 404)
    if not state.ai_available():
        return req.json({"detail": "no AI backend (manual mode)"}, 400)
    req.json(candidates.automate_image(name, req.body.get("method", "pseudoguard")))


# ------------------------------------------------------------------------ route tables
GET_PUBLIC = {"/api/config": _get_config, "/api/browse": _get_browse,
              "/api/datasets": _get_datasets}
GET_DATASET = {"/api/methods": _get_methods, "/api/status": _get_status,
               "/api/score_summary": _get_score_summary,
               "/api/train/status": _get_train_status, "/api/cycle/status": _get_cycle_status,
               "/api/train/crop_img": _get_crop_img, "/api/export": _get_export}
GET_DATASET_PREFIX = (("/api/file/", _get_file), ("/api/labels/", _get_labels),
                      ("/api/candidates/", _get_candidates))

POST_PUBLIC = {"/api/setup": _post_setup}
POST_DATASET = {"/api/classes": _post_classes, "/api/train": _post_train,
                "/api/cycle": _post_cycle, "/api/train/stop": _post_train_stop,
                "/api/train/regen": _post_train_regen,
                "/api/train/confirm": _post_train_confirm,
                "/api/train/cancel": _post_train_cancel,
                "/api/train/deviation_preview": _post_deviation_preview,
                "/api/automate_all": _post_automate_all}
POST_DATASET_PREFIX = (("/api/labels/", _post_labels), ("/api/automate/", _post_automate))


def _match_prefix(table, path):
    for prefix, fn in table:
        if path.startswith(prefix):
            return fn, urllib.parse.unquote(path[len(prefix):])
    return None, None


class Handler(BaseHTTPRequestHandler):
    """One request. ``self.query`` and ``self.body`` are parsed once, before dispatch."""

    server_version = "PG-Label"

    def log_message(self, *args):
        pass                                    # the app prints its own single startup line

    # ---------------------------------------------------------------- responses
    def send_bytes(self, data: bytes, ctype: str, code: int = 200, no_store: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # Never let a browser or proxy serve a remembered answer: the UI, the regenerated crops
        # and every api/*.json response describe model state that changes during a session, and
        # a heuristically-cached /api/status would show predictions the model no longer makes.
        if ctype.startswith("text/html") or ctype.startswith("application/json") or no_store:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def json(self, obj, code: int = 200):
        self.send_bytes(json.dumps(obj).encode(), "application/json", code)

    def _read_body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    # ----------------------------------------------------------------- dispatch
    def _dispatch(self, public, dataset, dataset_prefix):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        self.query = urllib.parse.parse_qs(parsed.query)
        fn = public.get(path)
        if fn is not None:
            return fn(self)
        if state.CFG["images"] is None:
            return self.json({"detail": "no dataset — set one on the setup screen first"}, 400)
        fn = dataset.get(path)
        if fn is not None:
            return fn(self)
        fn, suffix = _match_prefix(dataset_prefix, path)
        if fn is not None:
            return fn(self, suffix)
        self.json({"detail": "not found"}, 404)

    def do_GET(self):
        try:
            path = urllib.parse.urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self.send_bytes((paths.static_dir() / "index.html").read_bytes(),
                                       "text/html; charset=utf-8")
            if path.startswith("/static/"):
                return self._serve_static(path[len("/static/"):])
            self._dispatch(GET_PUBLIC, GET_DATASET, GET_DATASET_PREFIX)
        except Exception as e:
            self.json({"detail": str(e)}, 500)

    def do_POST(self):
        try:
            self.body = self._read_body()
            self._dispatch(POST_PUBLIC, POST_DATASET, POST_DATASET_PREFIX)
        except Exception as e:
            self.json({"detail": str(e)}, 500)

    # ------------------------------------------------------------------- static
    def _serve_static(self, rel: str):
        """Serve the UI's own css/js.

        The path is resolved and checked to be INSIDE the static folder, so a crafted
        ``/static/../../secrets`` cannot walk out of the bundle.
        """
        root = paths.static_dir().resolve()
        try:
            p = (root / urllib.parse.unquote(rel)).resolve()
            p.relative_to(root)
        except (ValueError, OSError):
            return self.json({"detail": "not found"}, 404)
        if not p.is_file():
            return self.json({"detail": "not found"}, 404)
        ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        if p.suffix == ".js":
            ctype = "text/javascript; charset=utf-8"
        elif p.suffix == ".css":
            ctype = "text/css; charset=utf-8"
        self.send_bytes(p.read_bytes(), ctype)
