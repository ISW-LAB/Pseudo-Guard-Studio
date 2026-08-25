#!/usr/bin/env python3
"""One place that decides which torch device to run on.

Every entry point takes ``--device auto`` by default. "auto" means: use the GPU when there
really is one, otherwise run on the CPU rather than crashing. A machine without CUDA — a
laptop, a CI runner, a reviewer reproducing the paper — therefore needs no flags at all, and
an explicitly requested ``cuda:0`` that is not present degrades to CPU with one warning
instead of a ``RuntimeError`` twenty minutes into a run.

Import-safe without torch: ``resolve()`` falls back to "cpu" if torch is missing, so the
label-only app (which never imports torch) can still call it.
"""

from __future__ import annotations

from typing import Optional

_WARNED = False


def cuda_available() -> bool:
    """True when torch is installed AND reports at least one usable CUDA device."""
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:            # a broken driver raises here rather than returning False
        return False


def resolve(preferred: Optional[str] = "auto", *, log=print) -> str:
    """Turn a requested device into one that actually exists.

    ``auto``/``None``  -> "cuda:0" when CUDA is usable, else "cpu"
    ``cuda*``          -> unchanged when CUDA is usable, else "cpu" (warned once)
    anything else      -> returned unchanged (``cpu``, ``mps``, an explicit index, …)
    """
    global _WARNED
    want = (preferred or "auto").strip().lower()
    if want in ("auto", ""):
        return "cuda:0" if cuda_available() else "cpu"
    if want.startswith("cuda") and not cuda_available():
        if log and not _WARNED:
            _WARNED = True
            log("[device] CUDA was requested but is not available -> running on CPU (slower)")
        return "cpu"
    return want


def describe(device: str) -> str:
    """Human-readable one-liner for the training log ("cuda:0 (NVIDIA RTX A6000, 48 GB)")."""
    if not str(device).startswith("cuda"):
        return str(device)
    try:
        import torch
        idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
        props = torch.cuda.get_device_properties(idx)
        return f"{device} ({props.name}, {props.total_memory / 1024**3:.0f} GB)"
    except Exception:
        return str(device)
