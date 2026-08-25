#!/usr/bin/env python3
r"""Filesystem operations that behave the same on Windows as everywhere else.

Two POSIX assumptions break on Windows, and both break during cleanup — the moment when an
error is least expected and most confusing:

1. **Read-only files refuse to be deleted.** ``shutil.rmtree`` raises ``PermissionError`` rather
   than removing them. Virtual environments, extracted Python runtimes and pip caches are full
   of read-only files, so ``build.py --clean`` and the uninstaller both walk straight into it.
   The fix is to clear the read-only bit and retry, which is what every installer does.

2. **A file that is open cannot be deleted or replaced at all.** Nothing can fix that from the
   outside, so the caller needs to know it happened rather than silently continuing with a
   half-removed tree — hence a return value instead of ``ignore_errors=True``.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path


def _clear_readonly(func, target, _exc):
    """rmtree error hook: drop the read-only bit and retry once."""
    try:
        os.chmod(target, stat.S_IWRITE)
        func(target)
    except OSError:
        pass                      # genuinely locked (in use) — reported by remove_tree's result


def remove_tree(path) -> bool:
    """Delete a directory tree. Returns True when nothing is left behind.

    Never raises: cleanup failing must not abort the operation that was cleaning up. Callers
    that care — an installer, a release build — check the result and tell the user which file
    is in use.
    """
    p = Path(path)
    if not p.exists():
        return True
    # Python 3.12 renamed the hook; passing the wrong one is silently ignored, so pick by version.
    if sys.version_info >= (3, 12):
        shutil.rmtree(p, onexc=_clear_readonly)
    else:
        shutil.rmtree(p, onerror=_clear_readonly)
    return not p.exists()


def remove_file(path) -> bool:
    """Delete one file, clearing the read-only bit if that is what is stopping it."""
    p = Path(path)
    try:
        p.unlink(missing_ok=True)
        return True
    except PermissionError:
        try:
            os.chmod(p, stat.S_IWRITE)
            p.unlink(missing_ok=True)
            return True
        except OSError:
            return False
    except OSError:
        return False


def replace_file(src, dst) -> bool:
    """Move ``src`` over ``dst``.

    ``Path.rename`` fails on Windows when the destination exists; ``os.replace`` is the
    cross-platform atomic form, and it is what log rotation and checkpoint writes need.
    """
    try:
        os.replace(str(src), str(dst))
        return True
    except OSError:
        return False
