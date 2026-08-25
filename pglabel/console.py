#!/usr/bin/env python3
r"""Make text output survive a Windows console, whatever code page it is set to.

The problem, precisely. Since Python 3.6 a real Windows console is written through
``WriteConsoleW`` and handles any character. But the moment stdout is REDIRECTED — piped into
a file, captured by a build script, read by a parent process — Python falls back to the
machine's ANSI code page, and that code page is small:

    character   cp949 (Korean)   cp1252 (Western)   cp437 (US OEM)
    —  em dash      FAILS              ok               FAILS
    →  arrow          ok              FAILS             FAILS
    ⏸ ✅ ⏹         FAILS             FAILS             FAILS

A single un-encodable character raises ``UnicodeEncodeError`` and takes the whole run with it.
That is why ``PG-Label.exe > log.txt`` on a Korean Windows could die where the same command in
a console window worked — the least reproducible kind of bug report.

``enable()`` forces UTF-8 with ``errors="replace"`` on both streams, so nothing can raise and
the worst case is one substituted glyph. Call it first thing in any entry point that prints.

It is also None-safe: a windowed PyInstaller build has no stdout at all, and this must not be
the thing that stops the app from starting.
"""

from __future__ import annotations

import sys


def enable(errors: str = "replace") -> bool:
    """Reconfigure stdout/stderr to UTF-8. Returns True if at least one stream was changed.

    Never raises: an unusable stream is left exactly as it was.
    """
    changed = False
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:            # None (windowed build) or a stream that cannot change
            continue
        try:
            reconfigure(encoding="utf-8", errors=errors)
            changed = True
        except (ValueError, OSError):      # detached, already closed, or a non-text stream
            pass
    return changed


def safe(text: str) -> str:
    """A copy of ``text`` that any Windows code page can render.

    For the rare place that must write through a stream this module does not own — a .vbs
    handed to cscript, a .bat, a subprocess argument. Decorative characters become their ASCII
    equivalents rather than being dropped, so the message stays readable.
    """
    replacements = {
        "—": "-", "–": "-",       # em dash, en dash
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "…": "...", "→": "->", "≥": ">=", "≤": "<=",
        "·": "*", "•": "*", "▸": ">", "─": "-",
        "⏸": "[paused]", "⏹": "[stopped]", "✅": "[done]",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("ascii", "replace").decode("ascii")
