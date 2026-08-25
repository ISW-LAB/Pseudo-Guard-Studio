#!/usr/bin/env python3
"""Generate packaging/assets/pglabel.ico — the app icon baked into PG-Label.exe.

Drawn rather than shipped as a binary blob so it stays editable and reviewable. The mark is
the app in one glyph: a labeled bounding box (corner handles + a class tag) on the indigo the
UI uses for its accent colour (--acc #6366f1 on --bg #0e1016).

    python packaging/make_icon.py          # writes packaging/assets/pglabel.ico

Run once; the build only needs the .ico. Requires Pillow (already an app dependency).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

BG = (14, 16, 22, 255)          # --bg
PANEL = (28, 32, 41, 255)       # --panel2
ACC = (99, 102, 241, 255)       # --acc
ACC2 = (143, 152, 251, 255)     # --acc2
FG = (241, 243, 248, 255)       # --fg
SIZES = [256, 128, 64, 48, 32, 16]


def rounded(size: int) -> Image.Image:
    """One square icon layer at `size` px, drawn at 4× and downsampled for clean edges."""
    s = size * 4
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    r = int(s * 0.22)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=r, fill=BG, outline=PANEL, width=max(1, s // 64))

    # the annotation box: a dashed-free rectangle with four solid corner handles
    m = int(s * 0.24)
    box = [m, int(s * 0.30), s - m, s - m]
    d.rectangle(box, outline=ACC, width=max(2, s // 22))
    hs = int(s * 0.055)                                  # handle size
    for cx, cy in ((box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])):
        d.rectangle([cx - hs, cy - hs, cx + hs, cy + hs], fill=ACC2)

    # the class tag riding on the top-left corner — what makes it a *labeling* box
    tw, th = int(s * 0.34), int(s * 0.15)
    tx, ty = box[0] - int(s * 0.02), box[1] - th - int(s * 0.055)
    d.rounded_rectangle([tx, ty, tx + tw, ty + th], radius=int(th * 0.35), fill=ACC)
    # three ticks inside the tag read as text at any size (real glyphs vanish below 32 px)
    for i in range(3):
        y = ty + th * (0.32 + 0.2 * i)
        d.line([tx + tw * 0.16, y, tx + tw * (0.84 if i < 2 else 0.6), y],
               fill=FG, width=max(1, int(th * 0.10)))

    return im.resize((size, size), Image.LANCZOS)


def main() -> None:
    out = Path(__file__).resolve().parent / "assets" / "pglabel.ico"
    out.parent.mkdir(parents=True, exist_ok=True)
    layers = [rounded(n) for n in SIZES]
    # Pillow writes every requested size into the one .ico; Windows picks per context
    # (16 px in the taskbar, 256 px in Explorer's extra-large view).
    layers[0].save(out, format="ICO", sizes=[(n, n) for n in SIZES])
    print(f"wrote {out}  ({out.stat().st_size} bytes, sizes={SIZES})")


if __name__ == "__main__":
    main()
