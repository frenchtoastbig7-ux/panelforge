"""Render the app icon and pack it into AppIcon.icns.

Drawn in code so the icon rebuilds with the app and needs no external asset.
Palette matches the UI accents (#7c5cff violet, #4ad0c0 teal).
"""
from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
S = 4                       # supersample, then downscale for clean edges
SIZE = 1024 * S


def lerp(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def render() -> Image.Image:
    violet, teal, ink = (124, 92, 255), (74, 208, 192), (22, 20, 38)
    # Diagonal gradient tile.
    grad = Image.new("RGB", (SIZE, SIZE))
    px = grad.load()
    for y in range(SIZE):
        for x in range(0, SIZE, S):
            c = lerp(violet, teal, min(1.0, (x + y) / (2 * SIZE) * 1.15))
            for dx in range(S):
                px[x + dx, y] = c

    # macOS icon grid: ~80% body with a continuous-corner rounded square.
    margin = int(SIZE * 0.1)
    body = (margin, margin, SIZE - margin, SIZE - margin)
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle(body, radius=int(SIZE * 0.18), fill=255)

    icon = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (body[0], body[1] + SIZE // 90, body[2], body[3] + SIZE // 90),
        radius=int(SIZE * 0.18), fill=(0, 0, 0, 90))
    icon.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(SIZE // 60)))
    icon.paste(grad, (0, 0), mask)

    d = ImageDraw.Draw(icon)
    # Manga speech bubble with a tail.
    bx0, by0, bx1, by1 = (int(SIZE * v) for v in (0.23, 0.26, 0.77, 0.66))
    d.polygon([(SIZE * 0.36, SIZE * 0.60), (SIZE * 0.30, SIZE * 0.78), (SIZE * 0.48, SIZE * 0.63)],
              fill=(255, 255, 255, 255))
    d.ellipse((bx0, by0, bx1, by1), fill=(255, 255, 255, 255))

    # Four-point sparkle inside the bubble: the "generate" mark.
    cx, cy, r = SIZE * 0.5, SIZE * 0.46, SIZE * 0.13
    pts = []
    for i in range(8):
        a = math.pi / 4 * i - math.pi / 2
        rr = r if i % 2 == 0 else r * 0.28
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    d.polygon(pts, fill=violet + (255,))
    sr = r * 0.34
    sx, sy = SIZE * 0.63, SIZE * 0.35
    small = []
    for i in range(8):
        a = math.pi / 4 * i - math.pi / 2
        rr = sr if i % 2 == 0 else sr * 0.3
        small.append((sx + rr * math.cos(a), sy + rr * math.sin(a)))
    d.polygon(small, fill=teal + (255,))
    # Ink outline on the bubble for manga flavour.
    d.ellipse((bx0, by0, bx1, by1), outline=ink + (255,), width=SIZE // 110)
    return icon.resize((1024, 1024), Image.LANCZOS)


def main() -> int:
    master = render()
    master.save(HERE / "AppIcon.png")
    iconset = HERE / "AppIcon.iconset"
    shutil.rmtree(iconset, ignore_errors=True)
    iconset.mkdir()
    for pt in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            px = pt * scale
            name = f"icon_{pt}x{pt}{'@2x' if scale == 2 else ''}.png"
            master.resize((px, px), Image.LANCZOS).save(iconset / name)
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(HERE / "AppIcon.icns")], check=True)
    shutil.rmtree(iconset)
    print(f"wrote {HERE / 'AppIcon.icns'} ({(HERE / 'AppIcon.icns').stat().st_size // 1024} KB) and AppIcon.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
