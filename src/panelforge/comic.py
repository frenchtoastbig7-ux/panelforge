"""Comic page composition: panel layouts, gutters, speech bubbles, lettering.

This is the part that makes the tool a manga studio rather than an image
generator. None of it touches the GPU — a page is composited from already
generated panel images, so layout and lettering stay instant and re-editable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont

BubbleStyle = Literal["speech", "thought", "shout", "narration", "whisper", "sfx"]
ReadingOrder = Literal["ltr", "rtl"]

# macOS system fonts. Each entry is a list of candidates tried in order, so a
# missing font degrades to the next rather than crashing the page render.
FONT_CANDIDATES: dict[str, list[str]] = {
    "comic": ["/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
              "/Library/Fonts/Comic Sans MS.ttf"],
    "comic_bold": ["/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf"],
    "impact": ["/System/Library/Fonts/Supplemental/Impact.ttf"],
    "marker": ["/System/Library/Fonts/Supplemental/MarkerFelt.ttc"],
    "sans": ["/System/Library/Fonts/Helvetica.ttc",
             "/System/Library/Fonts/HelveticaNeue.ttc"],
    # CJK, needed for vertical Japanese lettering.
    "cjk": ["/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"],
}

# Default font per bubble style, matching comic lettering convention.
STYLE_FONT: dict[BubbleStyle, str] = {
    "speech": "comic", "thought": "comic", "whisper": "comic",
    "narration": "sans", "shout": "comic_bold", "sfx": "impact",
}


def resolve_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a font by logical name, falling back through candidates."""
    for path in FONT_CANDIDATES.get(name, []):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    for group in FONT_CANDIDATES.values():
        for path in group:
            if Path(path).exists():
                try:
                    return ImageFont.truetype(path, size)
                except OSError:
                    continue
    return ImageFont.load_default(size)


def available_fonts() -> list[str]:
    return [name for name, paths in FONT_CANDIDATES.items()
            if any(Path(p).exists() for p in paths)]


# --------------------------------------------------------------------- model

@dataclass
class Panel:
    """A panel rectangle in relative page coordinates (0-1)."""
    x: float
    y: float
    w: float
    h: float
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = -1
    image: str | None = None      # filename in the outputs dir
    # Framing nudge when the generated image and the panel differ in aspect.
    focus_x: float = 0.5
    focus_y: float = 0.45         # faces sit above centre more often than not

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Bubble:
    """A lettering element in relative page coordinates (0-1)."""
    x: float
    y: float
    w: float
    h: float
    text: str = ""
    style: BubbleStyle = "speech"
    font: str | None = None
    font_size: int | None = None  # auto-fit when None
    vertical: bool = False        # top-to-bottom, columns right-to-left
    # Tail target in page coords; None draws no tail.
    tail_x: float | None = None
    tail_y: float | None = None
    rotation: float = 0.0         # degrees, mainly for sfx

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Page:
    width: int = 1448             # B5 at 200dpi, a common manga page size
    height: int = 2048
    panels: list[Panel] = field(default_factory=list)
    bubbles: list[Bubble] = field(default_factory=list)
    gutter: float = 0.018         # relative to the short edge
    margin: float = 0.04
    border_width: int = 5
    border_color: str = "#000000"
    background: str = "#ffffff"
    reading_order: ReadingOrder = "rtl"

    def to_dict(self) -> dict:
        return {
            "width": self.width, "height": self.height,
            "panels": [p.to_dict() for p in self.panels],
            "bubbles": [b.to_dict() for b in self.bubbles],
            "gutter": self.gutter, "margin": self.margin,
            "border_width": self.border_width, "border_color": self.border_color,
            "background": self.background, "reading_order": self.reading_order,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Page":
        return cls(
            width=int(data.get("width", 1448)),
            height=int(data.get("height", 2048)),
            panels=[Panel(**p) for p in data.get("panels", [])],
            bubbles=[Bubble(**b) for b in data.get("bubbles", [])],
            gutter=float(data.get("gutter", 0.018)),
            margin=float(data.get("margin", 0.04)),
            border_width=int(data.get("border_width", 5)),
            border_color=data.get("border_color", "#000000"),
            background=data.get("background", "#ffffff"),
            reading_order=data.get("reading_order", "rtl"),
        )


# ------------------------------------------------------------------- layouts

# Each preset is a list of (x, y, w, h) in relative coordinates before gutters
# and margins are applied.
LAYOUTS: dict[str, list[tuple[float, float, float, float]]] = {
    "single":       [(0, 0, 1, 1)],
    "two-across":   [(0, 0, .5, 1), (.5, 0, .5, 1)],
    "two-stack":    [(0, 0, 1, .5), (0, .5, 1, .5)],
    "three-strip":  [(0, 0, 1, 1/3), (0, 1/3, 1, 1/3), (0, 2/3, 1, 1/3)],
    "grid-2x2":     [(0, 0, .5, .5), (.5, 0, .5, .5), (0, .5, .5, .5), (.5, .5, .5, .5)],
    "grid-2x3":     [(0, 0, .5, 1/3), (.5, 0, .5, 1/3), (0, 1/3, .5, 1/3),
                     (.5, 1/3, .5, 1/3), (0, 2/3, .5, 1/3), (.5, 2/3, .5, 1/3)],
    # Establishing shot on top, beats below, payoff across the bottom.
    "manga-classic": [(0, 0, 1, .28), (0, .28, .5, .24), (.5, .28, .5, .24),
                      (0, .52, .55, .24), (.55, .52, .45, .24), (0, .76, 1, .24)],
    "action":       [(0, 0, 1, .45), (0, .45, .38, .27), (.38, .45, .62, .27),
                     (0, .72, 1, .28)],
    "4-koma":       [(0, 0, 1, .25), (0, .25, 1, .25), (0, .5, 1, .25), (0, .75, 1, .25)],
    "webtoon":      [(0, 0, 1, .2), (0, .2, 1, .2), (0, .4, 1, .2),
                     (0, .6, 1, .2), (0, .8, 1, .2)],
    "splash-inset": [(0, 0, 1, 1), (.62, .66, .34, .3)],
}

PAGE_PRESETS: dict[str, tuple[int, int]] = {
    "manga-b5": (1448, 2048),
    "comic-us": (1275, 1950),
    "square": (1600, 1600),
    "webtoon": (1200, 3600),
    "landscape": (2048, 1152),
}


def build_page(layout: str, page_preset: str = "manga-b5",
               reading_order: ReadingOrder = "rtl") -> Page:
    """Create a page with panels positioned according to a layout preset."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; choose from {sorted(LAYOUTS)}")
    w, h = PAGE_PRESETS.get(page_preset, PAGE_PRESETS["manga-b5"])
    cells = LAYOUTS[layout]
    if reading_order == "rtl" and layout != "splash-inset":
        # Manga reads right to left, so mirror the cells horizontally. The
        # panel order in the list stays the reading order either way.
        cells = [(1 - x - cw, y, cw, ch) for (x, y, cw, ch) in cells]
    return Page(width=w, height=h,
                panels=[Panel(x=x, y=y, w=cw, h=ch) for (x, y, cw, ch) in cells],
                reading_order=reading_order)


def layout_names() -> list[str]:
    return list(LAYOUTS)


# ------------------------------------------------------------------ geometry

def panel_rect(page: Page, panel: Panel) -> tuple[int, int, int, int]:
    """Absolute pixel rect for a panel, after margins and gutters."""
    short = min(page.width, page.height)
    m = page.margin * short
    g = page.gutter * short / 2
    inner_w = page.width - 2 * m
    inner_h = page.height - 2 * m
    x0 = m + panel.x * inner_w + g
    y0 = m + panel.y * inner_h + g
    x1 = m + (panel.x + panel.w) * inner_w - g
    y1 = m + (panel.y + panel.h) * inner_h - g
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def panel_pixel_size(page: Page, panel: Panel) -> tuple[int, int]:
    x0, y0, x1, y1 = panel_rect(page, panel)
    return max(1, x1 - x0), max(1, y1 - y0)


def nearest_bucket(width: int, height: int,
                   buckets: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """Pick the training bucket whose aspect ratio best matches a panel.

    Generating at the panel's exact pixel size would drift off the resolutions
    the model was trained on and degrade anatomy; generating at the closest
    bucket and cropping keeps quality.
    """
    target = width / height
    return min(buckets, key=lambda b: abs(math.log((b[0] / b[1]) / target)))


def _cover_crop(img: Image.Image, w: int, h: int,
                focus_x: float, focus_y: float) -> Image.Image:
    """Scale to cover the rect, then crop around the focus point."""
    scale = max(w / img.width, h / img.height)
    new = img.resize((max(1, round(img.width * scale)),
                      max(1, round(img.height * scale))), Image.LANCZOS)
    max_left = max(0, new.width - w)
    max_top = max(0, new.height - h)
    left = int(round(max_left * min(max(focus_x, 0.0), 1.0)))
    top = int(round(max_top * min(max(focus_y, 0.0), 1.0)))
    return new.crop((left, top, left + w, top + h))


# ----------------------------------------------------------------- lettering

def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: int,
          draw: ImageDraw.ImageDraw) -> list[str]:
    """Greedy word wrap, falling back to character wrap for CJK runs."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split(" ")
        current = ""
        for word in words:
            trial = f"{current} {word}".strip()
            if draw.textlength(trial, font=font) <= max_w or not current:
                # A single word wider than the bubble must break by character.
                if draw.textlength(trial, font=font) > max_w and not current:
                    part = ""
                    for ch in word:
                        if draw.textlength(part + ch, font=font) > max_w and part:
                            lines.append(part)
                            part = ch
                        else:
                            part += ch
                    current = part
                else:
                    current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def _fit_font(text: str, bubble: Bubble, box_w: int, box_h: int,
              draw: ImageDraw.ImageDraw) -> tuple[ImageFont.FreeTypeFont, list[str], int]:
    """Shrink the font until the wrapped text fits the bubble interior."""
    name = bubble.font or STYLE_FONT.get(bubble.style, "comic")
    if bubble.vertical:
        name = "cjk"
    start = bubble.font_size or max(10, int(box_h * 0.28))
    for size in range(start, 7, -1):
        font = resolve_font(name, size)
        lines = _wrap(text, font, box_w, draw)
        line_h = int(size * 1.25)
        if len(lines) * line_h <= box_h:
            return font, lines, line_h
        if bubble.font_size:      # explicit size: honour it even if it overflows
            return font, lines, line_h
    font = resolve_font(name, 8)
    return font, _wrap(text, font, box_w, draw), 10


def _vertical_columns(text: str, font: ImageFont.FreeTypeFont,
                      box_h: int) -> list[str]:
    """Split text into top-to-bottom columns for vertical CJK lettering."""
    ascent, descent = font.getmetrics()
    ch_h = ascent + descent
    per_col = max(1, box_h // max(1, ch_h))
    stripped = text.replace("\n", "")
    return [stripped[i:i + per_col] for i in range(0, len(stripped), per_col)] or [""]


def _bubble_outline(draw: ImageDraw.ImageDraw, style: BubbleStyle,
                    box: tuple[int, int, int, int], width: int) -> None:
    x0, y0, x1, y1 = box
    if style == "narration":
        draw.rectangle(box, fill="#ffffff", outline="#000000", width=width)
    elif style == "thought":
        # A ring of overlapping circles reads as a thought cloud.
        draw.ellipse(box, fill="#ffffff", outline=None)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
        bumps = max(8, int((rx + ry) / 18))
        for i in range(bumps):
            a = 2 * math.pi * i / bumps
            bx, by = cx + rx * math.cos(a), cy + ry * math.sin(a)
            r = min(rx, ry) * 0.22
            draw.ellipse((bx - r, by - r, bx + r, by + r),
                         fill="#ffffff", outline="#000000", width=max(1, width - 1))
        draw.ellipse((x0 + rx * .22, y0 + ry * .22, x1 - rx * .22, y1 - ry * .22),
                     fill="#ffffff", outline=None)
    elif style in ("shout", "sfx"):
        # Spiky burst.
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
        spikes = 16
        pts = []
        for i in range(spikes * 2):
            a = math.pi * i / spikes - math.pi / 2
            f = 1.0 if i % 2 == 0 else 0.72
            pts.append((cx + rx * f * math.cos(a), cy + ry * f * math.sin(a)))
        draw.polygon(pts, fill="#ffffff", outline="#000000")
        if width > 1:
            draw.line(pts + [pts[0]], fill="#000000", width=width, joint="curve")
    elif style == "whisper":
        for dash in _dashed_ellipse(box, 26):
            draw.arc(box, dash[0], dash[1], fill="#000000", width=width)
        draw.ellipse((x0 + width, y0 + width, x1 - width, y1 - width), fill="#ffffff")
        for dash in _dashed_ellipse(box, 26):
            draw.arc(box, dash[0], dash[1], fill="#000000", width=width)
    else:  # speech
        draw.ellipse(box, fill="#ffffff", outline="#000000", width=width)


def _dashed_ellipse(box, segments: int) -> list[tuple[int, int]]:
    step = 360 / segments
    return [(int(i * step), int(i * step + step * 0.55)) for i in range(segments)]


def _draw_tail(draw: ImageDraw.ImageDraw, bubble: Bubble, page: Page,
               box: tuple[int, int, int, int], width: int) -> None:
    if bubble.tail_x is None or bubble.tail_y is None:
        return
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    tx, ty = bubble.tail_x * page.width, bubble.tail_y * page.height
    ang = math.atan2(ty - cy, tx - cx)
    # Two base points on the bubble edge, straddling the tail direction.
    spread = 0.22
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    base = [(cx + rx * math.cos(ang + s), cy + ry * math.sin(ang + s))
            for s in (-spread, spread)]
    if bubble.style == "thought":
        # Thought tails are a trail of shrinking bubbles, not a spike.
        for i in range(1, 4):
            f = i / 4
            bx = cx + (tx - cx) * f
            by = cy + (ty - cy) * f
            r = max(2.0, min(rx, ry) * 0.16 * (1 - f * 0.6))
            draw.ellipse((bx - r, by - r, bx + r, by + r),
                         fill="#ffffff", outline="#000000", width=max(1, width - 1))
        return
    draw.polygon([base[0], base[1], (tx, ty)], fill="#ffffff", outline="#000000")
    draw.line([base[0], (tx, ty)], fill="#000000", width=width)
    draw.line([base[1], (tx, ty)], fill="#000000", width=width)


def draw_bubble(page_img: Image.Image, page: Page, bubble: Bubble) -> None:
    """Render one bubble (outline, tail and text) onto the page."""
    if not bubble.text.strip() and bubble.style != "sfx":
        return
    x0 = int(bubble.x * page.width)
    y0 = int(bubble.y * page.height)
    x1 = int((bubble.x + bubble.w) * page.width)
    y1 = int((bubble.y + bubble.h) * page.height)
    if x1 <= x0 or y1 <= y0:
        return
    box = (x0, y0, x1, y1)
    width = max(2, page.border_width // 2)

    # SFX is lettering with no container: draw it on its own layer so it can
    # be rotated without dragging a white background along.
    if bubble.style == "sfx":
        _draw_sfx(page_img, page, bubble, box)
        return

    layer = Image.new("RGBA", page_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    _draw_tail(draw, bubble, page, box, width)
    _bubble_outline(draw, bubble.style, box, width)
    page_img.alpha_composite(layer)

    draw = ImageDraw.Draw(page_img)
    pad = int(min(x1 - x0, y1 - y0) * (0.20 if bubble.style in ("thought", "shout") else 0.13))
    bw, bh = (x1 - x0) - 2 * pad, (y1 - y0) - 2 * pad
    if bw <= 0 or bh <= 0:
        return

    if bubble.vertical:
        font = resolve_font("cjk", bubble.font_size or max(12, int(bh * 0.16)))
        cols = _vertical_columns(bubble.text, font, bh)
        ascent, descent = font.getmetrics()
        ch_h = ascent + descent
        col_w = int(ch_h * 1.15)
        # Columns run right to left.
        start_x = (x0 + x1) / 2 + (len(cols) - 1) * col_w / 2
        for ci, col in enumerate(cols):
            cx = start_x - ci * col_w
            cy = (y0 + y1) / 2 - len(col) * ch_h / 2
            for chi, ch in enumerate(col):
                draw.text((cx, cy + chi * ch_h), ch, font=font,
                          fill="#000000", anchor="mm")
        return

    font, lines, line_h = _fit_font(bubble.text, bubble, bw, bh, draw)
    total = len(lines) * line_h
    cy = (y0 + y1) / 2 - total / 2 + line_h / 2
    for i, line in enumerate(lines):
        draw.text(((x0 + x1) / 2, cy + i * line_h), line, font=font,
                  fill="#000000", anchor="mm")


def _draw_sfx(page_img: Image.Image, page: Page, bubble: Bubble,
              box: tuple[int, int, int, int]) -> None:
    """Sound effect lettering: heavy outlined text, optionally rotated."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    pad = int(max(bw, bh) * 0.25)
    layer = Image.new("RGBA", (bw + 2 * pad, bh + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    name = bubble.font or "impact"
    size = bubble.font_size or max(14, int(bh * 0.8))
    for _ in range(24):
        font = resolve_font(name, size)
        if draw.textlength(bubble.text, font=font) <= bw or size <= 10:
            break
        size -= max(1, size // 12)
    font = resolve_font(name, size)
    cx, cy = layer.width / 2, layer.height / 2
    stroke = max(3, size // 10)
    draw.text((cx, cy), bubble.text, font=font, fill="#ffffff", anchor="mm",
              stroke_width=stroke, stroke_fill="#000000")
    if bubble.rotation:
        layer = layer.rotate(bubble.rotation, resample=Image.BICUBIC,
                             expand=False, center=(cx, cy))
    page_img.alpha_composite(layer, (x0 - pad, y0 - pad))


# ------------------------------------------------------------------- compose

def render_page(page: Page, outputs_dir: Path,
                draw_placeholders: bool = True) -> Image.Image:
    """Composite panels and lettering into a finished page."""
    img = Image.new("RGBA", (page.width, page.height), page.background)

    for index, panel in enumerate(page.panels):
        x0, y0, x1, y1 = panel_rect(page, panel)
        w, h = max(1, x1 - x0), max(1, y1 - y0)
        art = None
        if panel.image:
            path = (outputs_dir / panel.image).resolve()
            # Confine to the outputs directory: panel.image arrives from a client.
            if path.is_file() and outputs_dir.resolve() in path.parents:
                try:
                    art = Image.open(path).convert("RGBA")
                except OSError:
                    art = None
        if art is not None:
            img.paste(_cover_crop(art, w, h, panel.focus_x, panel.focus_y), (x0, y0))
        elif draw_placeholders:
            _placeholder(img, (x0, y0, x1, y1), index, panel, page)
        if page.border_width > 0:
            ImageDraw.Draw(img).rectangle((x0, y0, x1, y1),
                                          outline=page.border_color,
                                          width=page.border_width)

    for bubble in page.bubbles:
        draw_bubble(img, page, bubble)
    return img.convert("RGB")


def _placeholder(img: Image.Image, box: tuple[int, int, int, int],
                 index: int, panel: Panel, page: Page) -> None:
    draw = ImageDraw.Draw(img)
    draw.rectangle(box, fill="#eef0f4")
    x0, y0, x1, y1 = box
    font = resolve_font("sans", max(14, int(min(x1 - x0, y1 - y0) * 0.10)))
    label = f"{index + 1}"
    draw.text(((x0 + x1) / 2, (y0 + y1) / 2), label, font=font,
              fill="#aab0bd", anchor="mm")
    if panel.prompt:
        small = resolve_font("sans", max(10, int(min(x1 - x0, y1 - y0) * 0.045)))
        text = panel.prompt[:70] + ("…" if len(panel.prompt) > 70 else "")
        draw.text(((x0 + x1) / 2, (y0 + y1) / 2 + (y1 - y0) * 0.12), text,
                  font=small, fill="#99a0ad", anchor="mm")
