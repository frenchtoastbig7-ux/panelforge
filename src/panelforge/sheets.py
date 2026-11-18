"""Character reference sheets.

A sheet is several views of one character generated from a single locked seed
and composited into a labelled grid. Danbooru-trained anime models respond to
explicit view and expression tags, which is what makes a consistent turnaround
possible without any extra conditioning model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .comic import resolve_font

# Tags that push each cell toward one clean view of one character.
#
# Deliberately NOT "character sheet" or "reference sheet": the app composes the
# sheet itself, and Danbooru-trained models answer those tags by drawing an
# entire multi-view sheet inside every single cell. Measured with Animagine XL
# 4.0: the "Front" cell came back as a full-body view plus a floating jacket,
# detached legs and shoes, and four separate head studies.
SHEET_BASE = "solo, single view, simple background, white background, full lighting"
SHEET_NEGATIVE = ("multiple views, reference sheet, character sheet, split screen, collage, "
                  "multiple persons, detailed background, scenery, text, watermark")


@dataclass
class SheetCell:
    label: str
    tags: str
    # Portrait cells want a tighter crop than full-body ones.
    portrait: bool = False
    # View-specific negatives. Anime models lean hard toward showing the face;
    # without these a "Back" cell came out turned round, looking over her
    # shoulder, and a "3/4" cell came out as a plain side profile.
    negative: str = ""


SHEET_PRESETS: dict[str, list[SheetCell]] = {
    "turnaround": [
        SheetCell("Front", "full body, standing, (front view:1.2), facing viewer, arms at sides",
                  negative="from side, from behind, profile"),
        SheetCell("3/4", "full body, standing, (three-quarter view:1.3), body turned slightly, looking at viewer",
                  negative="profile, from side, from behind"),
        SheetCell("Side", "full body, standing, (from side:1.3), profile, facing to the side",
                  negative="facing viewer, looking at viewer, from behind"),
        SheetCell("Back", "full body, standing, (from behind:1.4), back view, facing away",
                  negative="looking at viewer, looking back, facing viewer"),
    ],
    "expressions": [
        SheetCell("Neutral", "portrait, close-up, neutral expression, facing viewer", True),
        SheetCell("Smile", "portrait, close-up, smiling, happy, facing viewer", True),
        SheetCell("Angry", "portrait, close-up, angry, frowning, v-shaped eyebrows", True),
        SheetCell("Sad", "portrait, close-up, sad, teary eyes, downcast", True),
        SheetCell("Surprised", "portrait, close-up, surprised, wide eyes, open mouth", True),
        SheetCell("Embarrassed", "portrait, close-up, embarrassed, blush, averted gaze", True),
    ],
    "full": [
        SheetCell("Front", "full body, standing, (front view:1.2), facing viewer",
                  negative="from side, from behind, profile"),
        SheetCell("Side", "full body, standing, (from side:1.3), profile, facing to the side",
                  negative="facing viewer, looking at viewer, from behind"),
        SheetCell("Back", "full body, standing, (from behind:1.4), back view, facing away",
                  negative="looking at viewer, looking back, facing viewer"),
        SheetCell("Bust", "portrait, upper body, neutral expression, facing viewer", True),
        SheetCell("Smile", "portrait, close-up, smiling, happy", True),
        SheetCell("Angry", "portrait, close-up, angry, frowning", True),
    ],
    "outfit": [
        SheetCell("Full", "full body, standing, front view, facing viewer"),
        SheetCell("Back", "full body, standing, from behind"),
        SheetCell("Detail", "upper body, clothing focus, detailed fabric", True),
        SheetCell("Accessories", "close-up, accessories focus, hands, detailed", True),
    ],
}


def sheet_presets() -> list[str]:
    return list(SHEET_PRESETS)


def cells_for(preset: str) -> list[SheetCell]:
    if preset not in SHEET_PRESETS:
        raise ValueError(f"unknown sheet preset {preset!r}; "
                         f"choose from {sorted(SHEET_PRESETS)}")
    return SHEET_PRESETS[preset]


def cell_prompt(character_tags: str, cell: SheetCell, extra: str = "") -> str:
    """Compose one cell's prompt: character first, then view, then sheet tags."""
    parts = [p for p in (character_tags.strip(), cell.tags, extra.strip(), SHEET_BASE) if p]
    return ", ".join(parts)


def cell_negative(character_negative: str, default_negative: str, cell: SheetCell) -> str:
    """Negative prompt for one cell: character, defaults, sheet-wide, then view."""
    parts = (character_negative.strip(), default_negative.strip(), SHEET_NEGATIVE, cell.negative.strip())
    return ", ".join(p for p in parts if p)


def grid_shape(count: int) -> tuple[int, int]:
    """Columns x rows for a cell count, preferring wide-ish sheets."""
    if count <= 2:
        return count, 1
    if count <= 4:
        return 2, (count + 1) // 2
    if count <= 6:
        return 3, (count + 2) // 3
    cols = 4
    return cols, (count + cols - 1) // cols


def compose_sheet(name: str, cells: list[SheetCell], images: list[Image.Image | None],
                  cell_size: tuple[int, int] = (512, 768),
                  title: str | None = None) -> Image.Image:
    """Lay the generated views out in a labelled grid."""
    cols, rows = grid_shape(len(cells))
    cw, ch = cell_size
    label_h = 34
    pad = 18
    header = 56 if title is not False else 0
    width = cols * cw + pad * (cols + 1)
    height = header + rows * (ch + label_h) + pad * (rows + 1)
    sheet = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(sheet)

    if header:
        font = resolve_font("sans", 30)
        draw.text((pad, header // 2), title or name, font=font, fill="#111111", anchor="lm")
        draw.line([(pad, header - 6), (width - pad, header - 6)], fill="#cccccc", width=2)

    label_font = resolve_font("sans", 18)
    for i, cell in enumerate(cells):
        col, row = i % cols, i // cols
        x = pad + col * (cw + pad)
        y = header + pad + row * (ch + label_h + pad)
        img = images[i] if i < len(images) else None
        if img is not None:
            sheet.paste(_fit(img, cw, ch), (x, y))
        else:
            draw.rectangle((x, y, x + cw, y + ch), fill="#f1f2f5", outline="#d5d8de")
            draw.text((x + cw / 2, y + ch / 2), "—", font=label_font,
                      fill="#aab0bd", anchor="mm")
        draw.rectangle((x, y, x + cw, y + ch), outline="#333333", width=2)
        draw.text((x + cw / 2, y + ch + label_h / 2), cell.label,
                  font=label_font, fill="#333333", anchor="mm")
    return sheet


def _fit(img: Image.Image, w: int, h: int) -> Image.Image:
    """Contain the image in the cell, letterboxed on white."""
    scale = min(w / img.width, h / img.height)
    new = img.convert("RGB").resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
        Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), "#ffffff")
    canvas.paste(new, ((w - new.width) // 2, (h - new.height) // 2))
    return canvas
