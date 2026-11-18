"""Style presets.

A style is a prompt wrapper plus sampler hints, matching the "pick a style"
selector a comic tool needs. Built-ins are tuned for Animagine/Illustrious-class
anime SDXL finetunes; `styles.json` in the project root overrides or extends
them so a user's own styles survive upgrades.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Style:
    name: str
    label: str
    prefix: str = ""          # leads the prompt: quality and medium tags
    suffix: str = ""          # trails it: rendering and lighting tags
    negative: str = ""        # appended to the negative prompt
    cfg: float | None = None
    steps: int | None = None
    sampler: str | None = None

    def apply(self, prompt: str, negative: str) -> tuple[str, str]:
        parts = [p for p in (self.prefix, prompt.strip(), self.suffix) if p]
        neg = ", ".join(p for p in (negative.strip(), self.negative) if p)
        return ", ".join(parts), neg

    def to_dict(self) -> dict:
        return asdict(self)


# Quality tags Animagine-class models are trained to respond to.
_Q = "masterpiece, best quality, very aesthetic, absurdres"
_NEG = "lowres, bad anatomy, bad hands, extra digits, fewer digits, jpeg artifacts, signature, watermark"

BUILTIN: list[Style] = [
    Style("anime-color", "Anime (colour)", _Q, "detailed background, cinematic lighting", _NEG),
    Style("manga-bw", "Manga (black & white)",
          f"{_Q}, monochrome, greyscale, manga",
          "screentone, halftone shading, clean lineart, high contrast inking",
          f"{_NEG}, color, colored, saturated"),
    Style("shoujo", "Shoujo",
          f"{_Q}, shoujo manga style",
          "delicate lineart, sparkles, soft pastel palette, flowing hair, dreamy",
          f"{_NEG}, harsh shadows, gritty"),
    Style("shounen-action", "Shounen action",
          f"{_Q}, shounen manga style, dynamic angle",
          "speed lines, motion blur, dramatic perspective, bold inking, impact frame",
          f"{_NEG}, static, dull"),
    Style("webtoon", "Webtoon",
          f"{_Q}, webtoon style, clean digital art",
          "soft cel shading, bright palette, simple background",
          f"{_NEG}, sketchy, rough lineart"),
    Style("noir", "Noir comic",
          f"{_Q}, noir comic style, monochrome",
          "heavy blacks, chiaroscuro, deep shadows, ink wash, rain",
          f"{_NEG}, bright, pastel, flat lighting"),
    Style("chibi", "Chibi",
          f"{_Q}, chibi, deformed",
          "big head, small body, cute, simple background, thick outlines",
          f"{_NEG}, realistic proportions, detailed anatomy"),
    Style("watercolour", "Watercolour",
          f"{_Q}, watercolor painting",
          "soft washes, paper texture, bleeding pigment, delicate linework",
          f"{_NEG}, digital, sharp edges, 3d render"),
    Style("sketch", "Pencil sketch",
          f"{_Q}, pencil sketch, rough draft",
          "graphite, hatching, construction lines, unfinished",
          f"{_NEG}, color, clean lineart, polished"),
    # Distilled checkpoints need their own sampler and a near-zero CFG; folding
    # that into a style stops it being a footgun.
    Style("lightning", "Lightning (fast draft)", _Q, "", _NEG,
          cfg=1.5, steps=8, sampler="euler_trailing"),
]


def load_styles(root: Path) -> list[Style]:
    """Built-ins, with any user styles from styles.json merged over them."""
    styles = {s.name: s for s in BUILTIN}
    path = root / "styles.json"
    if path.exists():
        try:
            for raw in json.loads(path.read_text()).get("styles", []):
                style = Style(**raw)
                styles[style.name] = style
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            log.error("styles.json ignored (%s)", exc)
    return list(styles.values())


def get_style(root: Path, name: str | None) -> Style | None:
    if not name or name == "none":
        return None
    return next((s for s in load_styles(root) if s.name == name), None)
