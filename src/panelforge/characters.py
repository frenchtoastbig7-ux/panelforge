"""Character library.

Keeping a cast consistent across panels is the hard part of making a comic with
a diffusion model. A character bundles the appearance tags, LoRAs and a locked
seed that together make the same person recognisable from panel to panel.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Character:
    id: str
    name: str
    # Danbooru-style appearance tags: "1girl, silver hair, blue eyes, ...".
    prompt: str = ""
    negative_prompt: str = ""
    loras: list[dict] = field(default_factory=list)
    # A fixed seed reproduces the same face when the rest of the prompt is
    # stable; it is a strong nudge, not a guarantee.
    seed: int = -1
    reference_image: str | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class CharacterLibrary:
    """A small JSON-backed store. Plain text on disk, easy to hand-edit."""

    def __init__(self, path: Path):
        self.path = path
        self._chars: dict[str, Character] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._chars = {}
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.error("character library unreadable (%s); starting empty", exc)
            self._chars = {}
            return
        self._chars = {}
        for raw in data.get("characters", []):
            try:
                self._chars[raw["id"]] = Character(**raw)
            except TypeError as exc:
                log.warning("skipping malformed character %r: %s", raw.get("name"), exc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"characters": [c.to_dict() for c in self._chars.values()]}
        # Write via a temp file so an interrupted save cannot truncate the library.
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        tmp.replace(self.path)

    def list(self) -> list[Character]:
        return sorted(self._chars.values(), key=lambda c: c.name.lower())

    def get(self, char_id: str) -> Character | None:
        return self._chars.get(char_id)

    def upsert(self, data: dict) -> Character:
        char_id = data.get("id") or uuid.uuid4().hex[:10]
        existing = self._chars.get(char_id)
        char = Character(
            id=char_id,
            name=data.get("name") or (existing.name if existing else "unnamed"),
            prompt=data.get("prompt", existing.prompt if existing else ""),
            negative_prompt=data.get("negative_prompt",
                                     existing.negative_prompt if existing else ""),
            loras=data.get("loras", existing.loras if existing else []),
            seed=int(data.get("seed", existing.seed if existing else -1)),
            reference_image=data.get("reference_image",
                                     existing.reference_image if existing else None),
            notes=data.get("notes", existing.notes if existing else ""),
        )
        self._chars[char_id] = char
        self.save()
        return char

    def delete(self, char_id: str) -> bool:
        if char_id in self._chars:
            del self._chars[char_id]
            self.save()
            return True
        return False


def _dedupe_tags(prompt: str) -> str:
    """Drop repeated tags while preserving order.

    Two characters in one panel commonly share tags such as `school uniform`;
    repeating a tag also silently doubles its weight.
    """
    seen: set[str] = set()
    out: list[str] = []
    for tag in re.split(r",\s*", prompt):
        tag = tag.strip()
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return ", ".join(out)


def compose_prompt(scene_prompt: str, characters: list[Character]) -> tuple[str, str, list[dict]]:
    """Merge a scene description with the selected cast.

    Returns (prompt, negative_prompt, loras). Character tags lead, because
    tokens early in a CLIP prompt carry more weight, and the subject matters
    more than the setting.
    """
    parts = [c.prompt.strip() for c in characters if c.prompt.strip()]
    if scene_prompt.strip():
        parts.append(scene_prompt.strip())
    prompt = _dedupe_tags(", ".join(parts))

    negatives = [c.negative_prompt.strip() for c in characters if c.negative_prompt.strip()]
    negative = _dedupe_tags(", ".join(negatives))

    # Merge LoRA stacks, keeping the highest weight when two characters share one.
    merged: dict[str, float] = {}
    for c in characters:
        for lora in c.loras:
            name = lora.get("name")
            if not name:
                continue
            weight = float(lora.get("weight", 1.0))
            merged[name] = max(merged.get(name, weight), weight)
    loras = [{"name": n, "weight": w} for n, w in merged.items()]

    if len(characters) > 1:
        # Multi-character panels need an explicit count tag or SDXL tends to
        # merge the cast into a single figure.
        count = f"{len(characters)}girls" if _all_female(characters) else "multiple people"
        prompt = _dedupe_tags(f"{count}, {prompt}")
    return prompt, negative, loras


def _all_female(characters: list[Character]) -> bool:
    return all("1girl" in c.prompt.lower() for c in characters)
