"""Comic composition and character library — no model weights needed."""
from __future__ import annotations

import json

import pytest
from PIL import Image

from panelforge.characters import Character, CharacterLibrary, compose_prompt
from panelforge.comic import (
    LAYOUTS, PAGE_PRESETS, Bubble, Page, Panel, build_page, layout_names,
    nearest_bucket, panel_rect, render_page,
)
from panelforge.pipelines import SDXL_BUCKETS


class TestLayouts:
    def test_every_layout_builds_and_tiles_without_overlap(self):
        for name in layout_names():
            page = build_page(name, "manga-b5", "ltr")
            assert page.panels, name
            for p in page.panels:
                assert 0 <= p.x <= 1 and 0 <= p.y <= 1
                assert p.x + p.w <= 1.0001 and p.y + p.h <= 1.0001, name

    def test_layout_areas_cover_the_page(self):
        """Every preset except the deliberate overlap should tile to ~1.0."""
        for name in layout_names():
            if name == "splash-inset":   # an inset sits on top of the splash
                continue
            total = sum(p.w * p.h for p in build_page(name, "square", "ltr").panels)
            assert total == pytest.approx(1.0, abs=0.01), f"{name} covers {total}"

    def test_rtl_mirrors_panels_for_manga_reading_order(self):
        ltr = build_page("grid-2x2", "manga-b5", "ltr")
        rtl = build_page("grid-2x2", "manga-b5", "rtl")
        # The first panel should be leftmost for ltr and rightmost for rtl.
        assert ltr.panels[0].x == pytest.approx(0.0)
        assert rtl.panels[0].x == pytest.approx(0.5)

    def test_unknown_layout_is_rejected(self):
        with pytest.raises(ValueError, match="unknown layout"):
            build_page("not-a-layout")

    def test_panel_rects_stay_inside_the_page(self):
        page = build_page("manga-classic", "manga-b5", "rtl")
        for panel in page.panels:
            x0, y0, x1, y1 = panel_rect(page, panel)
            assert 0 <= x0 < x1 <= page.width
            assert 0 <= y0 < y1 <= page.height

    def test_gutters_actually_separate_panels(self):
        page = build_page("two-across", "square", "ltr")
        left = panel_rect(page, page.panels[0])
        right = panel_rect(page, page.panels[1])
        assert left[2] < right[0], "panels must not touch"


class TestBuckets:
    def test_tall_panel_picks_a_portrait_bucket(self):
        w, h = nearest_bucket(600, 1200, SDXL_BUCKETS)
        assert h > w

    def test_wide_panel_picks_a_landscape_bucket(self):
        w, h = nearest_bucket(1600, 600, SDXL_BUCKETS)
        assert w > h

    def test_square_panel_picks_the_square_bucket(self):
        assert nearest_bucket(800, 800, SDXL_BUCKETS) == (1024, 1024)


class TestRender:
    def test_empty_page_renders_at_the_requested_size(self, tmp_path):
        page = build_page("grid-2x2", "manga-b5", "rtl")
        img = render_page(page, tmp_path)
        assert img.size == (page.width, page.height)
        assert img.mode == "RGB"

    def test_every_bubble_style_renders(self, tmp_path):
        page = build_page("single", "square", "ltr")
        for i, style in enumerate(
                ["speech", "thought", "shout", "whisper", "narration", "sfx"]):
            page.bubbles.append(Bubble(x=.05, y=.05 + i * .14, w=.35, h=.11,
                                       text=f"{style} test", style=style,
                                       tail_x=.5, tail_y=.5))
        img = render_page(page, tmp_path)
        assert img.size == (1600, 1600)

    def test_vertical_cjk_renders(self, tmp_path):
        page = build_page("single", "square", "ltr")
        page.bubbles.append(Bubble(x=.1, y=.1, w=.12, h=.4, text="行くぞ",
                                   style="speech", vertical=True))
        assert render_page(page, tmp_path).size == (1600, 1600)

    def test_panel_image_outside_outputs_is_ignored(self, tmp_path):
        """panel.image comes from the client and must not read arbitrary files."""
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        secret = tmp_path / "secret.png"
        Image.new("RGB", (8, 8), "red").save(secret)
        page = build_page("single", "square", "ltr")
        page.panels[0].image = "../secret.png"
        img = render_page(page, outputs)          # placeholder, not the secret
        assert img.getpixel((800, 800)) != (255, 0, 0)

    def test_missing_panel_image_falls_back_to_a_placeholder(self, tmp_path):
        page = build_page("single", "square", "ltr")
        page.panels[0].image = "does-not-exist.png"
        assert render_page(page, tmp_path).size == (1600, 1600)

    def test_page_survives_a_json_roundtrip(self):
        page = build_page("action", "comic-us", "rtl")
        page.bubbles.append(Bubble(x=.1, y=.1, w=.2, h=.1, text="hi"))
        back = Page.from_dict(json.loads(json.dumps(page.to_dict())))
        assert len(back.panels) == len(page.panels)
        assert back.bubbles[0].text == "hi"
        assert back.reading_order == "rtl"


class TestCharacters:
    def test_roundtrip_through_disk(self, tmp_path):
        lib = CharacterLibrary(tmp_path / "characters.json")
        char = lib.upsert({"name": "Aoi", "prompt": "1girl, silver hair", "seed": 7})
        assert CharacterLibrary(tmp_path / "characters.json").get(char.id).name == "Aoi"

    def test_delete(self, tmp_path):
        lib = CharacterLibrary(tmp_path / "characters.json")
        char = lib.upsert({"name": "Temp"})
        assert lib.delete(char.id)
        assert not lib.delete(char.id)

    def test_corrupt_library_does_not_crash_startup(self, tmp_path):
        path = tmp_path / "characters.json"
        path.write_text("{not json")
        assert CharacterLibrary(path).list() == []

    def test_character_tags_lead_the_prompt(self):
        """Tokens early in a CLIP prompt carry more weight than late ones."""
        aoi = Character(id="a", name="Aoi", prompt="1girl, silver hair")
        prompt, _, _ = compose_prompt("standing on a rooftop", [aoi])
        assert prompt.startswith("1girl, silver hair")
        assert "rooftop" in prompt

    def test_shared_tags_are_not_repeated(self):
        """A repeated tag silently doubles its own weight."""
        a = Character(id="a", name="A", prompt="1girl, school uniform")
        b = Character(id="b", name="B", prompt="1girl, school uniform, twintails")
        prompt, _, _ = compose_prompt("", [a, b])
        assert prompt.count("school uniform") == 1
        assert prompt.count("1girl,") <= 1

    def test_multiple_characters_get_a_count_tag(self):
        """Without one, SDXL tends to merge the cast into a single figure."""
        a = Character(id="a", name="A", prompt="1girl, red hair")
        b = Character(id="b", name="B", prompt="1girl, blue hair")
        prompt, _, _ = compose_prompt("in a cafe", [a, b])
        assert prompt.startswith("2girls")

    def test_lora_stacks_merge_keeping_the_strongest(self):
        a = Character(id="a", name="A", loras=[{"name": "/x.safetensors", "weight": 0.6}])
        b = Character(id="b", name="B", loras=[{"name": "/x.safetensors", "weight": 0.9}])
        _, _, loras = compose_prompt("", [a, b])
        assert loras == [{"name": "/x.safetensors", "weight": 0.9}]
