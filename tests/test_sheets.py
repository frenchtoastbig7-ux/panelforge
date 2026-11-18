"""Character sheet prompts and composition — no model weights needed."""
from __future__ import annotations

import pytest
from PIL import Image

from panelforge.sheets import (
    SHEET_NEGATIVE, cell_negative, cell_prompt, cells_for, compose_sheet, grid_shape, sheet_presets,
)


class TestCellPrompts:
    @pytest.mark.parametrize("preset", sheet_presets())
    def test_no_cell_asks_the_model_for_a_whole_sheet(self, preset):
        """Those tags made every cell a full multi-view sheet of its own."""
        for cell in cells_for(preset):
            prompt = cell_prompt("1girl, silver hair", cell).lower()
            assert "character sheet" not in prompt, (preset, cell.label)
            assert "reference sheet" not in prompt, (preset, cell.label)
            assert "solo" in prompt and "single view" in prompt

    def test_character_tags_lead_and_view_tags_follow(self):
        cell = cells_for("turnaround")[0]
        prompt = cell_prompt("1girl, silver hair", cell)
        assert prompt.startswith("1girl, silver hair")
        assert cell.tags in prompt

    def test_negative_actively_suppresses_multi_view_layouts(self):
        for tag in ("multiple views", "character sheet", "reference sheet", "collage"):
            assert tag in SHEET_NEGATIVE

    def test_unknown_preset_is_rejected(self):
        with pytest.raises(ValueError, match="unknown sheet preset"):
            cells_for("nope")


class TestComposition:
    @pytest.mark.parametrize("count,shape", [(1, (1, 1)), (2, (2, 1)), (4, (2, 2)), (6, (3, 2)), (8, (4, 2))])
    def test_grid_shapes(self, count, shape):
        assert grid_shape(count) == shape

    def test_missing_cells_render_as_placeholders(self):
        cells = cells_for("turnaround")
        images = [Image.new("RGB", (832, 1216), "red"), None, None, None]
        sheet = compose_sheet("Aoi", cells, images, cell_size=(128, 192))
        cols, rows = grid_shape(len(cells))
        assert sheet.width > cols * 128 and sheet.height > rows * 192



class TestViewAccuracy:
    """A Back cell came out looking over her shoulder; a 3/4 cell came out as a profile."""

    def _cell(self, preset, label):
        return next(c for c in cells_for(preset) if c.label == label)

    def test_back_view_forbids_showing_the_face(self):
        neg = cell_negative("", "lowres", self._cell("turnaround", "Back"))
        for tag in ("looking at viewer", "looking back"):
            assert tag in neg
        assert "(from behind:" in self._cell("turnaround", "Back").tags

    def test_three_quarter_view_forbids_a_plain_profile(self):
        assert "profile" in cell_negative("", "", self._cell("turnaround", "3/4"))

    def test_negative_keeps_character_defaults_and_sheet_terms(self):
        neg = cell_negative("red eyes", "lowres", self._cell("turnaround", "Side"))
        assert neg.startswith("red eyes, lowres")
        assert "multiple views" in neg and "facing viewer" in neg
