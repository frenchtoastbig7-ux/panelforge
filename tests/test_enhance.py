"""Face enhancement bookkeeping — no model weights or detector needed."""
from __future__ import annotations

import numpy as np
from PIL import Image

from panelforge.enhance import Detection, degenerate_reason, enhance_faces


class FakeDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, image, confidence=0.3):
        return list(self._detections)


FACE = Detection(40, 40, 120, 120, confidence=0.9)


def _noise(size=(256, 256), tint=(0, 0, 0), seed=0):
    """Textured image: a solid colour has zero variance and reads as collapsed."""
    rng = np.random.default_rng(seed)
    a = rng.integers(40, 200, (size[1], size[0], 3)).astype(np.int16) + np.array(tint)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def test_failed_refine_is_reported_not_claimed_as_success():
    """The first version logged "touched 1 face" after an OOM changed nothing."""
    def refine(crop, size, mask):
        raise RuntimeError("MPS backend out of memory")

    src = _noise()
    out, enhanced, errors = enhance_faces(src, FakeDetector([FACE]), refine)
    assert enhanced == []
    assert len(errors) == 1 and "out of memory" in errors[0]
    assert np.array_equal(np.asarray(out), np.asarray(src)), "a failed face must be left untouched"


def test_white_refine_output_is_rejected_not_pasted():
    """A Metal OOM returned a solid white crop that was pasted and counted as enhanced."""
    def refine(crop, size, mask):
        return Image.new("RGB", (size, size), (255, 255, 255))

    src = _noise()
    out, enhanced, errors = enhance_faces(src, FakeDetector([FACE]), refine)
    assert enhanced == []
    assert len(errors) == 1 and "collapsed" in errors[0]
    assert np.array_equal(np.asarray(out), np.asarray(src))


def test_successful_refine_changes_the_face_and_is_counted():
    def refine(crop, size, mask):
        return _noise((size, size), tint=(120, -40, -40), seed=1)

    src = _noise()
    out, enhanced, errors = enhance_faces(src, FakeDetector([FACE]), refine)
    assert len(enhanced) == 1 and errors == []
    r, g, _ = out.getpixel((80, 80))
    assert r > g + 40, "face centre should carry the refined pixels"
    assert out.getpixel((250, 250)) == src.getpixel((250, 250)), "outside the face must not change"


def test_refine_receives_a_square_crop_at_work_size():
    seen = []

    def refine(crop, size, mask):
        seen.append((crop.size, size))
        return crop

    enhance_faces(_noise(), FakeDetector([FACE]), refine, work_size=512)
    assert seen == [((512, 512), 512)]


def test_tiny_faces_are_skipped():
    tiny = Detection(10, 10, 20, 20, confidence=0.9)
    calls = []
    out, enhanced, errors = enhance_faces(
        _noise(), FakeDetector([tiny]), lambda c, s, m: calls.append(1) or c, min_size=24)
    assert calls == [] and enhanced == [] and errors == []


def test_face_cap_keeps_the_largest():
    small = Detection(0, 0, 30, 30, confidence=0.9)
    big = Detection(100, 100, 220, 220, confidence=0.9)
    # FaceDetector sorts largest-first; the fake mirrors that contract.
    _, enhanced, _ = enhance_faces(_noise(), FakeDetector([big, small]),
                                   lambda c, s, m: c, max_faces=1)
    assert enhanced == [big]


class TestDegenerateReason:
    def test_real_image_passes(self):
        assert degenerate_reason(_noise(), _noise(seed=3)) is None

    def test_mostly_white_background_is_allowed_when_the_input_was_too(self):
        """Character-sheet crops are legitimately full of white background."""
        def sheetish(seed):
            a = np.full((128, 128, 3), 255, np.uint8)
            a[40:90, 40:90] = np.asarray(_noise((50, 50), seed=seed))
            return Image.fromarray(a)
        assert degenerate_reason(sheetish(0), sheetish(1)) is None

    def test_newly_white_output_is_flagged(self):
        before = _noise((128, 128))
        a = np.asarray(_noise((128, 128), seed=2)).copy()
        a[:80] = 255
        assert "white" in degenerate_reason(before, Image.fromarray(a))

    def test_black_output_is_flagged(self):
        assert degenerate_reason(_noise(), Image.new("RGB", (256, 256), (0, 0, 0))) is not None


def test_refine_receives_a_face_mask_that_keeps_the_context():
    seen = {}

    def refine(crop, size, mask):
        seen["mask"] = mask
        return crop

    enhance_faces(_noise(), FakeDetector([FACE]), refine, work_size=256, padding=0.5)
    mask = seen["mask"]
    assert mask.mode == "L" and mask.size == (256, 256)
    assert mask.getpixel((128, 128)) > 200, "the face centre must be regenerated"
    assert mask.getpixel((2, 2)) < 30, "the crop corner is context and must be kept"
