"""Face enhancement (ADetailer-style).

Small faces are where diffusion models fail most visibly, and in a comic panel
a face is often only a couple of hundred pixels across. This detects faces,
crops each one, re-renders it at a resolution the model handles well, and
composites it back through a feathered mask.

Two re-render modes. img2img refines the whole padded crop at low strength.
inpaint regenerates only a feathered ellipse over the detected face and keeps
the surrounding hair and background verbatim, so the crop border cannot drift
away from the pixels around it. (A grey-blob failure once blamed on inpainting
was really the shared model being silently recast to fp32 and returning zeros;
see pipelines.PipelineManager.pipe_for.)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

log = logging.getLogger(__name__)


@dataclass
class Detection:
    x0: int
    y0: int
    x1: int
    y1: int
    confidence: float

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


class FaceDetector:
    """YOLO face detector, loaded lazily so ultralytics stays optional."""

    def __init__(self, weights: Path):
        self.weights = weights
        self._model = None

    @property
    def available(self) -> bool:
        return self.weights.exists()

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.available:
            raise FileNotFoundError(f"face detector not found: {self.weights}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "face enhancement needs ultralytics: uv add ultralytics"
            ) from exc
        self._model = YOLO(str(self.weights))
        return self._model

    def detect(self, image: Image.Image, confidence: float = 0.3) -> list[Detection]:
        model = self._load()
        # ultralytics prints a per-image summary line otherwise.
        results = model.predict(image, conf=confidence, verbose=False)
        out: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x0, y0, x1, y1 = (float(v) for v in box.xyxy[0].tolist())
                out.append(Detection(int(x0), int(y0), int(x1), int(y1),
                                     float(box.conf[0])))
        # Largest first: the focal face should be enhanced even if a cap is hit.
        out.sort(key=lambda d: d.area(), reverse=True)
        return out


def expand_box(det: Detection, image: Image.Image, padding: float = 0.35
               ) -> tuple[int, int, int, int]:
    """Grow a face box to include hair and chin, clamped to the image."""
    pad_x = int(det.width * padding)
    pad_y = int(det.height * padding)
    return (
        max(0, det.x0 - pad_x), max(0, det.y0 - pad_y),
        min(image.width, det.x1 + pad_x), min(image.height, det.y1 + pad_y),
    )


def _square(box: tuple[int, int, int, int], image: Image.Image
            ) -> tuple[int, int, int, int]:
    """Square the crop: the model was trained on square-ish aspect ratios."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = side // 2
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(image.width, x0 + side), min(image.height, y0 + side)
    # Re-clamp the origin if the far edge hit the boundary.
    x0, y0 = max(0, x1 - side), max(0, y1 - side)
    return x0, y0, x1, y1


def _face_mask(det: Detection, box: tuple[int, int, int, int], size: int) -> Image.Image:
    """White over the detected face inside the crop, feathered; black elsewhere.

    Used by inpaint mode: only the face is regenerated, while the hair and
    background in the padded margin are kept as context.
    """
    sx = size / max(1, box[2] - box[0])
    sy = size / max(1, box[3] - box[1])
    gx, gy = det.width * 0.12, det.height * 0.12
    ellipse = ((det.x0 - gx - box[0]) * sx, (det.y0 - gy - box[1]) * sy,
               (det.x1 + gx - box[0]) * sx, (det.y1 + gy - box[1]) * sy)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse(ellipse, fill=255)
    return mask.filter(ImageFilter.GaussianBlur(max(2, size // 64)))


def _feathered_mask(size: tuple[int, int], feather: int) -> Image.Image:
    """A soft-edged mask so the composite has no visible seam."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((feather, feather, size[0] - feather, size[1] - feather), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(feather / 2))


def degenerate_reason(before: Image.Image, after: Image.Image) -> str | None:
    """Why a refined crop is unusable, or None if it looks like a real image.

    A failed GPU pass does not always raise. On an M2, once the shared model
    had been silently recast to fp32 and memory ran out, the text encoder and
    VAE encoder returned all-zero tensors with no exception; the refine came
    back as a solid white crop that was pasted over the face and counted as
    enhanced. Judged relative to the input crop, because character-sheet
    crops legitimately contain a lot of white.
    """
    b = np.asarray(before.convert("RGB"), dtype=np.float32)
    a = np.asarray(after.convert("RGB"), dtype=np.float32)
    if a.std() < max(3.0, 0.25 * b.std()):
        return f"output collapsed (std {a.std():.1f} vs input {b.std():.1f})"
    white_in, white_out = (b.min(axis=2) > 250).mean(), (a.min(axis=2) > 250).mean()
    if white_out - white_in > 0.30:
        return f"output {white_out * 100:.0f}% pure white (input {white_in * 100:.0f}%)"
    black_in, black_out = (b.max(axis=2) < 5).mean(), (a.max(axis=2) < 5).mean()
    if black_out - black_in > 0.30:
        return f"output {black_out * 100:.0f}% pure black (input {black_in * 100:.0f}%)"
    return None


def enhance_faces(
    image: Image.Image,
    detector: FaceDetector,
    refine,                        # callable(crop, size, mask) -> Image
    confidence: float = 0.3,
    padding: float = 0.35,
    max_faces: int = 4,
    min_size: int = 24,
    work_size: int = 768,
) -> tuple[Image.Image, list[Detection], list[str]]:
    """Detect faces and re-render each one at `work_size`, then composite back.

    Returns the image, the faces that were successfully enhanced, and any
    per-face errors. A face that fails is left untouched rather than aborting
    the whole image, but the failure is reported rather than swallowed: a pass
    that silently changes nothing after minutes of work is worse than an error.
    """
    detections = [d for d in detector.detect(image, confidence)
                  if d.width >= min_size and d.height >= min_size][:max_faces]
    if not detections:
        return image, [], []

    result = image.convert("RGB").copy()
    enhanced: list[Detection] = []
    errors: list[str] = []
    for det in detections:
        box = _square(expand_box(det, image, padding), image)
        crop = result.crop(box)
        if min(crop.size) < 8:
            continue
        original_size = crop.size
        # Upscale the crop so the model works at a resolution it is good at.
        crop_big = crop.resize((work_size, work_size), Image.LANCZOS)
        mask_big = _face_mask(det, box, work_size)
        try:
            fixed = refine(crop_big, work_size, mask_big)
        except Exception as exc:
            log.error("face refine failed, leaving face unchanged: %s", exc)
            errors.append(str(exc).split("\n")[0][:200])
            continue
        reason = degenerate_reason(crop_big, fixed)
        if reason is not None:
            log.error("face refine produced an unusable image, leaving face unchanged: %s", reason)
            errors.append(reason)
            continue
        fixed = fixed.resize(original_size, Image.LANCZOS)
        feather = max(4, min(original_size) // 12)
        result.paste(fixed, box, _feathered_mask(original_size, feather))
        enhanced.append(det)
    return result, enhanced, errors
