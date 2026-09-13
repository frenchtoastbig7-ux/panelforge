"""OpenPose skeleton rendering and pose presets.

ControlNet-openpose expects a skeleton image drawn in a specific keypoint
layout and colour scheme. Rendering one directly from coordinates means pose
control works without installing a pose *estimator*, and it makes poses
editable — which is what a comic tool actually needs.

Keypoints follow the COCO-18 order used by the original OpenPose release:
  0 nose      1 neck      2 r-shoulder 3 r-elbow  4 r-wrist
  5 l-shoulder 6 l-elbow  7 l-wrist    8 r-hip    9 r-knee
 10 r-ankle   11 l-hip   12 l-knee    13 l-ankle 14 r-eye
 15 l-eye     16 r-ear   17 l-ear
Coordinates are normalised to 0-1; None marks a keypoint that is not visible.
"""
from __future__ import annotations

import math
from typing import Sequence

from PIL import Image, ImageDraw

Keypoint = tuple[float, float] | None

# The 19 limb connections OpenPose draws, and the colour of each. These values
# are part of the ControlNet input contract, not a style choice.
LIMBS: list[tuple[int, int]] = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
    (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
    (0, 15), (15, 17), (2, 16), (5, 17),
]
LIMB_COLORS: list[tuple[int, int, int]] = [
    (153, 0, 0), (153, 51, 0), (153, 102, 0), (153, 153, 0), (102, 153, 0),
    (51, 153, 0), (0, 153, 0), (0, 153, 51), (0, 153, 102), (0, 153, 153),
    (0, 102, 153), (0, 51, 153), (0, 0, 153), (51, 0, 153), (102, 0, 153),
    (153, 0, 153), (153, 0, 102), (153, 0, 51), (153, 0, 0),
]
POINT_COLORS: list[tuple[int, int, int]] = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]


def _p(x: float, y: float) -> Keypoint:
    return (x, y)


# A standing figure, used as the base for the other presets.
_STANDING: list[Keypoint] = [
    _p(.50, .12), _p(.50, .22), _p(.42, .22), _p(.39, .34), _p(.37, .45),
    _p(.58, .22), _p(.61, .34), _p(.63, .45), _p(.45, .50), _p(.44, .68),
    _p(.44, .86), _p(.55, .50), _p(.56, .68), _p(.56, .86), _p(.48, .10),
    _p(.52, .10), _p(.45, .11), _p(.55, .11),
]


def _edit(base: list[Keypoint], **changes: Keypoint) -> list[Keypoint]:
    """Copy a pose, replacing named joints. Keeps the presets readable."""
    names = {
        "nose": 0, "neck": 1, "rsho": 2, "relb": 3, "rwri": 4, "lsho": 5,
        "lelb": 6, "lwri": 7, "rhip": 8, "rkne": 9, "rank": 10, "lhip": 11,
        "lkne": 12, "lank": 13, "reye": 14, "leye": 15, "rear": 16, "lear": 17,
    }
    out = list(base)
    for name, value in changes.items():
        out[names[name]] = value
    return out


POSE_PRESETS: dict[str, list[Keypoint]] = {
    "standing": _STANDING,
    "arms-up": _edit(_STANDING, relb=_p(.36, .14), rwri=_p(.33, .04),
                     lelb=_p(.64, .14), lwri=_p(.67, .04)),
    "waving": _edit(_STANDING, lelb=_p(.66, .16), lwri=_p(.70, .06)),
    "pointing": _edit(_STANDING, relb=_p(.34, .30), rwri=_p(.22, .27)),
    "arms-crossed": _edit(_STANDING, relb=_p(.40, .34), rwri=_p(.56, .40),
                          lelb=_p(.60, .34), lwri=_p(.44, .40)),
    "running": _edit(_STANDING, relb=_p(.36, .30), rwri=_p(.41, .21),
                     lelb=_p(.63, .36), lwri=_p(.58, .45),
                     rkne=_p(.38, .64), rank=_p(.32, .78),
                     lkne=_p(.60, .70), lank=_p(.66, .84)),
    "sitting": _edit(_STANDING, rhip=_p(.45, .56), lhip=_p(.55, .56),
                     rkne=_p(.36, .70), lkne=_p(.64, .70),
                     rank=_p(.34, .86), lank=_p(.66, .86)),
    "fighting": _edit(_STANDING, relb=_p(.36, .30), rwri=_p(.44, .26),
                      lelb=_p(.64, .32), lwri=_p(.56, .28),
                      rkne=_p(.38, .66), rank=_p(.34, .86),
                      lkne=_p(.60, .68), lank=_p(.66, .86)),
    "walking": _edit(_STANDING, relb=_p(.38, .33), rwri=_p(.36, .43),
                     lelb=_p(.62, .35), lwri=_p(.64, .45),
                     rkne=_p(.41, .68), rank=_p(.38, .86),
                     lkne=_p(.58, .67), lank=_p(.62, .86)),
    # Portrait framing: only the head and shoulders are in frame.
    "bust": [
        _p(.50, .38), _p(.50, .62), _p(.30, .64), None, None,
        _p(.70, .64), None, None, None, None, None, None, None, None,
        _p(.45, .33), _p(.55, .33), _p(.36, .36), _p(.64, .36),
    ],
}


def pose_names() -> list[str]:
    return list(POSE_PRESETS)


def draw_openpose(keypoints: Sequence[Keypoint], width: int = 1024,
                  height: int = 1024, line_width: int | None = None,
                  point_radius: int | None = None) -> Image.Image:
    """Render a skeleton on black, the format ControlNet-openpose expects."""
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    scale = min(width, height)
    lw = line_width or max(2, scale // 100)
    pr = point_radius or max(2, scale // 130)

    # Map the unit square through a single scale factor and centre the result.
    # Scaling each axis to the canvas independently stretches the figure, and
    # ControlNet then reproduces those distorted proportions faithfully.
    span = min(width, height)
    off_x = (width - span) / 2
    off_y = (height - span) / 2

    def to_px(point: Keypoint) -> tuple[float, float] | None:
        if point is None:
            return None
        return off_x + point[0] * span, off_y + point[1] * span

    # OpenPose draws limbs as tapered ellipses; a thick line is close enough
    # for ControlNet and far simpler.
    for (a, b), color in zip(LIMBS, LIMB_COLORS):
        if a >= len(keypoints) or b >= len(keypoints):
            continue
        pa, pb = to_px(keypoints[a]), to_px(keypoints[b])
        if pa is None or pb is None:
            continue
        draw.line([pa, pb], fill=color, width=lw)

    for i, point in enumerate(keypoints[:len(POINT_COLORS)]):
        px = to_px(point)
        if px is None:
            continue
        x, y = px
        draw.ellipse((x - pr, y - pr, x + pr, y + pr), fill=POINT_COLORS[i])
    return canvas


def preset_image(name: str, width: int = 1024, height: int = 1024) -> Image.Image:
    if name not in POSE_PRESETS:
        raise ValueError(f"unknown pose {name!r}; choose from {sorted(POSE_PRESETS)}")
    return draw_openpose(POSE_PRESETS[name], width, height)


def mirror(keypoints: Sequence[Keypoint]) -> list[Keypoint]:
    """Flip a pose horizontally, swapping left and right joints."""
    swap = {2: 5, 5: 2, 3: 6, 6: 3, 4: 7, 7: 4, 8: 11, 11: 8,
            9: 12, 12: 9, 10: 13, 13: 10, 14: 15, 15: 14, 16: 17, 17: 16}
    out: list[Keypoint] = [None] * len(keypoints)
    for i, point in enumerate(keypoints):
        target = swap.get(i, i)
        out[target] = None if point is None else (1.0 - point[0], point[1])
    return out
