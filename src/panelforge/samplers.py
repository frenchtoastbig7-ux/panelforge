"""Scheduler presets, named the way image-generation UIs name them."""
from __future__ import annotations

from diffusers import (
    DDIMScheduler,
    LCMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    UniPCMultistepScheduler,
)

# name -> (scheduler class, config overrides)
SAMPLERS: dict[str, tuple[type, dict]] = {
    "euler": (EulerDiscreteScheduler, {}),
    "euler_a": (EulerAncestralDiscreteScheduler, {}),
    "heun": (HeunDiscreteScheduler, {}),
    "lms": (LMSDiscreteScheduler, {}),
    "ddim": (DDIMScheduler, {}),
    "unipc": (UniPCMultistepScheduler, {}),
    "dpmpp_2m": (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++"}),
    "dpmpp_2m_karras": (
        DPMSolverMultistepScheduler,
        {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True},
    ),
    "dpmpp_sde_karras": (
        DPMSolverSinglestepScheduler,
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
    ),
    "dpmpp_2m_sde_karras": (
        DPMSolverMultistepScheduler,
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
    ),
    # Distilled checkpoints (Lightning/Turbo/LCM) need a trailing-timestep
    # schedule and almost no guidance; see README.
    "euler_trailing": (
        EulerDiscreteScheduler,
        {"timestep_spacing": "trailing"},
    ),
    # Several popular anime checkpoints ship LCM variants, which need their
    # own scheduler rather than a re-timed Euler.
    "lcm": (LCMScheduler, {}),
}

DEFAULT_SAMPLER = "dpmpp_2m_karras"


def build_scheduler(name: str, current_config):
    """Return a scheduler built from the checkpoint's own training config.

    Reusing `current_config` preserves per-checkpoint values such as
    `prediction_type` (v-prediction models break badly if this is dropped).
    """
    if name not in SAMPLERS:
        raise ValueError(f"unknown sampler {name!r}; choose from {sorted(SAMPLERS)}")
    cls, overrides = SAMPLERS[name]
    overrides = dict(overrides)
    # v-prediction checkpoints (NoobAI-vpred and friends) also need zero-SNR
    # rescaling and a trailing schedule, or they render washed out and grey.
    if _is_v_prediction(current_config):
        for key, value in (("rescale_betas_zero_snr", True),
                           ("timestep_spacing", "trailing")):
            if _accepts(cls, key):
                overrides.setdefault(key, value)
    return cls.from_config(current_config, **overrides)


def _is_v_prediction(config) -> bool:
    try:
        return getattr(config, "prediction_type", None) == "v_prediction"
    except Exception:
        return False


def _accepts(cls, name: str) -> bool:
    import inspect
    try:
        return name in inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def sampler_names() -> list[str]:
    return sorted(SAMPLERS)
