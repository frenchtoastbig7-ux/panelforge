"""Device, dtype and memory policy.

Everything hardware-specific lives here. The rest of the codebase asks this
module what to do and never branches on `torch.backends.mps` itself, so moving
from Apple Silicon to an NVIDIA box is a policy change, not a rewrite.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import torch

log = logging.getLogger(__name__)

# Several ops used by diffusers schedulers still have no MPS kernel. Without the
# fallback they raise instead of quietly running on CPU. Must be set before the
# first MPS allocation, so it lives at import time.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


@dataclass(frozen=True)
class DevicePolicy:
    """Resolved answers to every hardware question the pipeline layer asks."""

    device: str
    dtype: torch.dtype
    vae_dtype: torch.dtype
    total_memory_gb: float
    # Memory-saving measures, ordered roughly by how much speed they cost.
    attention_slicing: bool = False
    vae_slicing: bool = True
    vae_tiling: bool = False
    sequential_offload: bool = False
    model_offload: bool = False
    # MPS reuses system RAM, so a "VRAM" figure is misleading there.
    unified_memory: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    @property
    def is_mps(self) -> bool:
        return self.device == "mps"

    def summary(self) -> str:
        mem = f"{self.total_memory_gb:.0f}GB"
        mem += " unified" if self.unified_memory else " VRAM"
        opts = [
            name
            for name, on in (
                ("attn-slicing", self.attention_slicing),
                ("vae-slicing", self.vae_slicing),
                ("vae-tiling", self.vae_tiling),
                ("model-offload", self.model_offload),
                ("seq-offload", self.sequential_offload),
            )
            if on
        ]
        return (
            f"{self.device} / {_dtype_name(self.dtype)} "
            f"(vae {_dtype_name(self.vae_dtype)}) / {mem}"
            + (f" / {', '.join(opts)}" if opts else "")
        )


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def detect_device(prefer: str | None = None) -> str:
    """Pick a backend. `prefer` (or PANELFORGE_DEVICE) wins if actually usable."""
    prefer = prefer or os.environ.get("PANELFORGE_DEVICE")
    if prefer:
        prefer = prefer.lower()
        if prefer == "cuda" and not torch.cuda.is_available():
            log.warning("device=cuda requested but no CUDA runtime; falling back")
        elif prefer == "mps" and not torch.backends.mps.is_available():
            log.warning("device=mps requested but MPS unavailable; falling back")
        else:
            return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _total_memory_gb(device: str) -> float:
    if device == "cuda":
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    # MPS and CPU both draw on system RAM.
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError):
        return 8.0


def resolve_policy(
    prefer_device: str | None = None,
    prefer_dtype: str | None = None,
    low_memory: bool | None = None,
) -> DevicePolicy:
    """Decide device, dtype and which memory tricks to enable."""
    device = detect_device(prefer_device)
    total = _total_memory_gb(device)
    notes: list[str] = []

    if device == "cpu":
        # fp16 on CPU is emulated and slower than fp32.
        return DevicePolicy(
            device="cpu",
            dtype=torch.float32,
            vae_dtype=torch.float32,
            total_memory_gb=total,
            attention_slicing=True,
            vae_tiling=True,
            notes=("CPU backend: expect minutes per image, not seconds.",),
        )

    dtype = _resolve_dtype(device, prefer_dtype, notes)

    if device == "mps":
        # MPS shares RAM with the OS and every other app, so the usable slice is
        # well under the installed total.
        # Two different thresholds, because the two measures trade differently.
        # Attention slicing is a measured 21% slower and only pays off when
        # memory is genuinely scarce. VAE tiling targets the single largest
        # transient allocation — the fp32 decode — for a much smaller cost, and
        # an M2/16GB running SDXL is already swapping (3.8GB of swap observed),
        # so protecting the peak is worth more than the last few percent.
        slice_attention = total < 12 if low_memory is None else low_memory
        tile_vae = total < 24 if low_memory is None else low_memory
        vae_dtype = torch.float32
        notes.append(
            "VAE kept in fp32: the SDXL VAE overflows in fp16 and decodes to "
            "black images. Costs ~200MB, avoids a silent failure."
        )
        return DevicePolicy(
            device=device,
            dtype=dtype,
            vae_dtype=vae_dtype,
            total_memory_gb=total,
            # Measured on an M2/16GB at 1024x1024: slicing cost +21%
            # (17.04 vs 14.08 s/step), because torch's SDPA is already
            # memory-efficient on MPS.
            attention_slicing=slice_attention,
            vae_slicing=True,
            vae_tiling=tile_vae,
            # Offloading to CPU on unified memory buys little and costs a lot of
            # copying, since "CPU RAM" and "GPU RAM" are the same chips.
            model_offload=False,
            sequential_offload=False,
            unified_memory=True,
            notes=tuple(notes),
        )

    # CUDA.
    tight = total < 12 if low_memory is None else low_memory
    very_tight = total < 8 if low_memory is None else low_memory
    if dtype is torch.float16:
        vae_dtype = torch.float32
        notes.append("VAE in fp32 to avoid SDXL fp16 overflow; use a fp16-fix VAE to drop this.")
    else:
        vae_dtype = dtype
    return DevicePolicy(
        device=device,
        dtype=dtype,
        vae_dtype=vae_dtype,
        total_memory_gb=total,
        attention_slicing=False,  # SDPA is faster than slicing when VRAM allows.
        vae_slicing=True,
        vae_tiling=tight,
        model_offload=tight and not very_tight,
        sequential_offload=very_tight,
        unified_memory=False,
        notes=tuple(notes),
    )


def _resolve_dtype(device: str, prefer: str | None, notes: list[str]) -> torch.dtype:
    prefer = (prefer or os.environ.get("PANELFORGE_DTYPE") or "").lower()
    explicit = {"float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float32": torch.float32, "fp32": torch.float32}
    if prefer in explicit:
        return explicit[prefer]
    if device == "cuda":
        # bf16 has fp32's exponent range, so it sidesteps the overflow issues
        # fp16 hits in SDXL. Ampere (SM80) and newer only.
        if torch.cuda.is_bf16_supported():
            notes.append("Using bf16: same range as fp32, no overflow guards needed.")
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        # MPS bf16 support is incomplete across ops in current torch builds;
        # fp16 is the well-trodden path on Apple Silicon.
        return torch.float16
    return torch.float32


def make_generator(device: str, seed: int) -> torch.Generator:
    """Seeded RNG for reproducible runs.

    MPS generators do not reproduce across torch versions and cannot be seeded
    identically to CUDA, so latents are always sampled on CPU and moved. This
    costs nothing and makes a seed mean the same thing on both backends.
    """
    gen_device = "cuda" if device == "cuda" else "cpu"
    return torch.Generator(device=gen_device).manual_seed(int(seed))


def empty_cache(device: str) -> None:
    """Release cached allocator blocks. Safe to call between jobs."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def apply_memory_policy(pipe, policy: DevicePolicy) -> None:
    """Enable the memory measures the policy selected, tolerating absent hooks."""
    def _try(name: str, *args) -> None:
        fn = getattr(pipe, name, None)
        if fn is None:
            return
        try:
            fn(*args)
        except Exception as exc:  # a missing optimisation must not kill a job
            log.debug("%s failed: %s", name, exc)

    if policy.attention_slicing:
        _try("enable_attention_slicing", "auto")
    else:
        _try("disable_attention_slicing")
    if policy.vae_slicing:
        _try("enable_vae_slicing")
    if policy.vae_tiling:
        _try("enable_vae_tiling")

    # Offload hooks move modules themselves; they conflict with an explicit .to().
    if policy.sequential_offload:
        _try("enable_sequential_cpu_offload")
    elif policy.model_offload:
        _try("enable_model_cpu_offload")
