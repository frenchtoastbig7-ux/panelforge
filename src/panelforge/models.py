"""Model discovery.

Checkpoints are identified by reading only the safetensors header, which is a
few kilobytes at the front of the file. Loading a 6GB checkpoint just to learn
whether it is SD 1.5 or SDXL makes the model list unusable.
"""
from __future__ import annotations

import json
import logging
import re
import struct
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Literal

log = logging.getLogger(__name__)

Architecture = Literal["sd15", "sdxl", "sd3", "flux", "unknown"]

# Marker keys, checked in order. Single-file checkpoints keep the original
# LDM/SGM prefixes; diffusers-format directories use their own layout.
_ARCH_MARKERS: tuple[tuple[Architecture, tuple[str, ...]], ...] = (
    ("flux", ("double_blocks.0.img_attn.proj.weight",
              "model.diffusion_model.double_blocks.0.img_attn.proj.weight")),
    ("sd3", ("model.diffusion_model.joint_blocks.0.context_block.attn.qkv.weight",
             "joint_blocks.0.context_block.attn.qkv.weight")),
    # SDXL's second text encoder is the reliable tell.
    ("sdxl", ("conditioner.embedders.1.model.ln_final.weight",
              "conditioner.embedders.1.model.token_embedding.weight",
              "model.diffusion_model.label_emb.0.0.weight")),
    ("sd15", ("cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
              "cond_stage_model.transformer.text_model.final_layer_norm.weight")),
)


@dataclass
class ModelInfo:
    name: str
    path: str
    kind: Literal["checkpoint", "lora", "vae", "embedding"]
    architecture: Architecture
    size_bytes: int
    single_file: bool
    # LoRAs only: the dimensionality hints at which base it was trained against.
    lora_rank: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["size_gb"] = round(self.size_bytes / 1024**3, 2)
        return d


def read_safetensors_keys(path: Path, limit: int | None = None) -> list[str]:
    """Read key names from a safetensors header without mapping the tensors.

    Layout: 8-byte little-endian header length, then that many bytes of JSON.
    """
    try:
        with path.open("rb") as fh:
            raw_len = fh.read(8)
            if len(raw_len) < 8:
                return []
            (header_len,) = struct.unpack("<Q", raw_len)
            # A sane header is well under 100MB; anything larger is not one.
            if header_len <= 0 or header_len > 100 * 1024**2:
                return []
            header = json.loads(fh.read(header_len))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        log.warning("could not read safetensors header from %s: %s", path, exc)
        return []
    keys = [k for k in header if k != "__metadata__"]
    return keys[:limit] if limit else keys


def detect_architecture(path: Path) -> Architecture:
    """Classify a checkpoint file or a diffusers-format directory."""
    if path.is_dir():
        return _detect_diffusers_dir(path)
    keys = read_safetensors_keys(path)
    if not keys:
        return "unknown"
    keyset = set(keys)
    for arch, markers in _ARCH_MARKERS:
        if any(m in keyset for m in markers):
            return arch
    # Diffusers-format single files carry no `cond_stage_model` prefix; fall
    # back to the SDXL-only added-conditioning projection on the UNet.
    if any("add_embedding.linear_1" in k for k in keys):
        return "sdxl"
    if any(k.startswith(("down_blocks.", "model.diffusion_model.")) for k in keys):
        return "sd15"
    return "unknown"


def _detect_diffusers_dir(path: Path) -> Architecture:
    index = path / "model_index.json"
    if not index.exists():
        return "unknown"
    try:
        data = json.loads(index.read_text())
    except (OSError, json.JSONDecodeError):
        return "unknown"
    cls = str(data.get("_class_name", ""))
    if "Flux" in cls:
        return "flux"
    if "StableDiffusion3" in cls:
        return "sd3"
    if "XL" in cls or "text_encoder_2" in data:
        return "sdxl"
    if "StableDiffusion" in cls:
        return "sd15"
    return "unknown"


def detect_lora_target(path: Path) -> tuple[Architecture, int | None]:
    """Guess which base architecture a LoRA was trained for, plus its rank.

    LoRAs ship under two different naming conventions — LDM/kohya
    (`lora_unet_input_blocks_*`) and diffusers (`lora_unet_down_blocks_*`) — and
    many are UNet-only, so the presence of a second text encoder is not enough
    on its own. The structural tell is block count: SD 1.5's UNet has four
    down/up blocks with attention in the first, SDXL's has three with none.
    """
    keys = read_safetensors_keys(path)
    if not keys:
        return "unknown", None
    # Two naming conventions in the wild: kohya/LDM uses underscores
    # (`lora_unet_down_blocks_1_...`) while diffusers uses dots
    # (`unet.down_blocks.1....lora.down.weight`). Normalising the separator
    # lets one set of structural rules cover both.
    blob = "\n".join(keys).replace(".", "_")

    arch: Architecture = "unknown"
    if re.search(r"text_encoder_2|lora_te2|\bte2_", blob):
        arch = "sdxl"
    elif "single_transformer_blocks" in blob:
        arch = "flux"
    elif re.search(r"down_blocks_3|up_blocks_3|input_blocks_(1[01])", blob):
        # A fourth down/up stage exists only in SD 1.5.
        arch = "sd15"
    elif re.search(r"down_blocks_0_attentions|input_blocks_1_1", blob):
        # SD 1.5 attends in the very first block; SDXL's is resnet-only.
        arch = "sd15"
    elif re.search(r"lora_te_", blob):
        # kohya names SD 1.5's single text encoder `lora_te_`; SDXL, having two,
        # numbers them `lora_te1_` / `lora_te2_`.
        arch = "sd15"
    elif "add_embedding" in blob:
        # SDXL's added time/size conditioning projection.
        arch = "sdxl"
    elif re.search(r"lora_unet|down_blocks_|input_blocks_", blob):
        arch = "sdxl"

    rank = _lora_rank(path, keys)
    return arch, rank


def _lora_rank(path: Path, keys: list[str]) -> int | None:
    """Rank is the output dimension of the down-projection.

    Not `min(shape)`: a conv LoRA's weight is [rank, in_ch, kh, kw], and the
    3x3 kernel would win the minimum and report a rank of 3.
    """
    try:
        with path.open("rb") as fh:
            (header_len,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(header_len))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    for key in keys:
        normalised = key.replace(".", "_")
        if "lora_down" in normalised or "lora_A" in normalised:
            shape = header.get(key, {}).get("shape")
            if shape:
                return int(shape[0])
        if "lora_up" in normalised or "lora_B" in normalised:
            shape = header.get(key, {}).get("shape")
            if shape and len(shape) >= 2:
                return int(shape[1])
    return None


def _iter_model_files(directory: Path) -> Iterator[Path]:
    if not directory.exists():
        return
    for entry in sorted(directory.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            # A diffusers-format model directory counts as one model.
            if (entry / "model_index.json").exists():
                yield entry
            continue
        if entry.suffix in {".safetensors", ".ckpt", ".pt", ".bin"}:
            yield entry


def _dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _dedupe(models: list[ModelInfo]) -> list[ModelInfo]:
    """First occurrence of a name wins, so user files override shipped ones."""
    seen: set[str] = set()
    out = []
    for m in models:
        if m.name in seen:
            continue
        seen.add(m.name)
        out.append(m)
    return out


def scan_checkpoints_in(directories: list[Path]) -> list[ModelInfo]:
    return _dedupe([m for d in directories for m in scan_checkpoints(d)])


def scan_loras_in(directories: list[Path]) -> list[ModelInfo]:
    return _dedupe([m for d in directories for m in scan_loras(d)])


def scan_simple_in(directories: list[Path], kind) -> list[ModelInfo]:
    return _dedupe([m for d in directories for m in scan_simple(d, kind)])


def scan_checkpoints(directory: Path) -> list[ModelInfo]:
    out = []
    for path in _iter_model_files(directory):
        out.append(ModelInfo(
            name=path.stem if path.is_file() else path.name,
            path=str(path),
            kind="checkpoint",
            architecture=detect_architecture(path),
            size_bytes=_dir_size(path),
            single_file=path.is_file(),
        ))
    return out


def scan_loras(directory: Path) -> list[ModelInfo]:
    out = []
    for path in _iter_model_files(directory):
        if path.is_dir():
            continue
        arch, rank = detect_lora_target(path)
        out.append(ModelInfo(
            name=path.stem,
            path=str(path),
            kind="lora",
            architecture=arch,
            size_bytes=path.stat().st_size,
            single_file=True,
            lora_rank=rank,
        ))
    return out


def scan_simple(directory: Path, kind) -> list[ModelInfo]:
    out = []
    for path in _iter_model_files(directory):
        out.append(ModelInfo(
            name=path.stem if path.is_file() else path.name,
            path=str(path),
            kind=kind,
            architecture="unknown",
            size_bytes=_dir_size(path),
            single_file=path.is_file(),
        ))
    return out
