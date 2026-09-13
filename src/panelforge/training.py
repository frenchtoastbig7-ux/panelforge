"""LoRA training.

The memory trick that makes SDXL LoRA training possible on a 16GB machine is
caching: every image is encoded to a latent and every caption to a text
embedding *before* training starts, after which the VAE and both text encoders
are freed. The training loop then only needs the UNet resident.
"""
from __future__ import annotations

import gc
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .device import DevicePolicy, empty_cache
from .models import detect_architecture

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ProgressFn = Callable[[int, int, float], None]   # step, total, loss


@dataclass
class TrainConfig:
    dataset_dir: str
    output_name: str
    checkpoint: str
    trigger_word: str = ""
    steps: int = 1000
    learning_rate: float = 1e-4
    rank: int = 16
    alpha: int | None = None          # defaults to rank
    resolution: int = 768
    batch_size: int = 1
    seed: int = 42
    # Text encoders are frozen: training them on 20 images overfits fast and
    # doubles the memory needed.
    train_text_encoder: bool = False
    save_every: int = 0               # 0 => only at the end
    caption_dropout: float = 0.05     # helps the LoRA generalise beyond captions
    warmup_steps: int = 50

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainSample:
    image_path: Path
    caption: str
    latent: torch.Tensor | None = None
    embeds: torch.Tensor | None = None
    pooled: torch.Tensor | None = None
    original_size: tuple[int, int] = (0, 0)


class DatasetError(RuntimeError):
    pass


# ------------------------------------------------------------------ dataset

def read_caption(image_path: Path, trigger: str) -> str:
    """Caption from a sibling .txt (kohya convention), else the trigger word."""
    for suffix in (".txt", ".caption"):
        side = image_path.with_suffix(suffix)
        if side.exists():
            try:
                text = side.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if text:
                # A trigger word must appear, or the LoRA has nothing to bind to.
                if trigger and trigger.lower() not in text.lower():
                    return f"{trigger}, {text}"
                return text
    return trigger or image_path.stem.replace("_", " ")


def scan_dataset(directory: Path, trigger: str = "") -> list[TrainSample]:
    if not directory.is_dir():
        raise DatasetError(f"dataset directory not found: {directory}")
    samples = [
        TrainSample(image_path=p, caption=read_caption(p, trigger))
        for p in sorted(directory.iterdir())
        if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")
    ]
    if not samples:
        raise DatasetError(
            f"no images in {directory}. Supported: "
            f"{', '.join(sorted(IMAGE_SUFFIXES))}")
    if len(samples) < 5:
        log.warning("only %d images: expect weak results (20-30 is typical)",
                    len(samples))
    return samples


def _center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    """Square centre crop. Buckets would be better; this keeps memory flat."""
    img = img.convert("RGB")
    scale = size / min(img.width, img.height)
    new = img.resize((max(size, round(img.width * scale)),
                      max(size, round(img.height * scale))), Image.LANCZOS)
    left = (new.width - size) // 2
    top = (new.height - size) // 2
    return new.crop((left, top, left + size, top + size))


# ------------------------------------------------------------------ trainer

class LoRATrainer:
    def __init__(self, config: TrainConfig, policy: DevicePolicy,
                 offline: bool = False, single_file_configs: dict | None = None):
        self.cfg = config
        self.policy = policy
        self.offline = offline
        self.samples: list[TrainSample] = []
        self._arch = "unknown"
        self._output_dir: Path | None = None
        self._unet_source = None
        self.single_file_configs = dict(single_file_configs or {})

    # -- phase 1: encode everything, then free the encoders ------------------

    def cache_inputs(self, progress: ProgressFn | None = None) -> None:
        from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline

        path = Path(self.cfg.checkpoint)
        self._arch = detect_architecture(path)
        if self._arch not in ("sdxl", "sd15"):
            raise DatasetError(
                f"training supports SD1.5 and SDXL; {path.name} is {self._arch}")
        is_xl = self._arch == "sdxl"
        cls = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline

        self.samples = scan_dataset(Path(self.cfg.dataset_dir), self.cfg.trigger_word)
        log.info("caching %d samples at %dpx", len(self.samples), self.cfg.resolution)

        common = {"torch_dtype": self.policy.dtype}
        if not is_xl:
            common |= {"safety_checker": None, "requires_safety_checker": False}
        if path.is_file():
            config_repo = self.single_file_configs.get("sdxl" if is_xl else "sd15") or (
                "stabilityai/stable-diffusion-xl-base-1.0" if is_xl
                else "runwayml/stable-diffusion-v1-5")
            pipe = cls.from_single_file(str(path), config=config_repo,
                                        local_files_only=self.offline, **common)
        else:
            pipe = cls.from_pretrained(str(path), local_files_only=self.offline,
                                       **common)
        pipe.to(self.policy.device)

        # The VAE encode must not run in fp16: the SDXL VAE overflows, and a
        # corrupt latent cache would poison the entire training run.
        vae = pipe.vae.to(dtype=torch.float32)
        scaling = vae.config.scaling_factor
        total = len(self.samples)
        with torch.no_grad():
            for i, sample in enumerate(self.samples):
                img = Image.open(sample.image_path)
                sample.original_size = (img.height, img.width)
                pixels = _center_crop_resize(img, self.cfg.resolution)
                array = np.asarray(pixels, dtype=np.float32) / 127.5 - 1.0
                tensor = (torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
                          .to(self.policy.device, dtype=torch.float32))
                latent = vae.encode(tensor).latent_dist.sample() * scaling
                sample.latent = latent.squeeze(0).to("cpu", dtype=torch.float32)
                if progress:
                    progress(i + 1, total * 2, 0.0)

            self._cache_text(pipe, is_xl, progress, total)

        # Free everything except what the training loop needs.
        self._unet_source = pipe
        pipe.vae = None
        del vae
        gc.collect()
        empty_cache(self.policy.device)

    def _cache_text(self, pipe, is_xl: bool, progress, total: int) -> None:
        for i, sample in enumerate(self.samples):
            if is_xl:
                embeds, _, pooled, _ = pipe.encode_prompt(
                    prompt=sample.caption, prompt_2=sample.caption,
                    device=self.policy.device, num_images_per_prompt=1,
                    do_classifier_free_guidance=False)
                sample.pooled = pooled.squeeze(0).to("cpu", dtype=torch.float32)
            else:
                embeds, _ = pipe.encode_prompt(
                    prompt=sample.caption, device=self.policy.device,
                    num_images_per_prompt=1, do_classifier_free_guidance=False)
            sample.embeds = embeds.squeeze(0).to("cpu", dtype=torch.float32)
            if progress:
                progress(total + i + 1, total * 2, 0.0)

        pipe.text_encoder = None
        if hasattr(pipe, "text_encoder_2"):
            pipe.text_encoder_2 = None
        gc.collect()
        empty_cache(self.policy.device)

    # -- phase 2: train ------------------------------------------------------

    def train(self, progress: ProgressFn | None = None,
              should_stop: Callable[[], bool] | None = None) -> dict:
        from diffusers import DDPMScheduler
        from peft import LoraConfig

        if not self.samples:
            raise DatasetError("cache_inputs() must run before train()")

        torch.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)

        pipe = self._unet_source
        unet = pipe.unet
        unet.requires_grad_(False)
        unet.train()
        if hasattr(unet, "enable_gradient_checkpointing"):
            # Trades ~30% speed for a large drop in activation memory. On a
            # 16GB machine that is the difference between training and OOM.
            unet.enable_gradient_checkpointing()

        rank = self.cfg.rank
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=self.cfg.alpha or rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet.add_adapter(lora_cfg)

        # LoRA parameters stay in fp32 even when the frozen base is fp16:
        # fp16 gradients on rank-16 updates underflow to zero.
        params = []
        for param in unet.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
                params.append(param)
        if not params:
            raise RuntimeError("no trainable LoRA parameters were created")
        log.info("training %d LoRA tensors (rank %d)", len(params), rank)

        optimizer = torch.optim.AdamW(params, lr=self.cfg.learning_rate,
                                      betas=(0.9, 0.999), weight_decay=1e-2,
                                      eps=1e-8)
        scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        lr_sched = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: min(1.0, (s + 1) / max(1, self.cfg.warmup_steps))
            * (0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, self.cfg.steps))))))

        is_xl = self._arch == "sdxl"
        device = self.policy.device
        losses: list[float] = []
        start = time.perf_counter()

        for step, batch in enumerate(self._batches()):
            if should_stop is not None and should_stop():
                log.info("training stopped at step %d", step)
                break
            if step >= self.cfg.steps:
                break

            latents = torch.stack([s.latent for s in batch]).to(
                device, dtype=self.policy.dtype)
            embeds = torch.stack([s.embeds for s in batch]).to(
                device, dtype=self.policy.dtype)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                      (latents.shape[0],), device=device).long()
            noisy = scheduler.add_noise(latents, noise, timesteps)

            kwargs = {}
            if is_xl:
                pooled = torch.stack([s.pooled for s in batch]).to(
                    device, dtype=self.policy.dtype)
                res = self.cfg.resolution
                # SDXL conditions on the source and target resolution; feeding
                # the true training size keeps it from learning a size bias.
                add_time = torch.tensor(
                    [[res, res, 0, 0, res, res]] * latents.shape[0],
                    device=device, dtype=self.policy.dtype)
                kwargs["added_cond_kwargs"] = {"text_embeds": pooled,
                                               "time_ids": add_time}

            pred = unet(noisy, timesteps, encoder_hidden_states=embeds, **kwargs).sample

            if scheduler.config.prediction_type == "v_prediction":
                target = scheduler.get_velocity(latents, noise, timesteps)
            else:
                target = noise
            # Loss in fp32: an fp16 MSE over a 128x128x4 latent loses precision.
            loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

            if not torch.isfinite(loss):
                log.warning("non-finite loss at step %d; skipping batch", step)
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())
            if progress:
                progress(step + 1, self.cfg.steps, loss.item())
            if self.cfg.save_every and (step + 1) % self.cfg.save_every == 0:
                self._save(unet, pipe, suffix=f"-step{step + 1}")

        path = self._save(unet, pipe)
        elapsed = time.perf_counter() - start
        recent = losses[-50:] or [0.0]
        return {
            "output": str(path),
            "steps_completed": len(losses),
            "elapsed_s": round(elapsed, 1),
            "final_loss": round(sum(recent) / len(recent), 4),
            "rank": rank,
            "size_mb": round(path.stat().st_size / 1024**2, 2) if path.exists() else 0,
        }

    def _batches(self) -> Iterator[list[TrainSample]]:
        """Shuffled, infinitely repeating batches."""
        pool: list[TrainSample] = []
        while True:
            if len(pool) < self.cfg.batch_size:
                order = list(self.samples)
                random.shuffle(order)
                pool.extend(order)
            batch, pool = pool[:self.cfg.batch_size], pool[self.cfg.batch_size:]
            yield batch

    def _save(self, unet, pipe, suffix: str = "") -> Path:
        from diffusers.utils import convert_state_dict_to_diffusers
        from peft.utils import get_peft_model_state_dict

        out_dir = Path(self.cfg.dataset_dir).parent
        state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        name = f"{self.cfg.output_name}{suffix}"
        target_dir = self._output_dir or out_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        type(pipe).save_lora_weights(
            save_directory=str(target_dir),
            unet_lora_layers=state,
            weight_name=f"{name}.safetensors",
            safe_serialization=True,
        )
        path = target_dir / f"{name}.safetensors"
        log.info("saved %s (%.1f MB)", path, path.stat().st_size / 1024**2)
        return path

    def set_output_dir(self, directory: Path) -> None:
        self._output_dir = directory


def estimate_feasibility(policy: DevicePolicy, resolution: int) -> list[str]:
    """Warnings worth showing before someone waits two hours for a bad run."""
    notes: list[str] = []
    if policy.device == "cpu":
        notes.append("CPU training is not practical — expect days, not hours.")
    if policy.unified_memory and policy.total_memory_gb < 24:
        notes.append(
            f"{policy.total_memory_gb:.0f}GB unified memory is tight for SDXL "
            "LoRA training. Use 512-768px, batch size 1, and close other apps; "
            "the machine will page to swap regardless.")
    if resolution > 768 and policy.total_memory_gb < 24:
        notes.append(f"{resolution}px is likely to OOM here; 768 is safer.")
    if policy.device == "mps":
        notes.append("MPS training is slower per step than CUDA and is best "
                     "left running unattended.")
    return notes
