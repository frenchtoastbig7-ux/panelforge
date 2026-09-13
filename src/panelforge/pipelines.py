"""Pipeline loading, caching and the three generation modes.

One checkpoint's weights are loaded once and shared across txt2img, img2img,
inpaint and ControlNet by constructing each variant directly from the base
pipeline's modules. `from_pipe` is deliberately avoided: it recasts the shared
modules in place (see `pipe_for`). On 16GB unified memory the difference
between sharing and reloading is the difference between working and swapping.
"""
from __future__ import annotations

import gc
import logging
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Literal

import torch
from PIL import Image

from .device import DevicePolicy, apply_memory_policy, empty_cache, make_generator
from .models import Architecture, detect_architecture
from .prompting import build_conditioning
from .samplers import build_scheduler

log = logging.getLogger(__name__)

Mode = Literal["txt2img", "img2img", "inpaint", "controlnet"]
ProgressFn = Callable[[int, int], None]

# SDXL was trained on these buckets. Off-bucket sizes produce anatomy artifacts,
# so the UI offers these rather than a free-form slider.
SDXL_BUCKETS = [
    (1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216),
    (1344, 768), (768, 1344), (1536, 640), (640, 1536),
]
SD15_BUCKETS = [
    (512, 512), (640, 512), (512, 640), (768, 512), (512, 768), (768, 768),
]


@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str = ""
    steps: int = 28
    cfg: float = 6.0
    width: int = 1024
    height: int = 1024
    seed: int = -1
    sampler: str = "dpmpp_2m_karras"
    batch_size: int = 1
    checkpoint: str | None = None
    loras: list[dict] = field(default_factory=list)  # [{"name":..., "weight":...}]
    mode: Mode = "txt2img"
    init_image: Image.Image | None = None
    mask_image: Image.Image | None = None
    strength: float = 0.75
    clip_skip: int | None = None
    # High-resolution pass: re-run the finished image through img2img at a
    # larger size. Denoising at the target size directly would exceed the
    # model's training buckets and duplicate limbs and faces.
    hires: bool = False
    hires_scale: float = 1.5
    hires_strength: float = 0.4
    hires_steps: int = 0          # 0 => reuse the base step count
    # ControlNet: a control image (pose skeleton, canny edges, depth map...)
    # plus the name of a model directory under models/controlnet/.
    controlnet: str | None = None
    control_image: Image.Image | None = None
    controlnet_scale: float = 0.8
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0
    # Face enhancement: detect faces and re-inpaint each at full resolution.
    face_enhance: bool = False
    face_strength: float = 0.4
    face_confidence: float = 0.3
    face_max: int = 4
    # Composition tags describe the whole frame and actively harm a face crop.
    face_prompt: str | None = None
    # "inpaint" regenerates only the face and keeps the surrounding hair and
    # background untouched; "img2img" refines the whole padded crop.
    face_mode: str = "inpaint"
    face_work_size: int = 0       # 0 => the model's native resolution


@dataclass
class GenerationResult:
    images: list[Image.Image]
    seeds: list[int]
    elapsed_s: float
    metadata: dict


class PipelineManager:
    """Holds at most one checkpoint resident and swaps modes cheaply."""

    def __init__(self, policy: DevicePolicy, offline: bool = False,
                 detectors_dir: Path | None = None,
                 single_file_configs: dict | None = None):
        self.policy = policy
        self.offline = offline
        self.detectors_dir = detectors_dir
        # arch -> local config folder; lets single-file checkpoints load with
        # no network and no Hugging Face cache (the packaged app ships these).
        self.single_file_configs = dict(single_file_configs or {})
        self._face_detector = None
        self._pipes: dict[Mode, object] = {}
        self._checkpoint: str | None = None
        self._arch: Architecture = "unknown"
        self._active_loras: list[tuple[str, float]] = []
        self._sampler: str | None = None
        self._fused = False
        self._controlnet = None
        self._controlnet_key: str | None = None
        self._last_face_note: str | None = None

    # ---------------------------------------------------------------- loading

    def _pipeline_classes(self, arch: Architecture):
        from diffusers import (
            StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline,
            StableDiffusionPipeline, StableDiffusionXLImg2ImgPipeline,
            StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline,
        )
        if arch == "sdxl":
            return {
                "txt2img": StableDiffusionXLPipeline,
                "img2img": StableDiffusionXLImg2ImgPipeline,
                "inpaint": StableDiffusionXLInpaintPipeline,
            }
        if arch == "sd15":
            return {
                "txt2img": StableDiffusionPipeline,
                "img2img": StableDiffusionImg2ImgPipeline,
                "inpaint": StableDiffusionInpaintPipeline,
            }
        raise ValueError(
            f"architecture {arch!r} is not supported yet (SD1.5 and SDXL are)"
        )

    def _single_file_config(self, arch: Architecture) -> str:
        # from_single_file needs a reference repo for the component configs.
        # These are small JSON files, fetched once and then cached.
        local = self.single_file_configs.get(arch)
        if local:
            return local
        return {
            "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
            "sd15": "runwayml/stable-diffusion-v1-5",
        }[arch]

    def load(self, checkpoint: str | Path, force: bool = False) -> None:
        """Make `checkpoint` the resident model, freeing the previous one."""
        checkpoint = str(checkpoint)
        if checkpoint == self._checkpoint and not force:
            return

        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

        arch = detect_architecture(path)
        if arch == "unknown":
            raise ValueError(
                f"could not identify the architecture of {path.name}. "
                "Is it a full checkpoint rather than a LoRA or VAE?"
            )
        classes = self._pipeline_classes(arch)

        self.unload()
        log.info("loading %s (%s) as %s", path.name, arch, self.policy.summary())
        t0 = time.perf_counter()

        common = {"torch_dtype": self.policy.dtype}
        if arch == "sd15":
            # The safety checker is a separate 1.2GB download and blanks
            # false-positive frames; this tool is local and offline by design.
            common |= {"safety_checker": None, "requires_safety_checker": False}

        cls = classes["txt2img"]
        if path.is_file():
            pipe = cls.from_single_file(
                str(path), config=self._single_file_config(arch),
                local_files_only=self.offline, **common,
            )
        else:
            pipe = cls.from_pretrained(
                str(path), local_files_only=self.offline, **common
            )

        self._prepare(pipe)
        self._pipes = {"txt2img": pipe}
        self._checkpoint = checkpoint
        self._arch = arch
        self._sampler = None
        log.info("loaded in %.1fs", time.perf_counter() - t0)

    def _prepare(self, pipe) -> None:
        """Move to device and apply the memory policy."""
        if not (self.policy.model_offload or self.policy.sequential_offload):
            pipe.to(self.policy.device)
        if getattr(pipe, "vae", None) is not None:
            # SDXL's VAE overflows fp16 and decodes to black frames. Rather than
            # keeping a permanent fp32 copy, let diffusers upcast only for the
            # decode step — same correctness, ~200MB less resident.
            if self.policy.vae_dtype != self.policy.dtype:
                pipe.vae.config.force_upcast = True
        pipe.set_progress_bar_config(disable=True)
        apply_memory_policy(pipe, self.policy)

    def load_controlnet(self, directory: str | Path):
        """Load and cache a ControlNet by directory, keeping one resident."""
        from diffusers import ControlNetModel

        directory = Path(directory)
        key = str(directory)
        if self._controlnet_key == key and self._controlnet is not None:
            return self._controlnet
        if not directory.is_dir():
            raise FileNotFoundError(f"controlnet not found: {directory}")
        log.info("loading controlnet %s", directory.name)
        # Only one ControlNet stays resident: each is ~2.3GB.
        self._controlnet = None
        gc.collect()
        empty_cache(self.policy.device)
        model = ControlNetModel.from_pretrained(
            str(directory), torch_dtype=self.policy.dtype,
            local_files_only=self.offline)
        if not (self.policy.model_offload or self.policy.sequential_offload):
            model.to(self.policy.device)
        self._controlnet = model
        self._controlnet_key = key
        # A cached ControlNet pipeline built on the previous model is now stale.
        self._pipes.pop("controlnet", None)
        return model

    def _controlnet_policy(self) -> DevicePolicy:
        """Tighten memory settings while a ControlNet is resident.

        A ControlNet adds ~2.3GB of weights on top of the base model. Measured
        on an M2/16GB at 832x1216: the checkpoint alone sits at 6.5GB allocated,
        the ControlNet takes it to 8.8GB, and generation then drives the driver
        reservation to the 20.13GB MPS ceiling and the process is killed.

        Slicing alone did not move that number. CPU offload does, and the
        reason is specific to how MPS accounts memory: offloaded weights live
        in CPU RAM, outside the MPS allocator pool that the watermark caps.
        On unified memory this costs little bandwidth — the bytes never leave
        the same physical chips — while removing gigabytes from the pool that
        actually runs out.
        """
        tightened = replace(self.policy, attention_slicing=True, vae_tiling=True)
        if self.policy.device == "mps" and self.policy.total_memory_gb < 32:
            tightened = replace(tightened, model_offload=True)
        elif self.policy.is_cuda and self.policy.total_memory_gb < 16:
            tightened = replace(tightened, model_offload=True)
        return tightened

    def _check_dtype(self, pipe, mode: str) -> None:
        """Restore, loudly, any shared weights whose precision a variant changed.

        Offload hooks manage placement themselves, so offloaded pipelines are
        left alone.
        """
        if self.policy.model_offload or self.policy.sequential_offload:
            return
        drifted = []
        for name in ("unet", "text_encoder", "text_encoder_2"):
            module = getattr(pipe, name, None)
            if module is None:
                continue
            try:
                actual = next(module.parameters()).dtype
            except (StopIteration, AttributeError):
                continue
            if actual != self.policy.dtype:
                drifted.append(f"{name}={actual}")
                module.to(dtype=self.policy.dtype)
        if drifted:
            log.error("building the %s pipeline changed shared weight precision (%s); "
                      "restored to %s", mode, ", ".join(drifted), self.policy.dtype)

    def pipe_for(self, mode: Mode):
        """Return the pipeline for `mode`, rewrapping shared weights if needed."""
        if self._checkpoint is None:
            raise RuntimeError("no checkpoint loaded")
        if mode in self._pipes:
            return self._pipes[mode]
        base = self._pipes["txt2img"]
        if mode == "controlnet":
            if self._controlnet is None:
                raise RuntimeError("no controlnet loaded")
            from diffusers import (StableDiffusionControlNetPipeline,
                                   StableDiffusionXLControlNetPipeline)
            cls = (StableDiffusionXLControlNetPipeline if self._arch == "sdxl"
                   else StableDiffusionControlNetPipeline)
            # Deliberately NOT from_pipe: it re-casts every component with
            # .to(), which transiently doubles resident weights. Measured on an
            # M2/16GB that took 8.8GB to ~17GB and the OS killed the process
            # before a single denoising step ran. The constructor just rebinds
            # the modules that are already on the device, allocating nothing.
            pipe = cls(**_shared_components(cls, base),
                       controlnet=self._controlnet)
            self._check_dtype(pipe, mode)
            pipe.set_progress_bar_config(disable=True)
            cn_policy = self._controlnet_policy()
            if cn_policy.model_offload:
                # Offload hooks move modules themselves and conflict with the
                # explicit .to(device) the base pipeline already had applied.
                gc.collect()
                empty_cache(self.policy.device)
            apply_memory_policy(pipe, cn_policy)
            self._pipes[mode] = pipe
            return pipe
        cls = self._pipeline_classes(self._arch)[mode]
        # Not from_pipe. It casts the shared modules with .to(), and here that
        # cast lands in float32 and happens *in place*: measured on an M2/16GB,
        # building the img2img pipeline flipped the UNet, VAE and both text
        # encoders of the base pipeline from fp16 to fp32 and took allocation
        # from 6.48GB to 12.99GB (driver 17.29GB). Past that point MPS passes
        # silently returned zeros -- the text encoder and the VAE encoder both
        # produced all-zero tensors -- so every face-enhance and hi-res image
        # decoded to solid white, and later txt2img jobs inherited the fp32
        # model. The constructor rebinds the existing modules untouched.
        pipe = cls(**_shared_components(cls, base))
        self._check_dtype(pipe, mode)
        pipe.set_progress_bar_config(disable=True)
        apply_memory_policy(pipe, self.policy)
        self._pipes[mode] = pipe
        return pipe

    def unload(self) -> None:
        self._pipes.clear()
        self._checkpoint = None
        self._active_loras = []
        self._fused = False
        self._controlnet = None
        self._controlnet_key = None
        gc.collect()
        empty_cache(self.policy.device)

    # ------------------------------------------------------------------ loras

    def _sync_loras(self, pipe, loras: list[dict]) -> None:
        """Load and weight the requested LoRA stack, unloading anything stale."""
        wanted = [(str(l["name"]), float(l.get("weight", 1.0)))
                  for l in loras if l.get("name")]
        if wanted == self._active_loras:
            return

        # Fused weights are baked into the base modules and must be undone
        # before the adapters they came from are unloaded.
        if self._fused:
            try:
                pipe.unfuse_lora()
            except Exception as exc:
                log.debug("unfuse_lora: %s", exc)
            self._fused = False
        try:
            pipe.unload_lora_weights()
        except Exception as exc:
            log.debug("unload_lora_weights: %s", exc)
        self._active_loras = []
        if not wanted:
            return

        names, weights = [], []
        for path_str, weight in wanted:
            path = Path(path_str)
            if not path.exists():
                log.warning("LoRA not found, skipping: %s", path_str)
                continue
            # Adapter names are identifiers in peft; punctuation breaks lookup.
            adapter = "".join(c if c.isalnum() else "_" for c in path.stem)
            try:
                pipe.load_lora_weights(
                    str(path.parent), weight_name=path.name, adapter_name=adapter
                )
            except Exception as exc:
                log.error("failed to load LoRA %s: %s", path.name, exc)
                continue
            names.append(adapter)
            weights.append(weight)

        if names:
            pipe.set_adapters(names, adapter_weights=weights)
            self._active_loras = wanted
            # Fusing folds the adapters into the base weights, removing a
            # per-layer matmul from every forward pass. Measured on an M2:
            # 19.06 -> 12.76 s/step at 1024x1024, a 33% saving.
            try:
                pipe.fuse_lora()
                self._fused = True
            except Exception as exc:
                # Unfused still produces correct images, just slower.
                log.warning("could not fuse LoRA (%s); continuing unfused", exc)

    # ------------------------------------------------------------- generation

    def _apply_sampler(self, pipe, name: str) -> None:
        if name == self._sampler:
            return
        pipe.scheduler = build_scheduler(name, pipe.scheduler.config)
        self._sampler = name
        # Mode variants share the scheduler object with the base pipeline.
        for other in self._pipes.values():
            other.scheduler = pipe.scheduler

    def _apply_clip_skip(self, pipe, clip_skip: int | None) -> dict:
        """Anime finetunes are almost universally trained at clip skip 2."""
        if not clip_skip or clip_skip <= 1:
            return {}
        return {"clip_skip": clip_skip}

    def generate(
        self, req: GenerationRequest, progress: ProgressFn | None = None
    ) -> GenerationResult:
        if self._checkpoint is None:
            raise RuntimeError("no checkpoint loaded")

        # Load the ControlNet here rather than at request time: switching
        # checkpoints calls unload(), which releases it, and the checkpoint
        # swap happens after the request is accepted.
        if req.mode == "controlnet" and req.controlnet:
            self.load_controlnet(req.controlnet)

        pipe = self.pipe_for(req.mode)
        self._apply_sampler(pipe, req.sampler)
        self._sync_loras(pipe, req.loras)

        seeds = self.resolve_seeds(req)
        is_sdxl = self._arch == "sdxl"
        cond = build_conditioning(pipe, req.prompt, req.negative_prompt, is_sdxl)

        kwargs: dict = dict(cond.as_kwargs())
        kwargs |= {
            "num_inference_steps": req.steps,
            "guidance_scale": req.cfg,
        }
        # clip_skip is incompatible with precomputed embeddings; compel already
        # applied the layer selection when it built them.
        if cond.prompt_embeds is None:
            kwargs |= self._apply_clip_skip(pipe, req.clip_skip)

        if req.mode == "controlnet":
            if req.control_image is None:
                raise ValueError("controlnet mode needs a control image")
            control = req.control_image.convert("RGB").resize(
                (req.width, req.height), Image.LANCZOS)
            kwargs |= {
                "image": control,
                "width": req.width, "height": req.height,
                "controlnet_conditioning_scale": req.controlnet_scale,
                "control_guidance_start": req.control_guidance_start,
                "control_guidance_end": req.control_guidance_end,
            }
        elif req.mode == "txt2img":
            kwargs |= {"width": req.width, "height": req.height}
        elif req.mode == "img2img":
            kwargs |= {"image": req.init_image, "strength": req.strength}
        else:
            kwargs |= {
                "image": req.init_image,
                "mask_image": req.mask_image,
                "strength": req.strength,
                "width": req.width,
                "height": req.height,
            }

        if progress is not None:
            kwargs["callback_on_step_end"] = _step_callback(progress, req.steps, req.mode, req.strength)

        images: list[Image.Image] = []
        t0 = time.perf_counter()
        for seed in seeds:
            generator = make_generator(self.policy.device, seed)
            with torch.inference_mode():
                out = pipe(generator=generator, **kwargs)
            batch = list(out.images)
            if req.hires and req.mode in ("txt2img", "controlnet"):
                batch = [self._hires_pass(img, req, cond, seed) for img in batch]
            if req.face_enhance:
                self._last_face_note = None
                batch = [self._face_pass(img, req, cond, seed) for img in batch]
                # Restore the base policy: the face pass tightened the shared
                # modules and the next job should not inherit that.
                apply_memory_policy(pipe, self.policy)
            images.extend(batch)
        elapsed = time.perf_counter() - t0

        empty_cache(self.policy.device)
        return GenerationResult(
            images=images,
            seeds=seeds,
            elapsed_s=elapsed,
            metadata={
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "steps": req.steps,
                "cfg": req.cfg,
                "sampler": req.sampler,
                "width": req.width,
                "height": req.height,
                "mode": req.mode,
                "checkpoint": Path(self._checkpoint).name,
                "architecture": self._arch,
                "loras": [{"name": Path(n).name, "weight": w}
                          for n, w in self._active_loras],
                "device": self.policy.device,
                "dtype": str(self.policy.dtype).replace("torch.", ""),
                "elapsed_s": round(elapsed, 2),
                "controlnet": Path(req.controlnet).name if req.controlnet else None,
                "controlnet_scale": req.controlnet_scale if req.controlnet else None,
                "face_enhance": req.face_enhance,
                "face_result": self._last_face_note if req.face_enhance else None,
                "hires": req.hires,
                "hires_scale": req.hires_scale if req.hires else None,
            },
        )

    def _hires_pass(self, image: Image.Image, req: GenerationRequest,
                    cond, seed: int) -> Image.Image:
        """Upscale then lightly re-denoise, the standard "hires fix"."""
        target = (int(image.width * req.hires_scale) // 8 * 8,
                  int(image.height * req.hires_scale) // 8 * 8)
        upscaled = image.resize(target, Image.LANCZOS)
        pipe = self.pipe_for("img2img")
        self._apply_sampler(pipe, req.sampler)
        steps = req.hires_steps or req.steps
        kwargs = dict(cond.as_kwargs())
        kwargs |= {
            "image": upscaled,
            # Low strength keeps the composition and only restores detail lost
            # to the resize; high strength reinvents the image.
            "strength": req.hires_strength,
            "num_inference_steps": steps,
            "guidance_scale": req.cfg,
        }
        with torch.inference_mode():
            out = pipe(generator=make_generator(self.policy.device, seed),
                       output_type="latent", **kwargs)
        return self._decode_latents(pipe, out.images)

    def face_detector(self):
        """Lazily construct the face detector; None when unavailable."""
        if self._face_detector is not None:
            return self._face_detector
        if self.detectors_dir is None:
            return None
        from .enhance import FaceDetector
        detector = FaceDetector(self.detectors_dir / "face_yolov8m.pt")
        if not detector.available:
            return None
        self._face_detector = detector
        return detector

    def _face_pass(self, image: Image.Image, req: GenerationRequest,
                   cond, seed: int) -> Image.Image:
        """Re-render each detected face through inpainting, then composite."""
        from .enhance import enhance_faces

        detector = self.face_detector()
        if detector is None:
            log.warning("face enhancement requested but no detector is installed")
            return image

        use_inpaint = req.face_mode == "inpaint"
        pipe = self.pipe_for("inpaint" if use_inpaint else "img2img")
        self._apply_sampler(pipe, req.sampler)
        # The base model plus a fused LoRA already fills most of the MPS pool;
        # an unsliced 768px inpaint on top of it exhausts the allocator, the
        # error is caught per-face, and the pass silently does nothing.
        apply_memory_policy(pipe, self._aux_policy())
        gc.collect()
        empty_cache(self.policy.device)

        # Re-encode with a face-appropriate prompt. Reusing the scene prompt
        # asks for "wide shot, full body" inside a close-up of a face, which is
        # exactly the wrong instruction.
        face_prompt = req.face_prompt or _face_prompt_from(req.prompt)
        face_cond = build_conditioning(pipe, face_prompt, req.negative_prompt,
                                       self._arch == "sdxl")
        base_kwargs = dict(face_cond.as_kwargs())

        # img2img-style inpainting runs steps * strength actual denoising
        # steps. At strength 0.4 a nominal 8 steps is 3 real ones, which
        # produces smeared artefacts rather than detail. Scale the nominal
        # count so the effective count matches the request's step budget --
        # this also keeps a distilled LoRA at the step count it was trained for.
        strength = max(0.15, min(1.0, req.face_strength))
        nominal = max(req.steps, math.ceil(req.steps / strength))

        def _refine(crop: Image.Image, size: int, mask: Image.Image | None = None) -> Image.Image:
            kwargs = dict(base_kwargs)
            kwargs |= {
                "image": crop,
                # Low strength: the goal is to add detail, not a different face.
                "strength": strength,
                "num_inference_steps": nominal,
                "guidance_scale": req.cfg,
            }
            if use_inpaint:
                kwargs |= {"mask_image": mask, "width": size, "height": size}
            with torch.inference_mode():
                latents = pipe(generator=make_generator(self.policy.device, seed),
                               output_type="latent", **kwargs).images
            return self._decode_latents(pipe, latents)

        try:
            result, enhanced, errors = enhance_faces(
                image, detector, _refine, confidence=req.face_confidence,
                max_faces=req.face_max, work_size=self._face_work_size(req.face_work_size))
        except Exception as exc:
            log.error("face enhancement failed, keeping the original: %s", exc)
            self._last_face_note = f"failed: {exc}"
            return image
        if errors and not enhanced:
            log.warning("face enhancement changed nothing (%d face(s) failed): %s",
                        len(errors), errors[0])
            self._last_face_note = f"0 enhanced, {len(errors)} failed: {errors[0]}"
        else:
            log.info("face enhancement improved %d face(s)%s", len(enhanced),
                     f", {len(errors)} failed" if errors else "")
            self._last_face_note = f"{len(enhanced)} enhanced" + (
                f", {len(errors)} failed" if errors else "")
        return result

    def _aux_policy(self) -> DevicePolicy:
        """Memory settings for a secondary pass running beside a loaded model."""
        return replace(self.policy, attention_slicing=True, vae_tiling=True)

    def _face_work_size(self, override: int = 0) -> int:
        """Resolution for the face re-render: the model's native resolution.

        Measured on an M2 with an 85px face: SDXL at 512px fragmented the face
        into zig-zag strokes at every strength and in both modes; at 768 and
        1024 the same passes were clean, 1024 the clearest. An earlier 512px cap
        answered out-of-memory failures that were really from_pipe recasting the
        model to fp32. Cost there: ~28s at 512, ~80s at 768, ~200s at 1024.
        """
        if override:
            return override
        return 1024 if self._arch == "sdxl" else 512

    def _decode_latents(self, pipe, latents: torch.Tensor) -> Image.Image:
        """Decode in fp32 rather than trusting the pipeline's dtype juggling.

        diffusers' SDXL decode only upcasts when it sees an fp16 VAE; on MPS it
        will otherwise cast the VAE to the latents' dtype. The SDXL VAE
        overflows in fp16 -- an fp16 decode on an M2 came back NaN in every
        element -- so the decode is pinned to fp32 here whatever state the
        shared VAE was left in.
        """
        vae = pipe.vae
        original = next(vae.parameters()).dtype
        vae.to(dtype=torch.float32)
        try:
            latents = latents.to(device=vae.device, dtype=torch.float32)
            cfg = vae.config
            mean = getattr(cfg, "latents_mean", None)
            std = getattr(cfg, "latents_std", None)
            if mean is not None and std is not None:
                mean_t = torch.tensor(mean).view(1, -1, 1, 1).to(latents)
                std_t = torch.tensor(std).view(1, -1, 1, 1).to(latents)
                latents = latents * std_t / cfg.scaling_factor + mean_t
            else:
                latents = latents / cfg.scaling_factor
            with torch.inference_mode():
                decoded = vae.decode(latents, return_dict=False)[0]
        finally:
            # The VAE is shared with every other pipeline variant. txt2img's own
            # decode only takes its safe upcast path when it finds the VAE in
            # fp16; left in fp32, the next txt2img decode on MPS would cast it
            # down to the fp16 latents' dtype and decode unguarded.
            vae.to(dtype=original)
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    def resolve_seeds(self, req: GenerationRequest) -> list[int]:
        if req.seed is not None and req.seed >= 0:
            # A batch from one seed must not be N identical images.
            return [req.seed + i for i in range(req.batch_size)]
        return [int(torch.randint(0, 2**31 - 1, (1,)).item())
                for _ in range(req.batch_size)]

    # ------------------------------------------------------------------ state

    @property
    def checkpoint(self) -> str | None:
        return self._checkpoint

    @property
    def architecture(self) -> Architecture:
        return self._arch

    def buckets(self) -> list[tuple[int, int]]:
        return SDXL_BUCKETS if self._arch == "sdxl" else SD15_BUCKETS


# Tags that describe framing or the wider scene. They make sense for the full
# image and mislead the model when only a face is in frame.
_COMPOSITION_TAGS = (
    "full body", "wide shot", "distant", "far away", "long shot", "cowboy shot",
    "from behind", "from above", "from below", "scenery", "landscape",
    "standing", "sitting", "walking", "running", "upper body", "lower body",
)


def _face_prompt_from(prompt: str) -> str:
    """Strip framing tags and steer the crop toward a detailed face."""
    kept = []
    for tag in prompt.split(","):
        cleaned = tag.strip()
        if not cleaned:
            continue
        if any(bad in cleaned.lower() for bad in _COMPOSITION_TAGS):
            continue
        kept.append(cleaned)
    kept = kept or ["1girl"]
    return ", ".join(kept + ["face focus", "portrait", "detailed face",
                             "detailed eyes", "best quality"])


def _shared_components(cls, base) -> dict:
    """The base pipeline's modules, filtered to what `cls` actually accepts."""
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters)
    out = {name: module for name, module in base.components.items()
           if name in accepted and name != "controlnet"}
    if "requires_safety_checker" in accepted:
        # SD 1.5 variants: the checker was deliberately omitted at load.
        out["requires_safety_checker"] = False
    return out


def _step_callback(progress: ProgressFn, steps: int, mode: Mode, strength: float):
    """Wrap a progress function in diffusers' callback_on_step_end signature.

    img2img and inpaint run only `steps * strength` denoising steps, so the
    reported total has to match or the bar stalls short of full.
    """
    total = steps if mode == "txt2img" else max(1, int(steps * strength))

    def _cb(pipe, step: int, timestep, callback_kwargs):
        try:
            progress(step + 1, total)
        except Exception:
            pass  # a UI hiccup must never abort a generation
        return callback_kwargs

    return _cb
