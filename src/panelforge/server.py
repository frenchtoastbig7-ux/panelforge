"""Local HTTP API and web UI.

Binds to loopback by default. Nothing here reaches the network: models are read
from disk and images are written to disk.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from .characters import Character, CharacterLibrary, compose_prompt
from .comic import (
    LAYOUTS, PAGE_PRESETS, Bubble, Page, Panel, available_fonts, build_page,
    layout_names, nearest_bucket, panel_pixel_size, render_page,
)
from .config import Settings, load_settings
from .device import resolve_policy
from .jobs import JobQueue
from .models import scan_checkpoints_in, scan_loras_in, scan_simple_in
from .pipelines import SD15_BUCKETS, SDXL_BUCKETS, GenerationRequest, PipelineManager
from .pose import POSE_PRESETS, draw_openpose, mirror, pose_names, preset_image
from .samplers import sampler_names
from .sheets import (
    cell_negative, cell_prompt, cells_for, compose_sheet, sheet_presets, SHEET_NEGATIVE,
)
from .styles import get_style, load_styles

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent / "web"
# img2img refines the whole face crop; inpaint regenerates only the face.
FACE_MODES = ("img2img", "inpaint")


class LoraSpec(BaseModel):
    name: str
    weight: float = 1.0


class GenerateBody(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = Field(default=28, ge=1, le=150)
    cfg: float = Field(default=6.0, ge=0.0, le=30.0)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    seed: int = -1
    sampler: str = "dpmpp_2m_karras"
    batch_size: int = Field(default=1, ge=1, le=16)
    checkpoint: str | None = None
    loras: list[LoraSpec] = Field(default_factory=list)
    mode: str = "txt2img"
    init_image: str | None = None  # data URL or bare base64
    mask_image: str | None = None
    strength: float = Field(default=0.75, ge=0.0, le=1.0)
    clip_skip: int | None = 2
    style: str | None = None
    hires: bool = False
    face_enhance: bool = False
    face_strength: float = Field(default=0.4, ge=0.05, le=1.0)
    face_prompt: str | None = None
    face_mode: str = "inpaint"
    face_work_size: int = Field(default=0, ge=0, le=2048)
    controlnet: str | None = None
    control_image: str | None = None
    controlnet_scale: float = Field(default=0.8, ge=0.0, le=2.0)


class LoadBody(BaseModel):
    checkpoint: str


class NewPageBody(BaseModel):
    layout: str = "manga-classic"
    page_preset: str = "manga-b5"
    reading_order: str = "rtl"


class RenderPageBody(BaseModel):
    page: dict
    name: str | None = None


class PageGenerateBody(BaseModel):
    page: dict
    checkpoint: str | None = None
    sampler: str = "dpmpp_2m_karras"
    steps: int = Field(default=28, ge=1, le=150)
    cfg: float = Field(default=6.0, ge=0.0, le=30.0)
    negative_prompt: str = ""
    characters: list[str] = Field(default_factory=list)
    loras: list[LoraSpec] = Field(default_factory=list)
    hires: bool = False
    face_enhance: bool = False
    face_mode: str = "inpaint"
    face_work_size: int = Field(default=0, ge=0, le=2048)
    style: str | None = None
    only_empty: bool = True     # skip panels that already have art
    panel_indices: list[int] | None = None


class CharacterBody(BaseModel):
    id: str | None = None
    name: str
    prompt: str = ""
    negative_prompt: str = ""
    loras: list[LoraSpec] = Field(default_factory=list)
    seed: int = -1
    reference_image: str | None = None
    notes: str = ""


class SheetBody(BaseModel):
    character_id: str
    preset: str = "full"
    checkpoint: str | None = None
    steps: int = Field(default=28, ge=1, le=150)
    cfg: float = Field(default=6.0, ge=0.0, le=30.0)
    sampler: str = "dpmpp_2m_karras"
    style: str | None = None
    extra: str = ""


class SheetComposeBody(BaseModel):
    character_id: str
    preset: str = "full"
    images: list[str | None] = Field(default_factory=list)
    name: str | None = None


class PoseBody(BaseModel):
    preset: str | None = None
    keypoints: list[list[float] | None] | None = None
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    mirror: bool = False


class TrainBody(BaseModel):
    dataset_dir: str
    output_name: str
    trigger_word: str = ""
    checkpoint: str | None = None
    steps: int = Field(default=1000, ge=10, le=10000)
    rank: int = Field(default=16, ge=1, le=128)
    learning_rate: float = Field(default=1e-4, gt=0, le=1e-2)
    resolution: int = Field(default=768, ge=256, le=1024)
    batch_size: int = Field(default=1, ge=1, le=8)
    save_every: int = 0


class SaveProjectBody(BaseModel):
    name: str
    page: dict


def _decode_image(payload: str | None) -> Image.Image | None:
    if not payload:
        return None
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        raw = base64.b64decode(payload)
    except Exception as exc:
        raise HTTPException(400, f"malformed image payload: {exc}") from exc
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"unreadable image: {exc}") from exc


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    policy = resolve_policy(settings.device, settings.dtype, settings.low_memory)
    detector = settings.find_detector("face_yolov8m.pt")
    manager = PipelineManager(policy, offline=settings.offline,
                              detectors_dir=detector.parent if detector else settings.detectors_dir,
                              single_file_configs=settings.single_file_configs)
    jobs = JobQueue(manager, settings.outputs_dir)
    characters = CharacterLibrary(settings.root / "characters.json")
    projects_dir = settings.root / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    training_state: dict = {"state": "idle", "phase": None, "step": 0,
                            "total": 0, "loss": None, "error": None,
                            "result": None, "cancel": False}

    log.info("hardware: %s", policy.summary())
    for note in policy.notes:
        log.info("  note: %s", note)

    app = FastAPI(title="panelforge", version="0.1.0")

    @app.middleware("http")
    async def revalidate_ui_assets(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            # The UI is plain files read from disk with no build step or content
            # hashes. Without revalidation the browser kept serving an old
            # studio.js after an upgrade, and a new button silently did nothing.
            # no-cache still permits cheap 304 responses via ETag/Last-Modified.
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/status")
    def status() -> dict:
        return {
            "device": policy.device,
            "dtype": str(policy.dtype).replace("torch.", ""),
            "memory_gb": round(policy.total_memory_gb, 1),
            "unified_memory": policy.unified_memory,
            "summary": policy.summary(),
            "notes": list(policy.notes),
            "checkpoint": manager.checkpoint,
            "checkpoint_name": Path(manager.checkpoint).name if manager.checkpoint else None,
            "architecture": manager.architecture,
            "queue_depth": jobs.depth(),
            "buckets": manager.buckets() if manager.checkpoint else SDXL_BUCKETS,
        }

    @app.get("/api/models")
    def models() -> dict:
        return {
            "checkpoints": [m.to_dict() for m in scan_checkpoints_in(settings.checkpoint_dirs)],
            "loras": [m.to_dict() for m in scan_loras_in(settings.lora_dirs)],
            "vaes": [m.to_dict() for m in scan_simple_in(settings.vae_dirs, "vae")],
            "samplers": sampler_names(),
            "styles": [s.to_dict() for s in load_styles(settings.root)],
            "buckets": {"sdxl": SDXL_BUCKETS, "sd15": SD15_BUCKETS},
            "defaults": {
                "steps": settings.default_steps,
                "cfg": settings.default_cfg,
                "width": settings.default_width,
                "height": settings.default_height,
                "sampler": settings.default_sampler,
                "negative_prompt": settings.default_negative,
            },
        }

    @app.post("/api/models/load")
    def load_model(body: LoadBody) -> dict:
        try:
            manager.load(body.checkpoint)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return status()

    @app.post("/api/generate")
    def generate(body: GenerateBody) -> dict:
        if body.mode not in ("txt2img", "img2img", "inpaint"):
            raise HTTPException(400, f"unknown mode {body.mode!r}")
        if body.controlnet and body.mode != "txt2img":
            raise HTTPException(400, "controlnet currently applies to text-to-image only")
        if body.face_mode not in FACE_MODES:
            raise HTTPException(400, f"unknown face_mode {body.face_mode!r}; choose from {list(FACE_MODES)}")
        if body.face_work_size and (body.face_work_size < 256 or body.face_work_size % 8):
            raise HTTPException(400, "face_work_size must be 0 (native) or a multiple of 8, at least 256")
        if body.mode != "txt2img" and not body.init_image:
            raise HTTPException(400, f"{body.mode} requires an init image")
        if body.mode == "inpaint" and not body.mask_image:
            raise HTTPException(400, "inpaint requires a mask image")

        checkpoint = body.checkpoint or manager.checkpoint or settings.default_checkpoint
        if not checkpoint:
            available = scan_checkpoints_in(settings.checkpoint_dirs)
            if not available:
                raise HTTPException(
                    400,
                    f"no checkpoints found in {settings.checkpoints_dir}. "
                    "Drop a .safetensors model there and reload.",
                )
            checkpoint = available[0].path

        style = get_style(settings.root, body.style)
        prompt = body.prompt
        negative = body.negative_prompt or settings.default_negative
        steps, cfg, sampler = body.steps, body.cfg, body.sampler
        if style is not None:
            prompt, negative = style.apply(prompt, negative)
            # A style's sampler hints are defaults, not overrides: they apply
            # only where the request left the value at its own default.
            if style.steps and body.steps == 28:
                steps = style.steps
            if style.cfg is not None and body.cfg == 6.0:
                cfg = style.cfg
            if style.sampler and body.sampler == "dpmpp_2m_karras":
                sampler = style.sampler

        mask = _decode_image(body.mask_image)
        if mask is not None:
            mask = mask.convert("L")

        mode = body.mode
        control = _decode_image(body.control_image)
        if body.controlnet:
            if control is None:
                raise HTTPException(400, "controlnet needs a control image")
            cn_dir = settings.find_controlnet(body.controlnet)
            if cn_dir is None:
                raise HTTPException(400, f"controlnet {body.controlnet!r} is not installed")
            mode = "controlnet"
        if body.face_enhance and manager.face_detector() is None:
            raise HTTPException(
                400, "face enhancement needs models/detectors/face_yolov8m.pt")

        req = GenerationRequest(
            prompt=prompt,
            negative_prompt=negative,
            steps=steps, cfg=cfg, width=body.width, height=body.height,
            seed=body.seed, sampler=sampler, batch_size=body.batch_size,
            checkpoint=checkpoint,
            loras=[l.model_dump() for l in body.loras],
            mode=mode,
            init_image=_decode_image(body.init_image),
            mask_image=mask,
            strength=body.strength,
            clip_skip=body.clip_skip,
            hires=body.hires,
            face_enhance=body.face_enhance,
            face_strength=body.face_strength,
            face_prompt=body.face_prompt,
            face_mode=body.face_mode,
            face_work_size=body.face_work_size,
            controlnet=str(cn_dir) if body.controlnet else None,
            control_image=control,
            controlnet_scale=body.controlnet_scale,
        )
        return jobs.submit(req).public()

    @app.get("/api/jobs")
    def list_jobs(limit: int = 50) -> dict:
        return {"jobs": jobs.recent(limit)}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.public()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        if not jobs.cancel(job_id):
            raise HTTPException(409, "job already finished")
        return {"cancelled": job_id}

    # ------------------------------------------------------------- comic

    @app.get("/api/poses")
    def poses() -> dict:
        return {"poses": pose_names(),
                "keypoints": {n: [list(k) if k else None for k in kps]
                              for n, kps in POSE_PRESETS.items()}}

    @app.post("/api/pose/render")
    def pose_render(body: PoseBody) -> dict:
        """Render a skeleton to a PNG usable as a ControlNet control image."""
        if body.keypoints is not None:
            points = [None if k is None else (float(k[0]), float(k[1]))
                      for k in body.keypoints]
        elif body.preset:
            if body.preset not in POSE_PRESETS:
                raise HTTPException(400, f"unknown pose {body.preset!r}")
            points = list(POSE_PRESETS[body.preset])
        else:
            raise HTTPException(400, "pass either a preset or keypoints")
        if body.mirror:
            points = mirror(points)
        img = draw_openpose(points, body.width, body.height)
        name = _safe_page_name(f"pose-{body.preset or 'custom'}-{int(time.time())}")
        img.save(settings.outputs_dir / name)
        return {"image": name, "width": img.width, "height": img.height}

    @app.get("/api/controlnets")
    def controlnets() -> dict:
        found, seen = [], set()
        for root_dir in settings.controlnet_dirs:
            if not root_dir.is_dir():
                continue
            for entry in sorted(root_dir.iterdir()):
                if entry.name in seen or not (entry / "config.json").is_file():
                    continue
                seen.add(entry.name)
                size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                found.append({"name": entry.name, "path": str(entry),
                              "size_gb": round(size / 1024**3, 2)})
        return {"controlnets": found,
                "face_detector": settings.find_detector("face_yolov8m.pt") is not None}

    @app.get("/api/layouts")
    def layouts() -> dict:
        return {
            "layouts": layout_names(),
            "page_presets": {k: list(v) for k, v in PAGE_PRESETS.items()},
            "bubble_styles": ["speech", "thought", "shout", "whisper",
                              "narration", "sfx"],
            "fonts": available_fonts(),
        }

    @app.post("/api/page/new")
    def page_new(body: NewPageBody) -> dict:
        if body.layout not in LAYOUTS:
            raise HTTPException(400, f"unknown layout {body.layout!r}")
        order = body.reading_order if body.reading_order in ("ltr", "rtl") else "rtl"
        return build_page(body.layout, body.page_preset, order).to_dict()

    @app.post("/api/page/render")
    def page_render(body: RenderPageBody) -> dict:
        try:
            page = Page.from_dict(body.page)
        except (TypeError, ValueError, KeyError) as exc:
            raise HTTPException(400, f"malformed page: {exc}") from exc
        img = render_page(page, settings.outputs_dir)
        name = _safe_page_name(body.name)
        img.save(settings.outputs_dir / name)
        return {"image": name, "width": img.width, "height": img.height}

    @app.post("/api/page/generate")
    def page_generate(body: PageGenerateBody) -> dict:
        """Queue one generation per panel, sized to that panel's aspect ratio."""
        if body.face_mode not in FACE_MODES:
            raise HTTPException(400, f"unknown face_mode {body.face_mode!r}; choose from {list(FACE_MODES)}")
        if body.face_work_size and (body.face_work_size < 256 or body.face_work_size % 8):
            raise HTTPException(400, "face_work_size must be 0 (native) or a multiple of 8, at least 256")
        try:
            page = Page.from_dict(body.page)
        except (TypeError, ValueError, KeyError) as exc:
            raise HTTPException(400, f"malformed page: {exc}") from exc

        checkpoint = body.checkpoint or manager.checkpoint or settings.default_checkpoint
        if not checkpoint:
            available = scan_checkpoints_in(settings.checkpoint_dirs)
            if not available:
                raise HTTPException(400, "no checkpoints available")
            checkpoint = available[0].path

        cast = [c for c in (characters.get(cid) for cid in body.characters) if c]
        style = get_style(settings.root, body.style)
        p_steps, p_cfg, p_sampler = body.steps, body.cfg, body.sampler
        if style is not None:
            if style.steps and body.steps == 28:
                p_steps = style.steps
            if style.cfg is not None and body.cfg == 6.0:
                p_cfg = style.cfg
            if style.sampler and body.sampler == "dpmpp_2m_karras":
                p_sampler = style.sampler
        # Buckets depend on the checkpoint's architecture, which may not be
        # loaded yet; detect from the file rather than forcing a load here.
        from .models import detect_architecture
        arch = manager.architecture if checkpoint == manager.checkpoint \
            else detect_architecture(Path(checkpoint))
        buckets = SDXL_BUCKETS if arch != "sd15" else SD15_BUCKETS

        queued = []
        for index, panel in enumerate(page.panels):
            if body.panel_indices is not None and index not in body.panel_indices:
                continue
            if body.only_empty and panel.image:
                continue
            if not panel.prompt.strip() and not cast:
                continue
            pw, ph = panel_pixel_size(page, panel)
            width, height = nearest_bucket(pw, ph, buckets)
            prompt, char_negative, char_loras = compose_prompt(panel.prompt, cast)
            negative = ", ".join(
                x for x in (panel.negative_prompt or body.negative_prompt
                            or settings.default_negative, char_negative) if x)
            if style is not None:
                prompt, negative = style.apply(prompt, negative)
            seed = panel.seed
            if seed < 0 and len(cast) == 1 and cast[0].seed >= 0:
                # A character's locked seed keeps their face stable panel to panel.
                seed = cast[0].seed
            loras = {l["name"]: l for l in char_loras}
            for spec in body.loras:
                loras[spec.name] = spec.model_dump()
            req = GenerationRequest(
                prompt=prompt, negative_prompt=negative,
                steps=p_steps, cfg=p_cfg, width=width, height=height,
                seed=seed, sampler=p_sampler, batch_size=1,
                checkpoint=checkpoint, loras=list(loras.values()),
                mode="txt2img", clip_skip=2, hires=body.hires,
                face_enhance=body.face_enhance, face_mode=body.face_mode, face_work_size=body.face_work_size,
            )
            job = jobs.submit(req)
            queued.append({"panel": index, "job_id": job.id,
                           "width": width, "height": height, "prompt": prompt})
        if not queued:
            raise HTTPException(400, "no panels to generate (all filled or empty prompts)")
        return {"queued": queued}

    # ------------------------------------------------------ character sheets

    @app.get("/api/sheets/presets")
    def sheet_preset_list() -> dict:
        return {"presets": {name: [{"label": c.label, "tags": c.tags}
                                   for c in cells_for(name)]
                            for name in sheet_presets()}}

    @app.post("/api/character-sheet")
    def character_sheet(body: SheetBody) -> dict:
        """Queue one generation per sheet cell, all sharing the character's seed."""
        char = characters.get(body.character_id)
        if char is None:
            raise HTTPException(404, "no such character")
        try:
            cells = cells_for(body.preset)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        checkpoint = body.checkpoint or manager.checkpoint or settings.default_checkpoint
        if not checkpoint:
            available = scan_checkpoints_in(settings.checkpoint_dirs)
            if not available:
                raise HTTPException(400, "no checkpoints available")
            checkpoint = available[0].path

        style = get_style(settings.root, body.style)
        steps, cfg, sampler = body.steps, body.cfg, body.sampler
        if style is not None:
            if style.steps and body.steps == 28:
                steps = style.steps
            if style.cfg is not None and body.cfg == 6.0:
                cfg = style.cfg
            if style.sampler and body.sampler == "dpmpp_2m_karras":
                sampler = style.sampler

        from .models import detect_architecture
        arch = manager.architecture if checkpoint == manager.checkpoint \
            else detect_architecture(Path(checkpoint))
        # Portrait cells are framed square; full-body cells need a tall canvas.
        tall = (832, 1216) if arch != "sd15" else (512, 768)
        square = (1024, 1024) if arch != "sd15" else (512, 512)

        # One seed across every cell is what keeps the face recognisable.
        seed = char.seed if char.seed >= 0 else int(time.time()) % (2**31 - 1)

        queued = []
        for index, cell in enumerate(cells):
            prompt = cell_prompt(char.prompt, cell, body.extra)
            negative = cell_negative(char.negative_prompt, settings.default_negative, cell)
            if style is not None:
                prompt, negative = style.apply(prompt, negative)
            width, height = square if cell.portrait else tall
            req = GenerationRequest(
                prompt=prompt, negative_prompt=negative,
                steps=steps, cfg=cfg, width=width, height=height,
                seed=seed, sampler=sampler, batch_size=1,
                checkpoint=checkpoint, loras=list(char.loras),
                mode="txt2img", clip_skip=2,
            )
            job = jobs.submit(req)
            queued.append({"cell": index, "label": cell.label,
                           "job_id": job.id, "width": width, "height": height})
        return {"queued": queued, "seed": seed, "character": char.name}

    @app.post("/api/character-sheet/compose")
    def character_sheet_compose(body: SheetComposeBody) -> dict:
        char = characters.get(body.character_id)
        if char is None:
            raise HTTPException(404, "no such character")
        try:
            cells = cells_for(body.preset)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        images = []
        for name in body.images:
            if not name:
                images.append(None)
                continue
            path = (settings.outputs_dir / name).resolve()
            # Names come from the client; keep them inside the outputs dir.
            if not path.is_file() or settings.outputs_dir.resolve() not in path.parents:
                images.append(None)
                continue
            try:
                images.append(Image.open(path))
            except OSError:
                images.append(None)

        sheet = compose_sheet(char.name, cells, images, title=char.name)
        name = _safe_page_name(body.name or f"sheet-{char.name}-{body.preset}")
        sheet.save(settings.outputs_dir / name)
        # The sheet doubles as the character's thumbnail in the cast list.
        characters.upsert({**char.to_dict(), "reference_image": name})
        return {"image": name, "width": sheet.width, "height": sheet.height}

    # -------------------------------------------------------- characters

    @app.get("/api/characters")
    def list_characters() -> dict:
        return {"characters": [c.to_dict() for c in characters.list()]}

    @app.post("/api/characters")
    def upsert_character(body: CharacterBody) -> dict:
        data = body.model_dump()
        data["loras"] = [l.model_dump() if hasattr(l, "model_dump") else l
                         for l in body.loras]
        return characters.upsert(data).to_dict()

    @app.delete("/api/characters/{char_id}")
    def delete_character(char_id: str) -> dict:
        if not characters.delete(char_id):
            raise HTTPException(404, "no such character")
        return {"deleted": char_id}

    # ---------------------------------------------------------- training

    @app.get("/api/train/status")
    def train_status() -> dict:
        return dict(training_state)

    @app.post("/api/train")
    def train_start(body: TrainBody) -> dict:
        """Start a LoRA training run in the background.

        Training holds the UNet for a long time, so only one run is allowed and
        it refuses to start while image jobs are queued.
        """
        if training_state["state"] == "running":
            raise HTTPException(409, "a training run is already in progress")
        if jobs.depth() > 0:
            raise HTTPException(409, "finish or cancel queued image jobs first")

        dataset = Path(body.dataset_dir).expanduser()
        if not dataset.is_dir():
            raise HTTPException(400, f"dataset directory not found: {dataset}")

        checkpoint = body.checkpoint or manager.checkpoint or settings.default_checkpoint
        if not checkpoint:
            available = scan_checkpoints_in(settings.checkpoint_dirs)
            if not available:
                raise HTTPException(400, "no checkpoints available")
            checkpoint = available[0].path

        from .training import (DatasetError, LoRATrainer, TrainConfig,
                               estimate_feasibility, scan_dataset)
        try:
            samples = scan_dataset(dataset, body.trigger_word)
        except DatasetError as exc:
            raise HTTPException(400, str(exc)) from exc

        cfg = TrainConfig(
            dataset_dir=str(dataset), output_name=_safe_stem(body.output_name),
            checkpoint=checkpoint, trigger_word=body.trigger_word,
            steps=body.steps, learning_rate=body.learning_rate, rank=body.rank,
            resolution=body.resolution, batch_size=body.batch_size,
            save_every=body.save_every,
        )

        # The resident generation model would otherwise sit alongside the
        # training copy; on unified memory that is a guaranteed OOM.
        manager.unload()

        training_state.update({
            "state": "running", "phase": "caching", "step": 0,
            "total": body.steps, "loss": None, "error": None, "result": None,
            "images": len(samples), "output_name": cfg.output_name,
            "warnings": estimate_feasibility(policy, body.resolution),
            "started": time.time(),
        })

        def _run() -> None:
            trainer = LoRATrainer(cfg, policy, offline=settings.offline,
                                  single_file_configs=settings.single_file_configs)
            trainer.set_output_dir(settings.loras_dir)
            try:
                def cache_progress(step: int, total: int, _loss: float) -> None:
                    training_state.update({"phase": "caching", "step": step,
                                           "total": total})
                trainer.cache_inputs(progress=cache_progress)

                def train_progress(step: int, total: int, loss: float) -> None:
                    training_state.update({"phase": "training", "step": step,
                                           "total": total, "loss": round(loss, 4)})
                result = trainer.train(
                    progress=train_progress,
                    should_stop=lambda: training_state.get("cancel", False))
                training_state.update({"state": "done", "phase": "done",
                                       "result": result})
            except Exception as exc:
                log.exception("training failed")
                training_state.update({"state": "error", "phase": "failed",
                                       "error": f"{type(exc).__name__}: {exc}"})
            finally:
                training_state["cancel"] = False

        threading.Thread(target=_run, daemon=True, name="panelforge-train").start()
        return dict(training_state)

    @app.post("/api/train/cancel")
    def train_cancel() -> dict:
        if training_state["state"] != "running":
            raise HTTPException(409, "no training run in progress")
        training_state["cancel"] = True
        return {"cancelling": True}

    # ---------------------------------------------------------- projects

    @app.get("/api/projects")
    def list_projects() -> dict:
        out = []
        for path in sorted(projects_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            out.append({"name": path.stem,
                        "panels": len(data.get("page", {}).get("panels", [])),
                        "modified": path.stat().st_mtime})
        return {"projects": out}

    @app.post("/api/projects")
    def save_project(body: SaveProjectBody) -> dict:
        name = _safe_stem(body.name)
        path = projects_dir / f"{name}.json"
        path.write_text(json.dumps({"name": name, "page": body.page},
                                   indent=2, ensure_ascii=False))
        return {"saved": name}

    @app.get("/api/projects/{name}")
    def load_project(name: str) -> dict:
        path = (projects_dir / f"{_safe_stem(name)}.json").resolve()
        if not path.is_file() or projects_dir.resolve() not in path.parents:
            raise HTTPException(404, "no such project")
        return json.loads(path.read_text())

    @app.get("/outputs/{name}")
    def output(name: str) -> FileResponse:
        # Resolve and confine to the outputs directory: the name arrives from
        # the client and must not escape via traversal.
        path = (settings.outputs_dir / name).resolve()
        if not path.is_file() or settings.outputs_dir.resolve() not in path.parents:
            raise HTTPException(404, "not found")
        return FileResponse(path)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.exception_handler(RuntimeError)
    def runtime_error(_request, exc: RuntimeError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=500)

    app.state.characters = characters
    app.state.settings = settings
    app.state.manager = manager
    app.state.jobs = jobs
    return app


def _safe_stem(name: str) -> str:
    """Reduce a client-supplied name to a single safe path segment."""
    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "", name).strip()[:60]
    return cleaned or "untitled"


def _safe_page_name(name: str | None) -> str:
    stem = _safe_stem(name or f"page-{time.strftime('%Y%m%d-%H%M%S')}")
    return f"{stem}.png"
