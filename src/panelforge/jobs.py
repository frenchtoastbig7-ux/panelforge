"""Single-worker job queue.

Only one checkpoint is resident at a time, so generations are serialized rather
than run concurrently: two parallel jobs on one GPU are slower than the same two
in sequence and risk exhausting memory mid-denoise.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image, PngImagePlugin

from .pipelines import GenerationRequest, PipelineManager

log = logging.getLogger(__name__)

JobState = Literal["queued", "running", "done", "error", "cancelled"]


class JobCancelled(Exception):
    """Raised inside the worker to unwind a cancelled generation."""


@dataclass
class Job:
    id: str
    request: GenerationRequest
    state: JobState = "queued"
    step: int = 0
    total_steps: int = 0
    error: str | None = None
    images: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    def public(self) -> dict:
        progress = self.step / self.total_steps if self.total_steps else 0.0
        if self.state == "done":
            progress = 1.0
        return {
            "id": self.id,
            "state": self.state,
            "step": self.step,
            "total_steps": self.total_steps,
            "progress": round(progress, 3),
            "error": self.error,
            "images": self.images,
            "metadata": self.metadata,
            "elapsed_s": round((self.finished or time.time()) - (self.started or self.created), 2),
            "prompt": self.request.prompt,
        }


class JobQueue:
    def __init__(self, manager: PipelineManager, outputs_dir: Path, keep: int = 200):
        self.manager = manager
        self.outputs_dir = outputs_dir
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._q: queue.Queue[str] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._keep = keep
        self._lock = threading.Lock()
        self._cancel: set[str] = set()
        self._current: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="panelforge-worker")
        self._thread.start()

    # ------------------------------------------------------------------- api

    def submit(self, request: GenerationRequest) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], request=request)
        job.total_steps = request.steps * max(1, request.batch_size)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._trim()
        self._q.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            ids = self._order[-limit:][::-1]
            return [self._jobs[i].public() for i in ids if i in self._jobs]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state in ("done", "error", "cancelled"):
                return False
            self._cancel.add(job_id)
            if job.state == "queued":
                job.state = "cancelled"
                job.finished = time.time()
        return True

    def depth(self) -> int:
        return self._q.qsize()

    def _trim(self) -> None:
        """Bound in-memory history; files on disk are left alone."""
        while len(self._order) > self._keep:
            old = self._order.pop(0)
            self._jobs.pop(old, None)
            self._cancel.discard(old)

    # ---------------------------------------------------------------- worker

    def _run(self) -> None:
        while True:
            job_id = self._q.get()
            job = self.get(job_id)
            if job is None or job.state == "cancelled":
                continue
            self._current = job_id
            try:
                self._execute(job)
            except JobCancelled:
                job.state = "cancelled"
            except Exception as exc:
                log.exception("job %s failed", job.id)
                job.state = "error"
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.finished = time.time()
                self._cancel.discard(job.id)
                self._current = None

    def _execute(self, job: Job) -> None:
        job.state = "running"
        job.started = time.time()
        req = job.request

        checkpoint = req.checkpoint or self.manager.checkpoint
        if not checkpoint:
            raise RuntimeError("no checkpoint selected")
        if checkpoint != self.manager.checkpoint:
            self.manager.load(checkpoint)

        batch = max(1, req.batch_size)
        completed_steps = 0

        def on_step(step: int, total: int) -> None:
            if job.id in self._cancel:
                raise JobCancelled()
            job.total_steps = total * batch
            job.step = completed_steps + step

        # Run images one at a time so progress and cancellation stay responsive
        # and peak memory does not scale with batch size.
        images, seeds, meta = [], [], {}
        per = GenerationRequest(**{**req.__dict__, "batch_size": 1})
        seed_list = self.manager.resolve_seeds(req)
        for i, seed in enumerate(seed_list):
            if job.id in self._cancel:
                raise JobCancelled()
            per.seed = seed
            result = self.manager.generate(per, progress=on_step)
            images.extend(result.images)
            seeds.extend(result.seeds)
            meta = result.metadata
            completed_steps = job.total_steps * (i + 1) // batch
            job.step = completed_steps

        job.images = [self._save(img, seed, meta) for img, seed in zip(images, seeds)]
        job.metadata = meta | {"seeds": seeds}
        job.step = job.total_steps
        job.state = "done"

    def _save(self, image: Image.Image, seed: int, meta: dict) -> str:
        """Write a PNG with generation parameters embedded, and return its name."""
        ts = time.strftime("%Y%m%d-%H%M%S")
        name = f"{ts}-{seed}-{uuid.uuid4().hex[:6]}.png"
        info = PngImagePlugin.PngInfo()
        # A1111-compatible parameter block, so these PNGs drop into other tools.
        params = (
            f"{meta.get('prompt','')}\n"
            f"Negative prompt: {meta.get('negative_prompt','')}\n"
            f"Steps: {meta.get('steps')}, Sampler: {meta.get('sampler')}, "
            f"CFG scale: {meta.get('cfg')}, Seed: {seed}, "
            f"Size: {meta.get('width')}x{meta.get('height')}, "
            f"Model: {meta.get('checkpoint','')}"
        )
        info.add_text("parameters", params)
        for key, value in meta.items():
            if isinstance(value, (str, int, float)):
                info.add_text(f"panelforge.{key}", str(value))
        info.add_text("panelforge.seed", str(seed))
        image.save(self.outputs_dir / name, pnginfo=info)
        return name
