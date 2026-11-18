"""Command line interface."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from .config import load_settings

app = typer.Typer(add_completion=False, help="Local anime image generation on diffusers.")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def info(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Show the resolved hardware policy and what is installed."""
    _setup_logging(verbose)
    from .device import resolve_policy
    from .models import scan_checkpoints_in, scan_loras_in

    settings = load_settings()
    policy = resolve_policy(settings.device, settings.dtype, settings.low_memory)

    typer.echo(f"device policy : {policy.summary()}")
    for note in policy.notes:
        typer.echo(f"  note        : {note}")
    typer.echo(f"models dirs   : {settings.models_dir}"
               + (f" (+ bundled {settings.bundled_models_dir})" if settings.bundled else ""))
    typer.echo(f"outputs dir   : {settings.outputs_dir}")

    ckpts = scan_checkpoints_in(settings.checkpoint_dirs)
    loras = scan_loras_in(settings.lora_dirs)
    typer.echo(f"\ncheckpoints ({len(ckpts)}):")
    for m in ckpts or []:
        typer.echo(f"  {m.name:<40} {m.architecture:<7} {m.size_bytes / 1024**3:.2f}GB")
    if not ckpts:
        typer.echo(f"  none — drop .safetensors files in {settings.checkpoints_dir}")
    typer.echo(f"\nloras ({len(loras)}):")
    for m in loras or []:
        rank = f"rank {m.lora_rank}" if m.lora_rank else "rank ?"
        typer.echo(f"  {m.name:<40} {m.architecture:<7} {rank}")
    if not loras:
        typer.echo(f"  none — drop .safetensors files in {settings.loras_dir}")


@app.command()
def serve(
    host: str = typer.Option(None, help="Bind address (default from config: 127.0.0.1)."),
    port: int = typer.Option(None, help="Port (default from config: 8188)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the local web UI."""
    _setup_logging(verbose)
    import uvicorn

    from .server import create_app

    settings = load_settings()
    host = host or settings.host
    port = port or settings.port
    if host not in ("127.0.0.1", "localhost", "::1"):
        typer.secho(
            f"binding to {host}: the UI has no authentication, so anyone who can "
            "reach this port can drive the model and read your outputs.",
            fg=typer.colors.YELLOW,
        )
    typer.echo(f"panelforge → http://{host}:{port}")
    uvicorn.run(create_app(settings), host=host, port=port, log_level="warning")


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Positive prompt."),
    negative: str = typer.Option("", "--negative", "-n"),
    checkpoint: str = typer.Option(None, "--checkpoint", "-c", help="Path or name."),
    steps: int = typer.Option(None, "--steps"),
    cfg: float = typer.Option(None, "--cfg"),
    width: int = typer.Option(None, "--width"),
    height: int = typer.Option(None, "--height"),
    seed: int = typer.Option(-1, "--seed"),
    sampler: str = typer.Option(None, "--sampler"),
    count: int = typer.Option(1, "--count", help="How many images."),
    lora: list[str] = typer.Option(None, "--lora", help="name[:weight], repeatable."),
    out: Path = typer.Option(None, "--out", help="Output directory."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate images without starting the server."""
    _setup_logging(verbose)
    from .device import resolve_policy
    from .jobs import JobQueue
    from .models import scan_checkpoints_in, scan_loras_in
    from .pipelines import GenerationRequest, PipelineManager

    settings = load_settings()
    ckpts = scan_checkpoints_in(settings.checkpoint_dirs)
    resolved = _resolve_checkpoint(checkpoint or settings.default_checkpoint, ckpts)
    if resolved is None:
        typer.secho(
            f"no checkpoint available. Put a .safetensors model in "
            f"{settings.checkpoints_dir}, or pass --checkpoint.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    policy = resolve_policy(settings.device, settings.dtype, settings.low_memory)
    typer.echo(f"hardware: {policy.summary()}")
    manager = PipelineManager(policy, offline=settings.offline,
                              single_file_configs=settings.single_file_configs)
    manager.load(resolved)

    bucket = manager.buckets()[0]
    req = GenerationRequest(
        prompt=prompt,
        negative_prompt=negative or settings.default_negative,
        steps=steps or settings.default_steps,
        cfg=cfg if cfg is not None else settings.default_cfg,
        width=width or bucket[0],
        height=height or bucket[1],
        seed=seed,
        sampler=sampler or settings.default_sampler,
        batch_size=count,
        checkpoint=resolved,
        loras=_parse_loras(lora or [], scan_loras_in(settings.lora_dirs)),
        clip_skip=2,
    )

    queue = JobQueue(manager, out or settings.outputs_dir)
    job = queue.submit(req)
    _wait(queue, job.id)


def _wait(queue, job_id: str) -> None:
    import time

    last = -1
    while True:
        job = queue.get(job_id)
        if job is None:
            typer.secho("job vanished", fg=typer.colors.RED)
            raise typer.Exit(1)
        if job.state == "running" and job.step != last:
            last = job.step
            pct = int(100 * job.step / max(1, job.total_steps))
            sys.stderr.write(f"\r  {pct:3d}%  step {job.step}/{job.total_steps}")
            sys.stderr.flush()
        if job.state in ("done", "error", "cancelled"):
            sys.stderr.write("\r" + " " * 40 + "\r")
            if job.state == "error":
                typer.secho(f"failed: {job.error}", fg=typer.colors.RED)
                raise typer.Exit(1)
            for name in job.images:
                typer.echo(str(queue.outputs_dir / name))
            typer.secho(
                f"{len(job.images)} image(s) in {job.metadata.get('elapsed_s')}s "
                f"(seeds: {', '.join(map(str, job.metadata.get('seeds', [])))})",
                fg=typer.colors.GREEN,
            )
            return
        time.sleep(0.2)


@app.command()
def train(
    dataset: Path = typer.Argument(..., help="Directory of training images (+ optional .txt captions)."),
    name: str = typer.Option(..., "--name", "-o", help="Output LoRA name."),
    trigger: str = typer.Option("", "--trigger", "-t", help="Trigger word to bind the concept to."),
    checkpoint: str = typer.Option(None, "--checkpoint", "-c"),
    steps: int = typer.Option(1000, "--steps"),
    rank: int = typer.Option(16, "--rank", help="LoRA rank; 8-32 is typical."),
    lr: float = typer.Option(1e-4, "--lr"),
    resolution: int = typer.Option(768, "--resolution"),
    batch_size: int = typer.Option(1, "--batch-size"),
    save_every: int = typer.Option(0, "--save-every", help="Checkpoint every N steps."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the feasibility prompt."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Train a character or style LoRA on your own images."""
    _setup_logging(verbose)
    from .device import resolve_policy
    from .models import scan_checkpoints_in
    from .training import LoRATrainer, TrainConfig, DatasetError, estimate_feasibility

    settings = load_settings()
    resolved = _resolve_checkpoint(checkpoint or settings.default_checkpoint,
                                   scan_checkpoints_in(settings.checkpoint_dirs))
    if resolved is None:
        typer.secho("no checkpoint available; pass --checkpoint", fg=typer.colors.RED)
        raise typer.Exit(1)

    policy = resolve_policy(settings.device, settings.dtype, settings.low_memory)
    typer.echo(f"hardware : {policy.summary()}")
    typer.echo(f"base     : {Path(resolved).name}")
    typer.echo(f"dataset  : {dataset}")

    warnings = estimate_feasibility(policy, resolution)
    for note in warnings:
        typer.secho(f"  ! {note}", fg=typer.colors.YELLOW)
    if warnings and not yes and not typer.confirm("Continue?", default=True):
        raise typer.Exit(0)

    cfg = TrainConfig(
        dataset_dir=str(dataset), output_name=name, checkpoint=resolved,
        trigger_word=trigger, steps=steps, learning_rate=lr, rank=rank,
        resolution=resolution, batch_size=batch_size, save_every=save_every,
    )
    trainer = LoRATrainer(cfg, policy, offline=settings.offline,
                          single_file_configs=settings.single_file_configs)
    trainer.set_output_dir(settings.loras_dir)

    try:
        typer.echo("caching latents and text embeddings...")
        trainer.cache_inputs(progress=_bar("cache"))
        sys.stderr.write("\r" + " " * 60 + "\r")
        typer.echo(f"training {steps} steps...")
        result = trainer.train(progress=_bar("train"))
    except DatasetError as exc:
        sys.stderr.write("\r" + " " * 60 + "\r")
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1)
    except (RuntimeError, torch_oom()) as exc:
        sys.stderr.write("\r" + " " * 60 + "\r")
        typer.secho(f"training failed: {exc}", fg=typer.colors.RED)
        typer.secho("Try --resolution 512 or a smaller --rank.", fg=typer.colors.YELLOW)
        raise typer.Exit(1)

    sys.stderr.write("\r" + " " * 60 + "\r")
    typer.secho(
        f"{result['output']}  ({result['size_mb']} MB, {result['steps_completed']} steps, "
        f"loss {result['final_loss']}, {result['elapsed_s']}s)", fg=typer.colors.GREEN)
    typer.echo(f"Use it with: panelforge generate \"{trigger or name}, ...\" --lora {name}:1.0")


def torch_oom():
    """The OOM exception class, which differs per backend."""
    import torch
    return getattr(torch, "OutOfMemoryError", MemoryError)


def _bar(label: str):
    last = [-1.0]

    def _fn(step: int, total: int, loss: float) -> None:
        pct = int(100 * step / max(1, total))
        if pct == last[0]:
            return
        last[0] = pct
        extra = f"  loss {loss:.4f}" if loss else ""
        sys.stderr.write(f"\r  {label} {pct:3d}%  {step}/{total}{extra}   ")
        sys.stderr.flush()

    return _fn


def _resolve_checkpoint(wanted: str | None, available: list) -> str | None:
    if wanted:
        path = Path(wanted)
        if path.exists():
            return str(path)
        for m in available:
            if wanted.lower() in m.name.lower():
                return m.path
        return None
    return available[0].path if available else None


def _parse_loras(specs: list[str], available: list) -> list[dict]:
    out = []
    for spec in specs:
        name, _, weight_s = spec.rpartition(":")
        if not name:  # no colon given
            name, weight_s = spec, "1.0"
        try:
            weight = float(weight_s)
        except ValueError:
            name, weight = spec, 1.0
        path = Path(name)
        if not path.exists():
            match = next((m for m in available if name.lower() in m.name.lower()), None)
            if match is None:
                typer.secho(f"no LoRA matching {name!r}, skipping", fg=typer.colors.YELLOW)
                continue
            name = match.path
        out.append({"name": str(name), "weight": weight})
    return out


if __name__ == "__main__":
    app()
