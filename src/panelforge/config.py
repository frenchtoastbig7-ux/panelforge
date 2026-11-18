"""Runtime settings: where models live, what the defaults are.

Two layouts share this code. From a source checkout everything lives under the
project root. Inside the macOS app, the launcher sets PANELFORGE_RESOURCES to the
bundle's Resources folder, which is read-only once signed: shipped models and
configs are read from there, while everything the user creates (outputs,
characters, projects, trained LoRAs, config.toml) lives in
~/Library/Application Support/panelforge.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, "
    "extra digit, fewer digits, cropped, worst quality, low quality, "
    "jpeg artifacts, signature, watermark, username, blurry"
)

APP_SUPPORT_NAME = "panelforge"


@dataclass
class Settings:
    root: Path
    models_dir: Path
    outputs_dir: Path
    device: str | None = None
    dtype: str | None = None
    low_memory: bool | None = None
    # Generation defaults, overridable per request.
    default_steps: int = 28
    default_cfg: float = 6.0
    default_width: int = 1024
    default_height: int = 1024
    default_sampler: str = "dpmpp_2m_karras"
    default_negative: str = DEFAULT_NEGATIVE
    default_checkpoint: str | None = None
    host: str = "127.0.0.1"
    port: int = 8188
    # Never contact the network once models are on disk.
    offline: bool = False
    extra: dict = field(default_factory=dict)
    # Read-only models shipped inside the app bundle, searched after models_dir.
    bundled_models_dir: Path | None = None
    # Architecture -> local diffusers config folder for single-file checkpoints.
    # Empty means the Hugging Face reference repo, which needs the network once.
    single_file_configs: dict = field(default_factory=dict)

    @property
    def bundled(self) -> bool:
        return self.bundled_models_dir is not None

    # User-writable locations.
    @property
    def checkpoints_dir(self) -> Path:
        return self.models_dir / "checkpoints"

    @property
    def loras_dir(self) -> Path:
        return self.models_dir / "loras"

    @property
    def vae_dir(self) -> Path:
        return self.models_dir / "vae"

    @property
    def embeddings_dir(self) -> Path:
        return self.models_dir / "embeddings"

    @property
    def controlnet_dir(self) -> Path:
        return self.models_dir / "controlnet"

    @property
    def detectors_dir(self) -> Path:
        return self.models_dir / "detectors"

    # Search paths: the user's folder first, so a user file overrides a shipped
    # one with the same name, then the bundle.
    def _search(self, sub: str) -> list[Path]:
        dirs = [self.models_dir / sub]
        if self.bundled_models_dir is not None:
            dirs.append(self.bundled_models_dir / sub)
        return dirs

    @property
    def checkpoint_dirs(self) -> list[Path]:
        return self._search("checkpoints")

    @property
    def lora_dirs(self) -> list[Path]:
        return self._search("loras")

    @property
    def vae_dirs(self) -> list[Path]:
        return self._search("vae")

    @property
    def controlnet_dirs(self) -> list[Path]:
        return self._search("controlnet")

    @property
    def detector_dirs(self) -> list[Path]:
        return self._search("detectors")

    def find_detector(self, filename: str) -> Path | None:
        for directory in self.detector_dirs:
            if (directory / filename).is_file():
                return directory / filename
        return None

    def find_controlnet(self, name: str | None) -> Path | None:
        """Resolve a ControlNet folder by name. The name arrives from a client."""
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            return None
        for directory in self.controlnet_dirs:
            if (directory / name / "config.json").is_file():
                return directory / name
        return None

    def ensure_dirs(self) -> None:
        # Only user-writable locations; the bundle is never written to.
        for d in (self.checkpoints_dir, self.loras_dir, self.vae_dir,
                  self.embeddings_dir, self.controlnet_dir, self.detectors_dir,
                  self.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)


def resources_dir() -> Path | None:
    """The app bundle's Resources folder, when running as the macOS app."""
    env = os.environ.get("PANELFORGE_RESOURCES")
    return Path(env).expanduser().resolve() if env else None


def _find_root(start: Path | None = None) -> Path:
    env = os.environ.get("PANELFORGE_HOME")
    if env:
        return Path(env).expanduser().resolve()
    if resources_dir() is not None:
        return Path.home() / "Library" / "Application Support" / APP_SUPPORT_NAME
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_settings(config_path: Path | None = None) -> Settings:
    root = _find_root()
    resources = resources_dir()
    data: dict = {}
    path = config_path or root / "config.toml"
    if path.exists():
        with path.open("rb") as fh:
            data = tomllib.load(fh)

    paths = data.get("paths", {})
    gen = data.get("generation", {})
    server = data.get("server", {})
    hardware = data.get("hardware", {})

    def _p(value: str | None, default: Path) -> Path:
        return Path(value).expanduser().resolve() if value else default

    bundled_models: Path | None = None
    configs: dict = {}
    if resources is not None:
        if (resources / "models").is_dir():
            bundled_models = resources / "models"
        for arch in ("sdxl", "sd15"):
            folder = resources / "configs" / arch
            if (folder / "model_index.json").is_file():
                configs[arch] = str(folder)

    settings = Settings(
        root=root,
        models_dir=_p(paths.get("models_dir"), root / "models"),
        outputs_dir=_p(paths.get("outputs_dir"), root / "outputs"),
        device=hardware.get("device"),
        dtype=hardware.get("dtype"),
        low_memory=hardware.get("low_memory"),
        default_steps=int(gen.get("steps", 28)),
        default_cfg=float(gen.get("cfg", 6.0)),
        default_width=int(gen.get("width", 1024)),
        default_height=int(gen.get("height", 1024)),
        default_sampler=gen.get("sampler", "dpmpp_2m_karras"),
        default_negative=gen.get("negative_prompt", DEFAULT_NEGATIVE),
        default_checkpoint=gen.get("checkpoint"),
        host=server.get("host", "127.0.0.1"),
        port=int(server.get("port", 8188)),
        # The packaged app ships everything it needs and never goes online.
        offline=bool(data.get("offline", False)) or resources is not None,
        extra=data,
        bundled_models_dir=bundled_models,
        single_file_configs=configs,
    )
    if settings.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    settings.ensure_dirs()
    return settings
