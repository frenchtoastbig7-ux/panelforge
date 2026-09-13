"""App-bundle layout: read-only Resources plus a user data folder."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch

from panelforge.config import load_settings
from panelforge.desktop import DesktopAPI
from panelforge.device import DevicePolicy
from panelforge.models import scan_checkpoints_in, scan_loras_in
from panelforge.pipelines import PipelineManager
from panelforge.training import LoRATrainer, TrainConfig


def _safetensors(path: Path, keys: dict[str, list[int]]) -> Path:
    header, offset = {}, 0
    for name, shape in keys.items():
        n = 1
        for dim in shape:
            n *= dim
        header[name] = {"dtype": "F16", "shape": shape, "data_offsets": [offset, offset + n * 2]}
        offset += n * 2
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)
    return path


SDXL_KEYS = {"conditioner.embedders.1.model.ln_final.weight": [4]}
TINY_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC"
SDXL_LORA_KEYS = {"lora_unet_down_blocks_1_attentions_0_proj_in.lora_down.weight": [8, 4]}


def _tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    res = tmp_path / "PanelForge.app" / "Contents" / "Resources"
    for sub in ("checkpoints", "loras", "controlnet/openpose-sdxl", "detectors"):
        (res / "models" / sub).mkdir(parents=True)
    (res / "models/controlnet/openpose-sdxl/config.json").write_text("{}")
    (res / "models/detectors/face_yolov8m.pt").write_bytes(b"weights")
    (res / "configs/sdxl").mkdir(parents=True)
    (res / "configs/sdxl/model_index.json").write_text("{}")
    _safetensors(res / "models/checkpoints/animagine.safetensors", SDXL_KEYS)
    _safetensors(res / "models/loras/lightning.safetensors", SDXL_LORA_KEYS)
    home = tmp_path / "Application Support" / "panelforge"
    monkeypatch.setenv("PANELFORGE_RESOURCES", str(res))
    monkeypatch.setenv("PANELFORGE_HOME", str(home))
    # load_settings exports these when offline; record them so they are undone.
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    return res, home


class TestBundleSettings:
    def test_user_data_goes_to_the_data_folder_never_the_bundle(self, bundle):
        res, home = bundle
        before = _tree(res)
        s = load_settings()
        assert s.root == home
        assert home in s.outputs_dir.parents and home in s.models_dir.parents
        assert s.loras_dir.is_dir() and s.outputs_dir.is_dir()
        assert _tree(res) == before, "the signed bundle must never be written to"

    def test_bundle_mode_is_offline_with_local_configs(self, bundle):
        res, _ = bundle
        s = load_settings()
        assert s.offline and s.bundled
        assert s.single_file_configs == {"sdxl": str(res / "configs" / "sdxl")}

    def test_shipped_models_are_discovered(self, bundle):
        res, _ = bundle
        s = load_settings()
        ckpts = scan_checkpoints_in(s.checkpoint_dirs)
        assert [(m.name, m.architecture) for m in ckpts] == [("animagine", "sdxl")]
        assert [m.name for m in scan_loras_in(s.lora_dirs)] == ["lightning"]
        assert s.find_controlnet("openpose-sdxl") == res / "models/controlnet/openpose-sdxl"
        assert s.find_detector("face_yolov8m.pt") == res / "models/detectors/face_yolov8m.pt"

    def test_a_user_file_overrides_the_shipped_one_by_name(self, bundle):
        s = load_settings()
        user = _safetensors(s.checkpoints_dir / "animagine.safetensors", SDXL_KEYS)
        ckpts = scan_checkpoints_in(s.checkpoint_dirs)
        assert [m.path for m in ckpts] == [str(user)]

    @pytest.mark.parametrize("name", ["", ".", "..", "../openpose-sdxl", "a/b", "..\\x"])
    def test_controlnet_names_cannot_escape(self, bundle, name):
        assert load_settings().find_controlnet(name) is None

    def test_source_checkout_layout_is_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PANELFORGE_RESOURCES", "")
        monkeypatch.delenv("PANELFORGE_RESOURCES")
        monkeypatch.setenv("PANELFORGE_HOME", str(tmp_path))
        s = load_settings()
        assert s.root == tmp_path and not s.bundled
        assert s.single_file_configs == {} and s.offline is False


class TestLocalConfigs:
    def _policy(self):
        return DevicePolicy(device="cpu", dtype=torch.float32, vae_dtype=torch.float32, total_memory_gb=16)

    def test_pipeline_manager_prefers_the_local_config_folder(self):
        m = PipelineManager(self._policy(), single_file_configs={"sdxl": "/bundle/configs/sdxl"})
        assert m._single_file_config("sdxl") == "/bundle/configs/sdxl"
        assert m._single_file_config("sd15") == "runwayml/stable-diffusion-v1-5"

    def test_trainer_carries_the_local_config_folder(self):
        cfg = TrainConfig(dataset_dir="/d", output_name="x", checkpoint="/c.safetensors")
        t = LoRATrainer(cfg, self._policy(), single_file_configs={"sdxl": "/cfg"})
        assert t.single_file_configs == {"sdxl": "/cfg"}


class TestBundleServer:
    def test_api_lists_shipped_models_and_rejects_bad_controlnet_names(self, bundle):
        from fastapi.testclient import TestClient

        from panelforge.server import create_app

        s = load_settings()
        s.device = "cpu"
        client = TestClient(create_app(s))
        models = client.get("/api/models").json()
        assert [m["name"] for m in models["checkpoints"]] == ["animagine"]
        nets = client.get("/api/controlnets").json()
        assert [n["name"] for n in nets["controlnets"]] == ["openpose-sdxl"]
        assert nets["face_detector"] is True
        # A valid image, so the request reaches the ControlNet name check.
        r = client.post("/api/generate", json={"prompt": "x", "controlnet": "../../etc",
                                               "control_image": TINY_PNG})
        assert r.status_code == 400 and "not installed" in r.json()["detail"]


class FakeWindow:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def create_file_dialog(self, dialog_type, **kwargs):
        self.calls.append((dialog_type, kwargs))
        return self.answer


class TestDesktopAPI:
    def test_save_copies_the_output_to_the_chosen_path(self, tmp_path):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        (outputs / "page.png").write_bytes(b"PNGDATA")
        dest = tmp_path / "Desktop" / "chapter-1"
        dest.parent.mkdir()
        api = DesktopAPI(outputs)
        api._attach(FakeWindow(str(dest)))
        result = api.save_output("page.png", "chapter-1.png")
        assert result == {"saved": str(dest.with_suffix(".png"))}
        assert dest.with_suffix(".png").read_bytes() == b"PNGDATA"

    @pytest.mark.parametrize("name", ["../secret.png", "/etc/hosts", ".hidden.png", "", None, "sub/x.png"])
    def test_save_refuses_names_outside_outputs(self, tmp_path, name):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        (tmp_path / "secret.png").write_bytes(b"s")
        window = FakeWindow(str(tmp_path / "out.png"))
        api = DesktopAPI(outputs)
        api._attach(window)
        assert api.save_output(name) == {"error": "no such output"}
        assert window.calls == [], "no dialog for a refused name"

    def test_cancelled_save_writes_nothing(self, tmp_path):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        (outputs / "a.png").write_bytes(b"x")
        api = DesktopAPI(outputs)
        api._attach(FakeWindow(None))
        assert api.save_output("a.png") == {"cancelled": True}

    def test_pick_folder_returns_the_chosen_path(self, tmp_path):
        api = DesktopAPI(tmp_path)
        api._attach(FakeWindow(("/Users/me/datasets/aoi",)))
        assert api.pick_folder() == "/Users/me/datasets/aoi"
