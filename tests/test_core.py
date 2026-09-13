"""Tests that do not need model weights."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch

from panelforge.device import DevicePolicy, make_generator, resolve_policy
from panelforge.models import detect_architecture, detect_lora_target, read_safetensors_keys
from panelforge.pipelines import GenerationRequest, PipelineManager
from panelforge.samplers import build_scheduler, sampler_names


def write_safetensors(path: Path, keys: dict[str, list[int]]) -> Path:
    """Write a header-only safetensors file; the tensor payload is zeros."""
    header, offset = {}, 0
    for name, shape in keys.items():
        n = 1
        for dim in shape:
            n *= dim
        header[name] = {"dtype": "F16", "shape": shape,
                        "data_offsets": [offset, offset + n * 2]}
        offset += n * 2
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)
    return path


class TestArchitectureDetection:
    def test_sdxl_identified_by_second_text_encoder(self, tmp_path):
        p = write_safetensors(tmp_path / "x.safetensors", {
            "conditioner.embedders.1.model.ln_final.weight": [1280],
            "model.diffusion_model.input_blocks.0.0.weight": [320, 4, 3, 3],
        })
        assert detect_architecture(p) == "sdxl"

    def test_sd15_identified(self, tmp_path):
        p = write_safetensors(tmp_path / "x.safetensors", {
            "cond_stage_model.transformer.text_model.final_layer_norm.weight": [768],
        })
        assert detect_architecture(p) == "sd15"

    def test_unreadable_file_is_unknown_not_a_crash(self, tmp_path):
        p = tmp_path / "junk.safetensors"
        p.write_bytes(b"not a safetensors file at all")
        assert detect_architecture(p) == "unknown"
        assert read_safetensors_keys(p) == []

    def test_absurd_header_length_rejected(self, tmp_path):
        """A corrupt length field must not trigger a huge allocation."""
        p = tmp_path / "bad.safetensors"
        p.write_bytes(struct.pack("<Q", 2**60) + b"{}")
        assert read_safetensors_keys(p) == []

    def test_diffusers_directory(self, tmp_path):
        d = tmp_path / "model"
        d.mkdir()
        (d / "model_index.json").write_text(json.dumps(
            {"_class_name": "StableDiffusionXLPipeline"}))
        assert detect_architecture(d) == "sdxl"


class TestLoraDetection:
    def test_sdxl_lora_and_rank(self, tmp_path):
        p = write_safetensors(tmp_path / "l.safetensors", {
            "lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": [32, 1280],
            "lora_unet_input_blocks_1_1_proj_in.lora_up.weight": [320, 32],
        })
        arch, rank = detect_lora_target(p)
        assert arch == "sdxl"
        assert rank == 32

    def test_sd15_lora(self, tmp_path):
        p = write_safetensors(tmp_path / "l.safetensors", {
            "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": [16, 768],
        })
        arch, rank = detect_lora_target(p)
        assert arch == "sd15"
        assert rank == 16


class TestLoraDetectionRegressions:
    """Both of these were misdetected by the first implementation."""

    def test_unet_only_sdxl_lora_with_diffusers_naming(self, tmp_path):
        """SDXL-Lightning has no text encoder and uses down_blocks_* naming.

        The original heuristic looked only for LDM-style `input_blocks_*` and a
        second text encoder, so it fell through and reported sd15 — which meant
        the UI hid the LoRA from every SDXL checkpoint.
        """
        p = write_safetensors(tmp_path / "lightning.safetensors", {
            "lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight": [64, 320, 3, 3],
            "lora_unet_down_blocks_1_attentions_0_proj_in.lora_down.weight": [64, 640],
            "lora_unet_mid_block_attentions_0_proj_in.lora_down.weight": [64, 1280],
        })
        arch, rank = detect_lora_target(p)
        assert arch == "sdxl"
        assert rank == 64

    def test_conv_lora_rank_ignores_the_kernel_dims(self, tmp_path):
        """[rank, in_ch, kh, kw]: min(shape) would report the 3x3 kernel as rank 3."""
        p = write_safetensors(tmp_path / "conv.safetensors", {
            "lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight": [64, 320, 3, 3],
        })
        assert detect_lora_target(p)[1] == 64

    def test_diffusers_format_lora_is_recognised(self, tmp_path):
        """Our own trainer writes dot-separated diffusers keys, not kohya ones.

        `unet.down_blocks.1....to_k.lora.down.weight` matched none of the
        underscore-based rules, so LoRAs trained by this tool came back as
        architecture "unknown" with no rank and were hidden from the UI.
        """
        p = write_safetensors(tmp_path / "trained.safetensors", {
            "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora.down.weight": [8, 640],
            "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora.up.weight": [640, 8],
            "unet.mid_block.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight": [8, 1280],
        })
        arch, rank = detect_lora_target(p)
        assert arch == "sdxl"
        assert rank == 8

    def test_diffusers_format_sd15_lora(self, tmp_path):
        p = write_safetensors(tmp_path / "trained15.safetensors", {
            "unet.down_blocks.3.resnets.0.conv1.lora.down.weight": [4, 1280, 3, 3],
        })
        assert detect_lora_target(p)[0] == "sd15"

    def test_fourth_down_block_means_sd15(self, tmp_path):
        """SD 1.5's UNet has down_blocks_3; SDXL stops at 2."""
        p = write_safetensors(tmp_path / "l.safetensors", {
            "lora_unet_down_blocks_3_resnets_0_conv1.lora_down.weight": [16, 1280, 3, 3],
        })
        assert detect_lora_target(p)[0] == "sd15"

    def test_attention_in_the_first_block_means_sd15(self, tmp_path):
        p = write_safetensors(tmp_path / "l.safetensors", {
            "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight": [8, 320],
        })
        assert detect_lora_target(p)[0] == "sd15"


class TestDevicePolicy:
    def test_detects_something_usable(self):
        policy = resolve_policy()
        assert policy.device in ("mps", "cuda", "cpu")
        assert policy.summary()

    def test_cpu_never_uses_fp16(self):
        policy = resolve_policy(prefer_device="cpu")
        assert policy.dtype is torch.float32

    def test_low_memory_forces_the_frugal_path(self):
        policy = resolve_policy(low_memory=True)
        if policy.device != "cpu":
            assert policy.vae_tiling

    def test_vae_guarded_against_fp16_overflow(self):
        policy = resolve_policy()
        if policy.dtype is torch.float16:
            assert policy.vae_dtype is torch.float32, "SDXL VAE would decode black"

    def test_seed_is_reproducible_and_not_on_mps(self):
        """Latents must come from a CPU generator so seeds port across backends."""
        a = make_generator("mps", 42)
        b = make_generator("mps", 42)
        assert a.device.type == "cpu"
        assert torch.randn(4, generator=a).equal(torch.randn(4, generator=b))


class TestSamplers:
    def test_every_preset_builds(self):
        from diffusers import DDIMScheduler
        base = DDIMScheduler()
        for name in sampler_names():
            assert build_scheduler(name, base.config) is not None

    def test_unknown_sampler_is_rejected_clearly(self):
        from diffusers import DDIMScheduler
        with pytest.raises(ValueError, match="unknown sampler"):
            build_scheduler("not_a_sampler", DDIMScheduler().config)

    def test_prediction_type_survives_a_sampler_swap(self):
        """v-prediction checkpoints break badly if this is dropped."""
        from diffusers import DDIMScheduler
        base = DDIMScheduler(prediction_type="v_prediction")
        for name in ("euler_a", "dpmpp_2m_karras", "unipc"):
            assert build_scheduler(name, base.config).config.prediction_type == "v_prediction"


class TestSeeding:
    def _manager(self):
        return PipelineManager(DevicePolicy(
            device="cpu", dtype=torch.float32, vae_dtype=torch.float32,
            total_memory_gb=16,
        ))

    def test_batch_from_one_seed_is_not_n_identical_images(self):
        seeds = self._manager().resolve_seeds(
            GenerationRequest(prompt="x", seed=100, batch_size=4))
        assert seeds == [100, 101, 102, 103]

    def test_negative_seed_randomises_each_image(self):
        seeds = self._manager().resolve_seeds(
            GenerationRequest(prompt="x", seed=-1, batch_size=3))
        assert len(set(seeds)) == 3


class TestServerGuards:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        from panelforge.config import Settings
        from panelforge.server import create_app

        settings = Settings(root=tmp_path, models_dir=tmp_path / "models",
                            outputs_dir=tmp_path / "out", device="cpu")
        settings.ensure_dirs()
        return TestClient(create_app(settings))

    def test_traversal_out_of_outputs_is_refused(self, client):
        assert client.get("/outputs/../../pyproject.toml").status_code == 404
        assert client.get("/outputs/%2e%2e%2fconfig.toml").status_code == 404

    def test_missing_checkpoint_gives_an_actionable_message(self, client):
        r = client.post("/api/generate", json={"prompt": "hi"})
        assert r.status_code == 400
        assert "checkpoints" in r.json()["detail"]

    def test_img2img_without_an_image_is_rejected(self, client):
        r = client.post("/api/generate", json={"prompt": "hi", "mode": "img2img"})
        assert r.status_code == 400
        assert "init image" in r.json()["detail"]

    def test_inpaint_without_a_mask_is_rejected(self, client):
        r = client.post("/api/generate", json={
            "prompt": "hi", "mode": "inpaint", "init_image": "data:image/png;base64,iVBOR"})
        assert r.status_code == 400
        assert "mask" in r.json()["detail"]

    def test_malformed_image_payload_is_a_400_not_a_500(self, client):
        r = client.post("/api/generate", json={
            "prompt": "hi", "mode": "img2img", "init_image": "data:image/png;base64,!!!!"})
        assert r.status_code == 400

    def test_unknown_face_mode_is_rejected(self, client):
        r = client.post("/api/generate", json={"prompt": "hi", "face_enhance": True,
                                               "face_mode": "sideways"})
        assert r.status_code == 400
        assert "face_mode" in r.json()["detail"]

    def test_ui_assets_are_revalidated_not_served_stale(self, client):
        """A cached studio.js once hid a new button's handler after an upgrade."""
        for path in ("/", "/static/studio.js", "/static/app.js"):
            r = client.get(path)
            assert r.status_code == 200, path
            assert r.headers.get("cache-control") == "no-cache", path

    def test_api_responses_are_left_to_their_own_cache_policy(self, client):
        assert "no-cache" not in client.get("/api/status").headers.get("cache-control", "")

    def test_face_work_size_must_be_native_or_a_valid_multiple_of_8(self, client):
        for bad in (100, 300):
            r = client.post("/api/generate", json={"prompt": "hi", "face_enhance": True,
                                                   "face_work_size": bad})
            assert r.status_code == 400, bad
            assert "face_work_size" in r.json()["detail"]

    def test_status_reports_the_hardware_policy(self, client):
        body = client.get("/api/status").json()
        assert body["device"] == "cpu"
        assert body["summary"]
