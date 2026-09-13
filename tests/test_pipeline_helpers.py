"""Pipeline helpers that don't need real weights."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from panelforge.device import DevicePolicy
from panelforge.pipelines import PipelineManager, _face_prompt_from


class FakeVAE(torch.nn.Module):
    """Records the dtypes the decode actually ran in."""

    def __init__(self, dtype=torch.float16, mean=None, std=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.config = SimpleNamespace(scaling_factor=0.5, latents_mean=mean, latents_std=std)
        self.seen: dict = {}

    @property
    def device(self):
        return self.weight.device

    def decode(self, latents, return_dict=False):
        self.seen = {"input_dtype": latents.dtype, "vae_dtype": self.weight.dtype,
                     "input": latents.clone()}
        b, _, h, w = latents.shape
        return (torch.zeros(b, 3, h * 8, w * 8),)


def _manager():
    return PipelineManager(DevicePolicy(device="cpu", dtype=torch.float16,
                                        vae_dtype=torch.float32, total_memory_gb=16))


def _pipe(vae):
    return SimpleNamespace(vae=vae, image_processor=VaeImageProcessor(vae_scale_factor=8))


class TestDecodeLatents:
    def test_decodes_in_fp32_even_when_latents_arrive_fp16(self):
        """The diffusers MPS fallback decoded the SDXL VAE in fp16: a white image."""
        vae = FakeVAE(dtype=torch.float16)
        latents = torch.ones(1, 4, 8, 8, dtype=torch.float16)
        img = _manager()._decode_latents(_pipe(vae), latents)
        assert vae.seen["input_dtype"] is torch.float32
        assert vae.seen["vae_dtype"] is torch.float32
        assert isinstance(img, Image.Image) and img.size == (64, 64)

    def test_restores_the_shared_vae_dtype_afterwards(self):
        """Leaving the shared VAE in fp32 sends the next txt2img decode down an unguarded fp16 path."""
        vae = FakeVAE(dtype=torch.float16)
        _manager()._decode_latents(_pipe(vae), torch.ones(1, 4, 8, 8, dtype=torch.float16))
        assert vae.seen["vae_dtype"] is torch.float32, "decode itself must run in fp32"
        assert vae.weight.dtype is torch.float16, "shared VAE must be handed back as found"

    def test_restores_the_vae_dtype_even_when_decode_fails(self):
        class Exploding(FakeVAE):
            def decode(self, latents, return_dict=False):
                raise RuntimeError("boom")
        vae = Exploding(dtype=torch.float16)
        try:
            _manager()._decode_latents(_pipe(vae), torch.ones(1, 4, 8, 8))
        except RuntimeError:
            pass
        assert vae.weight.dtype is torch.float16

    def test_unscales_by_the_vae_scaling_factor(self):
        vae = FakeVAE()
        _manager()._decode_latents(_pipe(vae), torch.ones(1, 4, 8, 8))
        assert torch.allclose(vae.seen["input"], torch.full((1, 4, 8, 8), 2.0))

    def test_denormalises_when_the_vae_has_latent_statistics(self):
        vae = FakeVAE(mean=[1.0] * 4, std=[3.0] * 4)
        _manager()._decode_latents(_pipe(vae), torch.ones(1, 4, 8, 8))
        # latents * std / scaling + mean = 1 * 3 / 0.5 + 1
        assert torch.allclose(vae.seen["input"], torch.full((1, 4, 8, 8), 7.0))


class TestFacePrompt:
    def test_framing_tags_are_removed_and_identity_kept(self):
        out = _face_prompt_from(
            "1girl, silver hair, blue eyes, full body, standing, wide shot, distant, park")
        for gone in ("full body", "standing", "wide shot", "distant"):
            assert gone not in out
        for kept in ("1girl", "silver hair", "blue eyes"):
            assert kept in out
        assert "face focus" in out and "detailed eyes" in out

    def test_a_prompt_of_only_framing_tags_still_yields_a_subject(self):
        out = _face_prompt_from("full body, wide shot")
        assert out.startswith("1girl") and "wide shot" not in out


class _Module(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(2, dtype=dtype))


class _Variant:
    """Stands in for a diffusers pipeline class built from shared components."""

    def __init__(self, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet, scheduler):
        self.vae, self.text_encoder, self.text_encoder_2 = vae, text_encoder, text_encoder_2
        self.tokenizer, self.tokenizer_2 = tokenizer, tokenizer_2
        self.unet, self.scheduler = unet, scheduler

    @classmethod
    def from_pipe(cls, *args, **kwargs):
        raise AssertionError("from_pipe recast the shared model to fp32; use the constructor")

    def set_progress_bar_config(self, **kwargs):
        pass


class _Recasting(_Variant):
    """A variant whose construction changes shared precision, as from_pipe did."""

    def __init__(self, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet, scheduler):
        super().__init__(vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet, scheduler)
        self.unet.to(torch.float32)
        self.text_encoder.to(torch.float32)


def _base_pipe():
    comps = dict(vae=_Module(torch.float16), text_encoder=_Module(torch.float16),
                 text_encoder_2=_Module(torch.float16), tokenizer=object(),
                 tokenizer_2=object(), unet=_Module(torch.float16), scheduler=object(),
                 image_encoder=None, feature_extractor=None)
    return SimpleNamespace(**comps, components=comps)


def _loaded_manager(monkeypatch, variant_cls):
    m = _manager()
    base = _base_pipe()
    m._pipes, m._checkpoint, m._arch = {"txt2img": base}, "fake.safetensors", "sdxl"
    monkeypatch.setattr(m, "_pipeline_classes",
                        lambda arch: {"img2img": variant_cls, "inpaint": variant_cls})
    return m, base


class TestVariantConstruction:
    def test_img2img_shares_modules_without_from_pipe(self, monkeypatch):
        """from_pipe flipped the base model to fp32 and doubled memory (6.48 -> 12.99GB)."""
        m, base = _loaded_manager(monkeypatch, _Variant)
        pipe = m.pipe_for("img2img")
        assert pipe.unet is base.unet and pipe.vae is base.vae
        assert next(base.unet.parameters()).dtype is torch.float16

    def test_dtype_guard_restores_recast_shared_weights(self, monkeypatch):
        m, base = _loaded_manager(monkeypatch, _Recasting)
        pipe = m.pipe_for("img2img")
        assert next(pipe.unet.parameters()).dtype is torch.float16
        assert next(base.text_encoder.parameters()).dtype is torch.float16


class TestFaceWorkSize:
    def test_native_resolution_per_architecture(self):
        """512px SDXL fragmented faces; native resolution rendered them cleanly."""
        m = _manager()
        m._arch = "sdxl"
        assert m._face_work_size() == 1024
        m._arch = "sd15"
        assert m._face_work_size() == 512

    def test_explicit_override_wins(self):
        m = _manager()
        m._arch = "sdxl"
        assert m._face_work_size(768) == 768
