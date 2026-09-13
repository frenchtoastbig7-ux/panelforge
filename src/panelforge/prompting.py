"""Prompt weighting.

Anime prompts lean heavily on `(tag:1.2)` emphasis and routinely exceed CLIP's
77-token window. compel handles both. It reaches into the text encoders
directly, so if a transformers release changes those internals we fall back to
plain prompts rather than failing the job.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

log = logging.getLogger(__name__)

_WARNED = False


@dataclass
class Conditioning:
    """Either embeddings (weighted path) or raw strings (fallback path)."""
    prompt_embeds: torch.Tensor | None = None
    negative_prompt_embeds: torch.Tensor | None = None
    pooled_prompt_embeds: torch.Tensor | None = None
    negative_pooled_prompt_embeds: torch.Tensor | None = None
    prompt: str | None = None
    negative_prompt: str | None = None

    def as_kwargs(self) -> dict:
        if self.prompt_embeds is not None:
            kwargs = {
                "prompt_embeds": self.prompt_embeds,
                "negative_prompt_embeds": self.negative_prompt_embeds,
            }
            if self.pooled_prompt_embeds is not None:
                kwargs["pooled_prompt_embeds"] = self.pooled_prompt_embeds
                kwargs["negative_pooled_prompt_embeds"] = self.negative_pooled_prompt_embeds
            return kwargs
        return {"prompt": self.prompt, "negative_prompt": self.negative_prompt}


def build_conditioning(pipe, prompt: str, negative: str, is_sdxl: bool) -> Conditioning:
    """Encode prompts, using weighted embeddings when compel is usable."""
    global _WARNED
    try:
        from compel import Compel, ReturnedEmbeddingsType
    except ImportError:
        return Conditioning(prompt=prompt, negative_prompt=negative)

    try:
        if is_sdxl:
            compel = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False,
            )
            cond, pooled = compel(prompt)
            uncond, neg_pooled = compel(negative)
            cond, uncond = compel.pad_conditioning_tensors_to_same_length([cond, uncond])
            return Conditioning(
                prompt_embeds=cond,
                negative_prompt_embeds=uncond,
                pooled_prompt_embeds=pooled,
                negative_pooled_prompt_embeds=neg_pooled,
            )
        compel = Compel(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            truncate_long_prompts=False,
        )
        cond = compel(prompt)
        uncond = compel(negative)
        cond, uncond = compel.pad_conditioning_tensors_to_same_length([cond, uncond])
        return Conditioning(prompt_embeds=cond, negative_prompt_embeds=uncond)
    except Exception as exc:
        if not _WARNED:
            log.warning(
                "prompt weighting unavailable (%s); falling back to plain prompts. "
                "(tag:1.2) emphasis and >77-token prompts will be ignored.", exc,
            )
            _WARNED = True
        return Conditioning(prompt=prompt, negative_prompt=negative)
