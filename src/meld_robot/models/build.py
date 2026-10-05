"""Construct / save / load FusionEmotionModel."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoModel

from .fusion import FusionEmotionModel, ModelConfig


def _load_encoder(name: str, pretrained: bool, override_cfg=None):
    if override_cfg is not None:  # tests: tiny random encoder, no download
        return AutoModel.from_config(override_cfg)
    if pretrained:
        return AutoModel.from_pretrained(name)
    return AutoModel.from_config(AutoConfig.from_pretrained(name))


def build_model(cfg: ModelConfig, pretrained: bool = True, text_cfg=None, audio_cfg=None) -> FusionEmotionModel:
    text = _load_encoder(cfg.text_name, pretrained, text_cfg) if cfg.mode in ("text", "fusion") else None
    audio = _load_encoder(cfg.audio_name, pretrained, audio_cfg) if cfg.mode in ("audio", "fusion") else None
    return FusionEmotionModel(text, audio, cfg)


def save_checkpoint(model: FusionEmotionModel, path: str | Path, extra: Optional[dict] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.cfg.to_dict(), "extra": extra or {}}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[FusionEmotionModel, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = build_model(cfg, pretrained=False)  # architecture/config only; weights come from the checkpoint
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck.get("extra", {})


def init_encoders_from(model: FusionEmotionModel, ckpt_path: str | Path, prefixes: tuple[str, ...]) -> int:
    """Copy encoder (and projection) weights from a single-modality checkpoint. Returns #tensors copied."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    own = model.state_dict()
    n = 0
    for k, v in ck["state_dict"].items():
        if k.startswith(prefixes) and k in own and own[k].shape == v.shape:
            own[k] = v
            n += 1
    model.load_state_dict(own)
    return n
