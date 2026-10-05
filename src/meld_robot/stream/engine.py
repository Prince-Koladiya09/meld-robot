"""Emotion inference engine: wraps the fused text+audio model for single-utterance / window prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..config import pick_device
from ..data import build_text_pair, load_tokenizer, normalize_wave
from ..models.build import load_checkpoint
from ..models.fusion import count_params

SR = 16000


@dataclass
class EngineOutput:
    probs: np.ndarray                   # fused 7-way probabilities
    text_probs: Optional[np.ndarray]    # text-branch auxiliary prediction (None for single-modality checkpoints)
    audio_probs: Optional[np.ndarray]
    gate_text: float                    # learned weight on the text stream, in [0, 1]


class EmotionEngine:
    def __init__(self, ckpt: str, device: str = "auto", max_text_len: int = 128, window_s: float = 6.0):
        self.device = pick_device(device)
        self.model, _ = load_checkpoint(ckpt, self.device)
        self.mode = self.model.cfg.mode
        self.tok = load_tokenizer(self.model.cfg.text_name) if self.mode != "audio" else None
        self.max_len, self.window = max_text_len, int(window_s * SR)

    @torch.no_grad()
    def predict(self, text: str, context: list[str], audio: np.ndarray) -> EngineOutput:
        kw: dict = {}
        if self.tok is not None:
            a, b = build_text_pair(text.strip() or "...", context)
            enc = self.tok(a, b, truncation="only_second" if b else True, max_length=self.max_len, return_tensors="pt")
            kw["input_ids"], kw["attention_mask"] = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        if self.mode != "text":
            wav = normalize_wave(audio[-self.window:])
            kw["input_values"] = torch.from_numpy(wav)[None].to(self.device)
            kw["audio_lengths"] = torch.tensor([len(wav)], device=self.device)
        out = self.model(**kw)
        sm = lambda t: torch.softmax(t.float(), -1)[0].cpu().numpy()
        return EngineOutput(
            probs=sm(out["logits"]),
            text_probs=sm(out["text_logits"]) if "text_logits" in out else None,
            audio_probs=sm(out["audio_logits"]) if "audio_logits" in out else None,
            gate_text=float(out["gate"][0, 0]),
        )

    def param_count(self) -> int:
        return count_params(self.model)
