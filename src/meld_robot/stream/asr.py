"""Streaming-style ASR: faster-whisper re-transcribes the rolling window (partial) and the full turn (final)."""
from __future__ import annotations

import numpy as np

# Published parameter counts, used for the budget report (CTranslate2 models don't expose torch params).
WHISPER_PARAMS = {"tiny.en": 39e6, "base.en": 74e6, "small.en": 244e6, "medium.en": 769e6,
                  "tiny": 39e6, "base": 74e6, "small": 244e6, "medium": 769e6}


class FasterWhisperASR:
    def __init__(self, model: str = "small.en", device: str = "auto", compute_type: str = "auto", beam_size: int = 1):
        from faster_whisper import WhisperModel

        self.name, self.beam = model, beam_size
        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, final: bool = False) -> str:
        if len(audio) < 1600:  # < 100 ms
            return ""
        segs, _ = self.model.transcribe(
            audio.astype(np.float32), beam_size=self.beam if not final else max(self.beam, 2),
            language="en" if self.name.endswith(".en") else None, vad_filter=False,
            condition_on_previous_text=False, without_timestamps=True, temperature=0.0)
        return " ".join(s.text.strip() for s in segs).strip()

    def param_count(self) -> int:
        return int(WHISPER_PARAMS.get(self.name, 0))


def make_asr(cfg: dict):
    a = cfg.get("asr", {})
    return FasterWhisperASR(a.get("model", "small.en"), a.get("device", "auto"), a.get("compute_type", "auto"),
                            a.get("beam_size", 1))
