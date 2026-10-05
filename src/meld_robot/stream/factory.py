"""Build the heavy components from a stream config (lazy imports keep tests/CLI light)."""
from __future__ import annotations

from .pipeline import Components


def build_components(cfg: dict) -> Components:
    from ..respond.llm import make_responder
    from .asr import make_asr
    from .engine import EmotionEngine
    from .vad import make_vad

    from .prosody import warmup

    warmup()
    s = cfg.get("streaming", {})
    return Components(
        vad=make_vad(cfg),
        asr=make_asr(cfg),
        engine=EmotionEngine(cfg["checkpoint"], cfg.get("asr", {}).get("device", "auto"), window_s=s.get("window_s", 6.0)),
        responder=make_responder(cfg),
    )
