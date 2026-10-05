"""Voice activity detection backends. Both return P(speech) for one 512-sample (32 ms @ 16 kHz) window."""
from __future__ import annotations

import numpy as np

WINDOW = 512  # samples (Silero's required window size at 16 kHz)


class EnergyVAD:
    """Dependency-free fallback: frame RMS above a dBFS threshold. Used in tests and for quick demos."""

    def __init__(self, threshold_db: float = -45.0):
        self.threshold_db = threshold_db

    def prob(self, chunk: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2) + 1e-12))
        return 1.0 if 20 * np.log10(rms + 1e-9) > self.threshold_db else 0.0

    def reset(self) -> None:
        pass

    def param_count(self) -> int:
        return 0


class SileroVAD:
    def __init__(self):
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self.model = load_silero_vad()

    def prob(self, chunk: np.ndarray) -> float:
        with self._torch.no_grad():
            return float(self.model(self._torch.from_numpy(chunk.astype(np.float32)), 16000).item())

    def reset(self) -> None:
        self.model.reset_states()

    def param_count(self) -> int:
        try:
            return sum(p.numel() for p in self.model.parameters())
        except Exception:
            return 1_800_000  # published size of Silero VAD v5 (~1.8M) if the jit module hides its params


def make_vad(cfg: dict):
    v = cfg.get("vad", {})
    if v.get("backend", "silero") == "energy":
        return EnergyVAD(v.get("energy_threshold_db", -45.0))
    return SileroVAD()
