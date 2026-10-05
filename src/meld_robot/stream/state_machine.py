"""Turn detection and label smoothing (pure Python / NumPy, fully unit-tested)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from ..labels import EMOTIONS


class Turn(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    RESPONDING = "responding"


class VadEvent(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass
class TurnDetector:
    """Converts per-window speech probabilities into start / end-of-turn events.

    - speech must persist `min_speech_ms` before SPEECH_START is confirmed (debounce)
    - `endpoint_silence_ms` of continuous non-speech ends the turn
    """
    threshold: float = 0.5
    min_speech_ms: float = 300.0
    endpoint_silence_ms: float = 500.0
    in_speech: bool = False
    _speech_ms: float = 0.0
    _silence_ms: float = 0.0
    _candidate: bool = False

    def update(self, prob: float, window_ms: float) -> Optional[VadEvent]:
        speech = prob >= self.threshold
        if not self.in_speech:
            if speech:
                self._speech_ms += window_ms
                self._candidate = True
                if self._speech_ms >= self.min_speech_ms:
                    self.in_speech, self._silence_ms = True, 0.0
                    return VadEvent.SPEECH_START
            else:
                if self._candidate and self._speech_ms > 0:
                    self._silence_ms += window_ms
                    if self._silence_ms >= 150:  # a brief dropout is tolerated, a real gap resets the candidate
                        self._speech_ms, self._silence_ms, self._candidate = 0.0, 0.0, False
            return None
        # in speech
        if speech:
            self._silence_ms = 0.0
            return None
        self._silence_ms += window_ms
        if self._silence_ms >= self.endpoint_silence_ms:
            self.reset()
            return VadEvent.SPEECH_END
        return None

    def reset(self) -> None:
        self.in_speech = self._candidate = False
        self._speech_ms = self._silence_ms = 0.0


class EmotionSmoother:
    """EMA over probabilities + hysteresis on the reported label.

    The raw model output flickers between windows; the reported label only switches when the
    new leader beats the current label's smoothed probability by `margin`.
    """

    def __init__(self, alpha: float = 0.5, margin: float = 0.10):
        self.alpha, self.margin = alpha, margin
        self.p: Optional[np.ndarray] = None
        self.label: Optional[int] = None

    def update(self, probs: np.ndarray) -> tuple[str, float, np.ndarray]:
        probs = np.asarray(probs, dtype=np.float64)
        self.p = probs.copy() if self.p is None else self.alpha * probs + (1 - self.alpha) * self.p
        top = int(self.p.argmax())
        if self.label is None or top == self.label or self.p[top] - self.p[self.label] >= self.margin:
            self.label = top
        return EMOTIONS[self.label], float(self.p[self.label]), self.p

    def reset(self) -> None:
        self.p, self.label = None, None


def count_flips(labels: list[int]) -> int:
    return sum(1 for a, b in zip(labels[:-1], labels[1:]) if a != b)
