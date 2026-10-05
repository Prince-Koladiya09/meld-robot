"""Interpretable prosody cues from raw audio (no learned parameters).

These cues are passed to the response LLM so the reply can reference *how* something was said.
Thresholds are rough, speaker-independent heuristics: they are cues, not ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

SR = 16000


@dataclass
class Prosody:
    pitch: str = "unknown"      # low | mid | high   (median F0)
    pitch_var: str = "unknown"  # flat | normal | expressive (F0 std in semitones)
    energy: str = "unknown"     # low | mid | high   (RMS dBFS)
    rate: str = "unknown"       # slow | normal | fast (words or onsets per second)
    pauses: str = "unknown"     # few | some | many  (silent fraction inside the utterance)
    raw: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        parts = [f"pitch {self.pitch}", f"pitch movement {self.pitch_var}", f"energy {self.energy}",
                 f"speaking rate {self.rate}", f"pauses {self.pauses}"]
        return ", ".join(p for p in parts if "unknown" not in p) or "no reliable prosody"


def _bucket(x: float, lo: float, hi: float, names: tuple[str, str, str]) -> str:
    return names[0] if x < lo else names[2] if x > hi else names[1]


def _frame_rms_db(wav: np.ndarray, frame: int = 400, hop: int = 160) -> np.ndarray:
    if len(wav) < frame:
        wav = np.pad(wav, (0, frame - len(wav)))
    n = 1 + (len(wav) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt((wav[idx] ** 2).mean(1) + 1e-12)
    return 20 * np.log10(rms + 1e-9)


def extract_prosody(wav: np.ndarray, text: str = "", sr: int = SR) -> Prosody:
    """wav: float32 mono in [-1, 1] at 16 kHz (un-normalised, so loudness is meaningful)."""
    dur = len(wav) / sr
    if dur < 0.3:
        return Prosody()
    db = _frame_rms_db(wav)
    peak = db.max()
    voiced = db > max(peak - 30.0, -60.0)           # relative energy gate
    if voiced.sum() < 5:
        return Prosody()
    energy_db = float(db[voiced].mean())

    # F0 with librosa.yin (fast); keep only frames the energy gate calls voiced
    f0_med = f0_std_st = None
    try:
        import librosa
        f0 = librosa.yin(wav.astype(np.float32), fmin=65, fmax=400, sr=sr, frame_length=1024, hop_length=160)
        n = min(len(f0), len(voiced))
        f0v = f0[:n][voiced[:n]]
        f0v = f0v[(f0v > 70) & (f0v < 390)]  # yin saturates at its bounds when unvoiced
        if len(f0v) >= 5:
            f0_med = float(np.median(f0v))
            st = 12 * np.log2(f0v / f0_med)
            f0_std_st = float(np.std(st))
    except Exception:
        pass

    # Speaking rate: words/s from the transcript if we have it, else onset density.
    voiced_s = voiced.sum() * 0.01
    words = len(text.split())
    if words >= 2:
        rate_val = words / max(voiced_s, 0.3)
        rate = _bucket(rate_val, 2.0, 3.8, ("slow", "normal", "fast"))
    else:
        try:
            import librosa
            on = librosa.onset.onset_detect(y=wav, sr=sr, hop_length=160, units="time")
            rate_val = len(on) / max(voiced_s, 0.3)
            rate = _bucket(rate_val, 2.0, 5.0, ("slow", "normal", "fast"))
        except Exception:
            rate_val, rate = float("nan"), "unknown"

    # Pauses: fraction of unvoiced frames between first and last voiced frame
    first, last = np.argmax(voiced), len(voiced) - 1 - np.argmax(voiced[::-1])
    span = voiced[first:last + 1]
    sil_frac = float(1 - span.mean()) if len(span) else 0.0

    return Prosody(
        pitch=_bucket(f0_med, 120, 200, ("low", "mid", "high")) if f0_med else "unknown",
        pitch_var=_bucket(f0_std_st, 2.0, 4.5, ("flat", "normal", "expressive")) if f0_std_st is not None else "unknown",
        energy=_bucket(energy_db, -32.0, -20.0, ("low", "mid", "high")),
        rate=rate,
        pauses=_bucket(sil_frac, 0.12, 0.30, ("few", "some", "many")),
        raw={"f0_median_hz": f0_med, "f0_std_semitones": f0_std_st, "energy_db": energy_db,
             "rate_per_s": rate_val, "silence_frac": sil_frac, "duration_s": dur},
    )


def warmup() -> None:
    """librosa/numba JIT-compile on first use (~0.7 s). Call once at startup so the first user turn doesn't pay for it."""
    rng = np.random.default_rng(0)
    t = np.arange(SR) / SR
    extract_prosody((0.2 * np.sin(2 * np.pi * 150 * t) + 0.01 * rng.standard_normal(SR)).astype(np.float32), "warm up now")
    extract_prosody((0.2 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), "")  # onset-rate path
