"""MELD dataset + collate (text with dialogue context, 16 kHz audio)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .labels import EMO2ID, SENT2ID

SR = 16000


def load_tokenizer(name: str):
    """Tokenizer that truncates from the LEFT, so when text+context exceeds max_length the OLDEST context goes first
    (HF's default right-side truncation would keep the oldest context and cut the newest turn instead)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.truncation_side = "left"
    return tok


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_wav(path: str | Path, max_seconds: float | None = None) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    if max_seconds:
        wav = wav[: int(max_seconds * SR)]
    return wav


def build_text_pair(text: str, context: list[str]) -> tuple[str, Optional[str]]:
    """(current utterance, context string). Context goes second so truncation cuts old context first."""
    return text, (" </s> ".join(context) if context else None)


def normalize_wave(wav: np.ndarray) -> np.ndarray:
    if wav.size == 0:
        return np.zeros(SR // 10, dtype=np.float32)
    wav = wav - wav.mean()
    return (wav / (wav.std() + 1e-7)).astype(np.float32)


def augment(wav: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:  # gain
        wav = wav * rng.uniform(0.6, 1.4)
    if rng.random() < 0.3:  # additive noise
        wav = wav + np.random.randn(len(wav)).astype(np.float32) * rng.uniform(0.001, 0.01) * (np.abs(wav).max() + 1e-6)
    if rng.random() < 0.3:  # speed +-5 % (naive resample)
        f = rng.uniform(0.95, 1.05)
        idx = np.clip((np.arange(int(len(wav) / f)) * f).astype(int), 0, len(wav) - 1)
        wav = wav[idx]
    return wav.astype(np.float32)


class MeldDataset(Dataset):
    def __init__(self, manifest: str | Path, tokenizer, max_text_len: int = 128, max_audio_seconds: float = 6.0,
                 train: bool = False, use_audio: bool = True, use_text: bool = True, augment_audio: bool = False,
                 text_override: Optional[dict[str, str]] = None, limit: int = 0):
        self.rows = read_manifest(manifest)
        if limit:
            self.rows = self.rows[:limit]
        self.tok, self.max_len, self.max_s = tokenizer, max_text_len, max_audio_seconds
        self.use_audio, self.use_text, self.aug = use_audio, use_text, augment_audio and train
        self.override = text_override or {}  # {row id: asr transcript} for ASR-noised evaluation
        self.rng = random.Random(0)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        item = {"id": r["id"], "emotion": EMO2ID[r["emotion"]], "sentiment": SENT2ID[r["sentiment"]]}
        if self.use_text:
            text = self.override.get(r["id"], r["text"])
            a, b = build_text_pair(text, r["context"])
            enc = self.tok(a, b, truncation="only_second" if b else True, max_length=self.max_len)
            item["input_ids"] = enc["input_ids"]
        if self.use_audio:
            wav = load_wav(r["wav"], self.max_s)
            if self.aug:
                wav = augment(wav, self.rng)
            item["wave"] = normalize_wave(wav)
        return item


def make_collate(pad_token_id: int):
    def collate(batch: list[dict]) -> dict:
        out: dict = {
            "ids": [b["id"] for b in batch],
            "labels": torch.tensor([b["emotion"] for b in batch]),
            "sent_labels": torch.tensor([b["sentiment"] for b in batch]),
        }
        if "input_ids" in batch[0]:
            L = max(len(b["input_ids"]) for b in batch)
            ids = torch.full((len(batch), L), pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(batch), L), dtype=torch.long)
            for i, b in enumerate(batch):
                n = len(b["input_ids"])
                ids[i, :n] = torch.tensor(b["input_ids"])
                mask[i, :n] = 1
            out["input_ids"], out["attention_mask"] = ids, mask
        if "wave" in batch[0]:
            T = max(len(b["wave"]) for b in batch)
            wav = torch.zeros((len(batch), T))
            lens = torch.zeros(len(batch), dtype=torch.long)
            for i, b in enumerate(batch):
                wav[i, : len(b["wave"])] = torch.from_numpy(b["wave"])
                lens[i] = len(b["wave"])
            out["input_values"], out["audio_lengths"] = wav, lens
        return out

    return collate
