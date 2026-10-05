#!/usr/bin/env python
"""Turn raw MELD into training manifests.

For every utterance this script
  1. extracts mono 16 kHz audio from the mp4 with ffmpeg,
  2. attaches the previous N utterances of the same dialogue as text context,
  3. writes one JSONL manifest per split (train/dev/test).

Utterances whose video is missing or whose audio extraction fails are skipped and counted
(MELD is known to contain a few broken clips).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meld_robot.config import load_config  # noqa: E402

SPLITS = ["train", "dev", "test"]


def index_videos(root: Path) -> dict[str, dict[str, Path]]:
    """{split: {'dia0_utt0': path}} found by walking the MELD tree once."""
    idx: dict[str, dict[str, Path]] = {s: {} for s in SPLITS}
    for p in root.rglob("*.mp4"):
        low = str(p.parent).lower()
        for s in SPLITS:
            if s in low:
                idx[s][p.stem] = p
                break
    return idx


def find_csv(root: Path, split: str) -> Path:
    hits = list(root.rglob(f"{split}_sent_emo.csv"))
    if not hits:
        raise FileNotFoundError(f"{split}_sent_emo.csv not found under {root}")
    return hits[0]


def extract_audio(src: Path, dst: Path, sr: int) -> bool:
    if dst.exists() and dst.stat().st_size > 1000:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr), str(dst)]
    try:
        subprocess.run(cmd, check=True, timeout=60)
        return dst.exists() and dst.stat().st_size > 1000
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N rows per split")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root, out = Path(cfg["meld_root"]), Path(cfg["out_dir"])
    sr, ctx_n = int(cfg["sample_rate"]), int(cfg["context_turns"])
    out.mkdir(parents=True, exist_ok=True)

    videos = index_videos(root)
    summary: dict[str, dict] = {}

    for split in SPLITS:
        df = pd.read_csv(find_csv(root, split))
        if args.limit:
            df = df.head(args.limit)
        df = df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)

        jobs, rows = [], []
        history: dict[int, list[str]] = {}
        for r in df.itertuples(index=False):
            d, u = int(r.Dialogue_ID), int(r.Utterance_ID)
            name = f"dia{d}_utt{u}"
            text = str(r.Utterance).replace("\x92", "'").strip()
            hist = history.setdefault(d, [])
            context = hist[-ctx_n:] if ctx_n > 0 else []
            hist.append(f"{r.Speaker}: {text}")
            src = videos[split].get(name)
            wav = out / "audio" / split / f"{name}.wav"
            rows.append({
                "id": f"{split}/{name}", "split": split, "dialogue": d, "utt": u,
                "speaker": str(r.Speaker), "text": text, "context": context,
                "emotion": str(r.Emotion).lower(), "sentiment": str(r.Sentiment).lower(),
                "wav": str(wav), "_src": src,
            })
            jobs.append((src, wav))

        def run(job):
            src, wav = job
            return bool(src) and extract_audio(src, wav, sr)

        with ThreadPoolExecutor(args.workers) as ex:
            ok = list(ex.map(run, jobs))

        kept, skipped = [], 0
        for row, good in zip(rows, ok):
            row.pop("_src")
            if good:
                kept.append(row)
            else:
                skipped += 1
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[split] = {
            "kept": len(kept), "skipped": skipped,
            "emotions": dict(Counter(r["emotion"] for r in kept)),
        }
        print(f"[{split}] kept={len(kept)} skipped={skipped}")

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
