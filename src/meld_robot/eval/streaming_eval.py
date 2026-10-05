"""Streaming behaviour of the emotion model when it only hears part of an utterance.

For each test utterance we replay audio in 250 ms hops. The partial transcript is approximated by the
proportional word-prefix of the gold transcript (isolates the emotion model from ASR error; use
scripts/transcribe_meld.py + offline_eval --asr-text for the ASR gap).

Reports:  accuracy / W-F1 vs fraction of the utterance heard, label-flip rate with and without the EMA +
hysteresis smoother, and per-call latency percentiles.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from ..data import load_wav, read_manifest
from ..labels import EMO2ID
from ..stream.engine import EmotionEngine
from ..stream.state_machine import EmotionSmoother, count_flips

SR, HOP = 16000, 4000  # 250 ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--manifest-dir", default="data/processed")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--out", default="results/streaming_eval.json")
    args = ap.parse_args()

    eng = EmotionEngine(args.ckpt)
    rows = read_manifest(Path(args.manifest_dir) / f"{args.split}.jsonl")[: args.limit]
    fracs = [0.25, 0.5, 0.75, 1.0]
    y_true, y_by_frac = [], {f: [] for f in fracs}
    flips_raw, flips_smooth, lat = [], [], []

    for r in rows:
        wav = load_wav(r["wav"], 6.0)
        if len(wav) < SR // 2:
            continue
        words = r["text"].split()
        y_true.append(EMO2ID[r["emotion"]])
        for f in fracs:
            n = max(1, int(round(len(words) * f)))
            t0 = time.perf_counter()
            o = eng.predict(" ".join(words[:n]), r["context"], wav[: max(int(len(wav) * f), SR // 4)])
            lat.append((time.perf_counter() - t0) * 1000)
            y_by_frac[f].append(int(o.probs.argmax()))
        sm, raw, smooth = EmotionSmoother(args.alpha, args.margin), [], []
        for end in range(HOP, len(wav) + HOP, HOP):
            end = min(end, len(wav))
            n = max(1, int(round(len(words) * end / len(wav))))
            o = eng.predict(" ".join(words[:n]), r["context"], wav[:end])
            raw.append(int(o.probs.argmax()))
            smooth.append(EMO2ID[sm.update(o.probs)[0]])
        flips_raw.append(count_flips(raw)); flips_smooth.append(count_flips(smooth))

    res = {
        "n_utterances": len(y_true),
        "by_fraction_heard": {str(f): {"accuracy": float(np.mean(np.array(y_true) == np.array(y_by_frac[f]))),
                                        "weighted_f1": float(f1_score(y_true, y_by_frac[f], average="weighted", zero_division=0))}
                              for f in fracs},
        "mean_label_flips_per_utterance": {"raw_argmax": float(np.mean(flips_raw)), "ema_hysteresis": float(np.mean(flips_smooth))},
        "predict_latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                               "device": eng.device},
        "smoother": {"alpha": args.alpha, "margin": args.margin},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
