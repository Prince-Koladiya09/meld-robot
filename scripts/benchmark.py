#!/usr/bin/env python
"""Measure latency, real-time factor and memory of each stage on THIS machine.

Run it after training and paste results/benchmark.json into the README's hardware table.
Uses the first N test clips if data/processed exists, otherwise synthetic audio (latency only, meaningless text).
"""
import argparse, asyncio, json, platform, resource, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meld_robot.config import load_config  # noqa: E402
from meld_robot.data import load_wav, read_manifest  # noqa: E402
from meld_robot.stream.factory import build_components  # noqa: E402
from meld_robot.stream.prosody import extract_prosody  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/stream.yaml")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--manifest", default="data/processed/test.jsonl")
ap.add_argument("--out", default="results/benchmark.json")
args = ap.parse_args()

import torch  # noqa: E402

cfg = load_config(args.config)
t0 = time.perf_counter()
comp = build_components(cfg)
load_s = time.perf_counter() - t0

if Path(args.manifest).exists():
    rows = read_manifest(args.manifest)[: args.n]
    clips = [(load_wav(r["wav"], 6.0), r["text"], r["context"]) for r in rows]
else:
    rng = np.random.default_rng(0)
    clips = [((0.1 * rng.standard_normal(16000 * 3)).astype(np.float32), "hello there", []) for _ in range(args.n)]

pct = lambda a, q: float(np.percentile(a, q))
asr_ms, eng_ms, pro_ms, rtf = [], [], [], []
comp.asr.transcribe(clips[0][0], True); comp.engine.predict("warm up", [], clips[0][0])   # warm-up
for wav, text, ctx in clips:
    t = time.perf_counter(); hyp = comp.asr.transcribe(wav, True); dt = time.perf_counter() - t
    asr_ms.append(dt * 1000); rtf.append(dt / (len(wav) / 16000))
    t = time.perf_counter(); comp.engine.predict(hyp or text, ctx, wav); eng_ms.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter(); extract_prosody(wav, hyp); pro_ms.append((time.perf_counter() - t) * 1000)


async def llm_latency():
    ft, tot, ntok = [], [], []
    for emo in ["anger", "joy", "sadness", "neutral", "surprise"] * 2:
        st = {"emotion": {"label": emo, "confidence": 0.7}, "sentiment": "neutral",
              "prosody_summary": "energy high", "mismatch": False, "text_emotion": emo}
        t = time.perf_counter(); first = None; n = 0
        async for _ in comp.responder.stream(st, "I had a long day.", []):
            first = first or time.perf_counter(); n += 1
        ft.append((first - t) * 1000 if first else float("nan")); tot.append((time.perf_counter() - t) * 1000); ntok.append(n)
    return ft, tot, ntok

ft, tot, ntok = asyncio.run(llm_latency())
cuda = torch.cuda.is_available()
res = {
    "hardware": {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
                 "cpu_cores": __import__("os").cpu_count(),
                 "gpu": torch.cuda.get_device_name(0) if cuda else None,
                 "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if cuda else None},
    "model_load_seconds": round(load_s, 1),
    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None,
    "peak_ram_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),  # KB on Linux
    "asr_final_ms": {"p50": pct(asr_ms, 50), "p95": pct(asr_ms, 95)},
    "asr_real_time_factor": {"p50": pct(rtf, 50), "p95": pct(rtf, 95)},
    "emotion_predict_ms": {"p50": pct(eng_ms, 50), "p95": pct(eng_ms, 95)},
    "prosody_ms": {"p50": pct(pro_ms, 50), "p95": pct(pro_ms, 95)},
    "llm_first_token_ms": {"p50": pct(ft, 50), "p95": pct(ft, 95)},
    "llm_total_ms": {"p50": pct(tot, 50), "p95": pct(tot, 95)},
    "llm_pieces_per_reply_mean": float(np.mean(ntok)),
    "n_clips": len(clips), "real_data": Path(args.manifest).exists(),
}
res["end_of_utterance_to_first_token_ms_estimate_p50"] = round(
    res["asr_final_ms"]["p50"] + res["emotion_predict_ms"]["p50"] + res["prosody_ms"]["p50"] + res["llm_first_token_ms"]["p50"], 1)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
