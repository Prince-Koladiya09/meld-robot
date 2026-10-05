#!/usr/bin/env python
"""Replay a wav/mp4 clip as live 20 ms audio frames and print every event.

In-process (default; loads models locally):
    python demo/replay_demo.py --config configs/stream.yaml --clip data/processed/audio/test/dia0_utt3.wav
Against a running server:
    python demo/replay_demo.py --ws ws://127.0.0.1:8000/stream --clip some.wav
Add --speed 0 to run as fast as possible (audio clock is used, so results are unchanged).
Add --template-llm / --energy-vad to skip the LLM / Silero download.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meld_robot.config import load_config  # noqa: E402
from meld_robot.data import load_wav  # noqa: E402

SR, FRAME = 16000, 320  # 20 ms


def load_clip(path: str) -> np.ndarray:
    if path.lower().endswith((".mp4", ".mkv", ".avi", ".mov")):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), tmp], check=True)
        path = tmp
    return load_wav(path)


def show(e: dict, verbose: bool) -> None:
    t = e["type"]
    if t == "state_update":
        em = e["emotion"]
        print(f"  [{e['t_ms']:>6} ms] partial={e['partial_text']!r:<34} {em['label']:<9} {em['confidence']:.2f}"
              f"  text/audio={e['modality_weight']['text']:.2f}/{e['modality_weight']['audio']:.2f}")
    elif t == "reply_token":
        print(e["token"], end="", flush=True)
    elif t == "reply_done":
        print(f"\n  -> reply latency: {e['latency_ms']}")
    elif t == "state_final":
        print("\n=== state_final ===")
        print(json.dumps({k: v for k, v in e.items() if k not in ("type", "prosody") or verbose}, indent=2))
        print("--- streamed reply ---")
    elif t in ("turn", "barge_in", "reply_cancelled"):
        print(f"[{t}] {e.get('turn', '')}")


async def run_inproc(args, audio: np.ndarray) -> None:
    from meld_robot.stream.pipeline import Components, Session

    cfg = load_config(args.config)
    if args.energy_vad:
        cfg["vad"]["backend"] = "energy"
    if args.template_llm:
        cfg["llm"]["backend"] = "template"
    from meld_robot.stream.factory import build_components
    comp: Components = build_components(cfg)
    q: asyncio.Queue = asyncio.Queue()
    s = Session(cfg, comp, q, realtime=args.speed > 0)

    async def printer():
        while True:
            show(await q.get(), args.verbose)

    pt = asyncio.create_task(printer())
    pad = np.concatenate([np.zeros(SR // 2, np.float32), audio, np.zeros(SR * 2, np.float32)])  # lead-in + endpoint silence
    for i in range(0, len(pad), FRAME):
        await s.feed(pad[i:i + FRAME])
        if args.speed > 0:
            await asyncio.sleep(0.02 / args.speed)
    await s.flush()
    await asyncio.sleep(0.1)
    pt.cancel()
    await s.close()


async def run_ws(args, audio: np.ndarray) -> None:
    import websockets

    async with websockets.connect(args.ws, max_size=None) as ws:
        async def reader():
            async for m in ws:
                show(json.loads(m), args.verbose)

        rt = asyncio.create_task(reader())
        pad = np.concatenate([np.zeros(SR // 2, np.float32), audio, np.zeros(SR * 2, np.float32)])
        pcm = (np.clip(pad, -1, 1) * 32767).astype(np.int16)
        for i in range(0, len(pcm), FRAME):
            await ws.send(pcm[i:i + FRAME].tobytes())
            if args.speed > 0:
                await asyncio.sleep(0.02 / args.speed)
        await asyncio.sleep(8)  # allow the reply to stream
        rt.cancel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--config", default="configs/stream.yaml")
    ap.add_argument("--ws", default="", help="ws:// URL of a running server; omit for in-process")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0 = as fast as possible")
    ap.add_argument("--energy-vad", action="store_true")
    ap.add_argument("--template-llm", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    audio = load_clip(args.clip)
    print(f"Replaying {args.clip} ({len(audio)/SR:.1f} s)")
    asyncio.run(run_ws(args, audio) if args.ws else run_inproc(args, audio))


if __name__ == "__main__":
    main()
