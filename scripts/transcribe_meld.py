#!/usr/bin/env python
"""Transcribe a MELD split with Whisper -> data/processed/asr_<split>.json  ({utterance id: transcript}).

Used for the ASR-noised evaluation (train/serve gap: live input is ASR text, MELD training text is gold).
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tqdm import tqdm  # noqa: E402

from meld_robot.data import load_wav, read_manifest  # noqa: E402
from meld_robot.stream.asr import FasterWhisperASR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--manifest-dir", default="data/processed")
ap.add_argument("--model", default="small.en")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

rows = read_manifest(Path(args.manifest_dir) / f"{args.split}.jsonl")
rows = rows[: args.limit] if args.limit else rows
asr = FasterWhisperASR(args.model, beam_size=2)
out = {r["id"]: asr.transcribe(load_wav(r["wav"], 20.0), final=True) for r in tqdm(rows)}
dst = Path(args.manifest_dir) / f"asr_{args.split}.json"
dst.write_text(json.dumps(out, ensure_ascii=False, indent=1))

# word error rate vs gold, as a sanity number
import re
def norm(s): return re.sub(r"[^a-z0-9' ]", "", s.lower()).split()
errs = words = 0
for r in rows:
    g, h = norm(r["text"]), norm(out[r["id"]])
    d = list(range(len(h) + 1))
    for i, gw in enumerate(g, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (gw != hw))
    errs += d[len(h)]; words += len(g)
print(f"saved {dst}; WER vs gold transcripts = {errs / max(words, 1):.3f}")
