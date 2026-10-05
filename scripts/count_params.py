#!/usr/bin/env python
"""Verify the <= 6B total-parameter budget of the complete local inference path.

Counts TOTAL learned parameters (not active / trainable / effective) of every component that runs at inference:
VAD + ASR + text encoder + audio encoder + fusion/heads + response LLM. Quantised LLMs are counted at their full
logical size (from the architecture config), not at the packed tensor size.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meld_robot.config import load_config  # noqa: E402

BUDGET = 6_000_000_000

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/stream.yaml")
ap.add_argument("--out", default="results/params.json")
args = ap.parse_args()
cfg = load_config(args.config)

import torch  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM  # noqa: E402

from meld_robot.models.build import load_checkpoint  # noqa: E402
from meld_robot.stream.asr import WHISPER_PARAMS  # noqa: E402

rows: list[tuple[str, int]] = []
count = lambda m: sum(p.numel() for p in m.parameters())

# VAD
if cfg["vad"]["backend"] == "silero":
    from meld_robot.stream.vad import SileroVAD
    rows.append(("VAD: Silero", SileroVAD().param_count()))
else:
    rows.append(("VAD: energy gate (no parameters)", 0))

# ASR (CTranslate2 model has no torch params -> published size of the chosen Whisper variant)
rows.append((f"ASR: faster-whisper {cfg['asr']['model']}", int(WHISPER_PARAMS[cfg["asr"]["model"]])))

# Fused emotion model, broken down by part
model, _ = load_checkpoint(cfg["checkpoint"])
parts = {"text_enc": "Text encoder", "audio_enc": "Audio encoder"}
enc_total = 0
for attr, label in parts.items():
    m = getattr(model, attr, None)
    if m is not None:
        n = count(m); enc_total += n
        rows.append((f"{label}: {getattr(model.cfg, 'text_name' if attr == 'text_enc' else 'audio_name')}", n))
rows.append(("Fusion, pooling, projections, heads", count(model) - enc_total))

# Response LLM
if cfg["llm"]["backend"] == "transformers":
    with torch.device("meta"):
        llm = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(cfg["llm"]["model"]))
    rows.append((f"Response LLM: {cfg['llm']['model']} ({cfg['llm']['quantization']})", count(llm)))
else:
    rows.append(("Response LLM: rule-based templates (no parameters)", 0))

total = sum(n for _, n in rows)
print(f"{'component':<62}{'params':>16}")
for name, n in rows:
    print(f"{name:<62}{n:>16,}")
print("-" * 78)
print(f"{'TOTAL':<62}{total:>16,}   ({total / 1e9:.2f} B; budget {BUDGET / 1e9:.0f} B -> {100 * total / BUDGET:.0f}% used)")
assert total <= BUDGET, "parameter budget exceeded!"
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps({"components": dict(rows), "total": total, "budget": BUDGET}, indent=2))
print(f"saved {args.out}")
