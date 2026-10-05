"""Offline MELD evaluation of a checkpoint (weighted-F1, macro-F1, per-class F1, confusion matrix, ECE)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from ..config import pick_device
from ..data import MeldDataset, load_tokenizer, make_collate
from ..models.build import load_checkpoint
from ..train.train import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--manifest-dir", default="data/processed")
    ap.add_argument("--asr-text", default="", help="JSON {id: transcript} from scripts/transcribe_meld.py -> ASR-noised eval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = pick_device()
    model, _ = load_checkpoint(args.ckpt, device)
    mode = model.cfg.mode
    tok = load_tokenizer(model.cfg.text_name if mode != "audio" else "roberta-base")
    override = json.loads(Path(args.asr_text).read_text()) if args.asr_text else None
    ds = MeldDataset(Path(args.manifest_dir) / f"{args.split}.jsonl", tok, use_text=mode != "audio",
                     use_audio=mode != "text", text_override=override, limit=args.limit)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=make_collate(tok.pad_token_id), num_workers=2)
    rep = evaluate(model, dl, device, amp=device == "cuda")
    rep.update(ckpt=args.ckpt, mode=mode, split=args.split, text_source="asr" if override else "gold")
    name = args.out or f"results/offline_{mode}_{args.split}_{rep['text_source']}.json"
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    Path(name).write_text(json.dumps(rep, indent=2))
    print(f"[{mode}/{args.split}/{rep['text_source']}] W-F1={rep['weighted_f1']:.4f} macro-F1={rep['macro_f1']:.4f} "
          f"acc={rep['accuracy']:.4f} ECE={rep['ece']:.3f}\nper-class F1: " +
          ", ".join(f"{k}={v:.2f}" for k, v in rep["per_class_f1"].items()) + f"\nsaved {name}")


if __name__ == "__main__":
    main()
