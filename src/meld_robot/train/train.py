"""Staged training: --stage text | audio | fusion.

Each stage writes checkpoints/<stage>/best.pt (best dev weighted-F1) and results/<stage>_metrics.json
(test metrics from the best checkpoint). The fusion stage warm-starts its encoders from the text/audio
checkpoints when they exist.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from ..config import load_config, pick_device
from ..data import MeldDataset, load_tokenizer, make_collate, read_manifest
from ..labels import EMO2ID, NUM_EMOTIONS
from ..metrics import classification_report, expected_calibration_error
from ..models.build import build_model, init_encoders_from, save_checkpoint
from ..models.fusion import ModelConfig, count_params


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def class_weights(train_manifest: Path, power: float) -> torch.Tensor | None:
    if power <= 0:
        return None
    counts = Counter(r["emotion"] for r in read_manifest(train_manifest))
    freq = torch.tensor([max(counts.get(e, 1), 1) for e in EMO2ID], dtype=torch.float)
    w = (1.0 / freq) ** power
    return w / w.mean()


def set_encoder_trainable(model, flag: bool) -> None:
    for enc in (model.text_enc, model.audio_enc):
        if enc is not None:
            for p in enc.parameters():
                p.requires_grad = flag


def apply_audio_freezing(model, freeze_cnn: bool, freeze_layers: int) -> None:
    enc = model.audio_enc
    if enc is None:
        return
    if freeze_cnn and hasattr(enc, "freeze_feature_encoder"):
        enc.freeze_feature_encoder()
    for layer in enc.encoder.layers[:freeze_layers]:
        for p in layer.parameters():
            p.requires_grad = False


def to_device(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def compute_loss(out: dict, batch: dict, cw, cfg_model: dict, smoothing: float, mode: str) -> torch.Tensor:
    y, ys = batch["labels"], batch["sent_labels"]
    ce = lambda lg: F.cross_entropy(lg, y, weight=cw, label_smoothing=smoothing)
    loss = ce(out["logits"]) + cfg_model["sentiment_weight"] * F.cross_entropy(out["sent_logits"], ys)
    if mode == "fusion":
        loss = loss + cfg_model["aux_weight"] * (ce(out["text_logits"]) + ce(out["audio_logits"]))
    return loss


@torch.no_grad()
def evaluate(model, loader, device: str, amp: bool) -> dict:
    model.eval()
    probs, ys, gates = [], [], []
    for batch in tqdm(loader, leave=False, desc="eval"):
        b = to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda"):
            out = model(b.get("input_ids"), b.get("attention_mask"), b.get("input_values"), b.get("audio_lengths"))
        probs.append(torch.softmax(out["logits"].float(), -1).cpu())
        gates.append(out["gate"].float().cpu())
        ys.append(b["labels"].cpu())
    P, Y, G = torch.cat(probs).numpy(), torch.cat(ys).numpy(), torch.cat(gates).numpy()
    rep = classification_report(Y, P.argmax(1))
    rep["ece"] = expected_calibration_error(P, Y)
    rep["mean_text_gate"] = float(G.mean())
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["text", "audio", "fusion"], required=True)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--limit", type=int, default=0, help="debug: use first N rows per split")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for checkpoint/result names (ablations)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = pick_device()
    tcfg, dcfg, mcfg = cfg["train"], cfg["data"], cfg["model"]
    stage_name = args.stage + (f"_{args.tag}" if args.tag else "")
    epochs = args.epochs or tcfg["epochs"]
    amp = bool(tcfg["amp"]) and device == "cuda"
    print(f"stage={args.stage} device={device} amp={amp}")

    model = build_model(ModelConfig(text_name=mcfg["text_name"], audio_name=mcfg["audio_name"],
                                    hidden=mcfg["hidden"], heads=mcfg["heads"], dropout=mcfg["dropout"],
                                    mode=args.stage))
    if args.stage == "fusion":  # warm start from single-modality stages
        for key, prefixes in (("init_text_ckpt", ("text_enc.", "text_proj.", "text_pool.")),
                              ("init_audio_ckpt", ("audio_enc.", "audio_proj.", "audio_pool.", "layer_logits"))):
            p = Path(tcfg.get(key, ""))
            if p.is_file():
                print(f"warm-start from {p}: {init_encoders_from(model, p, prefixes)} tensors")
    apply_audio_freezing(model, tcfg["freeze_audio_cnn"], tcfg["freeze_audio_layers"])
    model.to(device)
    print(f"total params: {count_params(model)/1e6:.1f}M")

    use_t, use_a = args.stage in ("text", "fusion"), args.stage in ("audio", "fusion")
    tok = load_tokenizer(mcfg["text_name"])
    mdir = Path(dcfg["manifest_dir"])

    def loader(split: str, train: bool) -> DataLoader:
        ds = MeldDataset(mdir / f"{split}.jsonl", tok, dcfg["max_text_len"], dcfg["max_audio_seconds"], train=train,
                         use_audio=use_a, use_text=use_t, augment_audio=dcfg["audio_augment"], limit=args.limit)
        return DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=train, num_workers=dcfg["num_workers"],
                          collate_fn=make_collate(tok.pad_token_id), drop_last=False)

    dl_train, dl_dev, dl_test = loader("train", True), loader("dev", False), loader("test", False)
    cw = class_weights(mdir / "train.jsonl", tcfg["class_weight_power"])
    cw = cw.to(device) if cw is not None else None

    enc_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if n.startswith(("text_enc.", "audio_enc.")) else head_params).append(p)
    opt = torch.optim.AdamW([{"params": enc_params, "lr": tcfg["lr_encoder"]},
                             {"params": head_params, "lr": tcfg["lr_head"]}], weight_decay=tcfg["weight_decay"])
    steps = max(1, epochs * len(dl_train) // tcfg["grad_accum"])
    sched = get_linear_schedule_with_warmup(opt, int(tcfg["warmup_ratio"] * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 autocast needs no scaler

    ckpt_path = Path(cfg["out"]["ckpt_dir"]) / stage_name / "best.pt"
    best, history = -1.0, []
    for epoch in range(epochs):
        frozen = args.stage == "fusion" and epoch < tcfg["freeze_encoders_epochs"]
        for enc in (model.text_enc, model.audio_enc):
            if enc is not None:
                for p in enc.parameters():
                    p.requires_grad = (not frozen)
        if not frozen:
            apply_audio_freezing(model, tcfg["freeze_audio_cnn"], tcfg["freeze_audio_layers"])
        model.train()
        t0, run = time.time(), 0.0
        pbar = tqdm(dl_train, desc=f"epoch {epoch+1}/{epochs}" + (" (encoders frozen)" if frozen else ""))
        for i, batch in enumerate(pbar):
            b = to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(b.get("input_ids"), b.get("attention_mask"), b.get("input_values"), b.get("audio_lengths"))
                loss = compute_loss(out, b, cw, mcfg, tcfg["label_smoothing"], args.stage) / tcfg["grad_accum"]
            loss.backward()
            if (i + 1) % tcfg["grad_accum"] == 0 or i + 1 == len(dl_train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run = 0.98 * run + 0.02 * loss.item() * tcfg["grad_accum"] if i else loss.item()
            pbar.set_postfix(loss=f"{run:.3f}")
        dev = evaluate(model, dl_dev, device, amp)
        history.append({"epoch": epoch + 1, "dev_weighted_f1": dev["weighted_f1"], "dev_macro_f1": dev["macro_f1"],
                        "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch+1}: dev W-F1={dev['weighted_f1']:.4f} macro-F1={dev['macro_f1']:.4f}")
        if dev["weighted_f1"] > best:
            best = dev["weighted_f1"]
            save_checkpoint(model, ckpt_path, extra={"stage": args.stage, "dev": dev, "epoch": epoch + 1})

    # test with best checkpoint
    from ..models.build import load_checkpoint
    best_model, _ = load_checkpoint(ckpt_path, device)
    test = evaluate(best_model, dl_test, device, amp)
    res = {"stage": args.stage, "params_total": count_params(best_model), "best_dev_weighted_f1": best,
           "test": test, "history": history, "limit": args.limit}
    out_dir = Path(cfg["out"]["results_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stage_name}_metrics.json").write_text(json.dumps(res, indent=2))
    print(f"TEST  W-F1={test['weighted_f1']:.4f}  macro-F1={test['macro_f1']:.4f}  acc={test['accuracy']:.4f}")


if __name__ == "__main__":
    main()
