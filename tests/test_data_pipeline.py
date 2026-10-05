"""Runs scripts/prepare_meld.py on a synthetic MELD-shaped tree (real ffmpeg), then exercises the dataset,
collate, evaluate() and one optimisation step of the model with tiny random encoders."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from meld_robot.data import MeldDataset, make_collate, read_manifest
from meld_robot.labels import EMOTIONS
from meld_robot.models.build import build_model
from meld_robot.models.fusion import ModelConfig
from meld_robot.train.train import class_weights, compute_loss, evaluate
from tests.test_model import AUD, TXT

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

ROWS = {  # dialogue -> [(speaker, text, emotion, sentiment)]
    0: [("Joey", "Hey, how you doin'?", "joy", "positive"), ("Rachel", "I'm fine.", "neutral", "neutral"),
        ("Joey", "You don't look fine!", "surprise", "negative"), ("Rachel", "Leave me alone.", "anger", "negative")],
    1: [("Ross", "We were on a break!", "anger", "negative"), ("Rachel", "Ugh.", "disgust", "negative"),
        ("Ross", "I'm so sorry.", "sadness", "negative")],
}


class FakeTok:
    pad_token_id = 1
    def __call__(self, a, b=None, truncation=None, max_length=128, **kw):
        ids = [0] + [3 + (hash(w) % 90) for w in a.split()] + [2]
        if b:
            ids += [3 + (hash(w) % 90) for w in b.split()] + [2]
        return {"input_ids": ids[:max_length]}


@pytest.fixture(scope="module")
def meld(tmp_path_factory):
    root = tmp_path_factory.mktemp("meld")
    raw = root / "MELD.Raw"
    for split, subdir in [("train", "train/train_splits"), ("dev", "dev_splits_complete"), ("test", "output_repeated_splits_test")]:
        (raw / subdir).mkdir(parents=True)
        lines = ["Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime"]
        n = 1
        for d, utts in ROWS.items():
            for u, (spk, text, emo, sent) in enumerate(utts):
                lines.append(f'{n},"{text}",{spk},{emo},{sent},{d},{u},1,1,00:00:00,00:00:01'); n += 1
                if not (split == "test" and d == 1 and u == 2):       # one missing video -> must be skipped
                    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency={200 + 40 * u}:duration=1.2",
                                    "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1.2", "-shortest", "-pix_fmt", "yuv420p",
                                    str(raw / subdir / f"dia{d}_utt{u}.mp4")], check=True)
        (raw / f"{split}_sent_emo.csv").write_text("\n".join(lines))
    (root / "data.yaml").write_text(f"meld_root: {raw}\nout_dir: {root / 'processed'}\nsample_rate: 16000\ncontext_turns: 2\nmax_audio_seconds: 6.0\n")
    subprocess.run([sys.executable, "scripts/prepare_meld.py", "--config", str(root / "data.yaml"), "--workers", "2"], check=True)
    return root / "processed"


def test_prepare_outputs(meld):
    summ = json.loads((meld / "summary.json").read_text())
    assert summ["train"]["kept"] == 7 and summ["train"]["skipped"] == 0
    assert summ["test"]["kept"] == 6 and summ["test"]["skipped"] == 1       # missing video skipped, not crashed
    rows = read_manifest(meld / "train.jsonl")
    r = next(x for x in rows if x["id"] == "train/dia0_utt3")
    assert len(r["context"]) == 2                                            # context_turns = 2
    assert r["context"][-1] == "Joey: You don't look fine!" and r["emotion"] == "anger"
    first = next(x for x in rows if x["id"] == "train/dia0_utt0")
    assert first["context"] == []
    assert Path(r["wav"]).exists()


def test_dataset_collate_and_train_step(meld):
    ds = MeldDataset(meld / "train.jsonl", FakeTok(), max_audio_seconds=6.0, train=True, augment_audio=True)
    batch = make_collate(1)([ds[i] for i in range(4)])
    assert batch["input_ids"].shape[0] == 4 and batch["input_values"].shape[0] == 4
    assert batch["audio_lengths"].max() == batch["input_values"].shape[1]
    assert abs(float(batch["input_values"][0, : batch["audio_lengths"][0]].std()) - 1.0) < 0.3   # normalised audio

    m = build_model(ModelConfig(hidden=32, heads=2, mode="fusion"), pretrained=False, text_cfg=TXT, audio_cfg=AUD)
    cw = class_weights(meld / "train.jsonl", 0.5)
    assert cw.shape == (len(EMOTIONS),) and abs(float(cw.mean()) - 1) < 1e-5
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(30):                                   # overfit a tiny batch: loss must go down
        out = m(batch["input_ids"], batch["attention_mask"], batch["input_values"], batch["audio_lengths"])
        loss = compute_loss(out, batch, cw, {"sentiment_weight": 0.3, "aux_weight": 0.3}, 0.0, "fusion")
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    assert min(losses[-5:]) < losses[0] * 0.6

    dl = DataLoader(MeldDataset(meld / "test.jsonl", FakeTok()), batch_size=3, collate_fn=make_collate(1))
    rep = evaluate(m, dl, "cpu", amp=False)
    assert rep["n"] == 6 and 0 <= rep["weighted_f1"] <= 1 and len(rep["confusion_matrix"]) == 7
    assert 0 <= rep["mean_text_gate"] <= 1 and "ece" in rep


def test_context_truncation_drops_oldest_turn_first():
    """Regression: right-side truncation (HF default) would have cut the NEWEST context turn."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + [f"w{i}" for i in range(30)])}
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")); t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
                                          special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")
    tok.truncation_side = "left"                      # what load_tokenizer() sets
    inv = {v: k for k, v in vocab.items()}
    enc = tok("w1 w2", "w10 w11 w12 w13 w20 w21 w22 w23", truncation="only_second", max_length=8)
    assert [inv[i] for i in enc["input_ids"]] == ["[CLS]", "w1", "w2", "[SEP]", "w21", "w22", "w23", "[SEP]"]
