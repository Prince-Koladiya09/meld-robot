import pytest
import torch
from transformers import RobertaConfig, WavLMConfig

from meld_robot.labels import NUM_EMOTIONS, NUM_SENTIMENTS
from meld_robot.models.build import build_model, load_checkpoint, save_checkpoint
from meld_robot.models.fusion import ModelConfig, count_params
from meld_robot.train.train import compute_loss

TXT = RobertaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=64, max_position_embeddings=64)
AUD = WavLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                  conv_dim=(16, 16, 16), conv_stride=(5, 4, 2), conv_kernel=(10, 3, 2), num_feat_extract_layers=3,
                  num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, num_buckets=16, max_bucket_distance=32,
                  vocab_size=32)


def batch(B=3):
    ids = torch.randint(3, 100, (B, 9))
    mask = torch.ones(B, 9, dtype=torch.long); mask[0, 6:] = 0
    wav = torch.randn(B, 8000)
    lens = torch.tensor([8000, 5000, 6400])[:B]
    for i, l in enumerate(lens): wav[i, l:] = 0
    return dict(input_ids=ids, attention_mask=mask, input_values=wav, audio_lengths=lens,
                labels=torch.tensor([0, 4, 6])[:B], sent_labels=torch.tensor([0, 1, 2])[:B])


@pytest.mark.parametrize("mode", ["text", "audio", "fusion"])
def test_forward_backward_shapes(mode):
    m = build_model(ModelConfig(hidden=32, heads=2, mode=mode), pretrained=False, text_cfg=TXT, audio_cfg=AUD)
    b = batch()
    out = m(b["input_ids"], b["attention_mask"], b["input_values"], b["audio_lengths"])
    assert out["logits"].shape == (3, NUM_EMOTIONS)
    assert out["sent_logits"].shape == (3, NUM_SENTIMENTS)
    assert out["gate"].shape == (3, 1)
    if mode == "fusion":
        assert out["text_logits"].shape == out["audio_logits"].shape == (3, NUM_EMOTIONS)
        assert ((out["gate"] >= 0) & (out["gate"] <= 1)).all()
    loss = compute_loss(out, b, None, {"sentiment_weight": 0.3, "aux_weight": 0.3}, 0.05, mode)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_padding_does_not_change_prediction():
    """Zero-padding the audio / text must not alter the output of a given sample (masking works)."""
    m = build_model(ModelConfig(hidden=32, heads=2, mode="fusion"), pretrained=False, text_cfg=TXT, audio_cfg=AUD).eval()
    ids = torch.randint(3, 100, (1, 7)); mask = torch.ones(1, 7, dtype=torch.long)
    wav = torch.randn(1, 6400); lens = torch.tensor([6400])
    with torch.no_grad():
        a = m(ids, mask, wav, lens)["logits"]
        ids2 = torch.cat([ids, torch.ones(1, 4, dtype=torch.long)], 1)
        mask2 = torch.cat([mask, torch.zeros(1, 4, dtype=torch.long)], 1)
        b = m(ids2, mask2, wav, lens)["logits"]  # extra masked text tokens
    assert torch.allclose(a, b, atol=1e-4)


def test_checkpoint_roundtrip(tmp_path, monkeypatch):
    import meld_robot.models.build as B
    m = build_model(ModelConfig(hidden=32, heads=2, mode="fusion"), pretrained=False, text_cfg=TXT, audio_cfg=AUD)
    save_checkpoint(m, tmp_path / "c.pt", {"x": 1})
    # load_checkpoint rebuilds encoders from hub configs; swap in the tiny ones for an offline test
    orig = B._load_encoder
    monkeypatch.setattr(B, "_load_encoder", lambda name, pre, cfg=None: orig(name, pre, TXT if "roberta" in name else AUD))
    m2, extra = load_checkpoint(tmp_path / "c.pt")
    assert extra == {"x": 1} and count_params(m) == count_params(m2)
