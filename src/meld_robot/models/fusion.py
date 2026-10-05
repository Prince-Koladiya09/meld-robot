"""Text + audio emotion model.

    text  --RoBERTa--> H_t (B,L,Dt) --proj--> (B,L,d) --\
                                                         cross-attn (both directions) --> attentive pooling --> p_t, p_a
    audio --WavLM----> H_a (B,T,Da) --proj--> (B,T,d) --/

    gate g = sigmoid(W [p_t ; p_a]);  fused = g * p_t + (1 - g) * p_a  -> MLP -> 7 emotions (+3 sentiments)

Auxiliary heads on p_t and p_a give per-modality predictions (extra training signal, and the
text-only view used at inference time to flag "words and tone disagree").

`mode` selects what is used: "text", "audio" or "fusion" (the three ablation stages).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..labels import NUM_EMOTIONS, NUM_SENTIMENTS


@dataclass
class ModelConfig:
    text_name: str = "roberta-base"
    audio_name: str = "microsoft/wavlm-base-plus"
    hidden: int = 256
    heads: int = 4
    dropout: float = 0.2
    mode: str = "fusion"  # text | audio | fusion

    def to_dict(self) -> dict:
        return asdict(self)


class AttentivePool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        s = self.score(x).squeeze(-1).masked_fill(~mask, float("-inf"))  # x:(B,N,d) mask:(B,N) True=valid
        w = torch.softmax(s, dim=-1).unsqueeze(-1)
        return (w * x).sum(1)


class CrossBlock(nn.Module):
    """Queries attend to the other modality; residual + norm."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, q, q_mask, kv, kv_mask):
        out, _ = self.attn(q, kv, kv, key_padding_mask=~kv_mask, need_weights=False)
        return self.norm(q + out).masked_fill(~q_mask.unsqueeze(-1), 0.0)


def _mlp(d_in: int, d_hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_hidden, d_out))


class FusionEmotionModel(nn.Module):
    def __init__(self, text_enc: Optional[nn.Module], audio_enc: Optional[nn.Module], cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden
        self.text_enc, self.audio_enc = text_enc, audio_enc
        use_t, use_a = cfg.mode in ("text", "fusion"), cfg.mode in ("audio", "fusion")
        if (use_t and text_enc is None) or (use_a and audio_enc is None):
            raise ValueError(f"mode={cfg.mode!r} needs the corresponding encoder(s)")

        if use_t:
            self.text_proj = nn.Sequential(nn.Linear(text_enc.config.hidden_size, d), nn.LayerNorm(d))
            self.text_pool = AttentivePool(d)
        if use_a:
            self.layer_logits = nn.Parameter(torch.zeros(audio_enc.config.num_hidden_layers + 1))
            self.audio_proj = nn.Sequential(nn.Linear(audio_enc.config.hidden_size, d), nn.LayerNorm(d))
            self.audio_pool = AttentivePool(d)
        if cfg.mode == "fusion":
            self.t2a = CrossBlock(d, cfg.heads, cfg.dropout)
            self.a2t = CrossBlock(d, cfg.heads, cfg.dropout)
            self.gate = nn.Linear(2 * d, 1)
            self.text_head = _mlp(d, d, NUM_EMOTIONS, cfg.dropout)    # auxiliary, per-modality
            self.audio_head = _mlp(d, d, NUM_EMOTIONS, cfg.dropout)
        self.fused_head = _mlp(d, d, NUM_EMOTIONS, cfg.dropout)
        self.sent_head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(d, NUM_SENTIMENTS))
        self.drop = nn.Dropout(cfg.dropout)

    # ---- encoders -----------------------------------------------------------------------
    def encode_text(self, input_ids, attention_mask):
        h = self.text_enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.text_proj(h), attention_mask.bool()

    def encode_audio(self, input_values, audio_lengths):
        out = self.audio_enc(input_values, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)  # (layers+1, B, T, D)
        w = torch.softmax(self.layer_logits, 0).view(-1, 1, 1, 1)
        h = (w * hs).sum(0)
        T = h.shape[1]
        frame_len = self.audio_enc._get_feat_extract_output_lengths(audio_lengths).clamp(min=1, max=T)
        mask = torch.arange(T, device=h.device)[None, :] < frame_len[:, None]
        return self.audio_proj(h), mask

    # ---- forward ------------------------------------------------------------------------
    def forward(self, input_ids=None, attention_mask=None, input_values=None, audio_lengths=None) -> dict:
        mode, out = self.cfg.mode, {}
        ht = ha = mt = ma = None
        if mode in ("text", "fusion"):
            ht, mt = self.encode_text(input_ids, attention_mask)
        if mode in ("audio", "fusion"):
            ha, ma = self.encode_audio(input_values, audio_lengths)

        if mode == "text":
            p = self.drop(self.text_pool(ht, mt))
            out.update(logits=self.fused_head(p), sent_logits=self.sent_head(p),
                       gate=torch.ones(p.size(0), 1, device=p.device))
        elif mode == "audio":
            p = self.drop(self.audio_pool(ha, ma))
            out.update(logits=self.fused_head(p), sent_logits=self.sent_head(p),
                       gate=torch.zeros(p.size(0), 1, device=p.device))
        else:
            ct = self.t2a(ht, mt, ha, ma)  # text enriched with audio
            ca = self.a2t(ha, ma, ht, mt)  # audio enriched with text
            pt, pa = self.text_pool(ct, mt), self.audio_pool(ca, ma)
            g = torch.sigmoid(self.gate(torch.cat([pt, pa], -1)))  # g -> 1 means "trust text"
            fused = self.drop(g * pt + (1 - g) * pa)
            out.update(logits=self.fused_head(fused), sent_logits=self.sent_head(fused), gate=g,
                       text_logits=self.text_head(self.drop(pt)), audio_logits=self.audio_head(self.drop(pa)))
        return out


def count_params(module: nn.Module) -> int:
    """Total learned parameters (not only trainable ones)."""
    return sum(p.numel() for p in module.parameters())
