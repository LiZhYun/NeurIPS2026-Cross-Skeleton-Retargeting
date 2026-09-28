"""The flow transformer MoReFlow trains on top of the per-skeleton motion tokenizers.

  - 6 transformer encoder layers x 8 heads, d_model 512, feed-forward 2048, dropout 0.1
  - Inputs: the interpolant in codebook space, the flow time, the source and target
            skeleton ids and graph features, and the condition (its family and its value,
            with a dropout mask for classifier-free guidance)
  - Output: the velocity, projected straight back into codebook space

Reference: arXiv:2509.25600, section 4.2 and Table 3.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.common.layers import SinusoidalPositionalEncoding


class TimeEmbedding(nn.Module):
    """Turns the flow time, a number between 0 and 1, into something the model can read."""
    def __init__(self, d_model, max_period=1000.0):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, q):
        # q: [B] in [0, 1]
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, dtype=q.dtype, device=q.device) / half
        )
        args = q[:, None] * freqs[None, :] * 2.0 * math.pi  # broadcast freqs into time
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, d_model]
        if self.d_model % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


class DiscreteFlowTransformer(nn.Module):
    """Predicts the flow velocity in the tokenizer's codebook space.

    Inputs:
      z_q:           [B, T=8, codebook_dim=256]  interpolant in codebook space
      q:             [B]                          flow time ∈ [0, 1]
      src_skel_id:   [B]                          ∈ {0..69}
      tgt_skel_id:   [B]                          ∈ {0..69}
      src_graph:     [B, d_graph=128]             morphology features (encoded by SkelGraphEncoder)
      tgt_graph:     [B, d_graph=128]
      cond_type:     [B]                          ∈ {0..5}
      cond_vec:      [B, d_cond_padded=24]        condition value (zero-padded)
      cond_mask:     [B]                          1 = drop the condition

    Output:
      v_psi:         [B, T=8, codebook_dim=256]   continuous velocity (paper Eq. 11 target)
    """
    def __init__(self, codebook_dim=256, d_model=512, n_layers=6, n_heads=8,
                 dim_ff=2048, dropout=0.1, max_seq_len=8,
                 n_skels=70, d_skel_id_emb=128, d_graph=128, d_cond_padded=24,
                 n_cond_types=6, d_cond_type_emb=64, d_cond_value=64):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token projection, from codebook space into the model's width
        self.token_proj = nn.Linear(codebook_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)

        # Time embedding
        self.time_emb = TimeEmbedding(d_model)

        # Skeleton embedding
        self.skel_id_emb = nn.Embedding(n_skels, d_skel_id_emb)
        self.skel_proj = nn.Sequential(
            nn.Linear(2 * (d_skel_id_emb + d_graph), d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Condition embedding
        self.cond_type_emb = nn.Embedding(n_cond_types, d_cond_type_emb)
        self.cond_value_mlp = nn.Sequential(
            nn.Linear(d_cond_padded, d_cond_value),
            nn.SiLU(),
            nn.Linear(d_cond_value, d_cond_value),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(d_cond_type_emb + d_cond_value, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # The head projects straight back to codebook space and returns the velocity
        self.vocab_head = nn.Linear(d_model, codebook_dim)

    def forward(self, z_q, q, src_skel_id, tgt_skel_id, src_graph, tgt_graph,
                cond_type, cond_vec, cond_mask):
        """Inputs are as listed in the class docstring. Returns the velocity, [B, T, codebook_dim]."""
        B, T, _ = z_q.shape

        h = self.token_proj(z_q)                              # [B, T, d_model]
        h = self.pos_enc(h)

        # The conditioning is one vector, added to every token
        time_h = self.time_emb(q)                              # [B, d_model]

        src_id_emb = self.skel_id_emb(src_skel_id)             # [B, d_skel_id_emb]
        tgt_id_emb = self.skel_id_emb(tgt_skel_id)
        skel_in = torch.cat([src_id_emb, src_graph, tgt_id_emb, tgt_graph], dim=-1)
        skel_h = self.skel_proj(skel_in)                       # [B, d_model]

        ct_emb = self.cond_type_emb(cond_type)                 # [B, d_cond_type_emb]
        cv_emb = self.cond_value_mlp(cond_vec)                 # [B, d_cond_value]
        cond_h = self.cond_proj(torch.cat([ct_emb, cv_emb], dim=-1))  # [B, d_model]
        # Where the mask says to drop the condition, zero it out
        cond_h = cond_h * (1.0 - cond_mask.unsqueeze(-1).to(cond_h.dtype))

        cond_sum = (time_h + skel_h + cond_h).unsqueeze(1)     # [B, 1, d_model]
        h = h + cond_sum                                       # additive condition over all T

        h = self.encoder(h)                                    # [B, T, d_model]
        v_psi = self.vocab_head(h)                             # [B, T, codebook_dim]
        return v_psi


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    torch.manual_seed(0)
    model = DiscreteFlowTransformer()
    n = count_parameters(model)
    print(f"Params: {n:,} ({n/1e6:.1f}M)")

    # One forward pass
    B = 4
    z_q = torch.randn(B, 8, 256)
    q = torch.rand(B)
    src_id = torch.randint(0, 70, (B,))
    tgt_id = torch.randint(0, 70, (B,))
    src_graph = torch.randn(B, 128)
    tgt_graph = torch.randn(B, 128)
    cond_type = torch.randint(0, 6, (B,))
    cond_vec = torch.randn(B, 24)
    cond_mask = torch.zeros(B)
    cond_mask[0] = 1  # drop the condition for the first sample

    v = model(z_q, q, src_id, tgt_id, src_graph, tgt_graph, cond_type, cond_vec, cond_mask)
    print(f"v shape: {v.shape} (expected [{B}, 8, 256])")
    print(f"v stats: mean={v.mean():.4f}, std={v.std():.4f}")
    # Dropping the condition should change the answer, keeping it should not
    cond_mask_off = torch.zeros(B)
    v_no_drop = model(z_q, q, src_id, tgt_id, src_graph, tgt_graph, cond_type, cond_vec, cond_mask_off)
    diff_dropped = (v[0] - v_no_drop[0]).abs().mean().item()
    diff_undropped = (v[1] - v_no_drop[1]).abs().mean().item()
    print(f"Sample 0 (mask=1 vs 0) diff: {diff_dropped:.4f}")
    print(f"Sample 1 (mask=0 vs 0) diff: {diff_undropped:.4f}")
    assert diff_dropped > diff_undropped, "the guidance mask has no effect"
    print("The guidance mask has an effect")
