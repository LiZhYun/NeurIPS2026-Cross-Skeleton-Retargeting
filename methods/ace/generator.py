"""ACE generator: predicts the target latent from the source motion and the previous
target chunk.

  Input:  z_src [B, 8, 256], prev_z_tgt [B, 8, 256], source/target skeleton id,
          source/target graph embedding
  Output: z_pred [B, 8, 256], the target latent for the current chunk

Architecture:
  - Concatenate (z_src, prev_z_tgt) into a [B, 16, 256] sequence
  - A positional embedding separates the source half (positions 0-7) from the
    previous-target half (positions 8-15)
  - Skeleton and graph conditioning is added at the input
  - 6 layers x 8 heads x 512 transformer encoder
  - The output head reads the 8 tokens at positions 8-15, the predicted target half

At the start of a clip there is no previous target chunk; a target-conditioned START
vector stands in for it (ACEStartTokens below).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from methods.common.layers import SinusoidalPositionalEncoding


class ACEStartTokens(nn.Module):
    """The vector that stands in for the previous chunk at the start of a clip.

    START[tgt_skel] = mean(z of the target skeleton's own clips) + learned_offset[tgt_skel]

    Skeletons at id >= n_train_skels get a zero offset, so a skeleton the model never
    trained on starts from its own latent mean alone. The mean is computed once from the
    latent cache and passed in at construction.

    Args:
        n_skels: total number of skeleton ids
        n_train_skels: how many of those were trained; the rest get a zero offset
        codebook_dim: 256
        n_tokens: 8
        z_means: Tensor[n_skels, n_tokens, codebook_dim], per-skeleton mean of the real
                 target latents
    """
    def __init__(self, n_skels, n_train_skels, codebook_dim, n_tokens, z_means):
        super().__init__()
        assert z_means.shape == (n_skels, n_tokens, codebook_dim), \
            f"z_means shape {z_means.shape} != ({n_skels}, {n_tokens}, {codebook_dim})"
        self.register_buffer('z_means', z_means.float())                  # frozen pre-computed
        self.n_skels = n_skels
        self.n_train_skels = n_train_skels
        # A learned offset per skeleton, starting at zero so the START vector begins
        # as the skeleton's own mean
        self.offset = nn.Parameter(torch.zeros(n_skels, n_tokens, codebook_dim))

    def forward(self, tgt_skel_id):
        """tgt_skel_id: [B] LongTensor → returns [B, n_tokens, codebook_dim]."""
        mean = self.z_means[tgt_skel_id]                                  # [B, n_tokens, d]
        offset = self.offset[tgt_skel_id]                                 # [B, n_tokens, d]
        # Zero the offset for skeletons beyond the trained set
        held_out = (tgt_skel_id >= self.n_train_skels).unsqueeze(-1).unsqueeze(-1)
        offset = torch.where(held_out, torch.zeros_like(offset), offset)
        return mean + offset


class ACEGenerator(nn.Module):
    """Predicts one chunk of the target animal's motion, in the tokenizer's own space."""

    def __init__(self, codebook_dim=256, d_model=512, n_layers=6, n_heads=8,
                 dim_ff=2048, dropout=0.1, n_tokens=8,
                 n_skels=70, d_skel_id_emb=128, d_graph=128):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.d_model = d_model
        self.n_tokens = n_tokens

        # Token projection
        self.token_proj = nn.Linear(codebook_dim, d_model)
        # Positional encoding over the 16-token sequence (8 source, 8 previous target)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=2 * n_tokens)
        # Marks each token as belonging to the source half or the previous-target half
        self.segment_emb = nn.Embedding(2, d_model)

        # Skeleton and graph conditioning, added at the input
        self.skel_id_emb = nn.Embedding(n_skels, d_skel_id_emb)
        self.skel_proj = nn.Sequential(
            nn.Linear(2 * (d_skel_id_emb + d_graph), d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Projects the predicted target tokens back into codebook space
        self.out_proj = nn.Linear(d_model, codebook_dim)

    def forward(self, z_src, prev_z_tgt, src_skel_id, tgt_skel_id, src_graph, tgt_graph):
        """
        z_src:        [B, n_tokens=8, codebook_dim=256]
        prev_z_tgt:   [B, n_tokens=8, codebook_dim=256]
        src_skel_id:  [B]
        tgt_skel_id:  [B]
        src_graph:    [B, d_graph=128]
        tgt_graph:    [B, d_graph=128]

        Returns:
          z_pred:     [B, n_tokens=8, codebook_dim=256]
        """
        B = z_src.shape[0]
        T = self.n_tokens

        # Source then previous target along the token axis, giving [B, 16, codebook_dim]
        seq = torch.cat([z_src, prev_z_tgt], dim=1)                       # [B, 16, 256]
        h = self.token_proj(seq)                                          # [B, 16, 512]

        # Positional + segment embeddings
        h = self.pos_enc(h)
        seg_ids = torch.cat([
            torch.zeros(T, device=h.device, dtype=torch.long),
            torch.ones(T, device=h.device, dtype=torch.long),
        ])                                                                # [16]
        seg = self.segment_emb(seg_ids).unsqueeze(0)                      # [1, 16, 512]
        h = h + seg

        # Skeleton and graph conditioning
        src_id_emb = self.skel_id_emb(src_skel_id)                        # [B, 128]
        tgt_id_emb = self.skel_id_emb(tgt_skel_id)
        skel_in = torch.cat([src_id_emb, src_graph, tgt_id_emb, tgt_graph], dim=-1)
        skel_h = self.skel_proj(skel_in)                                  # [B, 512]
        h = h + skel_h.unsqueeze(1)                                       # broadcast over 16 tokens

        # Transformer encoder
        h = self.encoder(h)                                               # [B, 16, 512]

        # The second half of the sequence has attended to all of the source and all of the
        # previous target, so it is read as the prediction
        h_target = h[:, T:, :]                                            # [B, 8, 512]
        z_pred = self.out_proj(h_target)                                  # [B, 8, 256]
        return z_pred


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    torch.manual_seed(0)
    G = ACEGenerator()
    n = count_parameters(G)
    print(f"Generator params: {n:,} ({n/1e6:.1f}M)")

    B = 4
    z_src = torch.randn(B, 8, 256)
    prev_z_tgt = torch.randn(B, 8, 256)
    src_id = torch.randint(0, 70, (B,))
    tgt_id = torch.randint(0, 70, (B,))
    src_g = torch.randn(B, 128)
    tgt_g = torch.randn(B, 128)
    z_pred = G(z_src, prev_z_tgt, src_id, tgt_id, src_g, tgt_g)
    print(f"z_pred shape: {z_pred.shape} (expect [{B}, 8, 256])")
    print(f"z_pred stats: mean={z_pred.mean():.4f}, std={z_pred.std():.4f}")

    # START vectors
    z_means = torch.randn(70, 8, 256)
    starts = ACEStartTokens(n_skels=70, n_train_skels=60, codebook_dim=256, n_tokens=8, z_means=z_means)
    n_starts = sum(p.numel() for p in starts.parameters() if p.requires_grad)
    print(f"ACEStartTokens learnable params: {n_starts:,}  (expected 70*8*256={70*8*256:,})")
    test_ids = torch.tensor([5, 10, 65, 30])  # 5, 10, 30 trained; 65 beyond the trained set
    out = starts(test_ids)
    print(f"START output shape: {out.shape} (expect [4, 8, 256])")
    # A skeleton beyond the trained set should get exactly the mean
    untrained_diff = (out[2] - z_means[65]).abs().max().item()
    print(f"id 65 was not trained, so its START vector should equal the mean: {untrained_diff:.6e}")
