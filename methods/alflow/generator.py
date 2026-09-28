"""The three AL-Flow generators.

All three are conditional flow-matching generators over the per-skeleton token latents, and
all three receive the same two action labels that ANCHOR retrieves with: the coarse action
cluster and the fine exact action. They differ in what else they see.

  ALFlowGenerator          the two labels and the target skeleton. Generation starts from
                           noise; nothing about the source clip reaches the model.
  ALFlowSrcGenerator       adds the source clip's latent as eight extra tokens in the
                           sequence, plus the source skeleton's id and graph features. This
                           is every input ANCHOR's retrieval side has.
  ALFlowSrcGraphGenerator  the same, with the per-skeleton id embeddings removed, so both
                           skeletons are described only by their graph features. Because
                           nothing is looked up by id, a skeleton the model never trained on
                           goes through exactly the same path as one it did.

Every conditioning channel has its own null index at 0 and its own dropout mask, so each can
be dropped independently for classifier-free guidance. In the source-conditioned variants
the source is dropped by swapping in a learned null token.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from methods.alflow.layers import TimeEmbedding, count_parameters
from methods.common.layers import SinusoidalPositionalEncoding


class ALFlowGenerator(nn.Module):
    """Generates a target latent from the action labels and the target skeleton."""

    def __init__(self, codebook_dim=256, d_model=512, n_layers=6, n_heads=8,
                 dim_ff=2048, dropout=0.1, max_seq_len=8,
                 n_skels=70, d_skel_id_emb=128, d_graph=128,
                 n_clusters=11, d_cluster_emb=128,
                 n_exact_actions=125, d_exact_emb=128):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Tokens and position
        self.token_proj = nn.Linear(codebook_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        self.time_emb = TimeEmbedding(d_model)

        # Target skeleton conditioning
        self.skel_id_emb = nn.Embedding(n_skels, d_skel_id_emb)
        self.skel_proj = nn.Sequential(
            nn.Linear(d_skel_id_emb + d_graph, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Coarse action cluster; index 0 is the null label
        self.cluster_emb = nn.Embedding(n_clusters, d_cluster_emb)
        self.cluster_proj = nn.Sequential(
            nn.Linear(d_cluster_emb, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Fine exact action; index 0 is the null label
        self.exact_emb = nn.Embedding(n_exact_actions, d_exact_emb)
        self.exact_proj = nn.Sequential(
            nn.Linear(d_exact_emb, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.vocab_head = nn.Linear(d_model, codebook_dim)

    def forward(self, z_q, q, tgt_skel_id, tgt_graph,
                cluster_id, exact_id,
                cluster_mask=None, exact_mask=None):
        """
        z_q:          [B, T=8, codebook_dim]
        q:            [B] flow time in [0, 1]
        tgt_skel_id:  [B] long
        tgt_graph:    [B, d_graph]
        cluster_id:   [B] long, in {0..n_clusters-1} (0 = null)
        exact_id:     [B] long, in {0..n_exact_actions-1} (0 = null)
        cluster_mask: [B] bool, True drops the cluster label to null
        exact_mask:   [B] bool, True drops the exact action to null
        Returns v:    [B, T=8, codebook_dim]
        """
        B, T, _ = z_q.shape
        h = self.token_proj(z_q)
        h = self.pos_enc(h)

        time_h = self.time_emb(q)
        skel_in = torch.cat([self.skel_id_emb(tgt_skel_id), tgt_graph], dim=-1)
        skel_h = self.skel_proj(skel_in)

        if cluster_mask is not None:
            cluster_id = torch.where(cluster_mask, torch.zeros_like(cluster_id), cluster_id)
        cluster_h = self.cluster_proj(self.cluster_emb(cluster_id))

        if exact_mask is not None:
            exact_id = torch.where(exact_mask, torch.zeros_like(exact_id), exact_id)
        exact_h = self.exact_proj(self.exact_emb(exact_id))

        # All conditioning is added to every token
        cond_sum = (time_h + skel_h + cluster_h + exact_h).unsqueeze(1)
        h = h + cond_sum

        h = self.encoder(h)
        return self.vocab_head(h)


class ALFlowSrcGenerator(nn.Module):
    """AL-Flow plus the source clip: its latent, its skeleton id and its graph features.

      16-token sequence = 8 source tokens then 8 target tokens
      global conditioning = time + target skeleton + source skeleton + cluster + exact action
      the output head reads the last 8 (target) positions
    """

    def __init__(self, codebook_dim=256, d_model=512, n_layers=6, n_heads=8,
                 dim_ff=2048, dropout=0.1, max_seq_len=8,
                 n_skels=70, d_skel_id_emb=128, d_graph=128,
                 n_clusters=11, d_cluster_emb=128,
                 n_exact_actions=125, d_exact_emb=128):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len  # tokens per skeleton (8)

        # Token projection, shared by the source and target halves
        self.token_proj = nn.Linear(codebook_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        self.time_emb = TimeEmbedding(d_model)

        # Marks each token as source or target within the concatenated sequence
        self.tok_type_emb = nn.Embedding(2, d_model)  # 0 = source, 1 = target

        # Target skeleton conditioning
        self.tgt_skel_id_emb = nn.Embedding(n_skels, d_skel_id_emb)
        self.tgt_skel_proj = nn.Sequential(
            nn.Linear(d_skel_id_emb + d_graph, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Source skeleton conditioning, with its own embedding
        self.src_skel_id_emb = nn.Embedding(n_skels, d_skel_id_emb)
        self.src_skel_proj = nn.Sequential(
            nn.Linear(d_skel_id_emb + d_graph, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.cluster_emb = nn.Embedding(n_clusters, d_cluster_emb)
        self.cluster_proj = nn.Sequential(
            nn.Linear(d_cluster_emb, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.exact_emb = nn.Embedding(n_exact_actions, d_exact_emb)
        self.exact_proj = nn.Sequential(
            nn.Linear(d_exact_emb, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Learned null token, swapped in when the source is dropped
        self.src_null_token = nn.Parameter(torch.randn(max_seq_len, codebook_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.vocab_head = nn.Linear(d_model, codebook_dim)

    def forward(self, z_q, q, tgt_skel_id, tgt_graph,
                src_z, src_skel_id, src_graph,
                cluster_id, exact_id,
                cluster_mask=None, exact_mask=None, src_mask=None):
        """
        z_q:          [B, T=8, codebook_dim]  the noisy target latent at flow time q
        q:            [B] flow time in [0, 1]
        tgt_skel_id:  [B] long
        tgt_graph:    [B, d_graph]
        src_z:        [B, T=8, codebook_dim]  the source clip's latent
        src_skel_id:  [B] long
        src_graph:    [B, d_graph]
        cluster_id:   [B] long, in {0..n_clusters-1} (0 = null)
        exact_id:     [B] long, in {0..n_exact_actions-1} (0 = null)
        cluster_mask: [B] bool, True drops the cluster label to null
        exact_mask:   [B] bool, True drops the exact action to null
        src_mask:     [B] bool, True replaces the source with the null token
        Returns v:    [B, T=8, codebook_dim], the velocity for the target tokens
        """
        B, T, _ = z_q.shape

        # Dropping the source replaces it with the learned null token
        if src_mask is not None:
            null_expanded = self.src_null_token.unsqueeze(0).expand(B, -1, -1)
            src_z = torch.where(src_mask.view(B, 1, 1), null_expanded, src_z)

        # Project both halves, add the positional encoding, then mark source and target
        src_h = self.token_proj(src_z)
        tgt_h = self.token_proj(z_q)
        src_h = self.pos_enc(src_h)
        tgt_h = self.pos_enc(tgt_h)
        src_h = src_h + self.tok_type_emb(torch.zeros(B, dtype=torch.long, device=z_q.device)).unsqueeze(1)
        tgt_h = tgt_h + self.tok_type_emb(torch.ones(B, dtype=torch.long, device=z_q.device)).unsqueeze(1)

        # [source, target] along the token axis, giving [B, 2T, d_model]
        h = torch.cat([src_h, tgt_h], dim=1)

        # Global conditioning
        time_h = self.time_emb(q)

        tgt_skel_in = torch.cat([self.tgt_skel_id_emb(tgt_skel_id), tgt_graph], dim=-1)
        tgt_skel_h = self.tgt_skel_proj(tgt_skel_in)

        src_skel_in = torch.cat([self.src_skel_id_emb(src_skel_id), src_graph], dim=-1)
        src_skel_h = self.src_skel_proj(src_skel_in)

        if cluster_mask is not None:
            cluster_id = torch.where(cluster_mask, torch.zeros_like(cluster_id), cluster_id)
        cluster_h = self.cluster_proj(self.cluster_emb(cluster_id))

        if exact_mask is not None:
            exact_id = torch.where(exact_mask, torch.zeros_like(exact_id), exact_id)
        exact_h = self.exact_proj(self.exact_emb(exact_id))

        cond_sum = (time_h + tgt_skel_h + src_skel_h + cluster_h + exact_h).unsqueeze(1)
        h = h + cond_sum  # added to all 2T tokens

        h = self.encoder(h)

        # The head reads the target half only
        target_h = h[:, T:, :]  # [B, T, d_model]
        return self.vocab_head(target_h)


class ALFlowSrcGraphGenerator(nn.Module):
    """AL-Flow-Src with no skeleton id embeddings: both skeletons are described only by
    their graph features, so a skeleton the model never trained on takes the same path as
    one it did.
    """

    def __init__(self, codebook_dim=256, d_model=512, n_layers=6, n_heads=8,
                 dim_ff=2048, dropout=0.1, max_seq_len=8,
                 d_graph=128,
                 n_clusters=11, d_cluster_emb=128,
                 n_exact_actions=125, d_exact_emb=128):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token projection, shared by the source and target halves
        self.token_proj = nn.Linear(codebook_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        self.time_emb = TimeEmbedding(d_model)

        # Graph features alone may not separate the halves, so the token type is explicit
        self.tok_type_emb = nn.Embedding(2, d_model)  # 0 = source, 1 = target

        # Target skeleton conditioning, graph only
        self.tgt_skel_proj = nn.Sequential(
            nn.Linear(d_graph, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Source skeleton conditioning, graph only
        self.src_skel_proj = nn.Sequential(
            nn.Linear(d_graph, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.cluster_emb = nn.Embedding(n_clusters, d_cluster_emb)
        self.cluster_proj = nn.Sequential(
            nn.Linear(d_cluster_emb, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.exact_emb = nn.Embedding(n_exact_actions, d_exact_emb)
        self.exact_proj = nn.Sequential(
            nn.Linear(d_exact_emb, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Learned null token, swapped in when the source is dropped
        self.src_null_token = nn.Parameter(torch.randn(max_seq_len, codebook_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.vocab_head = nn.Linear(d_model, codebook_dim)

    def forward(self, z_q, q, tgt_graph,
                src_z, src_graph,
                cluster_id, exact_id,
                cluster_mask=None, exact_mask=None, src_mask=None):
        """
        z_q:          [B, T=8, codebook_dim]
        q:            [B] flow time in [0, 1]
        tgt_graph:    [B, d_graph]   from the graph encoder on the target skeleton
        src_z:        [B, T=8, codebook_dim]
        src_graph:    [B, d_graph]   from the graph encoder on the source skeleton
        cluster_id, exact_id: [B] long
        cluster_mask, exact_mask, src_mask: [B] bool, drop to null when True
        Returns v:    [B, T=8, codebook_dim]
        """
        B, T, _ = z_q.shape

        if src_mask is not None:
            null_expanded = self.src_null_token.unsqueeze(0).expand(B, -1, -1)
            src_z = torch.where(src_mask.view(B, 1, 1), null_expanded, src_z)

        src_h = self.token_proj(src_z)
        tgt_h = self.token_proj(z_q)
        src_h = self.pos_enc(src_h)
        tgt_h = self.pos_enc(tgt_h)
        src_h = src_h + self.tok_type_emb(torch.zeros(B, dtype=torch.long, device=z_q.device)).unsqueeze(1)
        tgt_h = tgt_h + self.tok_type_emb(torch.ones(B, dtype=torch.long, device=z_q.device)).unsqueeze(1)
        h = torch.cat([src_h, tgt_h], dim=1)

        # Global conditioning: time + target graph + source graph + cluster + exact action
        time_h = self.time_emb(q)
        tgt_skel_h = self.tgt_skel_proj(tgt_graph)
        src_skel_h = self.src_skel_proj(src_graph)

        if cluster_mask is not None:
            cluster_id = torch.where(cluster_mask, torch.zeros_like(cluster_id), cluster_id)
        cluster_h = self.cluster_proj(self.cluster_emb(cluster_id))

        if exact_mask is not None:
            exact_id = torch.where(exact_mask, torch.zeros_like(exact_id), exact_id)
        exact_h = self.exact_proj(self.exact_emb(exact_id))

        cond_sum = (time_h + tgt_skel_h + src_skel_h + cluster_h + exact_h).unsqueeze(1)
        h = h + cond_sum

        h = self.encoder(h)

        target_h = h[:, T:, :]
        return self.vocab_head(target_h)


if __name__ == '__main__':
    torch.manual_seed(42)

    print("== AL-Flow ==")
    G = ALFlowGenerator(n_skels=70, n_clusters=11, n_exact_actions=125,
                        d_model=512, n_layers=6, n_heads=8)
    print(f"params: {count_parameters(G):,}")
    G.eval()
    z = torch.randn(4, 8, 256)
    q = torch.rand(4)
    tid = torch.randint(0, 70, (4,))
    tg = torch.randn(4, 128)
    cid = torch.randint(1, 11, (4,))   # avoid the null index, so dropping means something
    eid = torch.randint(1, 125, (4,))
    out_full = G(z, q, tid, tg, cid, eid)
    print('out shape:', out_full.shape)
    cmask = torch.zeros(4, dtype=torch.bool); cmask[1] = True
    emask = torch.zeros(4, dtype=torch.bool); emask[2] = True
    out_drop = G(z, q, tid, tg, cid, eid, cluster_mask=cmask, exact_mask=emask)
    assert torch.allclose(out_full[0], out_drop[0]), "row 0 (nothing dropped) must be identical"
    assert torch.allclose(out_full[3], out_drop[3]), "row 3 (nothing dropped) must be identical"
    assert not torch.allclose(out_full[1], out_drop[1]), "row 1 dropped the cluster, must differ"
    assert not torch.allclose(out_full[2], out_drop[2]), "row 2 dropped the exact action, must differ"
    cid2 = cid.clone(); cid2[1] = 0
    assert torch.allclose(out_drop[1], G(z, q, tid, tg, cid2, eid)[1]), \
        "dropping the cluster must equal setting it to the null index"
    eid2 = eid.clone(); eid2[2] = 0
    assert torch.allclose(out_drop[2], G(z, q, tid, tg, cid, eid2)[2]), \
        "dropping the exact action must equal setting it to the null index"
    print('dropout assertions PASS')

    print("== AL-Flow-Src ==")
    Gs = ALFlowSrcGenerator(n_skels=70, n_clusters=11, n_exact_actions=125,
                            d_model=512, n_layers=6, n_heads=8)
    print(f"params: {count_parameters(Gs):,}")
    Gs.eval()
    src_z = torch.randn(4, 8, 256)
    sid = torch.randint(0, 70, (4,))
    sg = torch.randn(4, 128)
    out_full = Gs(z, q, tid, tg, src_z, sid, sg, cid, eid)
    print('out shape:', out_full.shape)
    smask = torch.zeros(4, dtype=torch.bool); smask[3] = True
    out_drop = Gs(z, q, tid, tg, src_z, sid, sg, cid, eid,
                  cluster_mask=cmask, exact_mask=emask, src_mask=smask)
    assert torch.allclose(out_full[0], out_drop[0])
    assert not torch.allclose(out_full[1], out_drop[1])
    assert not torch.allclose(out_full[2], out_drop[2])
    assert not torch.allclose(out_full[3], out_drop[3])
    src_z3 = src_z.clone()
    src_z3[3] = Gs.src_null_token.detach()
    assert torch.allclose(out_drop[3], Gs(z, q, tid, tg, src_z3, sid, sg, cid, eid)[3]), \
        "dropping the source must equal passing the null token"
    print('dropout assertions PASS')

    print("== AL-Flow-Src-G ==")
    Gg = ALFlowSrcGraphGenerator(n_clusters=11, n_exact_actions=125,
                                 d_model=512, n_layers=6, n_heads=8)
    print(f"params: {count_parameters(Gg):,}")
    Gg.eval()
    out_full = Gg(z, q, tg, src_z, sg, cid, eid)
    print('out shape:', out_full.shape)
    out_drop = Gg(z, q, tg, src_z, sg, cid, eid,
                  cluster_mask=cmask, exact_mask=emask, src_mask=smask)
    assert torch.allclose(out_full[0], out_drop[0])
    assert not torch.allclose(out_full[1], out_drop[1])
    assert not torch.allclose(out_full[2], out_drop[2])
    assert not torch.allclose(out_full[3], out_drop[3])
    print('dropout assertions PASS')
