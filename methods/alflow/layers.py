"""Building blocks shared by the three AL-Flow generators."""
from __future__ import annotations
import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, d_model, max_period=1000.0):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        half = d_model // 2
        freqs = torch.exp(-math.log(max_period) *
                          torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer('freqs', freqs)

    def forward(self, q):
        args = q.float().unsqueeze(-1) * self.freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
