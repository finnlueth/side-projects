"""Value head: attention-weighted pooling over residues → scalar."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHead(nn.Module):
    """Attention-weighted pooling over residues followed by MLP → scalar."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        # Learned attention query for pooling
        self.attn_query = nn.Linear(d_model, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, N_residues, d_model)

        Returns:
            value: (batch, 1) scalar value estimate
        """
        # Attention weights: (batch, N_residues, 1)
        attn_weights = F.softmax(self.attn_query(features), dim=1)
        # Weighted sum: (batch, d_model)
        pooled = (attn_weights * features).sum(dim=1)
        return self.mlp(pooled)  # (batch, 1)
