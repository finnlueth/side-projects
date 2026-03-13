"""Transformer backbone with 3D RoPE for protein folding."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from folding_rl.model.rope_3d import RoPE3D


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer with 3D RoPE in self-attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, rope: RoPE3D):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

        self.rope = rope

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, n_tokens, d_model)
            positions: (batch, n_tokens, 3) integer voxel coords (int64)

        Returns:
            (batch, n_tokens, d_model)
        """
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head

        # --- Self-attention (pre-norm) ---
        h = self.norm1(x)
        Q = self.q_proj(h).reshape(B, T, H, Dh)
        K = self.k_proj(h).reshape(B, T, H, Dh)
        V = self.v_proj(h).reshape(B, T, H, Dh)

        # Apply 3D RoPE to Q and K
        Q = self.rope(Q, positions)
        K = self.rope(K, positions)

        # Scaled dot-product attention
        # (B, H, T, Dh)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(Q, K, V)  # (B, H, T, Dh)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.d_model)
        x = x + self.out_proj(attn_out)

        # --- FFN (pre-norm) ---
        h = self.norm2(x)
        x = x + self.ff2(F.gelu(self.ff1(h)))

        return x


class ProteinTransformer(nn.Module):
    """Transformer backbone: observation dict → per-residue features.

    Input:  observation dict from ProteinFoldingEnv
    Output: (batch, N_residues, d_model) feature vectors
    """

    def __init__(
        self,
        n_residues: int = 20,
        n_aa_types: int = 20,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        max_resolution: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_residues = n_residues

        # Input embeddings
        self.aa_embed = nn.Embedding(n_aa_types, 32)
        self.seq_pos_embed = nn.Embedding(n_residues, 16)

        # Position features (3) + resolution scalar (1) + step progress (1)
        input_dim = 32 + 16 + 3 + 1 + 1
        self.input_proj = nn.Linear(input_dim, d_model)
        self.max_resolution = max_resolution

        # Shared RoPE across all layers
        rope = RoPE3D(d_head=d_model // n_heads, max_resolution=max_resolution)

        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, n_heads, d_ff, rope) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs: dict with keys:
                residue_positions: (B, N, 3) int
                residue_types:     (B, N)    int
                grid_resolution:   (B, 1)    int
                step_count:        (B, 1)    int (unused in model)

        Returns:
            (B, N, d_model)
        """
        positions = obs["residue_positions"].long()  # (B, N, 3) int64
        aa_types = obs["residue_types"].long()     # (B, N) int64
        resolution = obs["grid_resolution"]        # (B, 1)
        step_count = obs["step_count"]             # (B, 1)

        B, N, _ = positions.shape
        device = positions.device

        # Amino acid embedding
        aa_feats = self.aa_embed(aa_types)         # (B, N, 32)

        # Sequence position embedding
        seq_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
        seq_feats = self.seq_pos_embed(seq_idx)    # (B, N, 16)

        # Normalized voxel positions: [0, 1] range
        pos_feats = positions.float() / float(self.max_resolution)  # (B, N, 3)

        # Resolution scalar: normalize to [0, 1]
        res_scalar = resolution.float() / 64.0    # (B, 1)
        res_feats = res_scalar.unsqueeze(1).expand(B, N, 1)  # (B, N, 1)

        # Step progress: normalize to [0, 1]
        step_feats = step_count.float() / 300.0   # (B, 1)
        step_feats = step_feats.unsqueeze(1).expand(B, N, 1)  # (B, N, 1)

        # Concatenate and project
        x = torch.cat([aa_feats, seq_feats, pos_feats, res_feats, step_feats], dim=-1)  # (B, N, 53)
        x = self.input_proj(x)                     # (B, N, d_model)

        # Transformer layers (resolution not needed — RoPE uses max_resolution table)
        for layer in self.layers:
            x = layer(x, positions)

        return self.final_norm(x)
