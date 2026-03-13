"""3D Rotary Positional Embedding (RoPE) for transformer attention.

The head dimension is split into 3 groups (x, y, z). Each group applies
standard 1D RoPE using the residue's voxel coordinate along that axis.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def _build_rope_cache(
    max_positions: int,
    dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build sin/cos tables for positions [0, max_positions).

    Returns:
        sin_table: (max_positions, dim/2)
        cos_table: (max_positions, dim/2)
    """
    half = dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(max_positions, device=device).float()
    freqs = torch.outer(pos, theta)  # (max_positions, half)
    return freqs.sin(), freqs.cos()


def _apply_rope_1d(
    x: torch.Tensor,
    sin_table: torch.Tensor,
    cos_table: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Apply 1D RoPE to tensor x at given positions.

    Args:
        x: (batch, n_tokens, n_heads, dim) or (batch, n_tokens, dim)
           – the last dimension must be even
        sin_table: (max_pos, dim/2)
        cos_table: (max_pos, dim/2)
        positions: (batch, n_tokens) integer position indices

    Returns:
        Tensor of same shape as x
    """
    half = x.shape[-1] // 2
    # Gather sin/cos for each position: (batch, n_tokens, half)
    sin = sin_table[positions]
    cos = cos_table[positions]

    # If x has a heads dimension, expand sin/cos
    if x.dim() == 4:
        sin = sin.unsqueeze(2)  # (batch, n_tokens, 1, half)
        cos = cos.unsqueeze(2)

    x1 = x[..., :half]
    x2 = x[..., half:]
    # Rotation: [x1, x2] → [x1*cos - x2*sin, x2*cos + x1*sin]
    x_rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return x_rot


class RoPE3D(nn.Module):
    """Apply 3D RoPE by splitting head dim into 3 groups (x, y, z).

    For d_head=32: allocate 12 dims to x, 10 to y, 10 to z (all even).
    """

    def __init__(
        self,
        d_head: int,
        max_resolution: int = 64,
        base: float = 10000.0,
    ):
        super().__init__()
        self.d_head = d_head

        # Split d_head into 3 even groups
        # e.g. d_head=32 → 12, 10, 10
        per_axis = d_head // 3
        # Make all even; adjust last to absorb remainder
        dx = (per_axis // 2) * 2
        dy = (per_axis // 2) * 2
        dz = d_head - dx - dy
        if dz % 2 != 0:
            dz -= 1
            dx += 1  # small adjustment
        self.dims = (dx, dy, dz)
        assert sum(self.dims) == d_head, f"dim split {self.dims} != {d_head}"

        self.max_resolution = max_resolution
        self.base = base
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _get_tables(
        self, device: torch.device
    ) -> tuple[
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        # Always build for max_resolution so any position index is valid
        key = str(device)
        if key not in self._cache:
            sx, cx = _build_rope_cache(self.max_resolution, self.dims[0], self.base, device)
            sy, cy = _build_rope_cache(self.max_resolution, self.dims[1], self.base, device)
            sz, cz = _build_rope_cache(self.max_resolution, self.dims[2], self.base, device)
            self._cache[key] = (sx, cx, sy, cy, sz, cz)
        return self._cache[key]

    def forward(
        self,
        qk: torch.Tensor,
        positions: torch.Tensor,
        resolution: int = 0,
    ) -> torch.Tensor:
        """Apply 3D RoPE to query or key tensor.

        Args:
            qk: (batch, n_tokens, n_heads, d_head)
            positions: (batch, n_tokens, 3) integer voxel coordinates (x, y, z)
            resolution: unused, kept for API compatibility

        Returns:
            (batch, n_tokens, n_heads, d_head)
        """
        sx, cx, sy, cy, sz, cz = self._get_tables(qk.device)
        dx, dy, dz = self.dims

        pos_x = positions[..., 0]  # (batch, n_tokens)
        pos_y = positions[..., 1]
        pos_z = positions[..., 2]

        # Split d_head into 3 segments
        qk_x = qk[..., :dx]
        qk_y = qk[..., dx:dx + dy]
        qk_z = qk[..., dx + dy:]

        qk_x = _apply_rope_1d(qk_x, sx, cx, pos_x)
        qk_y = _apply_rope_1d(qk_y, sy, cy, pos_y)
        qk_z = _apply_rope_1d(qk_z, sz, cz, pos_z)

        return torch.cat([qk_x, qk_y, qk_z], dim=-1)
