"""Mean-field policy head: per-residue categorical over 27 moves."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyHead(nn.Module):
    """Per-residue linear projection to 27 move logits.

    Mean-field approximation: each residue independently samples from
    Categorical(27). Joint log-prob = sum of per-residue log-probs.
    """

    def __init__(self, d_model: int = 128, n_moves: int = 27):
        super().__init__()
        self.proj = nn.Linear(d_model, n_moves)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, N_residues, d_model)

        Returns:
            logits: (batch, N_residues, 27)
        """
        return self.proj(features)

    def get_distribution(self, features: torch.Tensor) -> Categorical:
        """Return a batched Categorical distribution over moves.

        The returned distribution has batch_shape (batch, N_residues)
        and event_shape ().
        """
        logits = self.forward(features)
        return Categorical(logits=logits)

    def sample_actions(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample joint actions and compute log-probs + entropy.

        Returns:
            actions: (batch, N_residues) int64
            log_probs: (batch,) joint log-prob (sum over residues)
            entropy: (batch,) joint entropy (sum over residues)
        """
        dist = self.get_distribution(features)
        actions = dist.sample()                    # (B, N)
        log_probs = dist.log_prob(actions).sum(-1) # (B,)
        entropy = dist.entropy().sum(-1)           # (B,)
        return actions, log_probs, entropy

    def evaluate_actions(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log-probs and entropy for stored actions.

        Args:
            features: (batch, N_residues, d_model)
            actions:  (batch, N_residues) int64

        Returns:
            log_probs: (batch,) joint log-prob
            entropy:   (batch,) joint entropy
        """
        dist = self.get_distribution(features)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, entropy
