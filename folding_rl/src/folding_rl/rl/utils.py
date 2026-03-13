"""GAE computation and advantage normalization."""
from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation (GAE).

    Convention: dones[t] = 1.0 means the action at step t ended the episode.
    When dones[t] = 1, the observation at t+1 (or next_value for t=T-1) belongs
    to a new episode and must NOT be bootstrapped from.

    Args:
        rewards:    (T, num_envs) float
        values:     (T, num_envs) float  (value estimates at each step)
        dones:      (T, num_envs) float  (1.0 if action at step t ended episode)
        next_value: (num_envs,) float    (bootstrap value at end of rollout)
        next_done:  (num_envs,) float    (1.0 if last rollout step ended episode)
        gamma:      discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: (T, num_envs) float
        returns:    (T, num_envs) float  (advantages + values)
    """
    T, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(num_envs, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
            nonterminal = 1.0 - next_done
        else:
            next_val = values[t + 1]
            nonterminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_val * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages to zero mean, unit variance."""
    return (advantages - advantages.mean()) / (advantages.std() + eps)


def explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    """Fraction of return variance explained by the value function."""
    var_y = returns.var()
    if var_y < 1e-8:
        return float("nan")
    return float(1.0 - (returns - values).var() / var_y)
