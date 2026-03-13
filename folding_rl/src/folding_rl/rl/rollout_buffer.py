"""Fixed-size rollout buffer for PPO."""
from __future__ import annotations

import torch


class RolloutBuffer:
    """Stores trajectories from vectorized environments for PPO updates.

    All tensors are shaped (num_steps, num_envs, ...).
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        n_residues: int,
        obs_resolution: int = 33,
        device: torch.device | str = "cpu",
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.n_residues = n_residues
        self.device = torch.device(device)
        self.ptr = 0

        # Observations
        self.positions = torch.zeros(
            num_steps, num_envs, n_residues, 3, dtype=torch.int32, device=self.device
        )
        self.aa_types = torch.zeros(
            num_steps, num_envs, n_residues, dtype=torch.int32, device=self.device
        )
        self.grid_resolution = torch.zeros(
            num_steps, num_envs, 1, dtype=torch.int32, device=self.device
        )
        self.step_counts = torch.zeros(
            num_steps, num_envs, 1, dtype=torch.int32, device=self.device
        )

        # Actions and PPO quantities
        self.actions = torch.zeros(
            num_steps, num_envs, n_residues, dtype=torch.int64, device=self.device
        )
        self.log_probs = torch.zeros(
            num_steps, num_envs, device=self.device
        )
        self.values = torch.zeros(
            num_steps, num_envs, device=self.device
        )
        self.rewards = torch.zeros(
            num_steps, num_envs, device=self.device
        )
        self.dones = torch.zeros(
            num_steps, num_envs, device=self.device
        )

    def add(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Store one step of data from all environments."""
        t = self.ptr
        self.positions[t] = obs["residue_positions"]
        self.aa_types[t] = obs["residue_types"]
        self.grid_resolution[t] = obs["grid_resolution"]
        self.step_counts[t] = obs["step_count"]
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.ptr = (self.ptr + 1) % self.num_steps

    def get_obs_at(self, t: int) -> dict[str, torch.Tensor]:
        """Get observation dict for timestep t."""
        return {
            "residue_positions": self.positions[t],
            "residue_types": self.aa_types[t],
            "grid_resolution": self.grid_resolution[t],
            "step_count": self.step_counts[t],
        }

    def get_all_obs(self) -> dict[str, torch.Tensor]:
        """Flatten (T, E, ...) to (T*E, ...) for minibatch training."""
        T, E = self.num_steps, self.num_envs
        return {
            "residue_positions": self.positions.view(T * E, self.n_residues, 3),
            "residue_types": self.aa_types.view(T * E, self.n_residues),
            "grid_resolution": self.grid_resolution.view(T * E, 1),
            "step_count": self.step_counts.view(T * E, 1),
        }

    def get_flat(self) -> dict[str, torch.Tensor]:
        """Return all stored data flattened over (T, E) → (T*E,)."""
        T, E = self.num_steps, self.num_envs
        return {
            "obs": self.get_all_obs(),
            "actions": self.actions.view(T * E, self.n_residues),
            "log_probs": self.log_probs.view(T * E),
            "values": self.values.view(T * E),
        }
