"""Gymnasium environment for protein folding on a 3D voxel grid."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from folding_rl.config import Config
from folding_rl.data.fetch_pdb import load_ca_coords
from folding_rl.env.voxel_grid import VoxelGrid
from folding_rl.env.initialization import self_avoiding_walk
from folding_rl.env.scoring import compute_lddt, shaped_step_reward


# 27 displacement vectors in {-1, 0, 1}^3, indexed 0..26
_MOVES = np.array(
    [
        (a // 9 - 1, (a % 9) // 3 - 1, a % 3 - 1)
        for a in range(27)
    ],
    dtype=np.int32,
)  # shape (27, 3)


class ProteinFoldingEnv(gym.Env):
    """RL environment that folds a protein on a 3D voxel grid via PPO.

    Observation space:
        residue_positions: (N, 3) int32 voxel coordinates
        residue_types:     (N,)   int32  amino acid indices [0, 19]
        grid_resolution:   (1,)   int32  current grid size per axis
        step_count:        (1,)   int32  elapsed steps in episode

    Action space:
        MultiDiscrete([27] * N) — each residue picks one of 27 moves
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Config | None = None, pdb_id: str | None = None):
        super().__init__()
        self.cfg = config or Config()
        self.pdb_id = pdb_id or self.cfg.pdb_id

        # Load native structure (done once)
        self._native_coords, self._sequence = load_ca_coords(self.pdb_id)
        self._n = len(self._sequence)

        # Map amino acid letters to indices
        _aa_order = "ACDEFGHIKLMNPQRSTVWY"
        self._aa_to_idx = {aa: i for i, aa in enumerate(_aa_order)}
        self._residue_types = np.array(
            [self._aa_to_idx.get(aa, 0) for aa in self._sequence], dtype=np.int32
        )

        # Compute bounding box from native structure extent + padding
        from_center = np.linalg.norm(self._native_coords, axis=-1).max()
        self._bbox_size = 2 * from_center + self.cfg.bbox_padding * 2

        # Max grid resolution after 2 expansions: 9→17→33
        self._max_resolution = 33

        # Gym spaces
        self.observation_space = spaces.Dict(
            {
                "residue_positions": spaces.Box(
                    low=0,
                    high=self._max_resolution,
                    shape=(self._n, 3),
                    dtype=np.int32,
                ),
                "residue_types": spaces.Box(
                    low=0, high=19, shape=(self._n,), dtype=np.int32
                ),
                "grid_resolution": spaces.Box(
                    low=0, high=self._max_resolution, shape=(1,), dtype=np.int32
                ),
                "step_count": spaces.Box(
                    low=0,
                    high=self.cfg.max_steps_per_episode,
                    shape=(1,),
                    dtype=np.int32,
                ),
            }
        )
        self.action_space = spaces.MultiDiscrete([27] * self._n)

        # State
        self._grid: VoxelGrid | None = None
        self._native_voxel: np.ndarray | None = None  # discretized native at current res
        self._step_count = 0
        self._prev_coords: np.ndarray | None = None
        self._rng = np.random.default_rng(self.cfg.seed)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Fresh grid at initial resolution
        self._grid = VoxelGrid(self._bbox_size, self.cfg.n_init_resolution)

        # Discretize native structure to initial grid
        self._native_voxel = self._grid.discretize_coords(self._native_coords)

        # Self-avoiding walk initialization
        positions = self_avoiding_walk(
            self._n, self.cfg.n_init_resolution, rng=self._rng
        )
        self._grid.set_positions(positions)

        self._step_count = 0
        self._prev_coords = self._grid.get_real_coords().copy()

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.int32)
        displacements = _MOVES[action]  # (N, 3)

        # Propose new positions
        proposed = self._grid.positions + displacements

        # Reject out-of-bounds moves per residue (keep old position)
        oob = (proposed < 0) | (proposed > self._grid.max_coord)
        oob_mask = oob.any(axis=-1)  # (N,) True if any axis is OOB
        proposed[oob_mask] = self._grid.positions[oob_mask]

        # Reject moves that cause collisions (two residues at same voxel).
        # Process sequentially: each residue either moves to proposed or stays.
        n = len(proposed)
        old_positions = self._grid.positions.copy()
        final_positions = old_positions.copy()
        occupied = set()

        # First pass: place all residues at their OLD positions as baseline
        # Then try to upgrade each one to proposed if the target is free.
        for i in range(n):
            occupied.add(tuple(old_positions[i]))

        for i in range(n):
            old_key = tuple(old_positions[i])
            new_key = tuple(proposed[i])
            if new_key != old_key and new_key not in occupied:
                # Move succeeds: free old spot, claim new spot
                occupied.discard(old_key)
                occupied.add(new_key)
                final_positions[i] = proposed[i]
            # else: stay at old position (already in occupied)

        self._grid.set_positions(final_positions)

        self._step_count += 1

        # Grid expansion (at steps 50, 100, 150 with interval=50)
        if (
            self._step_count % self.cfg.expand_interval == 0
            and self._grid.resolution < self._max_resolution
        ):
            self._grid.expand()
            self._native_voxel = self._grid.discretize_coords(self._native_coords)

        curr_coords = self._grid.get_real_coords()
        ref_coords = self._native_coords

        # Shaped step reward
        reward, lddt_curr, _ = shaped_step_reward(
            prev_coords=self._prev_coords,
            curr_coords=curr_coords,
            ref_coords=ref_coords,
            lddt_delta_weight=self.cfg.lddt_delta_weight,
            bond_distance=self.cfg.ca_bond_distance,
            min_contact_distance=self.cfg.min_contact_distance,
            rg_target=self.cfg.rg_target,
            chain_weight=self.cfg.chain_penalty_weight,
            clash_weight=self.cfg.clash_penalty_weight,
            compact_weight=self.cfg.compact_penalty_weight,
        )

        done = self._step_count >= self.cfg.max_steps_per_episode

        if done:
            lddt_final = compute_lddt(curr_coords, ref_coords)
            reward += self.cfg.terminal_reward_scale * lddt_final
        else:
            lddt_final = lddt_curr

        self._prev_coords = curr_coords.copy()

        # Diagnostics
        bond_dists = np.linalg.norm(
            np.diff(curr_coords, axis=0), axis=-1
        )
        chain_breaks = int(np.sum(bond_dists > 4.5))
        center = curr_coords.mean(axis=0)
        rg = float(np.sqrt(np.mean(np.sum((curr_coords - center) ** 2, axis=-1))))

        info = {
            "lddt": lddt_final,
            "chain_breaks": chain_breaks,
            "rg": rg,
            "grid_resolution": self._grid.resolution,
        }

        return self._get_obs(), float(reward), done, False, info

    def _get_obs(self) -> dict:
        return {
            "residue_positions": self._grid.positions.copy(),
            "residue_types": self._residue_types.copy(),
            "grid_resolution": np.array([self._grid.resolution], dtype=np.int32),
            "step_count": np.array([self._step_count], dtype=np.int32),
        }
