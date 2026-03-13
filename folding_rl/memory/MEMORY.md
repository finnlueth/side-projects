# Project Memory: Protein Folding RL

## Project Summary
PPO + Transformer to fold Trp-cage (1L2Y, 20 residues) on 3D voxel grid.
Overfitting experiment to validate the formulation. Full design in IMPLEMENTATION_PLAN_1.md.

## Module Structure
```
src/folding_rl/
├── config.py               — Config dataclass with all hyperparameters
├── data/fetch_pdb.py       — load_ca_coords(pdb_id) → (coords, sequence)
├── env/
│   ├── protein_env.py      — ProteinFoldingEnv (Gymnasium)
│   ├── voxel_grid.py       — VoxelGrid: positions, expand(), get_real_coords()
│   ├── initialization.py   — self_avoiding_walk(n, grid_size, rng)
│   └── scoring.py          — compute_lddt(), shaped_step_reward(), potentials
├── model/
│   ├── transformer.py      — ProteinTransformer (4-layer, d_model=128, 4 heads)
│   ├── rope_3d.py          — RoPE3D: always builds tables for max_resolution (64)
│   ├── policy_head.py      — PolicyHead: per-residue Linear(128, 27) → Categorical
│   └── value_head.py       — ValueHead: attention-weighted pool → scalar
└── rl/
    ├── ppo.py              — PPOLightning (manual_optimization, DummyDataModule)
    ├── rollout_buffer.py   — RolloutBuffer(T, E, N)
    └── utils.py            — compute_gae(), normalize_advantages()
scripts/
├── train.py               — --fast-dev-run --no-wandb for quick test
└── evaluate.py            — greedy rollout, RMSD, optional matplotlib viz
```

## Key Bug Fixes Applied
1. **RoPE table size**: Must build for `max_resolution` (not current resolution).
   Minibatches contain mixed resolutions (positions from steps 0-200 across 4 expansion levels).
   Fixed in rope_3d.py: `_get_tables` always uses `self.max_resolution`.
2. **int32 vs int64**: `nn.Embedding` requires int64. Buffer stores int32, transformer must call `.long()` on aa_types and positions.
3. **SyncVectorEnv**: Pass `env_factory` function reference (not `env_factory()` call) in the list.
4. **DummyDataModule**: Must return a real `DataLoader(TensorDataset(...))`, not a plain list.

## Training Commands
```bash
uv run python scripts/train.py --fast-dev-run --no-wandb  # quick test
uv run python scripts/train.py --no-wandb                 # full 2M steps
uv run python scripts/train.py                            # with wandb
```

## Verified Working
- Training runs on RTX 4060 Ti (CUDA)
- 547K parameters
- PPO stable: approx_kl ~0.06, clip_fraction ~0.29
- ~2 updates/sec with num_envs=4, num_steps=64
