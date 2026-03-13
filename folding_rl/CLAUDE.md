# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a protein folding reinforcement learning system — an **overfitting experiment** to validate that a PPO + Transformer formulation can fold the Trp-cage miniprotein (PDB: 1L2Y, 20 residues) on a 3D voxel grid. The goal is not generalization, but proof-of-concept on a single small protein.

The full design is documented in `IMPLEMENTATION_PLAN_1.md`. Read it before implementing anything — it contains the complete scientific background, architecture decisions, and build order.

## Commands

```bash
uv sync                        # Install dependencies
uv run pytest tests/           # Run all tests
uv run pytest tests/test_x.py  # Run a single test file
uv run python scripts/train.py     # Main training entry point
uv run python scripts/evaluate.py  # Visualize predicted vs native fold
```

## Architecture

The system is structured as a Gymnasium RL environment with a Transformer policy, trained via PPO.

### Core Formulation

- **State:** 3D voxel grid with N=20 residues at integer voxel coordinates
- **Action:** `MultiDiscrete([27] * 20)` — each residue independently chooses one of 27 moves in {-1, 0, 1}³ (mean-field approximation)
- **Reward:** Potential-based shaped reward (chain connectivity + steric clash + compactness) + direct ΔlDDT signal each step, plus terminal `C · lDDT_final` (C=10)
- **Curriculum:** Grid resolution expands periodically: n → 2n-1 (5→9→17→33 voxels/side), physical bbox stays fixed, voxel spacing halves each time
- **Scoring:** Discretized lDDT — superposition-free metric comparing pairwise Cα distances at 4 thresholds (0.5, 1.0, 2.0, 4.0 Å)

### Planned Module Structure

```
src/folding_rl/
├── config.py                 # All hyperparameters in one dataclass
├── data/fetch_pdb.py         # Load 1L2Y.cif, extract Cα coords (20, 3), center to origin
├── env/
│   ├── protein_env.py        # Gymnasium env (ProteinFoldingEnv)
│   ├── voxel_grid.py         # VoxelGrid class: grid state, expand(), get_real_coords()
│   ├── initialization.py     # Self-avoiding walk (SAW) starting configuration
│   └── scoring.py            # lDDT (cα-only, ~30 lines numpy) + reward shaping potentials
├── model/
│   ├── transformer.py        # 4-layer Transformer backbone (d_model=128, 4 heads, pre-norm, GELU)
│   ├── rope_3d.py            # 3D RoPE: split head dims into 3 groups, apply per-axis RoPE
│   ├── policy_head.py        # Per-residue Linear(128, 27) → categorical distributions
│   └── value_head.py         # Attention-weighted pooling → MLP → scalar
└── rl/
    ├── ppo.py                # PPO (CleanRL-style, single-file pattern)
    ├── rollout_buffer.py     # Experience storage
    └── utils.py              # GAE (γ=0.99, λ=0.95), advantage normalization
scripts/
├── train.py
└── evaluate.py
data/
├── 1L2Y.cif                  # Target protein (NMR ensemble — use model 0)
└── 1BBL.cif
```

### Key Design Decisions

- **3D RoPE** encodes spatial position in attention: head dim split into 3 groups (x, y, z), each applying standard RoPE with the residue's voxel coordinate. Must be recomputed dynamically when grid expands mid-episode.
- **Sequence position embedding** (separate from 3D RoPE) encodes chain order (residue 1 bonded to 2, etc.)
- **SAW initialization:** Each episode starts with a random self-avoiding walk from grid center using face-adjacent moves (6 directions only)
- **Grid expansion mapping:** When n → 2n-1, positions are simply doubled (`positions *= 2`); voxel_spacing recomputed as `bbox_size / (new_resolution - 1)`
- **lDDT note:** At coarse resolution, 0.5 Å and 1.0 Å thresholds yield near-zero scores; 2.0 Å and 4.0 Å thresholds still provide learning signal — this is the natural curriculum
- **No existing RL libraries** (CleanRL adapted, not imported); custom lDDT (not Biotite); custom 3D RoPE (not `rotary-embedding-torch`)

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Model, training, CUDA |
| `gymnasium` | RL environment interface |
| `biopython` | Parse CIF files, extract Cα coords |
| `numpy` | Voxel grid math |
| `wandb` | Experiment tracking |
| `einops` | Tensor reshaping in transformer |
| `scipy` | Pairwise distance for lDDT |
| `pytest` | Testing (dev) |
