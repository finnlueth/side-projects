# Protein Folding via RL: Implementation Plan

## Overfitting Experiment — Trp-cage (1L2Y, 20 residues)

---

## 0. Problem Context & Scientific Background

This section provides the full scientific and architectural context needed to understand every design decision in this project. Read this section completely before implementing anything.

### 0.1 What We Are Building

We are building a reinforcement learning system that learns to fold a protein into its 3D structure by playing a "game" on a discrete voxel grid. The formulation is inspired by AlphaZero (the system that mastered Go and Chess), but applied to the protein structure prediction problem.

**The core idea:** A protein is a chain of amino acid residues. In nature, this chain folds into a specific 3D shape (its "native structure") determined by its amino acid sequence. We model this folding process as a game where an RL agent manipulates residue positions on a 3D grid, and is rewarded for reaching the known native structure.

**This is an overfitting experiment.** We are training on a single protein (Trp-cage, 20 residues) with the known answer as the reward signal. The goal is NOT to generalize — it is to validate that this RL formulation, transformer architecture, and reward structure can, in principle, discover the correct fold. If it cannot overfit to a single small protein, the approach has fundamental issues and should be redesigned before scaling.

### 0.2 The Game Formulation

**State:** A 3D voxel grid (like a 3D chessboard) with amino acid residues placed at specific voxel positions. Each residue occupies exactly one voxel. The grid has a physical size in Ångströms (Å, the standard unit for atomic distances), and the voxel spacing determines the resolution.

**Actions:** At each time step, every residue simultaneously and independently chooses one of 27 moves — the displacement vectors in {-1, 0, 1}³. This includes (0,0,0) which means "stay in place." This is a **mean-field approximation**: each residue's decision is sampled independently from a categorical distribution conditioned on the global state, even though the global state encodes all residues' positions via the transformer's self-attention. The joint action space is `MultiDiscrete([27] * N)` for N residues.

**Grid expansion:** Periodically (on a fixed schedule), the grid resolution increases: n → 2n-1 per axis. Each existing voxel coordinate (x,y,z) maps to (2x, 2y, 2z) in the new grid, with new voxels inserted between old ones. This creates a natural coarse-to-fine curriculum: early in the episode the agent works at low resolution (getting the gross shape right), later it refines at high resolution. The physical bounding box size stays fixed — only the number of voxels increases.

**Episode termination:** Fixed number of steps (no learned termination). The agent gets a terminal reward based on how closely its final configuration matches the known native structure.

### 0.3 The Protein: Trp-cage (PDB: 1L2Y)

Trp-cage is a 20-residue miniprotein commonly used as a benchmark in protein folding studies. Its sequence is NLYIQWLKDGGPSSGRPPPS. Key physical properties:

- 20 amino acid residues, represented as Cα (alpha-carbon) positions only
- Native structure spans approximately 15 Å
- Radius of gyration ≈ 7 Å
- Contains an alpha-helix, a 3₁₀-helix, and a polyproline II helix
- PDB 1L2Y is an NMR ensemble — we use model 0 (first conformer)
- Cα-Cα bond distance between sequential residues: 3.8 Å (constant for all protein backbones)

### 0.4 Scoring: Discretized lDDT

**lDDT (Local Distance Difference Test)** is a superposition-free metric for comparing protein structures. Unlike RMSD, it does not require aligning the structures, which makes it more robust and better-behaved as an optimization target.

The algorithm for Cα-only lDDT:

1. In the **reference** (native) structure, compute all pairwise Cα distances
2. Select all residue pairs where the reference distance is within an inclusion radius (R₀ = 15 Å)
3. For each selected pair, compute the corresponding distance in the **model** (predicted) structure
4. A pair is "preserved" at threshold t if |d_model - d_reference| < t
5. lDDT = average fraction of preserved pairs across four thresholds: {0.5, 1.0, 2.0, 4.0} Å
6. Score ranges from 0.0 (completely wrong) to 1.0 (perfect match)

**Discretization:** Both the native structure and the agent's positions exist on the voxel grid. The native Cα coordinates are snapped to the nearest voxel center at the current resolution. At coarse resolution (5 Å voxel spacing), the fine thresholds (0.5 Å, 1.0 Å) will yield near-zero scores while the coarse thresholds (2.0 Å, 4.0 Å) still provide signal. As the grid expands and voxel spacing shrinks, finer thresholds activate — this is the natural curriculum.

### 0.5 Reward Shaping

Raw terminal-only reward (lDDT at end of episode) would be extremely sparse. We use **potential-based reward shaping** (Ng et al., 1999), which provably preserves the optimal policy while accelerating learning.

Define a potential function Φ(s) over states. The shaped reward at each step is:
```
F(s_t, s_{t+1}) = γ · Φ(s_{t+1}) - Φ(s_t)
```

Three potential components for the initial implementation:

1. **Chain connectivity** (most critical): Penalizes Cα-Cα distances between sequential residues that deviate from the expected 3.8 Å bond distance. Without this, the mean-field agent will immediately shatter the chain by moving residues independently.
   ```
   Φ_chain = -δ · Σᵢ max(0, d(i, i+1) - d_bond)²
   ```

2. **Steric clash**: Penalizes residues that are too close together (< 3.0 Å). Prevents the agent from collapsing all residues into a single voxel.
   ```
   Φ_clash = -β · Σ_{i<j} max(0, d_min - d(i,j))²
   ```

3. **Compactness**: Penalizes deviation of the radius of gyration from the expected value for a folded protein of this length (~7.0 Å for N=20).
   ```
   Φ_compact = -α · (R_g - R_g_target)²
   ```

Additionally, since this is an overfitting experiment, we include a direct lDDT improvement signal:
```
r_step = F(s, s') + λ · (lDDT(s') - lDDT(s))
r_terminal = C · lDDT_final
```

All distances in the shaping potentials are computed in Ångströms by converting voxel coordinates using the current voxel spacing.

### 0.6 Model Architecture

**Why a Transformer?** Each residue is a "token." The transformer's self-attention lets each residue's policy be informed by the positions and types of all other residues. This is critical because protein folding is a cooperative process — moving one residue only makes sense in the context of where all other residues are.

**3D RoPE (Rotary Positional Embedding):** Standard RoPE encodes 1D sequence position. We extend it to 3D: the head dimension is split into 3 groups (one per spatial axis x, y, z), and each group applies standard RoPE using the residue's voxel coordinate along that axis. This encodes 3D spatial relationships directly into the attention computation — residues that are close in 3D space will attend to each other more naturally.

The model also has a **sequence position embedding** (standard learned embedding for position in the chain) which is distinct from the 3D RoPE. This tells the transformer that residue 1 is bonded to residue 2 (a chain constraint), independent of where they are in 3D space.

**Architecture summary:**
- Input: amino acid type embedding (20 types) + sequence position embedding + resolution scalar
- Backbone: 4-layer transformer, d_model=128, 4 heads, pre-norm, GELU FFN
- Policy head: per-residue Linear(128, 27) → independent categorical distributions
- Value head: attention-weighted pooling over residues → MLP → scalar

~500K parameters. Extremely fast inference — the bottleneck is environment stepping, not the model.

### 0.7 PPO (Proximal Policy Optimization)

We use PPO with a mean-field factorized policy. Key details:

- **Log-probability of a joint action** = sum of per-residue log-probs (mean-field assumption)
- **Entropy** = sum of per-residue entropies
- **GAE (Generalized Advantage Estimation)** for variance reduction with γ=0.99, λ=0.95
- **Clipped objective** with ε=0.2
- Adapted from CleanRL's single-file PPO pattern — we write our own because we need full control over the custom transformer policy and multi-discrete action space

### 0.8 Self-Avoiding Walk Initialization

Each episode starts with residues placed along a **self-avoiding walk (SAW)** on the voxel grid. This mimics an extended/unfolded polymer chain. The walk uses face-adjacent moves (6 directions, not 26) because consecutive residues should be one voxel apart, which at the initial voxel spacing (~5 Å) roughly corresponds to the Cα-Cα bond distance.

Algorithm: Start at grid center. For each subsequent residue, randomly choose from available face-adjacent neighbors that are in-bounds and unoccupied. If trapped, restart. For N=20 on a 5³ grid (125 voxels), this succeeds quickly.

Each episode gets a different random SAW, providing diversity in starting configurations.

---

## 1. Project Overview

Train a PPO agent with a Transformer policy (3D RoPE) to fold the Trp-cage miniprotein (PDB: 1L2Y, 20 residues) on a discrete 3D voxel grid. The agent moves all residues simultaneously (mean-field, 27 directions each including stay), periodically expands grid resolution on a fixed schedule, and is scored via discretized lDDT against the native Cα structure.

---

## 2. Dependencies

| Package | Purpose | Notes |
|---|---|---|
| `torch` (≥2.1) | Core framework — model, training, CUDA | |
| `gymnasium` | RL environment interface | |
| `biopython` | Parse PDB file, extract Cα coordinates | |
| `numpy` | Array ops, voxel grid math | |
| `wandb` | Experiment tracking, reward curves | |
| `einops` | Tensor reshaping in transformer | |
| `scipy` | Spatial distance computations for lDDT | |
| `pytest` | Testing | dev dependency |

**What we write ourselves** (not available as drop-in libraries for this exact use case):

- Custom Gymnasium environment (`ProteinFoldingEnv`)
- 3D RoPE implementation (extend standard RoPE to 3 spatial dims)
- Discretized lDDT scoring function (Cα-only, on voxel grid)
- Transformer policy/value network with 3D positional encoding
- PPO training loop (adapted from CleanRL's single-file pattern)
- Self-avoiding walk initialization
- Reward shaping potentials

**Why not use existing libraries:**

- **CleanRL**: Great reference, but not importable as a library. We adapt its PPO logic into our own training loop.
- **Biotite `lddt()`**: Full-featured but operates on `AtomArray` objects. We need a lightweight Cα-only lDDT on raw coordinate arrays. Simpler to implement the ~30 lines ourselves.
- **`rotary-embedding-torch`**: Supports axial N-D RoPE, but since grid resolution changes mid-episode, we need dynamic recomputation. We implement our own (~50 lines).

---

## 3. Project Structure (UV)

```
protein-fold-rl/
├── pyproject.toml
├── README.md
├── uv.lock
├── data
│   ├── 1BBL.cif
│   └── 1L2Y.cif
├── src/
│   └── protein_fold_rl/
│       ├── __init__.py
│       ├── config.py                 # All hyperparameters in one dataclass
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   └── fetch_pdb.py          # Download 1L2Y, extract Cα coords
│       │
│       ├── env/
│       │   ├── __init__.py
│       │   ├── protein_env.py        # Gymnasium environment
│       │   ├── voxel_grid.py         # Grid state, expansion logic
│       │   ├── initialization.py     # Self-avoiding walk on grid
│       │   └── scoring.py            # Discretized lDDT, reward shaping
│       │
│       ├── model/
│       │   ├── __init__.py
│       │   ├── transformer.py        # Transformer encoder backbone
│       │   ├── rope_3d.py            # 3D Rotary Positional Embedding
│       │   ├── policy_head.py        # Mean-field policy (N × 27 logits)
│       │   └── value_head.py         # Scalar value output
│       │
│       └── rl/
│           ├── __init__.py
│           ├── ppo.py                # PPO algorithm (CleanRL-style)
│           ├── rollout_buffer.py     # Experience storage
│           └── utils.py              # GAE, advantage normalization
│
├── scripts/
│   ├── train.py                      # Main training entry point
│   └── evaluate.py                   # Visualize predicted vs native fold
│
└── tests/
    ├── __init__.py
    ├── test_voxel_grid.py
    ├── test_initialization.py
    ├── test_scoring.py
    ├── test_environment.py
    ├── test_model.py
    └── test_ppo.py
```

### `pyproject.toml`

```toml
[project]
name = "protein-fold-rl"
version = "0.1.0"
description = "Protein folding via reinforcement learning on a 3D voxel grid"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.1",
    "gymnasium>=0.29",
    "biopython>=1.83",
    "numpy>=1.26",
    "wandb>=0.16",
    "einops>=0.7",
    "scipy>=1.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
]

[project.scripts]
train = "scripts.train:main"
evaluate = "scripts.evaluate:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/protein_fold_rl"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

### Project initialization commands

```bash
# Add dependencies
uv add torch gymnasium biopython numpy wandb einops scipy

# Add dev dependencies
uv add --dev pytest pytest-cov

# Create the source layout
mkdir -p src/protein_fold_rl/{data/pdb_cache,env,model,rl}
mkdir -p scripts tests
touch src/protein_fold_rl/__init__.py
touch src/protein_fold_rl/{data,env,model,rl}/__init__.py
touch tests/__init__.py
```

---

## 4. Implementation Modules (ordered by build sequence)

### Phase 1: Data & Scoring (Day 1)

#### 4.1 `src/protein_fold_rl/data/fetch_pdb.py` — PDB Loading

```
Input:  PDB ID string ("1L2Y")
Output: numpy array of Cα coordinates, shape (20, 3), in Ångströms
        amino acid sequence as list of 1-letter codes
```

- Use `Bio.PDB.PDBParser` to parse the structure
- Extract Cα atoms: iterate residues, get `residue['CA'].get_coord()`
- 1L2Y is an NMR ensemble — use model 0 (first conformer)
- Store sequence: map `residue.get_resname()` → 1-letter code via `Bio.Data.IUPACData.protein_letters_3to1`
- Center coordinates to origin (subtract mean)
- Cache the PDB file in `src/protein_fold_rl/data/pdb_cache/`

#### 4.2 `src/protein_fold_rl/env/scoring.py` — lDDT and Reward Shaping

**Discretized lDDT implementation:**

See Section 0.4 for the full algorithm. Implementation notes:
- Pure numpy: pairwise distance matrices via `scipy.spatial.distance.cdist` or manual broadcasting
- ~30 lines of code for Cα-only lDDT
- Must be fast since it's called every step for shaping and at episode end
- At coarse resolution, the 0.5 Å and 1.0 Å thresholds will yield near-zero scores, but the 2.0 Å and 4.0 Å thresholds still provide signal
- As the grid expands, finer thresholds activate — this is the natural curriculum

**Reward shaping potentials:**

See Section 0.5 for the full formulation. We use potential-based shaping (F(s, s') = γ·Φ(s') - Φ(s)) with three potentials: chain connectivity, steric clash, compactness. Plus a direct Δ_lDDT signal since this is an overfitting experiment.

All distances computed in Ångströms by converting voxel positions: `real_coords = voxel_coords * voxel_spacing + grid_origin`

**Step reward:** `r_step = F(s, s') + λ · Δ_lDDT`

**Terminal reward:** `r_terminal = C · lDDT_final` (C = 10.0)

---

### Phase 2: Environment (Day 1–2)

#### 4.3 `src/protein_fold_rl/env/voxel_grid.py` — Grid State Management

```python
class VoxelGrid:
    resolution: int          # current grid size per axis (n)
    voxel_spacing: float     # Å per voxel at current resolution
    bbox_size: float         # physical size of the grid in Å (fixed)
    positions: np.ndarray    # (N_residues, 3) integer voxel coordinates
```

**Grid sizing:**
- Physical bounding box: based on the native structure's extent + padding
  - `bbox_size = 2 * (max radius from center-of-mass in native structure) + margin`
  - For Trp-cage: the protein spans ~15 Å, so bbox ≈ 25 Å per side
- Initial resolution: start coarse, e.g. `n_init = 5` → voxel_spacing = 25/5 = 5.0 Å
- After expansion: n → 2n - 1 (e.g., 5 → 9 → 17 → 33)
  - voxel_spacing halves: 5.0 → 2.78 → 1.47 → 0.76 Å
  - 4 resolution levels; level 3 (0.76 Å) is finer than Cα spacing

**Expansion logic:**
- When expanding from n to 2n-1, each existing voxel coordinate (x,y,z) maps to (2x, 2y, 2z)
- New voxels are inserted between old ones
- Residue positions are simply doubled: `positions *= 2`
- voxel_spacing is recomputed: `bbox_size / (new_resolution - 1)`

**Key methods:**
- `get_real_coords() → np.ndarray`: Convert voxel indices to Ångström coordinates
- `expand()`: Double resolution, remap positions
- `discretize_coords(real_coords) → np.ndarray`: Snap real-space coords to nearest voxel
- `is_valid_position(pos) → bool`: Check bounds (0 ≤ coord < resolution)

#### 4.4 `src/protein_fold_rl/env/initialization.py` — Self-Avoiding Walk

See Section 0.8 for the algorithm and rationale.

**Output:** Integer array of shape (N, 3) — initial voxel coordinates for each residue.

**Key constraint:** Uses only 6 face-adjacent moves (±x, ±y, ±z), NOT the full 26 neighbors, because consecutive bonded residues should be exactly one voxel apart.

#### 4.5 `src/protein_fold_rl/env/protein_env.py` — Gymnasium Environment

```python
class ProteinFoldingEnv(gymnasium.Env):
    observation_space: Dict
    action_space: MultiDiscrete([27] * N_residues)
```

**Observation space:**
```python
{
    "residue_positions": Box(low=0, high=max_grid, shape=(20, 3), dtype=np.int32),
    "residue_types": Box(low=0, high=19, shape=(20,), dtype=np.int32),
    "grid_resolution": Box(low=0, high=100, shape=(1,), dtype=np.int32),
    "step_count": Box(low=0, high=max_steps, shape=(1,), dtype=np.int32),
}
```

**Action space:**
- `MultiDiscrete([27] * 20)` — each of 20 residues independently picks one of 27 moves
- Moves are the 27 vectors in {-1, 0, 1}³, enumerated as integers 0–26
- Action integer a maps to displacement: `(a // 9 - 1, (a % 9) // 3 - 1, a % 3 - 1)`
- Action 13 = (0,0,0) = stay in place

**Step logic:**
```
1. Decode actions → 20 displacement vectors from {-1, 0, 1}³
2. Compute new_positions = positions + displacements
3. Clip to grid bounds (residues that would exit the grid stay put)
4. Update positions (we allow collisions — clash penalty handles it)
5. If step_count % expand_interval == 0 and not at max resolution:
     call voxel_grid.expand()
     re-discretize the native target to the new resolution
6. Compute shaped reward (potential-based shaping + Δ_lDDT)
7. If step_count == max_steps: add terminal reward, done=True
8. Return obs, reward, done, truncated, info
```

**Episode configuration (from config.py):**
- `max_steps = 200`
- `expand_interval = 50` (expand at steps 50, 100, 150)
- `n_init = 5` (initial grid resolution, so 5×5×5 = 125 voxels)
- `max_resolution = 33` (after 3 expansions: 5 → 9 → 17 → 33)

---

### Phase 3: Model (Day 2–3)

#### 4.6 `src/protein_fold_rl/model/rope_3d.py` — 3D Rotary Positional Embedding

See Section 0.6 for the conceptual explanation.

**Implementation approach:** Concatenate separate per-axis RoPE. Split the head dimension d_head into 3 equal groups. For a token at voxel position (x, y, z):
```
Dimensions [0 : d_head//3]:            RoPE with position = x
Dimensions [d_head//3 : 2*d_head//3]:  RoPE with position = y
Dimensions [2*d_head//3 : d_head]:     RoPE with position = z
```

d_head must be divisible by 6 (3 axes × 2 per rotation pair). With d_model=128 and 4 heads, d_head=32, so each axis gets 10 dimensions (with 2 leftover — use 30 RoPE dims with 2 non-rotated dims, or adjust d_model to 144 for clean division).

**Simplest approach for d_head=32:** Allocate 12 dims to x-axis, 10 to y-axis, 10 to z-axis. Each group must have even size (pairs for rotation). This works: 12+10+10=32, all even. The exact split is an implementation detail — prioritize simplicity.

**Critical:** RoPE sin/cos tables must be recomputed when the grid expands (position indices change from [0, n) to [0, 2n-1)). Cache tables per resolution level.

**~60 lines of PyTorch.**

#### 4.7 `src/protein_fold_rl/model/transformer.py` — Backbone

```python
class ProteinTransformer(nn.Module):
    """
    Input:  observation dict from environment
    Output: (batch, N_residues, d_model) per-residue feature vectors
    """
```

**Architecture:**
- **Input embedding:**
  - Amino acid type: `nn.Embedding(20, 32)`
  - Sequence position: `nn.Embedding(max_residues, 16)` — encodes chain position, distinct from 3D RoPE
  - Resolution level: scalar, broadcast and concatenated
  - Total input dim → linear projection to d_model=128

- **Transformer layers (×4):**
  - Pre-LayerNorm
  - Multi-head self-attention (4 heads, d_head=32) with 3D RoPE on Q and K
  - Residual connection
  - Pre-LayerNorm
  - FFN: Linear(128, 256) → GELU → Linear(256, 128)
  - Residual connection

- **~500K parameters.** Forward pass <1ms on any GPU for 20 tokens.

#### 4.8 `src/protein_fold_rl/model/policy_head.py` — Mean-Field Policy

```python
class PolicyHead(nn.Module):
    # Input:  (batch, N_residues, d_model)
    # Output: (batch, N_residues, 27) logits
    # Each residue independently samples from Categorical(27)
    # Joint log-prob = sum of per-residue log-probs (mean-field)
    # Joint entropy = sum of per-residue entropies
```

#### 4.9 `src/protein_fold_rl/model/value_head.py` — Value Function

```python
class ValueHead(nn.Module):
    # Input:  (batch, N_residues, d_model)
    # Output: (batch, 1) scalar value estimate
    # Attention-weighted pooling over residues → MLP → scalar
```

---

### Phase 4: PPO Training Loop (Day 3–4)

#### 4.10 `src/protein_fold_rl/rl/ppo.py` — PPO Implementation

Adapted from CleanRL's single-file PPO pattern. See Section 0.7 for the algorithm.

**Rollout collection:**
```
For each of num_steps (128):
    obs → transformer → features → policy_head → logits
    sample actions from Categorical(logits) for each residue
    compute log_prob = sum of per-residue log_probs
    compute value from value_head
    env.step(actions) → next_obs, reward, done
    store (obs, actions, log_probs, rewards, values, dones) in buffer
```

**GAE advantage estimation:**
```
Standard λ-return with γ=0.99, λ=0.95
```

**PPO update:**
```
For K=4 epochs:
    For each minibatch (4 minibatches):
        Recompute log_probs, values, entropy from stored obs/actions
        ratio = exp(new_log_prob - old_log_prob)
        clipped_ratio = clip(ratio, 1-0.2, 1+0.2)
        policy_loss = -min(ratio * advantage, clipped_ratio * advantage).mean()
        value_loss = 0.5 * (returns - values)².mean()
        entropy_loss = -0.01 * entropy.mean()
        loss = policy_loss + 0.5 * value_loss + entropy_loss
        optimizer.step()
```

#### 4.11 `src/protein_fold_rl/rl/rollout_buffer.py`

Standard fixed-size buffer storing tensors for observations, actions, log-probs, rewards, values, dones. Shaped for vectorized environments: `(num_steps, num_envs, ...)`.

#### 4.12 `src/protein_fold_rl/config.py` — All Hyperparameters

```python
@dataclass
class Config:
    # Environment
    pdb_id: str = "1L2Y"
    num_envs: int = 8
    max_steps_per_episode: int = 200
    expand_interval: int = 50
    n_init_resolution: int = 5
    bbox_padding: float = 5.0        # Å of padding around native structure

    # Reward shaping
    chain_penalty_weight: float = 1.0
    clash_penalty_weight: float = 0.5
    compact_penalty_weight: float = 0.1
    lddt_delta_weight: float = 1.0
    terminal_reward_scale: float = 10.0
    ca_bond_distance: float = 3.8    # Å
    min_contact_distance: float = 3.0 # Å

    # Model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256

    # PPO
    num_steps: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    total_timesteps: int = 2_000_000
    seed: int = 42
    log_interval: int = 10           # log every N updates
    save_interval: int = 100         # checkpoint every N updates
    wandb_project: str = "protein-fold-rl"
```

#### 4.13 Parallel Environments

Use Gymnasium's `SyncVectorEnv` to run `num_envs=8` copies in parallel. Each environment independently runs episodes with its own random SAW initialization.

---

### Phase 5: Training Script & Evaluation (Day 4–5)

#### 4.14 `scripts/train.py`

```
1. Parse config (from CLI args or config file)
2. Set seeds for reproducibility
3. Load native Cα coords from PDB 1L2Y
4. Create SyncVectorEnv with num_envs copies of ProteinFoldingEnv
5. Initialize model (ProteinTransformer + PolicyHead + ValueHead)
6. Initialize optimizer (Adam, lr=3e-4 with linear annealing to 0)
7. Initialize wandb run
8. Training loop:
   a. Collect num_steps of rollout data across all envs
   b. Compute GAE advantages and returns
   c. Run PPO update for update_epochs with num_minibatches
   d. Log metrics to wandb
   e. Periodically save checkpoints
9. Final evaluation and save
```

**Key metrics to log:**
- `episode/lddt` — primary success metric (target: >0.8)
- `episode/reward` — total shaped reward per episode
- `episode/chain_breaks` — number of Cα-Cα distances > 4.5 Å
- `episode/rg` — radius of gyration
- `episode/clash_count` — number of steric clashes
- `loss/policy`, `loss/value`, `loss/entropy`
- `diagnostics/approx_kl` — PPO stability (should stay <0.05)
- `diagnostics/clip_fraction` — how often clipping activates
- `diagnostics/explained_variance` — value function quality

#### 4.15 `scripts/evaluate.py`

- Load trained model checkpoint
- Run greedy rollout (argmax actions, no sampling)
- Extract final Cα positions in Ångströms
- Compute lDDT, RMSD (with optimal superposition) against native
- Print per-residue lDDT scores
- Visualize using matplotlib 3D scatter: predicted vs native Cα trace
- Optionally write output as PDB file for PyMOL visualization

---

## 5. Build Order for Claude Code

Execute in this order. Each step should be independently testable before moving to the next. Run `uv run pytest tests/test_<module>.py` after each step.

```
Step 1:  Project scaffolding
         → Create pyproject.toml, directory structure, __init__.py files
         → RUN: uv sync

Step 2:  src/protein_fold_rl/config.py
         → All hyperparams in a single dataclass, no external dependencies
         → TEST: import config, instantiate, verify defaults

Step 3:  src/protein_fold_rl/data/fetch_pdb.py
         → Fetch and parse PDB 1L2Y, extract Cα coordinates and sequence
         → TEST (tests/test_scoring.py):
           - Verify 20 residues returned
           - Verify sequence matches NLYIQWLKDGGPSSGRPPPS
           - Verify coordinates are centered (mean ≈ 0)
           - Verify all Cα-Cα sequential distances ≈ 3.8 Å

Step 4:  src/protein_fold_rl/env/voxel_grid.py
         → VoxelGrid class with expansion
         → TEST (tests/test_voxel_grid.py):
           - Init with n=5, verify shape
           - Expand 3 times, verify 5→9→17→33
           - Verify positions double correctly on expansion
           - Verify real_coords conversion roundtrips
           - Verify discretize_coords snaps correctly

Step 5:  src/protein_fold_rl/env/initialization.py
         → Self-avoiding walk generator
         → TEST (tests/test_initialization.py):
           - Generate SAW for N=20 on 5³ grid
           - Verify no duplicate positions
           - Verify all consecutive positions are face-adjacent (Manhattan dist = 1)
           - Verify all positions within grid bounds
           - Run 100 times, verify it always succeeds (no infinite loops)

Step 6:  src/protein_fold_rl/env/scoring.py
         → lDDT computation + reward shaping potentials
         → TEST (tests/test_scoring.py):
           - lDDT of native coords against themselves = 1.0
           - lDDT of randomly permuted coords << 1.0
           - lDDT of slightly perturbed coords (add 1 Å noise) ≈ 0.5–0.8
           - Chain potential: perfectly bonded chain → 0 penalty
           - Chain potential: broken chain → large negative
           - Clash potential: overlapping residues → large negative
           - Compact potential: R_g at target → 0 penalty

Step 7:  src/protein_fold_rl/env/protein_env.py
         → Full Gymnasium environment
         → TEST (tests/test_environment.py):
           - env.reset() returns valid observation shapes
           - Random actions for 200 steps: no crashes, all rewards finite
           - Verify grid expansion occurs at steps 50, 100, 150
           - Verify observation grid_resolution updates after expansion
           - gymnasium.utils.env_checker.check_env(env) passes

Step 8:  src/protein_fold_rl/model/rope_3d.py
         → 3D RoPE implementation
         → TEST (tests/test_model.py):
           - Output shape matches input shape
           - Different positions produce different rotations
           - Same position produces same rotation (deterministic)

Step 9:  src/protein_fold_rl/model/transformer.py
         → Transformer backbone
         → TEST (tests/test_model.py):
           - Random obs dict → features of shape (B, 20, 128)
           - Gradients flow (no detached tensors, no NaN)
           - Verify parameter count is ~500K

Step 10: src/protein_fold_rl/model/policy_head.py + value_head.py
         → Policy and value heads
         → TEST (tests/test_model.py):
           - End-to-end: obs → (action_logits [B,20,27], value [B,1])
           - Sample actions, compute log_probs and entropy
           - Verify log_probs sum correctly (mean-field)

Step 11: src/protein_fold_rl/rl/rollout_buffer.py + rl/utils.py
         → Rollout buffer and GAE computation
         → TEST (tests/test_ppo.py):
           - Fill buffer with dummy data, verify shapes
           - GAE with known rewards/values → verify against hand-computed advantages
           - Advantage normalization: mean≈0, std≈1

Step 12: src/protein_fold_rl/rl/ppo.py
         → Full PPO training loop
         → TEST (tests/test_ppo.py):
           - Run 1000 timesteps, verify loss is finite (no NaN)
           - Verify approx_kl stays reasonable (<0.1)
           - Verify clip_fraction is non-zero but not 1.0

Step 13: scripts/train.py
         → Main training entry point
         → RUN: uv run python scripts/train.py
           (train for 2M steps, monitor wandb)

Step 14: scripts/evaluate.py
         → Visualization and final evaluation
         → RUN: uv run python scripts/evaluate.py --checkpoint <path>
```

---

## 6. Key Design Decisions & Rationale

### Why Gymnasium (not raw env loop)?
Standard interface lets us use `VectorEnv` for parallelism, makes the code testable, and is compatible with future integration of other RL libraries (SB3, TorchRL) if we want to benchmark against their PPO implementations.

### Why CleanRL-style PPO (not SB3 or TorchRL)?
We need full control over how observations are encoded (custom transformer), how the multi-discrete action space log-probs are summed (mean-field), and how 3D RoPE integrates with attention. SB3 and TorchRL assume standard architectures. Rolling our own PPO (~200 lines) gives us this control.

### Why all-residue simultaneous moves (mean-field)?
Reduces episode length dramatically (20 residues all moving each step vs one at a time). The transformer's self-attention allows residues to "coordinate" through shared representations even though their action sampling is independent. The action space is O(27 × N) rather than O(27^N).

### Why fixed expansion schedule?
Removes one learned decision from the agent, simplifying the problem. Expand every 50 steps over 200 total → ~50 steps at each of 4 resolution levels. This naturally creates a coarse-to-fine curriculum.

### Why include Δ_lDDT in step reward for overfitting?
Since we're explicitly testing whether the architecture CAN solve the problem (not whether it generalizes), giving a direct lDDT signal maximizes our chance of seeing positive results. If overfitting fails even with this strong signal, the approach has fundamental issues. Ablate this away later.

---

## 7. Success Criteria

| Metric | Threshold | Notes |
|---|---|---|
| lDDT (final) | > 0.6 | Moderate structural agreement |
| lDDT (final) | > 0.8 | Strong — confirms approach works |
| Chain connectivity | 0 breaks | All sequential Cα-Cα within ~4.5 Å |
| Training time | < 12 hours | On single RTX 3090/4090 |
| Stability | No NaN/divergence | PPO approx_kl < 0.1 |

---

## 8. Scaling Notes (for later multi-A100 phase)

Once the single-protein overfitting experiment succeeds:

- **More environments:** 64–256 parallel envs across GPUs
- **Larger model:** d_model=256–512, 8–12 layers, ~10M parameters
- **Multi-protein training:** Sample different proteins each episode
- **Distributed PPO:** PyTorch DDP (CleanRL has a `ppo_atari_multigpu.py` reference)
- **Physics-based reward:** Replace target lDDT with energy function
- **3D RoPE alternatives:** Interleaved frequencies, learned frequencies, GATr geometric algebra
- **Mixed action space:** Fragment moves and pivot moves alongside single-residue

---

## 9. Known Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Mean-field actions break chain constantly | High | Heavy chain connectivity penalty; consider hard constraint (reject bond-breaking moves) as fallback |
| Sparse lDDT signal at coarse resolution | Medium | Include Δ_lDDT in step reward; the 4.0 Å and 2.0 Å thresholds provide signal even at coarse resolution |
| PPO gets stuck in local minimum (compact blob) | Medium | Entropy bonus; diverse random SAW initializations; curriculum from grid expansion |
| Grid expansion disrupts learned policy | Medium | Physical coordinates stay stable across expansion (just doubled in voxel space); warm-start at new resolution |
| 200 steps insufficient for 20 residues | Low | Can increase; 20 residues × 200 steps = 4000 total residue-moves |
| Transformer overfits to specific SAW init | Low | Random SAW each reset; 8 parallel envs with different inits |
| Off-by-one errors in expansion | Medium | Extensive unit tests for voxel_grid.expand(); verify real-space coords are preserved |
