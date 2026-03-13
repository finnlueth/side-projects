from dataclasses import dataclass


@dataclass
class Config:
    # Environment
    pdb_id: str = "1L2Y"
    num_envs: int = 8
    max_steps_per_episode: int = 300
    expand_interval: int = 100         # expand at steps 100 and 200 → 9→17→33
    n_init_resolution: int = 9
    bbox_padding: float = 5.0        # Å of padding around native structure
    n_residues: int = 20

    # Reward shaping
    # Weights are small to keep |Φ(s)| ≈ O(1-10), preventing the (1-γ) leak
    # from drowning out the lDDT signal over 200-step episodes.
    chain_penalty_weight: float = 0.1
    clash_penalty_weight: float = 0.05
    compact_penalty_weight: float = 0.02
    lddt_delta_weight: float = 50.0
    terminal_reward_scale: float = 100.0
    ca_bond_distance: float = 3.8    # Å
    min_contact_distance: float = 3.0  # Å
    rg_target: float = 7.0           # Å, for Trp-cage (N=20)

    # Model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 512

    # PPO
    num_steps: int = 256
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
    total_timesteps: int = 5_000_000
    seed: int = 42
    log_interval: int = 10           # log every N updates
    save_interval: int = 100         # checkpoint every N updates
    wandb_project: str = "protein-fold-rl"
    use_wandb: bool = True
