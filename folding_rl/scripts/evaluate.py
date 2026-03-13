"""Evaluate a trained checkpoint: compute lDDT, RMSD, export PDB."""
from __future__ import annotations

import argparse
import numpy as np
import torch

from folding_rl.config import Config
from folding_rl.data.fetch_pdb import load_ca_coords
from folding_rl.env.protein_env import ProteinFoldingEnv
from folding_rl.rl.ppo import PPOLightning

torch.serialization.add_safe_globals([Config])


_AA3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


def superpose_rmsd(pred: np.ndarray, ref: np.ndarray) -> float:
    """Compute RMSD after optimal Kabsch superposition."""
    pred_c = pred - pred.mean(0)
    ref_c = ref - ref.mean(0)
    H = pred_c.T @ ref_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    rotated = pred_c @ R.T
    return float(np.sqrt(np.mean(np.sum((rotated - ref_c) ** 2, axis=-1))))


def write_ca_pdb(
    coords: np.ndarray,
    sequence: list[str],
    path: str,
    lddt: float | None = None,
) -> None:
    """Write Cα-only PDB file from coordinates and sequence."""
    lines = []
    if lddt is not None:
        lines.append(f"REMARK   lDDT = {lddt:.4f}")
    for i, (coord, aa) in enumerate(zip(coords, sequence)):
        resname = _AA3.get(aa, "UNK")
        lines.append(
            f"ATOM  {i+1:5d}  CA  {resname:3s} A{i+1:4d}    "
            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
            f"  1.00  0.00           C  "
        )
    lines.append("END")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def greedy_rollout(model: PPOLightning, env: ProteinFoldingEnv) -> tuple[np.ndarray, float]:
    """Run a greedy (argmax) episode. Returns (final_real_coords, lddt)."""
    obs, _ = env.reset()
    device = next(model.parameters()).device
    done = False

    while not done:
        obs_torch = {
            k: torch.tensor(v, dtype=torch.int32 if v.dtype != np.float32 else torch.float32,
                            device=device).unsqueeze(0)
            for k, v in obs.items()
        }
        with torch.no_grad():
            features = model.transformer(obs_torch)
            logits = model.policy_head(features)        # (1, N, 27)
            actions = logits.argmax(dim=-1).squeeze(0)  # (N,)

        obs, _, done, _, info = env.step(actions.cpu().numpy())

    final_coords = env._grid.get_real_coords()
    lddt = info["lddt"]
    return final_coords, lddt


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--output-pdb", type=str, default="prediction.pdb",
                        help="Output PDB file for best prediction")
    args = parser.parse_args()

    cfg = Config()
    native_coords, sequence = load_ca_coords(cfg.pdb_id)
    print(f"Sequence: {''.join(sequence)}")

    model = PPOLightning.load_from_checkpoint(
        args.checkpoint, config=cfg, env_factory=lambda: None, strict=False
    )
    model.eval()

    env = ProteinFoldingEnv(config=cfg)

    lddts, rmsds, all_coords = [], [], []
    for i in range(args.n_rollouts):
        final_coords, lddt = greedy_rollout(model, env)
        rmsd = superpose_rmsd(final_coords, native_coords)
        print(f"Rollout {i+1:3d}: lDDT={lddt:.4f}  RMSD={rmsd:.2f} Å")
        lddts.append(lddt)
        rmsds.append(rmsd)
        all_coords.append(final_coords)

    print(f"\nMean lDDT: {np.mean(lddts):.4f} ± {np.std(lddts):.4f}")
    print(f"Mean RMSD: {np.mean(rmsds):.2f} ± {np.std(rmsds):.2f} Å")

    # Export best prediction as PDB
    best_idx = int(np.argmax(lddts))
    best_coords = all_coords[best_idx]
    write_ca_pdb(best_coords, sequence, args.output_pdb, lddt=lddts[best_idx])
    print(f"\nSaved best prediction to {args.output_pdb} (lDDT={lddts[best_idx]:.4f})")

    # Also export native for comparison
    native_pdb = args.output_pdb.replace(".pdb", "_native.pdb")
    write_ca_pdb(native_coords, sequence, native_pdb, lddt=1.0)
    print(f"Saved native structure to {native_pdb}")


if __name__ == "__main__":
    main()
