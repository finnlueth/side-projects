"""lDDT scoring and potential-based reward shaping."""
import numpy as np
from scipy.spatial.distance import cdist


# lDDT thresholds in Ångströms
_LDDT_THRESHOLDS = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
_LDDT_R0 = 15.0  # inclusion radius in Å


def compute_lddt(
    model_coords: np.ndarray,
    ref_coords: np.ndarray,
    inclusion_radius: float = _LDDT_R0,
    thresholds: np.ndarray = _LDDT_THRESHOLDS,
) -> float:
    """Cα-only lDDT score (superposition-free).

    Args:
        model_coords: (N, 3) predicted Cα coordinates in Å
        ref_coords: (N, 3) reference (native) Cα coordinates in Å
        inclusion_radius: Only count pairs within this distance in the reference
        thresholds: Distance thresholds for "preserved" pairs

    Returns:
        lDDT score in [0, 1]
    """
    n = len(ref_coords)
    ref_dists = cdist(ref_coords, ref_coords)  # (N, N)
    model_dists = cdist(model_coords, model_coords)  # (N, N)

    # Mask: pairs within inclusion radius, excluding self (diagonal)
    mask = (ref_dists < inclusion_radius) & (np.eye(n, dtype=bool) == False)

    if mask.sum() == 0:
        return 0.0

    ref_d = ref_dists[mask]       # (M,)
    model_d = model_dists[mask]   # (M,)
    diff = np.abs(model_d - ref_d)  # (M,)

    # Fraction preserved for each threshold
    preserved = [(diff < t).mean() for t in thresholds]
    return float(np.mean(preserved))


# ---------------------------------------------------------------------------
# Reward shaping potentials
# ---------------------------------------------------------------------------

def chain_connectivity_potential(
    real_coords: np.ndarray,
    bond_distance: float = 3.8,
    weight: float = 1.0,
) -> float:
    """Penalize Cα-Cα bond length deviations from expected 3.8 Å.

    Only penalizes distances that are *larger* than bond_distance (chain breaks),
    not compression. Returns a non-positive value (0 = perfect).
    """
    diffs = real_coords[1:] - real_coords[:-1]
    dists = np.linalg.norm(diffs, axis=-1)
    penalty = np.sum(np.maximum(0.0, dists - bond_distance))
    return -weight * float(penalty)


def steric_clash_potential(
    real_coords: np.ndarray,
    min_distance: float = 3.0,
    weight: float = 0.5,
) -> float:
    """Penalize pairs of residues that are closer than min_distance.

    Returns a non-positive value (0 = no clashes).
    """
    n = len(real_coords)
    if n < 2:
        return 0.0
    dists = cdist(real_coords, real_coords)
    # Upper triangle only, exclude diagonal
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair_dists = dists[triu_mask]
    penalty = np.sum(np.maximum(0.0, min_distance - pair_dists))
    return -weight * float(penalty)


def compactness_potential(
    real_coords: np.ndarray,
    rg_target: float = 7.0,
    weight: float = 0.1,
) -> float:
    """Penalize deviation of radius of gyration from target.

    Returns a non-positive value (0 = exact match).
    """
    center = real_coords.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((real_coords - center) ** 2, axis=-1))))
    return -weight * abs(rg - rg_target)


def compute_potential(
    real_coords: np.ndarray,
    bond_distance: float = 3.8,
    min_contact_distance: float = 3.0,
    rg_target: float = 7.0,
    chain_weight: float = 1.0,
    clash_weight: float = 0.5,
    compact_weight: float = 0.1,
) -> float:
    """Sum of all shaping potentials."""
    return (
        chain_connectivity_potential(real_coords, bond_distance, chain_weight)
        + steric_clash_potential(real_coords, min_contact_distance, clash_weight)
        + compactness_potential(real_coords, rg_target, compact_weight)
    )


def shaped_step_reward(
    prev_coords: np.ndarray,
    curr_coords: np.ndarray,
    ref_coords: np.ndarray,
    lddt_delta_weight: float = 1.0,
    bond_distance: float = 3.8,
    min_contact_distance: float = 3.0,
    rg_target: float = 7.0,
    chain_weight: float = 1.0,
    clash_weight: float = 0.5,
    compact_weight: float = 0.1,
) -> tuple[float, float, float]:
    """Compute shaped step reward using potential difference + Δ_lDDT.

    Uses Φ(s') - Φ(s) (no gamma scaling) so the total shaped reward over an
    episode telescopes cleanly to Φ(s_T) - Φ(s_0) with no (1-γ) leak.

    Returns:
        (total_reward, lddt_curr, potential_curr)
    """
    phi_prev = compute_potential(
        prev_coords, bond_distance, min_contact_distance, rg_target,
        chain_weight, clash_weight, compact_weight,
    )
    phi_curr = compute_potential(
        curr_coords, bond_distance, min_contact_distance, rg_target,
        chain_weight, clash_weight, compact_weight,
    )
    shaping = phi_curr - phi_prev

    lddt_prev = compute_lddt(prev_coords, ref_coords)
    lddt_curr = compute_lddt(curr_coords, ref_coords)
    delta_lddt = lddt_curr - lddt_prev

    reward = shaping + lddt_delta_weight * delta_lddt
    return float(reward), float(lddt_curr), float(phi_curr)
