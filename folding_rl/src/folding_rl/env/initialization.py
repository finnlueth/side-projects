"""Self-avoiding walk initialization for residue positions on the voxel grid."""
import numpy as np


# 6 face-adjacent moves (±x, ±y, ±z)
_FACE_MOVES = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
], dtype=np.int32)


def self_avoiding_walk(
    n_residues: int,
    grid_size: int,
    rng: np.random.Generator | None = None,
    max_retries: int = 1000,
) -> np.ndarray:
    """Generate a random self-avoiding walk starting from the grid center.

    Uses only 6 face-adjacent moves (not 26) so consecutive residues are
    exactly 1 voxel apart, matching the Cα-Cα bond distance at initial spacing.

    Args:
        n_residues: Number of residues (chain length)
        grid_size: Number of voxels per axis
        rng: NumPy random generator for reproducibility
        max_retries: Maximum restart attempts if trapped

    Returns:
        (n_residues, 3) int32 array of voxel coordinates
    """
    if rng is None:
        rng = np.random.default_rng()

    center = grid_size // 2

    for _ in range(max_retries):
        positions = np.zeros((n_residues, 3), dtype=np.int32)
        positions[0] = center

        occupied = {(center, center, center)}
        success = True

        for i in range(1, n_residues):
            current = positions[i - 1]
            # Shuffle available moves
            move_order = rng.permutation(len(_FACE_MOVES))
            placed = False
            for m_idx in move_order:
                candidate = current + _FACE_MOVES[m_idx]
                key = tuple(candidate)
                if (
                    np.all(candidate >= 0)
                    and np.all(candidate < grid_size)
                    and key not in occupied
                ):
                    positions[i] = candidate
                    occupied.add(key)
                    placed = True
                    break

            if not placed:
                success = False
                break

        if success:
            return positions

    raise RuntimeError(
        f"Self-avoiding walk failed after {max_retries} retries for "
        f"n_residues={n_residues}, grid_size={grid_size}"
    )
