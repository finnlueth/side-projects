"""3D voxel grid state management with resolution expansion."""
import numpy as np


class VoxelGrid:
    """Manages residue positions on a 3D integer voxel grid.

    Physical bounding box is centered at the origin and stays fixed in size.
    Voxel spacing halves on each expansion.

    Coordinate system:
        voxel index 0 → real coordinate -bbox_size/2
        voxel index (resolution-1) → real coordinate +bbox_size/2
    """

    def __init__(self, bbox_size: float, n_init: int = 5):
        """
        Args:
            bbox_size: Physical size of the grid in Å (cubic)
            n_init: Initial grid resolution (voxels per axis)
        """
        self.bbox_size = bbox_size
        self.resolution = n_init
        self.voxel_spacing = bbox_size / (n_init - 1)
        self.origin = -bbox_size / 2.0  # real-space coordinate of voxel index 0
        self.positions: np.ndarray | None = None  # (N_residues, 3) int32

    def set_positions(self, positions: np.ndarray) -> None:
        """Set residue positions (N, 3) integer voxel coordinates."""
        self.positions = np.array(positions, dtype=np.int32)

    def get_real_coords(self) -> np.ndarray:
        """Convert voxel indices to Ångström coordinates (centered at origin).

        Returns:
            (N, 3) float32 array of Cα coordinates in Å
        """
        return self.positions.astype(np.float32) * self.voxel_spacing + self.origin

    def expand(self) -> None:
        """Double resolution: n → 2n-1. Voxel coordinates are doubled."""
        new_resolution = 2 * self.resolution - 1
        self.positions = self.positions * 2
        self.resolution = new_resolution
        self.voxel_spacing = self.bbox_size / (new_resolution - 1)
        # origin stays the same — grid spans [-bbox_size/2, +bbox_size/2]

    def discretize_coords(self, real_coords: np.ndarray) -> np.ndarray:
        """Snap real-space Å coordinates to nearest voxel indices.

        Args:
            real_coords: (N, 3) float array in Å (centered at origin)

        Returns:
            (N, 3) int32 array of voxel indices, clipped to [0, resolution-1]
        """
        voxel_coords = (real_coords - self.origin) / self.voxel_spacing
        voxel_coords = np.round(voxel_coords).astype(np.int32)
        voxel_coords = np.clip(voxel_coords, 0, self.resolution - 1)
        return voxel_coords

    def is_valid_position(self, pos: np.ndarray) -> bool:
        """Check that all coordinates are in [0, resolution-1]."""
        return bool(np.all(pos >= 0) and np.all(pos < self.resolution))

    @property
    def max_coord(self) -> int:
        return self.resolution - 1
