"""Load CIF files, extract Cα coordinates and amino acid sequence."""
from pathlib import Path
import numpy as np
from Bio import PDB
from Bio.Data.IUPACData import protein_letters_3to1


_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


def load_ca_coords(pdb_id: str = "1L2Y") -> tuple[np.ndarray, list[str]]:
    """Parse CIF file, return (coords, sequence) for model 0.

    Args:
        pdb_id: PDB ID string (e.g. "1L2Y")

    Returns:
        coords: (N, 3) float32 array of Cα coordinates in Å, centered at origin
        sequence: list of 1-letter amino acid codes
    """
    cif_path = _DATA_DIR / f"{pdb_id}.cif"
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(cif_path))

    # Use model 0 (first conformer for NMR ensembles)
    model = structure[0]

    coords = []
    sequence = []
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":  # skip HETATM / water
                continue
            if "CA" not in residue:
                continue
            ca = residue["CA"]
            coords.append(ca.get_coord())
            resname = residue.get_resname().strip()
            one_letter = protein_letters_3to1.get(resname.capitalize(), "X")
            sequence.append(one_letter)

    coords = np.array(coords, dtype=np.float32)
    # Center to origin
    coords -= coords.mean(axis=0)
    return coords, sequence
