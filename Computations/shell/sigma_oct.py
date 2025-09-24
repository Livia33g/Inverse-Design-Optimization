"""
Compute Symmetry Numbers for Octahedral Shell Subsets
----------------------------------------------------
This script calculates the symmetry number for each .pos file (octahedral subset)
in a specified folder, using all 24 octahedral group rotations.

- Outputs a CSV file with symmetry numbers for each input file.
- Requires scipy >= 1.8 for Rotation.create_group('O').
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import os


def compute_center_of_mass(coordinates):
    """Return the center of mass of a set of coordinates."""
    return np.mean(coordinates, axis=0)


def generate_octahedral_rotations():
    """Generate all 24 rotation matrices for the octahedral (O) group."""
    return R.create_group("O").as_matrix()


def validate_unique_rotations(rotations):
    """Check for uniqueness among rotation matrices (for debugging)."""
    unique_rotations = set()
    for rotation in rotations:
        key = tuple(np.round(rotation.flatten(), decimals=6))
        unique_rotations.add(key)
    return len(unique_rotations), unique_rotations


def align_coordinates(coords):
    """Sort and round coordinates for robust comparison."""
    return np.round(np.sort(coords, axis=0), decimals=6)


def compute_symmetry_number(coordinates, rotations, atol=0.558670741):
    """
    Compute the symmetry number for a set of coordinates under a group of rotations.
    Returns the number of rotations that leave the structure invariant (within atol).
    """
    com = compute_center_of_mass(coordinates)
    centered_coords = coordinates - com
    aligned_original = align_coordinates(centered_coords)

    symmetry_count = 0
    for rotation in rotations:
        rotated_coords = centered_coords @ rotation.T
        aligned_rotated = align_coordinates(rotated_coords)
        if np.allclose(aligned_rotated, aligned_original, atol=atol):
            symmetry_count += 1
    return symmetry_count


def load_coordinates(file_path):
    """Load vertex coordinates from a .pos file, skipping headers and non-vertex lines."""
    coordinates = []
    with open(file_path, "r") as file:
        for line in file:
            if line.strip() and not line.startswith(("boxMatrix", "def", "eof")):
                parts = line.split()
                if len(parts) == 4 and parts[0] in ("V", "P"):
                    coordinates.append(list(map(float, parts[1:])))
    return np.array(coordinates)


if __name__ == "__main__":
    # Folder containing .pos files for octahedral subsets
    pos_files_folder = "oct_files"
    output_file = "symmetry_numbers_oct.txt"

    # Generate all octahedral group rotations
    rotations = generate_octahedral_rotations()

    with open(output_file, "w") as f:
        f.write("File Name,Symmetry Number\n")
        for pos_file in sorted(os.listdir(pos_files_folder)):
            if not pos_file.endswith(".pos"):
                continue
            file_path = os.path.join(pos_files_folder, pos_file)
            coords = load_coordinates(file_path)
            if coords.size == 0:
                f.write(f"{pos_file},No vertices found\n")
                continue
            symmetry_number = compute_symmetry_number(coords, rotations)
            print(f"{pos_file}: Symmetry Number = {symmetry_number}")
            f.write(f"{pos_file},{symmetry_number}\n")
