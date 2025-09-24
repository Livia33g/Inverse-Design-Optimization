import pickle
import jax.numpy as jnp
import os
import argparse


def extract_tetramer_info(input_file, output_file):
    """
    Extract tetramer information from the input file and save it to a separate file.

    Parameters:
        input_file (str): Path to the input .pkl file containing species information.
        output_file (str): Path to save the extracted tetramer information.
    """
    # Load the existing data
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    # Extract tetramer-specific information
    tetramer_species_key = "4_pc_species"
    tetramer_sigma_key = "4_sigma"
    tetramer_data = {}

    if tetramer_species_key in data:
        tetramer_data[tetramer_species_key] = data[tetramer_species_key]
    else:
        raise KeyError(f"{tetramer_species_key} not found in the input file.")

    if tetramer_sigma_key in data:
        tetramer_data[tetramer_sigma_key] = data[tetramer_sigma_key]
    else:
        raise KeyError(f"{tetramer_sigma_key} not found in the input file.")

    # Extract monomer counts for tetramers
    monomer_keys = [key for key in data.keys() if key.endswith("_4_counts")]
    for key in monomer_keys:
        tetramer_data[key] = data[key]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the tetramer-specific information
    with open(output_file, "wb") as f:
        pickle.dump(tetramer_data, f)

    print(f"Tetramer information saved to {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tetramer data from species file.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input .pkl file containing species information.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the extracted tetramer information.",
    )
    args = parser.parse_args()
    extract_tetramer_info(args.input, args.output)
