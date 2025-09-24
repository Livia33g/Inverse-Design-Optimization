"""
Generates all unique polymer chain combinations, counts monomer occurrences, and calculates sigma values for each structure size.

"""

import itertools
import pickle
import jax.numpy as jnp
from concurrent.futures import ThreadPoolExecutor
import os


def create_monomers(num_monomers):
    """
    Create a dictionary of monomer types with associated numeric representations.
    Args:
        num_monomers (int): Number of unique monomer types.
    Returns:
        dict: Mapping from monomer name to numeric list.
    """
    monomers = {}
    for i in range(num_monomers):
        letter = chr(ord("A") + i)
        monomers[letter] = [2 * i + 1, 0, 2 * i + 2]
    return monomers


def valid_even_odd_sequence(numeric_combination):
    """
    Check if a numeric combination meets the even-odd condition.
    Args:
        numeric_combination (list): Numeric representation of a polymer chain.
    Returns:
        bool: True if valid, False otherwise.
    """
    for i in range(1, len(numeric_combination) - 1):
        if numeric_combination[i] != 0 and numeric_combination[i + 1] != 0:
            if numeric_combination[i] % 2 == numeric_combination[i + 1] % 2:
                return False
    return True


def generate_combinations(monomers, max_struc_size):
    """
    Generate unique combinations of monomers up to a given size.
    Args:
        monomers (dict): Monomer dictionary.
        max_struc_size (int): Maximum structure size.
    Returns:
        tuple: (unique_combinations, all_monomers)
    """
    monomers_prime = {f"{k}'": v[::-1] for k, v in monomers.items()}
    all_monomers = {**monomers, **monomers_prime}

    all_combinations = []
    for r in range(1, max_struc_size + 1):
        all_combinations.extend(itertools.product(all_monomers.keys(), repeat=r))

    unique_combinations = []
    seen = set()
    for comb in all_combinations:
        numeric_combination = sum([all_monomers[name] for name in comb], [])
        if valid_even_odd_sequence(numeric_combination):
            mirrored_comb = tuple(
                [k + "'" if k[-1] != "'" else k[:-1] for k in comb][::-1]
            )
            if mirrored_comb not in seen:
                unique_combinations.append(comb)
                seen.add(comb)
    return unique_combinations, all_monomers


def combination_to_string(comb):
    """Convert a tuple of monomer names to a space-separated string."""
    return " ".join(comb)


def get_numeric_combination(comb_str, all_monomers):
    """Convert a string of monomer names to a numeric combination."""
    monomer_names = comb_str.split()
    numeric_combination = sum([all_monomers[name] for name in monomer_names], [])
    return numeric_combination


def count_monomers(combinations, monomer_name):
    """Count the occurrences of a monomer in each combination."""
    monomer_counts = []
    for comb in combinations:
        count = sum(
            1 for mon in comb if mon == monomer_name or mon == f"{monomer_name}'"
        )
        monomer_counts.append(count)
    return monomer_counts


def calculate_sigma(size):
    """Calculate sigma based on structure size."""
    return 3 ** (size + 1)


def process_combinations(unique_combinations, all_monomers, monomers, max_struc_size):
    """
    Process combinations and calculate species, counts of each monomer type in combinations, and sigma for each size.
    Args:
        unique_combinations (list): List of unique combinations.
        all_monomers (dict): All monomer representations.
        monomers (dict): Original monomer dictionary.
        max_struc_size (int): Maximum structure size.
    Returns:
        dict: Combined results for all sizes.
    """
    species_dict = {}
    counts = {}
    sigmas = {}

    for size in range(1, max_struc_size + 1):
        structure_key = f"{size}_pc_species"
        current_combinations = [
            comb for comb in unique_combinations if len(comb) == size
        ]

        # Store species
        species_dict[structure_key] = jnp.array(
            [
                get_numeric_combination(combination_to_string(comb), all_monomers)
                for comb in current_combinations
            ]
        )

        # Store monomer counts
        for letter in monomers.keys():
            count_key = f"{letter}_{size}_counts"
            counts[count_key] = jnp.array(count_monomers(current_combinations, letter))

        # Store sigma value
        sigma_key = f"{size}_sigma"
        sigma_value = calculate_sigma(size)
        sigmas[sigma_key] = sigma_value
        print(f"Sigma for size {size}: {sigma_value}")

    return {**species_dict, **counts, **sigmas}


def save_results(data, filename):
    """Save the results dictionary to a pickle file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def main(num_monomers=3, target_size=4, output_dir="input_data"):
    """
    Main function to generate and save polymer chain data for a target size and for all sizes up to num_monomers.
    Args:
        num_monomers (int): Number of monomer types.
        target_size (int): Target structure size (e.g., 4 for tetramers).
        output_dir (str): Directory to save output files.
    """
    monomers = create_monomers(num_monomers)
    # 1. Save for the target size only
    unique_combinations, all_monomers = generate_combinations(monomers, target_size)
    with ThreadPoolExecutor() as executor:
        future = executor.submit(
            process_combinations,
            unique_combinations,
            all_monomers,
            monomers,
            target_size,
        )
        result = future.result()
    target_filename = os.path.join(output_dir, f"species_targetsize{target_size}.pkl")
    save_results(result, target_filename)
    print(f"Saved target size data to {target_filename}")

    # 2. Save for all sizes up to num_monomers
    unique_combinations_all, all_monomers_all = generate_combinations(
        monomers, num_monomers
    )
    with ThreadPoolExecutor() as executor:
        future_all = executor.submit(
            process_combinations,
            unique_combinations_all,
            all_monomers_all,
            monomers,
            num_monomers,
        )
        result_all = future_all.result()
    all_filename = os.path.join(output_dir, f"species_all_upto{num_monomers}.pkl")
    save_results(result_all, all_filename)
    print(f"Saved all sizes data to {all_filename}")

    print("Species combinations, counts, and sigma values saved successfully.")
    print(sum(len(result[f"{i}_pc_species"]) for i in range(1, target_size + 1)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate polymer chain species and save as pickle files."
    )
    parser.add_argument(
        "--num-monomers", type=int, default=3, help="Number of monomer types."
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=4,
        help="Target structure size (e.g., 4 for tetramers).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="input_data",
        help="Directory to save output .pkl files.",
    )
    args = parser.parse_args()
    main(num_monomers=args.num_monomers, target_size=args.target_size, output_dir=args.output_dir)
