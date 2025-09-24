"""
Pair Potential Parameter Setup for HOOMD Simulations
---------------------------------------------------
Defines the function to set Morse and tabulated pair potentials for all relevant type pairs.
- Uses job.sp.d_types and job.sp.D0 to set Morse D0 values for specific pairs.
- Sets r_cut and table parameters for excluded pairs.
"""

import numpy as np


def set_pair_potentials_params(job, morse, table, table_func, types):
    """
    Set Morse and tabulated pair potential parameters for all type pairs.
    - job: signac job with parameters (d_types, D0, alpha, r0, etc.)
    - morse: HOOMD Morse pair potential object
    - table: HOOMD Table pair potential object
    - table_func: (U, F) arrays for tabulated repulsion
    - types: list of all particle types
    """
    # Mapping: 1→AP1 (3), 2→AP2 (5), 3→BP1 (6), 4→BP2 (8), 5→CP1 (9), 6→CP2 (11)
    mapping = {1: 3, 2: 5, 3: 6, 4: 8, 5: 9, 6: 11}

    # Ordered list of pairs for which d_types are provided
    d_mores_pairs = [
        (2, 3), (4, 5),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
        (2, 4), (2, 5), (2, 6),
        (3, 4), (3, 5), (3, 6),
        (4, 6),
        (5, 6),
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)
    ]

    # Initialize Morse D0 matrix for all 12 types
    morse_d0s = np.zeros((12, 12))

    # Assign D0 values for each relevant pair
    for (a, b), potential in zip(d_mores_pairs, job.sp.d_types):
        i, j = mapping[a], mapping[b]
        if i < j:
            i, j = j, i  # Fill lower triangle for consistency
        morse_d0s[i, j] = job.sp.D0 * potential

    # Set Morse potential parameters for all pairs
    for i, type_i in enumerate(types):
        for j, type_j in enumerate(types):
            if i >= j:
                morse.params[(type_i, type_j)] = dict(
                    alpha=job.sp.alpha,
                    D0=morse_d0s[i, j],
                    r0=job.sp.r0
                )

    # Set r_cut=0 for pairs not meant to interact (from morse_r_cuts matrix)
    morse_r_cuts = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    ])
    for i, type_i in enumerate(types):
        for j, type_j in enumerate(types):
            if i >= j and morse_r_cuts[i, j] == 0.0:
                morse.r_cut[(type_i, type_j)] = 0

    # Set table (tabulated repulsion) parameters for all pairs
    for i, type_i in enumerate(types):
        for j, type_j in enumerate(types):
            if i >= j:
                table.params[(type_i, type_j)] = dict(
                    r_min=0, U=table_func[0], F=table_func[1]
                )

    # Set r_cut=0 for table pairs not meant to interact (from table_r_cuts matrix)
    table_r_cuts = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    for i, type_i in enumerate(types):
        for j, type_j in enumerate(types):
            if i >= j and table_r_cuts[i, j] == 0.0:
                table.params[(type_i, type_j)] = dict(r_min=0, U=[0], F=[0])
                table.r_cut[(type_i, type_j)] = 0
