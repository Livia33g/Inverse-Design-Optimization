"""
optimize.py
-----------

This script performs optimization of interaction parameters and concentrations for polymer chain assembly,
with a focus on maximizing the yield of a target structure under physical and mass action constraints.

Features:
- Supports user-facing configuration via argparse (interaction strengths, concentrations, output, etc.)
- Clear separation between general (all sizes) and tetramer-specific (n=4) data
- Mass action constraint can be toggled via command-line flag

Example usage:
    python Mass_all.py --eps_weak 2.0 --eps_init 6.2 --kt 1.0 --init_conc 0.001 \
        --desired_yield 0.4 --output results.txt --use_mass_action --sim_particles 300 --outer_iters 450

Reproducibility:
- The random seed, all key parameters, and results are printed and saved to the output file.
- All dependencies and environment details should be documented in the project README or environment file.
"""

import argparse
import numpy as onp
import pickle
import time
import jax.numpy as jnp
import optax
from jax import (
    random,
    vmap,
    hessian,
    jacfwd,
    jit,
    value_and_grad,
    grad,
    lax,
    checkpoint,
    clear_backends,
)
from tqdm import tqdm
from jax_md import space
import potentials
import utils
import jax_transformations3d as jts
from jaxopt import implicit_diff, GradientDescent
from checkpoint import checkpoint_scan
import functools
import itertools
from itertools import permutations, product
from functools import wraps
import matplotlib.pyplot as plt
from jax.config import config
import os
from jax.nn import relu, sigmoid, softplus
import networkx as nx

# Set up argument Parsing
parser = argparse.ArgumentParser(description="Simulation with adjustable parameters.")
parser.add_argument(
    "--eps_weak",
    type=float,
    default=2,
    help="weak epsilon values (attraction strengths).",
)
parser.add_argument(
    "--eps_init",
    type=float,
    default=6.2,
    help="init strong epsilon values (attraction strengths).",
)
parser.add_argument(
    "--kt", type=float, default=1.0, help="Thermal energy (kT). Default is 1.0."
)
parser.add_argument(
    "--init_conc",
    type=float,
    default=0.001,
    help="Initial concentration. Default is 0.001.",
)
parser.add_argument(
    "--desired_yield", type=float, default=0.4, help="desired yield of target."
)
parser.add_argument(
    "--output", type=str, default="results.txt", help="Output file to save the results."
)
parser.add_argument(
    "--use_mass_action",
    action="store_true",
    help="Enable the mass action constraint in the loss function.",
)
parser.add_argument(
    "--sim_particles",
    type=int,
    default=300,
    help="total number of particles wanted in the simulation.",
)
parser.add_argument(
    "--outer_iters",
    type=int,
    default=450,
    help="total number of optimization simulations.",
)
parser.add_argument(
    "--input_dir",
    type=str,
    default="input_data",
    help="Directory containing input .pkl data files (species_all_upto3.pkl, species_targetsize4.pkl).",
)
args = parser.parse_args()

V = args.sim_particles / args.init_conc

output_file = args.output
use_mass_action = args.use_mass_action
input_dir = args.input_dir


# Define constants
a = 1.0  # Radius placeholder
b = 0.3
separation = 2.0
noise = 1e-14


def safe_log(x, eps=1e-10):
    return jnp.log(jnp.clip(x, a_min=eps, a_max=None))


target = jnp.array([1, 0, 2, 3, 0, 4, 5, 0, 6])

use_custom_pairs = True
custom_pairs = [
    (2, 3),
    (4, 5),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (2, 4),
    (2, 5),
    (2, 6),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 6),
    (5, 6),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
]

custom_pairs_n = len(custom_pairs)


def load_species_combinations(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

# Load data files from the specified input directory
species_all_path = os.path.join(input_dir, "species_all_upto3.pkl")
species_target4_path = os.path.join(input_dir, "species_targetsize4.pkl")
data = load_species_combinations(species_all_path)
data_4 = load_species_combinations(species_target4_path)

num_monomers = max(
    int(k.split("_")[0]) for k in data.keys() if k.endswith("_pc_species")
)


species_data = {}
tot_num_structures = 0

for i in range(1, num_monomers + 1):
    key = f"{i}_pc_species"
    species_data[key] = data[key]
    tot_num_structures += species_data[key].shape[0]


# Function to find the index of a target structure
def indx_of_target(target, species_data):
    target_reversed = target[::-1]
    num_monomers = len(species_data)

    offset = 0
    for i in range(1, num_monomers + 1):
        key = f"{i}_pc_species"
        current_species = species_data[key]
        for j in range(current_species.shape[0]):
            if jnp.array_equal(current_species[j], target) or jnp.array_equal(
                current_species[j], target_reversed
            ):
                return j + offset
        offset += current_species.shape[0]

    return None


target_idx = indx_of_target(target, species_data)

euler_scheme = "sxyz"

SEED = 42
main_key = random.PRNGKey(SEED)

kT = args.kt
kT_val = jnp.array([kT])
n = num_monomers

# Shape and energy helper functions
a = 1.0  # distance of the center of the spheres from the BB COM
b = 0.3  # distance of the center of the patches from the BB COM
separation = 2.0
noise = 1e-14
vertex_radius = a
patch_radius = 0.2 * a
small_value = 1e-12
vertex_species = 0
n_patches = n * 2  # 2 species of patches per monomer type
n_species = n_patches + 1  # plus the common vertex species 0

n_morse_vals = (
    n_patches * (n_patches - 1) // 2 + n_patches
)  # all possible pair permutations plus same patch attraction (i,i)
patchy_vals_weak = jnp.full(
    custom_pairs_n - 2, args.eps_weak
)  # FIXME for optimization over specific attraction strengths
patchy_vals_strong = jnp.full(2, args.eps_init)

init_conc = args.init_conc
m_conc = init_conc / n
init_concs = jnp.full(n, m_conc)
# init_params = jnp.concatenate([patchy_vals, init_concs])
# init_params = patchy_vals

patchy_vals = jnp.concatenate([patchy_vals_strong, patchy_vals_weak])
init_params = jnp.concatenate([patchy_vals, kT_val, init_concs])


def make_shape(size):
    base_shape = jnp.array(
        [
            [-a, 0.0, b],  # first patch
            [-a, b * jnp.cos(jnp.pi / 6.0), -b * jnp.sin(jnp.pi / 6.0)],  # second patch
            [-a, -b * jnp.cos(jnp.pi / 6.0), -b * jnp.sin(jnp.pi / 6.0)],
            [0.0, 0.0, a],
            [
                0.0,
                a * jnp.cos(jnp.pi / 6.0),
                -a * jnp.sin(jnp.pi / 6.0),
            ],  # second sphere
            [
                0.0,
                -a * jnp.cos(jnp.pi / 6.0),
                -a * jnp.sin(jnp.pi / 6.0),
            ],  # third sphere
            [a, 0.0, b],  # first patch
            [a, b * jnp.cos(jnp.pi / 6.0), -b * jnp.sin(jnp.pi / 6.0)],  # second patch
            [a, -b * jnp.cos(jnp.pi / 6.0), -b * jnp.sin(jnp.pi / 6.0)],  # third patch
        ],
        dtype=jnp.float64,
    )
    return jnp.array([base_shape for _ in range(size)])


def make_rb(size, key, separation=2.0, noise=1e-14):
    if size == 1:
        return jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float64), key  # Return key

    key, subkey = random.split(key)  # Split the main_key properly
    rand_vals = random.normal(subkey, shape=(size,))

    rb = []
    half_size = size // 2

    for i in range(size):
        if size % 2 == 0:
            if i < half_size:
                rb.extend(
                    [
                        -separation / 2.0 * (size - 1 - 2 * i),
                        rand_vals[i] * noise,
                        0,
                        0,
                        0,
                        0,
                    ]
                )
            else:
                rb.extend(
                    [
                        separation / 2.0 * (size - 1 - 2 * (size - 1 - i)),
                        rand_vals[i] * noise,
                        0,
                        0,
                        0,
                        0,
                    ]
                )
        else:
            if i == half_size:
                rb.extend([0, 0, 0, 0, 0, 0])
            elif i < half_size:
                rb.extend(
                    [-separation * (half_size - i), rand_vals[i] * noise, 0, 0, 0, 0]
                )
            else:
                rb.extend(
                    [separation * (i - half_size), rand_vals[i] * noise, 0, 0, 0, 0]
                )

    return jnp.array(rb, dtype=jnp.float64), key


rep_rmax_table = jnp.full((n_species, n_species), 2 * vertex_radius)
rep_A_table = (
    jnp.full((n_species, n_species), small_value)
    .at[vertex_species, vertex_species]
    .set(500.0)
)
rep_alpha_table = jnp.full((n_species, n_species), 2.5)

morse_narrow_alpha = 5.0
morse_alpha_table = jnp.full((n_species, n_species), morse_narrow_alpha)


def generate_idx_pairs(n_species):
    idx_pairs = []
    for i in range(1, n_species):
        for j in range(i + 1, n_species):
            idx_pairs.append((i, j))
    return idx_pairs


generated_idx_pairs = generate_idx_pairs(n_species)


def make_tables(opt_params, use_custom_pairs=True, custom_pairs=custom_pairs):
    # morse_eps_table = jnp.full((n_species, n_species), args.eps_weak)
    morse_eps_table = jnp.full((n_species, n_species), 2.0)
    morse_eps_table = morse_eps_table.at[0, :].set(small_value)
    morse_eps_table = morse_eps_table.at[:, 0].set(small_value)

    if use_custom_pairs and custom_pairs is not None:
        idx_pairs = custom_pairs
    else:
        idx_pairs = generated_idx_pairs

    # Set off-diagonal elements
    for i, (idx1, idx2) in enumerate(idx_pairs):
        morse_eps_table = morse_eps_table.at[idx1, idx2].set(opt_params[i])
        morse_eps_table = morse_eps_table.at[idx2, idx1].set(opt_params[i])

    # Set diagonal elements excluding (0,0)
    if not use_custom_pairs:
        diagonal_start_idx = len(idx_pairs)
        for i in range(1, n_species):
            morse_eps_table = morse_eps_table.at[i, i].set(
                opt_params[diagonal_start_idx + i - 1]
            )

    return morse_eps_table


def pairwise_morse(ipos, jpos, i_species, j_species, opt_params):
    morse_eps_table = make_tables(opt_params)
    morse_d0 = morse_eps_table[i_species, j_species]
    morse_alpha = morse_alpha_table[i_species, j_species]
    morse_r0 = 0.0
    morse_rcut = 8.0 / morse_alpha + morse_r0
    dr = space.distance(ipos - jpos)
    return potentials.morse_x(
        dr,
        rmin=morse_r0,
        rmax=morse_rcut,
        D0=morse_d0,
        alpha=morse_alpha,
        r0=morse_r0,
        ron=morse_rcut / 2.0,
    )


morse_func = vmap(
    vmap(pairwise_morse, in_axes=(None, 0, None, 0, None)),
    in_axes=(0, None, 0, None, None),
)


def pairwise_repulsion(ipos, jpos, i_species, j_species):
    rep_rmax = rep_rmax_table[i_species, j_species]
    rep_a = rep_A_table[i_species, j_species]
    rep_alpha = rep_alpha_table[i_species, j_species]
    dr = space.distance(ipos - jpos)
    return potentials.repulsive(dr, rmin=0, rmax=rep_rmax, A=rep_a, alpha=rep_alpha)


inner_rep = vmap(pairwise_repulsion, in_axes=(None, 0, None, 0))
rep_func = vmap(inner_rep, in_axes=(0, None, 0, None))


def get_nmer_energy_fn(n):
    pairs = jnp.array(list(itertools.combinations(onp.arange(n), 2)))

    def nmer_energy_fn(q, pos, species, opt_params):
        positions = utils.get_positions(q, pos)
        pos_slices = [(i * 9, (i + 1) * 9) for i in range(n)]
        species_slices = [(i * 3, (i + 1) * 3) for i in range(n)]

        all_pos = jnp.stack([positions[start:end] for start, end in pos_slices])
        all_species = jnp.stack(
            [jnp.repeat(species[start:end], 3) for start, end in species_slices]
        )

        def pairwise_energy(pair):
            i, j = pair
            morse_energy = morse_func(
                all_pos[i], all_pos[j], all_species[i], all_species[j], opt_params
            ).sum()
            rep_energy = rep_func(
                all_pos[i], all_pos[j], all_species[i], all_species[j]
            ).sum()
            return morse_energy + rep_energy

        all_pairwise_energies = vmap(pairwise_energy)(pairs)
        return all_pairwise_energies.sum()

    return nmer_energy_fn


def hess(energy_fn, q, pos, species, opt_params):
    H = hessian(energy_fn)(q, pos, species, opt_params)
    evals, evecs = jnp.linalg.eigh(H)
    return evals, evecs


def compute_zvib(energy_fn, q, pos, species, opt_params):
    evals, evecs = hess(energy_fn, q, pos, species, opt_params)
    zvib = jnp.prod(
        jnp.sqrt(
            2.0 * jnp.pi / (opt_params[custom_pairs_n] * jnp.abs(evals[6:]) + 1e-12)
        )
    )
    return zvib


def compute_zrot_mod_sigma(energy_fn, q, pos, species, opt_params, key, nrandom=100000):
    Nbb = len(pos)
    evals, evecs = hess(energy_fn, q, pos, species, opt_params)

    def set_nu_random(key):
        quat = jts.random_quaternion(None, key)
        angles = jnp.array(jts.euler_from_quaternion(quat, euler_scheme))
        nu0 = jnp.full((Nbb * 6,), 0.0)
        return nu0.at[3:6].set(angles)

    def ftilde(nu):
        nu = nu.astype(jnp.float32)
        q_tilde = jnp.matmul(evecs.T[6:].T, nu[6:])
        nu_tilde = jnp.reshape(jnp.array([nu[:6] for _ in range(Nbb)]), nu.shape)
        return utils.add_variables_all(q_tilde, nu_tilde)

    key, *splits = random.split(key, nrandom + 1)
    nus = vmap(set_nu_random)(jnp.array(splits))
    nu_fn = lambda nu: jnp.abs(jnp.linalg.det(jacfwd(ftilde)(nu)))
    Js = vmap(nu_fn)(nus)
    J = jnp.mean(Js)
    Jtilde = 8.0 * (jnp.pi**2) * J
    return Jtilde, Js, key


def compute_zc(boltzmann_weight, z_rot_mod_sigma, z_vib, sigma, V=V):
    z_trans = V
    z_rot = z_rot_mod_sigma / sigma
    return boltzmann_weight * z_trans * z_rot * z_vib


sizes = range(1, n + 1)


rbs = {}

for size in sizes:
    main_key, subkey = random.split(main_key)
    rb, subkey = make_rb(size, subkey)
    rbs[size] = rb
    main_key = subkey

shapes = {size: make_shape(size) for size in sizes}
sigmas = {size: data[f"{size}_sigma"] for size in sizes if f"{size}_sigma" in data}
energy_fns = {size: jit(get_nmer_energy_fn(size)) for size in range(2, n + 1)}

energy_fn_4 = get_nmer_energy_fn(n + 1)


main_key, subkey = random.split(main_key)
rb_4, subkey = make_rb(n + 1, subkey)
main_key = subkey
main_key, subkey = random.split(main_key)
rb_plus_2, subkey = make_rb(n + 2, subkey)
main_key = subkey

shape_4 = make_shape(n + 1)



rb1 = rbs[1]
shape1 = shapes[1]
mon_energy_fn = lambda q, pos, species, opt_params: 0.0


zrot_mod_sigma_1, _, main_key = compute_zrot_mod_sigma(
    mon_energy_fn, rb1, shape1, jnp.array([1, 0, 2]), patchy_vals, main_key
)
zvib_1 = 1.0
boltzmann_weight = 1.0

z_1 = compute_zc(boltzmann_weight, zrot_mod_sigma_1, zvib_1, sigmas[1])
z_1s = jnp.full(n, z_1)
log_z_1 = jnp.log(z_1s)

zrot_mod_sigma_values = {}

for size in range(2, n + 1):

    zrot_mod_sigma, Js, main_key = compute_zrot_mod_sigma(
        energy_fns[size],
        rbs[size],
        shapes[size],
        jnp.array([1, 0, 2] * size),
        patchy_vals,
        main_key,
    )

    zrot_mod_sigma_values[size] = zrot_mod_sigma


def get_log_z_all(opt_params):
    def compute_log_z(size, species, sigma):
        energy_fn = energy_fns[size]
        shape = shapes[size]
        rb = rbs[size]
        zrot_mod_sigma = zrot_mod_sigma_values[size]
        zvib = compute_zvib(energy_fn, rb, shape, species, opt_params)
        e0 = energy_fn(rb, shape, species, opt_params)
        boltzmann_weight = jnp.exp(-e0 / opt_params[custom_pairs_n])
        z = compute_zc(boltzmann_weight, zrot_mod_sigma, zvib, sigma)
        return safe_log(z)

    log_z_all = []

    for size in range(2, n + 1):
        species = data[f"{size}_pc_species"]
        sigma = data[f"{size}_sigma"]

        # Repeat sigma for each structure in species of the current size
        sigma_array = jnp.full(species.shape[0], sigma)

        if size <= 4:
            log_z = vmap(lambda sp, sg: compute_log_z(size, sp, sg))(
                species, sigma_array
            )
        else:
            compute_log_z_ckpt = checkpoint(lambda sp, sg: compute_log_z(size, sp, sg))
            flat_species = species.reshape(species.shape[0], -1)
            xs = jnp.concatenate([flat_species, sigma_array[:, None]], axis=-1)

            def scan_fn(carry, x):
                flat_species, sigma_val = x[:-1], x[-1]
                species_new = flat_species.reshape(species.shape[1:])
                result = compute_log_z_ckpt(species_new, sigma_val)
                return carry, result

            checkpoint_freq = 10
            scan_with_ckpt = functools.partial(
                checkpoint_scan, checkpoint_every=checkpoint_freq
            )
            _, log_z = scan_with_ckpt(scan_fn, None, xs)
            log_z = jnp.array(log_z)

        log_z_all.append(log_z)

    log_z_all = jnp.concatenate(log_z_all, axis=0)
    print(log_z_all.shape)
    log_z_all = jnp.concatenate([log_z_1, log_z_all], axis=0)

    return log_z_all


# Example monomer counts
monomer_counts = []
for letter in "ABC":
    counts_list = []
    for i in range(1, n + 1):
        key = f"{letter}_{i}_counts"
        if key in data:
            counts_list.append(data[key])
    if counts_list:
        monomer_counts.append(jnp.concatenate(counts_list))


nper_structure = jnp.array(monomer_counts)
species_rb_4 = data_4[f"4_pc_species"]
species_4 = species_rb_4

counts_4 = {
    key.split("_")[0]: data_4[key]
    for key in data_4.keys()
    if key.endswith("_4_counts")
}

counts_4_array = jnp.array(list(counts_4.values()))


def loss_fn(log_concs_struc, log_z_list, opt_params):
    m_conc = opt_params[-n:]
    log_mon_conc = safe_log(m_conc)

    def mon_loss_fn(mon_idx):
        mon_val = safe_log(jnp.dot(nper_structure[mon_idx], jnp.exp(log_concs_struc)))
        return jnp.sqrt((mon_val - log_mon_conc[mon_idx]) ** 2)

    def struc_loss_fn(struc_idx):
        log_vcs = jnp.log(V) + log_concs_struc[struc_idx]

        def get_vcs_denom(mon_idx):
            n_sa = nper_structure[mon_idx][struc_idx]
            log_vca = jnp.log(V) + log_concs_struc[mon_idx]
            return n_sa * log_vca

        vcs_denom = vmap(get_vcs_denom)(jnp.arange(num_monomers)).sum()
        log_zs = log_z_list[struc_idx]

        def get_z_denom(mon_idx):
            n_sa = nper_structure[mon_idx][struc_idx]
            log_zalpha = log_z_list[mon_idx]
            return n_sa * log_zalpha

        z_denom = vmap(get_z_denom)(jnp.arange(num_monomers)).sum()

        return jnp.sqrt((log_vcs - vcs_denom - log_zs + z_denom) ** 2)

    final_mon_conc = jnp.exp(log_concs_struc[:n]).sum()

    def log_massact_loss_fn(opt_params, struc_idx):

        def mon_sum(mon_idx):

            conc_mon_cont = (
                counts_4_array[mon_idx, struc_idx] * log_concs_struc[mon_idx]
            )

            return conc_mon_cont

        energy_4 = energy_fn_4(
            rb_4, shape_4, species_4[struc_idx], opt_params
        )

        presum_mon = vmap(mon_sum)(jnp.arange(n))

        mass_act_loss = (
            -1 / opt_params[custom_pairs_n] * energy_4
            + jnp.sum(presum_mon)
            + jnp.log(n + 1)
        )

        return mass_act_loss

    mon_loss = vmap(mon_loss_fn)(jnp.arange(num_monomers))
    struc_loss = vmap(struc_loss_fn)(jnp.arange(num_monomers, tot_num_structures))

    loss_var = jnp.var(jnp.concatenate([mon_loss, struc_loss]))
    combined_losses = jnp.concatenate([mon_loss, struc_loss])
    combined_loss = jnp.linalg.norm(combined_losses)

    tot_loss = combined_loss + loss_var

    return tot_loss, combined_loss, loss_var


def optimality_fn(log_concs_struc, log_z_list, opt_params):
    return grad(
        lambda log_concs_struc, log_z_list, opt_params: loss_fn(
            log_concs_struc, log_z_list, opt_params
        )[0]
    )(log_concs_struc, log_z_list, opt_params)


@implicit_diff.custom_root(optimality_fn)
def inner_solver(init_guess, log_z_list, opt_params):
    gd = GradientDescent(
        fun=lambda log_concs_struc, log_z_list, opt_params: loss_fn(
            log_concs_struc, log_z_list, opt_params
        )[0],
        maxiter=80000,
        implicit_diff=True,
    )
    sol = gd.run(init_guess, log_z_list, opt_params)

    final_params = sol.params
    final_loss, combined_losses, loss_var = loss_fn(
        final_params, log_z_list, opt_params
    )

    return final_params


# --- Helper functions for optimization ---
def abs_array(par):
    """Elementwise absolute value for parameter arrays."""
    return jnp.abs(par)


def normalize(arr):
    """Normalize an array to sum to 1."""
    sum_arr = jnp.sum(arr)
    return arr / sum_arr


def normalize_logits(logits, total_concentration):
    """Normalize logits to concentrations summing to total_concentration."""
    norm_logits = normalize(logits)
    concentrations = norm_logits * total_concentration
    return concentrations


num_params = len(init_params)
mask = jnp.full(num_params, 1.0)
mask = mask.at[custom_pairs_n].set(0.0)


# --- Utility functions for robust and physically meaningful optimization ---

def safe_exp(x, lower_bound=-701.0, upper_bound=701.0):
    """
    Exponentiate with clipping to avoid overflow/underflow.
    This is essential for numerical stability in scientific optimization routines.
    """
    clipped_x = jnp.clip(x, a_min=lower_bound, a_max=upper_bound)
    return jnp.exp(clipped_x)


def masked_grads(grads):
    """
    Mask gradients for non-concentration parameters (if needed).
    This allows selective optimization (e.g., fixing temperature or certain concentrations),
    which is important for flexible experimentation and reproducibility.
    """
    return grads * mask


def project(params):
    """
    Project concentrations to be >= 9e-5.
    Ensures physical constraints (e.g., non-negative concentrations) are always enforced.
    """
    conc_min = 9e-5
    concs = jnp.clip(params[-n:], a_min=conc_min)
    return jnp.concatenate([params[:-n], concs])


def project_eps(params):
    """
    Project epsilon values to be >= 0.25 (if needed).
    Maintains physical realism for interaction strengths during optimization.
    """
    min_eps = 0.25
    eps = jnp.clip(params[2:-4], a_min=min_eps)
    return jnp.concatenate([params[:2], eps, params[-4:]])


# --- Main optimization function ---
def optimize(opt_params):
    """
    Compute the log yield of the target structure, monomer concentrations, and total concentration.
    Args:
        opt_params: Optimization parameters (interaction strengths, concentrations, etc.)
    Returns:
        target_yield: Log yield of the target structure
        mon_concs: Monomer concentrations
        fin_conc: Total concentration
    """
    log_z_list = get_log_z_all(opt_params)
    tot_conc = init_conc
    struc_concs_guess = jnp.full(
        tot_num_structures, safe_log(tot_conc / tot_num_structures)
    )
    fin_log_concs = inner_solver(struc_concs_guess, log_z_list, opt_params)
    fin_concs = jnp.exp(fin_log_concs)
    yields = fin_concs / jnp.sum(fin_concs)
    target_yield = safe_log(yields[target_idx])
    return target_yield, fin_concs[:n], fin_concs.sum()


def optimize_grad_fn(opt_params: jnp.ndarray, desired_yield_val: float):
    """
    Compute the loss and mass action loss for optimization, using the optimize() function.
    Args:
        opt_params: Optimization parameters (interaction strengths, concentrations, etc.)
        desired_yield_val: The target yield value for the optimization.
    Returns:
        loss: The total loss (objective for optimizer)
        mass_act_loss: The mass action constraint loss (if enabled)
    """
    target_yield, mon_concs, fin_conc = optimize(opt_params)

    def log_massact_loss_fn(opt_params, struc_idx):
        def mon_sum(mon_idx):
            conc_mon_cont = jnp.sum(
                counts_4_array[mon_idx, struc_idx] * safe_log(mon_concs[mon_idx])
            )
            return conc_mon_cont

        energy_4 = energy_fn_4(
            rb_4, shape_4, species_4[struc_idx], opt_params
        )
        presum_mons = vmap(mon_sum)(jnp.arange(n))
        mass_act_loss = (
            -1 / opt_params[custom_pairs_n] * energy_4
            + jnp.sum(presum_mons)
            + jnp.log(n + 1)
        )
        return mass_act_loss

    mass_act_loss_logs = vmap(log_massact_loss_fn, in_axes=(None, 0))
    mass_act_loss = jnp.sum(
        mass_act_loss_logs(opt_params, jnp.arange(species_4.shape[0]))
    )
    # Mass action constraint toggle: set mass_act_loss = 1 to disable constraint
    if not use_mass_action:
        mass_act_loss = 1.0
    loss = 1000 * (abs(0.99 - safe_exp(target_yield))) + 5 * softplus(
        mass_act_loss
    )  # Loss function can be tuned as needed
    return loss, mass_act_loss


our_grad_fn = jit(value_and_grad(optimize_grad_fn, has_aux=True))
params = init_params
outer_optimizer = optax.adam(2e-2)
opt_state = outer_optimizer.init(params)

n_outer_iters = args.outer_iters
outer_losses = []

if use_custom_pairs and custom_pairs is not None:
    param_names = [f"Eps({i},{j})" for i, j in custom_pairs]
else:
    param_names = [f"Eps({i},{j})" for i, j in generated_idx_pairs]
    param_names += [f"Eps({i},{i})" for i in range(1, n_patches + 1)]

# param_names += [f"conc_{chr(ord('A') + i)}" for i in range(n)]
param_names += ["Weak Eps"]
param_names += [f"A conc:"]
param_names += [f"B conc:"]
param_names += [f"C conc:"]

final_results = []
desired_yield = args.desired_yield

# --- Main optimization loop (moved to end for clarity and correctness) ---
with open(output_file, "w") as f:
    for i in tqdm(range(n_outer_iters)):
        (loss, mass_act_loss), grads = our_grad_fn(params, args.desired_yield)
        grads = masked_grads(grads)
        print(f"Iteration {i + 1}, Loss: {loss}")
        print(f"Mass action loss: {mass_act_loss}")
        print(f"Gradients: {grads}")
        print(f"Parameters: {params}")
        updates, opt_state = outer_optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        conc_params = normalize_logits
        params = abs_array(params)
        norm_conc = normalize_logits(params[-n:], init_conc)
        params = jnp.concatenate([params[:-n], norm_conc])
        params = project(params)
        params = project_eps(params)
        print("Updated Parameters:")
        for name, value in {
            name: params[idx] for idx, name in enumerate(param_names)
        }.items():
            print(f"{name}: {value}")
        print(params)
        fin_yield = optimize(params)
        fin_yield = jnp.exp(fin_yield[0])
        print(f"Desired Yield: {args.desired_yield}, Final Yield: {fin_yield}")

    final_params = params
    fin_yield = optimize(params)
    final_target_yields = jnp.exp(fin_yield[0])

    params_str = ",".join([f"{param}" for param in params])
    f.write(
        f"{args.desired_yield},{final_target_yields},{mass_act_loss},{params_str}\n"
    )
    f.flush()
    # Print summary for reproducibility
    print("\n--- Optimization Complete ---")
    print(f"Random seed: {SEED}")
    print(f"Final parameters: {params}")
    print(f"Final target yield: {final_target_yields}")
    print(f"Mass action loss: {mass_act_loss}")
    print(f"All arguments: {args}")
