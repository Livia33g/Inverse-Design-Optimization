"""
Dimer Yield Maximization Script
------------------------------
Maximizes dimer yield using JAX and partition function calculations.
Lightly cleaned for clarity and usability. All scientific logic is preserved.
"""

# ----------------------
# Imports
# ----------------------
import numpy as onp
import jax.numpy as jnp
import optax
from jax import (
    random, vmap, hessian, jacfwd, jit, value_and_grad, grad
)
from tqdm import tqdm
import potentials
import utils
import jax_transformations3d as jts
from jaxopt import implicit_diff, GradientDescent
import os
import argparse

# Euler scheme for rotations
# (used in jts.euler_from_matrix and jts.euler_from_quaternion)
euler_scheme = "sxyz"

# ----------------------
# User Parameters & Argparse
# ----------------------
parser = argparse.ArgumentParser(description="Maximize dimer yield.")
parser.add_argument("--init_patchy_vals", type=float, required=True, help="Initial patchy values")
parser.add_argument("--init_kt", type=float, required=True, help="Initial temperature value")
parser.add_argument("--init_conc_val", type=float, required=True, help="Initial concentration value")
args = parser.parse_args()

SEED = 42
main_key = random.PRNGKey(SEED)
a = 1.0  # sphere center distance
b = 0.3  # patch center distance
n = 2    # number of building blocks
separation = 2.0
vertex_radius = a
patch_radius = 0.2 * a
small_value = 1e-12
n_species = 4
V = 54000.0

# Set up parameters
init_patchy_vals = args.init_patchy_vals
init_kT = args.init_kt
init_conc_val = args.init_conc_val
patchy_vals = jnp.full(3, init_patchy_vals)
init_conc = jnp.full(2, init_conc_val)
init_kT = jnp.array([init_kT])
init_params = jnp.concatenate([patchy_vals, init_kT, init_conc])

# ----------------------
# Geometry Setup
# ----------------------
mon_shape1 = onp.array([
    [0., 0., a],
    [0., a*onp.cos(onp.pi/6.), -a*onp.sin(onp.pi/6.)],
    [0., -a*onp.cos(onp.pi/6.), -a*onp.sin(onp.pi/6.)],
    [a, 0., b],
    [a, b*onp.cos(onp.pi/6.), -b*onp.sin(onp.pi/6.)],
    [a, -b*onp.cos(onp.pi/6.), -b*onp.sin(onp.pi/6.)]
])
mon_shape2 = jts.matrix_apply(jts.reflection_matrix(jnp.array([0, 0, 0], dtype=jnp.float64),
                                                   jnp.array([1, 0, 0], dtype=jnp.float64)),
                             mon_shape1)
mon_shape2  = jts.matrix_apply(jts.reflection_matrix(jnp.array([0, 0, 0], dtype=jnp.float64),
                                                   jnp.array([0, 1, 0], dtype=jnp.float64)),
                             mon_shape2)
dimer_shape = jnp.array([mon_shape1, mon_shape2])
mon_species_1 = onp.array([0,0,0,1,2,3])
mon_species_2 = onp.array([0,0,0,1,3,2])
dimer_species = onp.array([0,0,0,1,3,2,0,0,0,1,2,3])
rb_1 = jnp.array([-separation/2.0, 1e-15, 0, 0, 0, 0])   
rb_2 = jnp.array([-separation/2.0, 1e-15, 0, 0, 0, 0,
                separation/2.0, 0, 0, 0, 0, 0], dtype=jnp.float64)   

# ----------------------
# Utility Functions
# ----------------------
def safe_log(x, eps=1e-10):
    """Safe logarithm function that avoids log(0)"""
    return jnp.log(jnp.clip(x, a_min=eps, a_max=None))

def distance(dR):
    """Computes the distance between points, safely handling zero distances"""
    dr = jnp.sum(dR ** 2, axis=-1)
    return jnp.where(dr > 0, jnp.sqrt(dr), 0)

def get_positions(q, ppos):
    """Gets the real positions of the monomers in the dimer"""
    Mat = [utils.convert_to_matrix(q[i*6:(i+1)*6]) for i in range(2)]
    return [jts.matrix_apply(Mat[i], ppos[i]) for i in range(2)]

# ----------------------
# Energy and Partition Function
# ----------------------
@jit
def get_energy_fns(q, pos, species, opt_params):
    """Calculates the energy of the dimer configuration"""
    Nbb = 2
    morse_rcut = 8. / 5.
    sphere_radius = 1.0
    real_ppos = get_positions(q, pos)
    tot_energy = 0.0
    # Repulsive sphere-sphere
    for i in range(3):
        for j in range(3):
            r = distance(real_ppos[0][i] - real_ppos[1][j])
            tot_energy += potentials.repulsive(r, rmin=0, rmax=sphere_radius*2, A=500., alpha=2.5)
    # Morse patch-patch
    patch_pairs = [(3,3,0), (5,4,1), (4,5,2)]
    for idx, (i1, i2, pidx) in enumerate(patch_pairs):
        r = distance(real_ppos[0][i1] - real_ppos[1][i2])
        tot_energy += potentials.morse_x(r, rmin=0, rmax=morse_rcut, D0=opt_params[pidx], alpha=5., r0=0.0, ron=morse_rcut/2.)
    return tot_energy

def add_variables(ma, mb):
    """
    given two vectors of length (6,) corresponding to x,y,z,alpha,beta,gamma,
    convert to transformation matrixes, 'add' them via matrix multiplication,
    and convert back to x,y,z,alpha,beta,gamma

    note: add_variables(ma,mb) != add_variables(mb,ma)
    """

    Ma = utils.convert_to_matrix(ma)
    Mb = utils.convert_to_matrix(mb)
    Mab = jnp.matmul(Mb, Ma)
    trans = jnp.array(jts.translation_from_matrix(Mab))
    angles = jnp.array(jts.euler_from_matrix(Mab, euler_scheme))

    return jnp.concatenate((trans, angles))


def add_variables_all(mas, mbs):
    """
    Given two vectors of length (6*n,), 'add' them per building block according
    to add_variables().
    """

    mas_temp = jnp.reshape(mas, (mas.shape[0] // 6, 6))
    mbs_temp = jnp.reshape(mbs, (mbs.shape[0] // 6, 6))

    return jnp.reshape(
        vmap(add_variables, in_axes=(0, 0))(mas_temp, mbs_temp), mas.shape
    )


def hess(energy_fn, q, pos, species, opt_params):
    """Computes the Hessian matrix of the energy function"""
    H = hessian(energy_fn)(q, pos, species, opt_params)
    evals, evecs = jnp.linalg.eigh(H)
    return evals, evecs


def compute_zvib(energy_fn, q, pos, species, opt_params):
    """Computes the vibrational partition function component"""
    evals, evecs = hess(energy_fn, q, pos, species, opt_params)
    zvib = jnp.prod(
        jnp.sqrt(2.0 * jnp.pi / (opt_params[3] * (jnp.abs(evals[6:]) + 1e-12)))
    )
    return zvib


def compute_zrot_mod_sigma(
    energy_fn, q, pos, species, opt_params, key, size, nrandom=100000
):
    """Computes the rotational and configurational partition function component"""
    Nbb = size
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


def compute_zc(boltzmann_weight, z_rot_mod_sigma, z_vib, sigma=3, V=V):
    """Computes the complete partition function component"""
    z_trans = V
    z_rot = z_rot_mod_sigma / sigma
    return boltzmann_weight * z_trans * z_rot * z_vib


mon_energy_fn = lambda q, pos, species, opt_params: 0.0

zrot_mod_sigma_1, _, main_key = compute_zrot_mod_sigma(
    mon_energy_fn, rb_1, mon_shape1, mon_species_1, patchy_vals, main_key, 1
)
zvib_1 = 1.0
boltzmann_weight = 1.0

z_1 = compute_zc(boltzmann_weight, zrot_mod_sigma_1, zvib_1)
z_1s = jnp.full(n, z_1)
log_z_1 = safe_log(z_1s)


def get_log_z_all(opt_params, key, rb=rb_2, shape=dimer_shape, species=dimer_species):
    """Gets the logarithm of the partition function for all structures"""
    dim_energy_fn = get_energy_fns
    zvib = compute_zvib(dim_energy_fn, rb, shape, species, opt_params)
    e0 = dim_energy_fn(rb, shape, species, opt_params)
    boltzmann_weight = jnp.exp(-jnp.clip(e0 / opt_params[3], a_min=-100, a_max=100))
    zrot_mod_sigma, _, new_key = compute_zrot_mod_sigma(
        dim_energy_fn, rb, shape, species, opt_params, key, 2
    )
    z = compute_zc(boltzmann_weight, zrot_mod_sigma, zvib)
    z_log = safe_log(z)
    z_log_2 = jnp.array([z_log])
    z_all = jnp.concatenate([log_z_1, z_log_2], axis=0)
    return z_all, new_key


nper_structure = nper_structure = jnp.array([[0, 1, 1], [1, 0, 1]])


def loss_fn(log_concs_struc, log_z_list, opt_params):
    """Calculates the loss for the optimization problem"""
    m_conc = opt_params[-n:]
    tot_conc = jnp.sum(m_conc)
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

        vcs_denom = vmap(get_vcs_denom)(jnp.arange(n)).sum()
        log_zs = log_z_list[struc_idx]

        def get_z_denom(mon_idx):
            n_sa = nper_structure[mon_idx][struc_idx]
            log_zalpha = log_z_list[mon_idx]
            return n_sa * log_zalpha

        z_denom = vmap(get_z_denom)(jnp.arange(n)).sum()

        return jnp.sqrt((log_vcs - vcs_denom - log_zs + z_denom) ** 2)

    mon_loss = vmap(mon_loss_fn)(jnp.arange(n))
    struc_loss = vmap(struc_loss_fn)(jnp.arange(n, 3))
    combined_loss = jnp.concatenate([mon_loss, struc_loss])
    loss_var = jnp.var(combined_loss)
    loss_max = jnp.max(combined_loss)
    tot_loss = jnp.linalg.norm(combined_loss) + loss_var
    return tot_loss, combined_loss, loss_var


def optimality_fn(log_concs_struc, log_z_list, opt_params):
    """Gradient of the loss function for optimization"""
    return grad(
        lambda log_concs_struc, log_z_list, opt_params: loss_fn(
            log_concs_struc, log_z_list, opt_params
        )[0]
    )(log_concs_struc, log_z_list, opt_params)


@implicit_diff.custom_root(optimality_fn)
def inner_solver(init_guess, log_z_list, opt_params):
    """Inner solver for the implicit differentiation"""
    gd = GradientDescent(
        fun=lambda log_concs_struc, log_z_list, opt_params: loss_fn(
            log_concs_struc, log_z_list, opt_params
        )[0],
        maxiter=50000,
        implicit_diff=True,
    )
    sol = gd.run(init_guess, log_z_list, opt_params)

    final_params = sol.params
    final_loss, combined_losses, loss_var = loss_fn(
        final_params, log_z_list, opt_params
    )
    max_loss = jnp.max(combined_losses)
    second_max_loss = jnp.partition(combined_losses, -2)[-2]

    return final_params


def ofer(opt_params, key):
    """Outer function for optimization, orchestrating the yield maximization"""
    log_z_list, new_key = get_log_z_all(opt_params, key)
    tot_conc = jnp.sum(opt_params[-n:])
    struc_concs_guess = jnp.full(3, safe_log(tot_conc / 3))
    fin_log_concs = inner_solver(struc_concs_guess, log_z_list, opt_params)
    fin_concs = jnp.exp(fin_log_concs)
    yields = fin_concs / jnp.sum(fin_concs)
    target_yield = safe_log(yields[2])
    return target_yield, new_key


def ofer_grad_fn(opt_params, key):
    """Gradient function for the ofer function"""
    target_yield, new_key = ofer(opt_params, key)
    loss = target_yield
    return -loss, new_key


def project(params):
    """Projects the parameters onto their feasible set"""
    conc_min, conc_max = 1e-6, 3.0
    kbt_min, kbt_max = 1e-6, 3.0
    kbt_idx = 3
    concs = jnp.clip(params[-n:], a_min=conc_min, a_max=conc_max)
    kbt_val = jnp.clip(params[kbt_idx], a_min=kbt_min, a_max=kbt_max)
    kbt = jnp.array([kbt_val])
    return jnp.concatenate([params[:kbt_idx], kbt, concs])


num_params = len(init_params)
mask = jnp.zeros(num_params)
mask = mask.at[:3].set(1.0)


def masked_grads(grads):
    """Applies the mask to the gradients"""
    return grads * mask


our_grad_fn = jit(value_and_grad(ofer_grad_fn, has_aux=True))
params = init_params
outer_optimizer = optax.adam(1e-2)
opt_state = outer_optimizer.init(params)

n_outer_iters = 400
outer_losses = []

use_custom_pairs = True
custom_pairs = [[1, 1], [2, 2], [3, 3]]
n_patches = 3

if use_custom_pairs and custom_pairs is not None:
    param_names = [f"Eps({i},{j})" for i, j in custom_pairs]
else:
    param_names = [f"Eps({i},{j})" for i, j in generated_idx_pairs]
    param_names += [f"Eps({i},{i})" for i in range(1, n_patches + 1)]

param_names += ["kT"]

param_names += [f"conc_{chr(ord('A') + i)}" for i in range(n)]

final_results = []


os.makedirs("Maximized_agnese", exist_ok=True)
output_filename = f"Maximized_agnese/kbt{init_kT[0]:.2f}.txt"
#os.makedirs("Nonmaximized_agnese", exist_ok=True)
#output_filename = f"Nonmaximized_agnese/kbt{init_kT[0]:.2f}.txt"

with open(output_filename, "w") as f:


    for i in tqdm(range(n_outer_iters)):
        (loss, main_key), grads = our_grad_fn(params, main_key)
        # outer_losses.append(loss)
        grads = masked_grads(grads)
        print(f"Iteration {i + 1}: Loss = {loss}")
        updates, opt_state = outer_optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        # params = project(params)
        print("Updated Parameters:")
        for name, value in {
            name: params[idx] for idx, name in enumerate(param_names)
        }.items():
            print(f"{name}: {value}")
        print(params)
        fin_yield, main_key = ofer(params, main_key)
        fin_yield = jnp.exp(fin_yield)
        print(f"Yield: {fin_yield}")

    final_params = params
    fin_yield, main_key = ofer(params, main_key)
    final_target_yields = jnp.exp(fin_yield)

    f.write(
        f"{final_target_yields},{params[0]},{params[1]},{params[2]}\n"
    )
    # f.write(f"{des_yield}, {final_target_yields}, {params[3]}\n")
    f.flush()


print("All results saved.")
