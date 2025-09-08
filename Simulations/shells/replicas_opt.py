import os

os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import pdb
import functools
from tqdm import tqdm
import numpy as onp
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
from jax import jit, random, lax, vmap
import jax.numpy as jnp

from jax_md.util import *
from jax_md import space, energy, simulate
from jax_md import rigid_body as orig_rigid_body

import catalyst.octahedron.rigid_body as rigid_body
from catalyst.octahedron import utils


box = (300 / 0.001) ** (1 / 3)


def traj_to_pos_file(traj, shell_info, traj_path, box_size=67.0):
    assert len(traj.center.shape) == 3
    n_states = traj.center.shape[0]

    patch_radius = 0.5
    vertex_color = "43a5be"
    patch1_color = "D81B60"
    patch2_color = "1E88E5"

    patch_colors = [patch1_color, patch2_color]

    traj_injavis_lines = list()
    for i in tqdm(range(n_states), desc="Generating injavis output"):
        s = traj[i]

        assert len(s.center.shape) == 2
        body_pos = shell_info.get_body_frame_positions(s)

        assert len(body_pos.shape) == 2
        assert body_pos.shape[0] % 5 == 0
        n_vertices = body_pos.shape[0] // 5
        if n_vertices != 6:
            print(f"WARNING: writing shell body with only {n_vertices} vertices")

        assert body_pos.shape[1] == 3

        box_def = f"boxMatrix {box_size} 0 0 0 {box_size} 0 0 0 {box_size}"
        vertex_def = f'def V "sphere {shell_info.vertex_radius*2} {vertex_color}"'
        patch1_def = f'def P1 "sphere {patch_radius*2} {patch1_color}"'
        patch2_def = f'def P2 "sphere {patch_radius*2} {patch2_color}"'

        position_lines = list()
        for num_vertex in range(n_vertices):
            vertex_start_idx = num_vertex * 5

            # vertex center
            vertex_center_pos = body_pos[vertex_start_idx]
            vertex_line = f"V {vertex_center_pos[0]} {vertex_center_pos[1]} {vertex_center_pos[2]}"
            position_lines.append(vertex_line)

            for num_patch in range(4):
                patch_pos = body_pos[vertex_start_idx + num_patch + 1]
                # if num_patch < 2:
                if num_patch % 2 == 0:
                    patch_line = f"P1 {patch_pos[0]} {patch_pos[1]} {patch_pos[2]}"
                else:
                    patch_line = f"P2 {patch_pos[0]} {patch_pos[1]} {patch_pos[2]}"
                position_lines.append(patch_line)

        all_lines = (
            [box_def, vertex_def, patch1_def, patch2_def] + position_lines + ["eof"]
        )

        traj_injavis_lines += all_lines

    with open(traj_path, "w+") as of:
        of.write("\n".join(traj_injavis_lines))


class ShellInfo:
    def __init__(self, displacement_fn, shift_fn, obj_basedir="obj/", verbose=True):
        self.displacement_fn = displacement_fn
        self.shift_fn = shift_fn
        self.obj_dir = Path(obj_basedir) / "octahedron"
        assert self.obj_dir.exists()
        self.set_path_names()
        self.vertex_radius = 2.0

        self.verbose = verbose

        self.load()  # populate self.rigid_body and self.vertex_shape

    def set_path_names(self):
        self.rb_center_path = self.obj_dir / "rb_center.npy"
        self.rb_orientation_vec_path = self.obj_dir / "rb_orientation_vec.npy"
        self.vertex_shape_points_path = self.obj_dir / "vertex_shape_points.npy"
        self.vertex_shape_masses_path = self.obj_dir / "vertex_shape_masses.npy"
        self.vertex_shape_point_count_path = (
            self.obj_dir / "vertex_shape_point_count.npy"
        )
        self.vertex_shape_point_offset_path = (
            self.obj_dir / "vertex_shape_point_offset.npy"
        )
        self.vertex_shape_point_species_path = (
            self.obj_dir / "vertex_shape_point_species.npy"
        )
        self.vertex_shape_point_radius_path = (
            self.obj_dir / "vertex_shape_point_radius.npy"
        )

    def load(self):
        rb_paths_exist = (
            self.rb_center_path.exists() and self.rb_orientation_vec_path.exists()
        )
        vertex_shape_paths_exist = (
            self.vertex_shape_points_path.exists()
            and self.vertex_shape_masses_path.exists()
            and self.vertex_shape_point_count_path.exists()
            and self.vertex_shape_point_offset_path.exists()
            and self.vertex_shape_point_species_path.exists()
            and self.vertex_shape_point_radius_path.exists()
        )

        if not (rb_paths_exist and vertex_shape_paths_exist):
            raise RuntimeError("Requires a minimized octahedron")

        self.load_from_file()

    def get_vertex_shape(self, vertex_coords):
        # Get the vertex shape (i.e. the coordinates of a vertex for defining a rigid body)

        anchor = vertex_coords[0]
        d = vmap(self.displacement_fn, (0, None))

        # Compute all pairwise distances
        dists = space.distance(d(vertex_coords, anchor))

        # Mask the diagonal
        self_distance_tolerance = 1e-5
        large_mask_distance = 100.0
        dists = jnp.where(
            dists < self_distance_tolerance, large_mask_distance, dists
        )  # mask the diagonal

        # Find nearest neighbors
        # note: we use min because the distances to the nearest neighbors are all the same (they should be 1 diameter away)
        # note: this step is not differentiable, but that's fine: we keep the octahedron fixed for the optimization
        nbr_ids = jnp.where(dists == jnp.min(dists))[0]
        nbr_coords = vertex_coords[nbr_ids]

        # Compute displacements to neighbors to determine patch positions
        vec = d(nbr_coords, anchor)
        norm = jnp.linalg.norm(vec, axis=1).reshape(-1, 1)
        vec /= norm
        patch_pos = anchor - self.vertex_radius * vec

        # Collect shape in an array and return
        shape_coords = jnp.concatenate([jnp.array([anchor]), patch_pos]) - anchor
        return shape_coords

    def get_unminimized_shell(self, vertex_mass=1.0, patch_mass=1e-8):

        d = vmap(self.displacement_fn, (0, None))

        # Compute the coordinates of the vertices (i.e. no patches)
        vertex_coords = (
            2.0
            / jnp.sqrt(2.0)
            * self.vertex_radius
            * jnp.array(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                ]
            )
        )

        # note: we rotate by a random quaternion to avoid numerical issues
        rand_quat = orig_rigid_body.random_quaternion(random.PRNGKey(0), jnp.float64)
        vertex_coords = orig_rigid_body.quaternion_rotate(rand_quat, vertex_coords)

        # Compute the vertex shape positions
        # note: first position is vertex, rest are patches
        vertex_rb_positions = self.get_vertex_shape(vertex_coords)
        num_patches = vertex_rb_positions.shape[0] - 1  # don't count vertex particle
        assert num_patches == 4

        # Set the species
        # species = jnp.zeros(num_patches + 1, dtype=jnp.int32)
        # species = species.at[1:].set(1) # first particle is the vertex, rest are patches
        # species = jnp.array([0, 1, 2, 3, 4]) # each have unique patches

        species = jnp.array([0, 1, 2, 1, 2])  # each have unique patches
        # species = jnp.array([0, 1, 1, 2, 2]) # each have unique patches

        # Get the masses
        # note: patch mass should be zero, but zeros cause nans in the gradient
        patch_mass = jnp.ones(num_patches) * patch_mass
        mass = jnp.concatenate((jnp.array([vertex_mass]), patch_mass), axis=0)

        # Set the shape
        vertex_shape = orig_rigid_body.point_union_shape(vertex_rb_positions, mass).set(
            point_species=species
        )
        self.shape = vertex_shape
        self.shape_species = None

        """
        Now we orient rigid body particles (vertex + patches) within the rigid body.
        We don't orient the rotation about the z axis (where the z axis points
        toward the center of the icosahedron). We correct this by running a short
        simulation with just the icosahedron.

        We reference this stack overflow link to handle reorientation with
        quaternions:
        https://math.stackexchange.com/questions/60511/quaternion-for-an-object-that-to-point-in-a-direction
        """

        # Get vectors that point towards the center
        central_point = jnp.mean(vertex_coords, axis=0)  # center of the icosahedron
        reoriented_vectors = d(vertex_coords, central_point)
        norm = jnp.linalg.norm(reoriented_vectors, axis=1).reshape(-1, 1)
        reoriented_vectors /= norm

        # Now we have to compute a quaternion such that we rotate the current directions towards the center (i.e. the reoriented vectors)
        orig_vec = self.displacement_fn(
            vertex_shape.points[0], jnp.mean(vertex_shape.points[1:], axis=0)
        )
        orig_vec /= jnp.linalg.norm(orig_vec)
        crossed = vmap(jnp.cross, (None, 0))(orig_vec, reoriented_vectors)
        dotted = vmap(jnp.dot, (0, None))(reoriented_vectors, orig_vec)

        theta = jnp.arccos(dotted)
        cos_part = jnp.cos(theta / 2).reshape(-1, 1)
        mult = vmap(lambda v, s: s * v, (0, 0))
        sin_part = mult(crossed, jnp.sin(theta / 2))
        orientation = jnp.concatenate([cos_part, sin_part], axis=1)
        norm = jnp.linalg.norm(orientation, axis=1).reshape(-1, 1)
        orientation /= norm
        orientation = orig_rigid_body.Quaternion(orientation)

        octahedron_rb = orig_rigid_body.RigidBody(vertex_coords, orientation)

        return octahedron_rb

    def load_from_file(self):
        if self.verbose:
            print(
                f"Loading minimized octahedron rigid body and vertex shape from data directory: {self.obj_dir}"
            )
        rigid_body_center = jnp.load(self.rb_center_path).astype(jnp.float64)
        rigid_body_orientation_vec = jnp.load(self.rb_orientation_vec_path).astype(
            jnp.float64
        )
        octahedron_rigid_body = rigid_body.RigidBody(
            center=rigid_body_center,
            orientation=rigid_body.Quaternion(vec=rigid_body_orientation_vec),
        )

        vertex_shape_points = jnp.load(self.vertex_shape_points_path)
        vertex_shape_masses = jnp.load(self.vertex_shape_masses_path)
        vertex_shape_point_count = jnp.load(self.vertex_shape_point_count_path)
        vertex_shape_point_offset = jnp.load(self.vertex_shape_point_offset_path)
        vertex_shape_point_species = jnp.load(self.vertex_shape_point_species_path)
        vertex_shape_point_radius = jnp.load(self.vertex_shape_point_radius_path)

        # Changed from non-specific octahedron
        vertex_shape_point_species = jnp.array([0, 1, 2, 1, 2])

        vertex_shape = rigid_body.RigidPointUnion(
            points=vertex_shape_points,
            masses=vertex_shape_masses,
            point_count=vertex_shape_point_count,
            point_offset=vertex_shape_point_offset,
            point_species=vertex_shape_point_species,
            point_radius=vertex_shape_point_radius,
        )

        self.rigid_body = octahedron_rigid_body
        self.shape = vertex_shape
        self.shape_species = None

        return

    def get_body_frame_positions(self, body):
        return utils.get_body_frame_positions(body, self.shape).reshape(-1, 3)

    # note: body is only a single state, not a trajectory
    def body_to_injavis_lines(
        self,
        body,
        box_size,
        patch_radius=0.5,
        vertex_color="43a5be",
        patch_color="4fb06d",
    ):

        assert len(body.center.shape) == 2
        body_pos = self.get_body_frame_positions(body)

        assert len(body_pos.shape) == 2
        assert body_pos.shape[0] % 5 == 0
        n_vertices = body_pos.shape[0] // 5
        if n_vertices != 6:
            print(f"WARNING: writing shell body with only {n_vertices} vertices")

        assert body_pos.shape[1] == 3

        box_def = f"boxMatrix {box_size} 0 0 0 {box_size} 0 0 0 {box_size}"
        vertex_def = f'def V "sphere {self.vertex_radius*2} {vertex_color}"'
        patch_def = f'def P "sphere {patch_radius*2} {patch_color}"'

        position_lines = list()
        for num_vertex in range(n_vertices):
            vertex_start_idx = num_vertex * 5

            # vertex center
            vertex_center_pos = body_pos[vertex_start_idx]
            if num_vertex == utils.vertex_to_bind_idx:
                vertex_line = f"T {vertex_center_pos[0]} {vertex_center_pos[1]} {vertex_center_pos[2]}"
            else:
                vertex_line = f"V {vertex_center_pos[0]} {vertex_center_pos[1]} {vertex_center_pos[2]}"
            position_lines.append(vertex_line)

            for num_patch in range(4):
                patch_pos = body_pos[vertex_start_idx + num_patch + 1]
                patch_line = f"P {patch_pos[0]} {patch_pos[1]} {patch_pos[2]}"
                position_lines.append(patch_line)

        # Return: all lines, box info, particle types, positions
        all_lines = [box_def, vertex_def, patch_def] + position_lines + ["eof"]
        return all_lines, box_def, [vertex_def, patch_def], position_lines

    def get_energy_fn(
        self,
        morse_ii_eps=10.0,
        morse_ii_alpha=2.0,
        soft_eps=10000.0,
        morse_r_onset=10.0,
        morse_r_cutoff=12.0,
    ):

        n_point_species = 5  # hardcoded for clarity

        zero_interaction = jnp.zeros((n_point_species, n_point_species))

        # icosahedral patches attract eachother
        morse_eps = zero_interaction.at[1, 1].set(morse_ii_eps)
        morse_alpha = zero_interaction.at[1, 1].set(morse_ii_alpha)

        morse_eps = morse_eps.at[2, 2].set(morse_ii_eps)
        morse_alpha = morse_alpha.at[2, 2].set(morse_ii_alpha)

        # icosahedral centers repel each other
        soft_sphere_eps = zero_interaction.at[0, 0].set(soft_eps)

        soft_sphere_sigma = zero_interaction.at[0, 0].set(self.vertex_radius * 2)
        soft_sphere_sigma = jnp.where(
            soft_sphere_sigma == 0.0, 1e-5, soft_sphere_sigma
        )  # avoids nans

        pair_energy_soft = energy.soft_sphere_pair(
            self.displacement_fn,
            species=n_point_species,
            sigma=soft_sphere_sigma,
            epsilon=soft_sphere_eps,
        )
        pair_energy_morse = energy.morse_pair(
            self.displacement_fn,
            species=n_point_species,
            sigma=0.0,
            epsilon=morse_eps,
            alpha=morse_alpha,
            r_onset=morse_r_onset,
            r_cutoff=morse_r_cutoff,
        )
        pair_energy_fn = lambda R, **kwargs: pair_energy_soft(
            R, **kwargs
        ) + pair_energy_morse(R, **kwargs)

        # will accept body where body.center has dimensions (12, 3)
        # and body.orientation.vec has dimensions (12, 4)
        energy_fn = rigid_body.point_energy(
            pair_energy_fn,
            self.shape,
            # jnp.zeros(12) # FIXME: check if we need this
        )

        return energy_fn


def tree_stack(trees):
    return jax.tree_util.tree_map(lambda *v: jnp.stack(v), *trees)


def sq_lattice_3d(side_length, particle_count, padding=0):
    # Approximate the number of points per dimension
    points_per_dim = int(round(particle_count ** (1 / 3)))

    # Adjust the actual number of particles based on the grid dimensions
    actual_particle_count = points_per_dim**3

    # Compute the lattice spacing
    usable_length = side_length - 2 * padding
    lattice_spacing = (
        usable_length / (points_per_dim - 1) if points_per_dim > 1 else usable_length
    )

    # Generate the coordinates
    xs = jnp.linspace(padding, side_length - padding, points_per_dim)
    coords = jnp.meshgrid(xs, xs, xs)
    grid = jnp.stack(
        [coords[0].reshape(-1), coords[1].reshape(-1), coords[2].reshape(-1)], axis=-1
    )

    return grid[:particle_count]  # Trim to exactly match the requested particle count


if __name__ == "__main__":
    import argparse

    # --- 1. SET UP ARGUMENT PARSER ---
    # This section defines all the arguments the script expects from the Slurm command.
    parser = argparse.ArgumentParser(
        description="Run an NVT simulation of self-assembling octahedra."
    )

    # Arguments passed from the Slurm script
    parser.add_argument(
        "--kT",
        type=float,
        required=True,
        help="Temperature of the simulation in kT units.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        required=True,
        help="Morse potential epsilon (attraction strength).",
    )
    parser.add_argument(
        "--percent",
        type=float,
        required=True,
        help="Percentage identifier for the parameter set.",
    )
    parser.add_argument(
        "--kT_init",
        type=float,
        required=True,
        help="Initial temperature of the simulation.",
    )
    parser.add_argument(
        "--replica_id",
        type=int,
        required=True,
        help="The replica number (e.g., 1, 2, or 3) for a unique random seed.",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        required=True,
        help="Total number of simulation steps.",
    )

    # --- 2. PARSE ARGUMENTS AND ASSIGN VARIABLES ---
    args = parser.parse_args()

    # Simulation parameters from arguments
    kT = args.kT
    morse_ii_eps = args.eps
    n_steps = args.n_steps
    percent = args.percent
    kT_init = args.kT_init

    # Fixed simulation settings
    box_size = box
    morse_ii_alpha = 2.0  # Using a fixed value for potential steepness

    # --- 3. SET UP SIMULATION ENVIRONMENT ---
    if box_size is None:
        displacement_fn, shift_fn = space.free()
    else:
        assert box_size > 0.0
        displacement_fn, shift_fn = space.periodic(box_size)

    shell_info = ShellInfo(displacement_fn=displacement_fn, shift_fn=shift_fn)
    shell_info.shape = shell_info.shape.set(
        masses=jnp.array([0.5, 1e-8, 1e-8, 1e-8, 1e-8])
    )  # Lighter

    shell_energy_fn = shell_info.get_energy_fn(
        morse_ii_eps=morse_ii_eps, morse_ii_alpha=morse_ii_alpha
    )

    # --- 4. INITIALIZE SIMULATION STATE ---
    # The replica_id is CRITICAL here to ensure each replica is a unique simulation
    key = random.PRNGKey(args.replica_id)

    dt = 1e-3
    gamma = 1.0
    gamma_rb = rigid_body.RigidBody(jnp.array([gamma]), jnp.array([gamma / 3]))

    init_fn, step_fn = simulate.nvt_langevin(
        shell_energy_fn, shift_fn, dt, kT, gamma=gamma_rb
    )
    step_fn = jit(step_fn)
    mass = shell_info.shape.mass(shell_info.shape_species)

    # Create random initial positions for the rigid bodies
    assert box_size is not None
    n_try_particles = 300
    padding = 5
    rand_center = sq_lattice_3d(box_size, n_try_particles, padding=padding)
    n_particles = rand_center.shape[0]

    key, quat_key = random.split(key)
    quat_keys = random.split(quat_key, n_particles)
    rand_quaternion = jax.vmap(rigid_body.random_quaternion, (0, None))(
        quat_keys, jnp.float64
    )

    init_body = rigid_body.RigidBody(center=rand_center, orientation=rand_quaternion)

    state = init_fn(key, init_body, mass=mass)

    # --- 5. RUN SIMULATION ---
    do_step = lambda state, t: (step_fn(state), state.position)
    do_step = jit(do_step)

    print(f"🚀 Running replica {args.replica_id} for {n_steps} steps...")
    final_state, traj = lax.scan(do_step, state, jnp.arange(n_steps))
    print("✅ Simulation complete.")

    # --- 6. PROCESS AND SAVE RESULTS ---
    vis_freq = 10000
    vis_traj_idxs = jnp.arange(0, n_steps, vis_freq)
    vis_traj = traj[vis_traj_idxs]

    # Define unique output filenames using replica_id
    job_name = os.getenv("SLURM_JOB_NAME", "local_run")
    base_filename = f"{job_name}__{percent}_kT_{kT_init}_lowT_rep{args.replica_id}"

    # Create the directory structure
    output_dir = f"optimized/opt_{kT_init}/{percent}"
    os.makedirs(output_dir, exist_ok=True)
    pos_filename = os.path.join(output_dir, f"{base_filename}.pos")

    print(f"🎨 Saving visualization file to {pos_filename}")

    # Center trajectory for visualization and save file
    if box_size is not None:
        vis_traj_centered = vis_traj.set(center=vis_traj.center - box_size / 2)
        traj_to_pos_file(vis_traj_centered, shell_info, pos_filename, box_size=box_size)
    else:
        traj_to_pos_file(vis_traj, shell_info, pos_filename)

    print(f"✅ Replica {args.replica_id} finished and saved successfully.")
