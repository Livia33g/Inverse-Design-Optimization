import numpy as np
import jax.numpy as jnp
from jax import vmap, grad
import signac
from flow import FlowProject
from flow import directives
import flow.environments
import json
import os
import gsd.hoomd

class Project(FlowProject):
    pass

@Project.label
def preint(job):
    return True

@Project.label
def initialized(job):
    return os.path.isfile(job.fn("init.gsd"))

@Project.post(initialized)
#@Project.operation
@Project.operation(directives={'walltime': 1, 'nranks': 1})
def initialize(job):
    import hoomd
    from hoomd import md
    import gsd.hoomd
    import random
    from scipy.spatial.transform import Rotation
    from pair_potentials import set_pair_potentials_params
    import lattice

    r = job.sp.r
    p_mass_A = [1.0,1.0,1.0,0,0,0]
    p_mass_B = [0,0,0,1.0,1.0,1.0]
    mass = 3.0 

    a = job.sp.a
    b = job.sp.b
    # Agnese monomer shape
    positions_A = np.array([
                    [0., 0., a], # first sphere
                    [0., a*np.cos(np.pi/6.), -a*np.sin(np.pi/6.)],  # second sphere
                    [0., -a*np.cos(np.pi/6.), -a*np.sin(np.pi/6.)],  # third sphere
                    [a, 0., b],  # first patch
                    [a, b*np.cos(np.pi/6.), -b*np.sin(np.pi/6.)],  # second patch
                    [a, -b*np.cos(np.pi/6.), -b*np.sin(np.pi/6.)]  # third patch
                ], dtype=np.float64)
    positions_B = np.array([
                    [-a, 0., b],  # first patch
                    [-a, b*np.cos(np.pi/6.), -b*np.sin(np.pi/6.)],  # second patch
                    [-a, -b*np.cos(np.pi/6.), -b*np.sin(np.pi/6.)], # third patch
                    [0., 0., a], # first sphere
                    [0., a*np.cos(np.pi/6.), -a*np.sin(np.pi/6.)],  # second sphere
                    [0., -a*np.cos(np.pi/6.), -a*np.sin(np.pi/6.)]  # third sphere
                ], dtype=np.float64)
    

    MOI_A = [0,0,0]
    MOI_B = [0,0,0]
    for i, p in enumerate(positions_A):
        MOI_A[0] += p_mass_A[i]*(p[1]**2 + p[2]**2) + 0.4*p_mass_A[i]*r**2
        MOI_A[1] += p_mass_A[i]*(p[0]**2 + p[2]**2) + 0.4*p_mass_A[i]*r**2
        MOI_A[2] += p_mass_A[i]*(p[0]**2 + p[1]**2) + 0.4*p_mass_A[i]*r**2
    for i, p in enumerate(positions_B):
        MOI_B[0] += p_mass_B[i]*(p[1]**2 + p[2]**2) + 0.4*p_mass_B[i]*r**2
        MOI_B[1] += p_mass_B[i]*(p[0]**2 + p[2]**2) + 0.4*p_mass_B[i]*r**2
        MOI_B[2] += p_mass_B[i]*(p[0]**2 + p[1]**2) + 0.4*p_mass_B[i]*r**2

    typesA = ['AM', 'AM', 'AM', 'AP1', 'AP2', 'AP3']
    typesB = ['BP1', 'BP2', 'BP3', 'BM', 'BM', 'BM']

    # Default direction of dipole: +x
    orientations = np.array([[1, 0, 0, 0]]*6, dtype = np.float64)

    uc = lattice.unitcell(N = 2,
                            a1 = np.array([job.sp.lat_a,0,0]),
                            a2 = np.array([0,job.sp.lat_a,0]),
                            a3 = np.array([0,0,job.sp.lat_a]),
                            dimensions = 3,
                            position = [[4.0,0.0,0.0], [-4.0, 0.0, 0.0]],
                            type_name = ['A', 'B'],
                            mass = [3.0, 3.0],
                            charge = [0.0, 0.0],
                            diameter = [1.0, 1.0],
                            moment_inertia = [MOI_A, MOI_B],
                            orientation = [[1,0,0,0]]*2)
    system = lattice.create_lattice(unitcell=uc,
                                       irange=job.sp.Nx, 
                                       jrange=job.sp.Ny, 
                                       krange=job.sp.Nz)


    random_rotations = Rotation.random(num=system.particles.N, random_state=job.sp.seed)
    system.particles.orientation = random_rotations.as_quat(scalar_first=True)
    system.particles.mass = np.array([mass]*system.particles.N)
    system.particles.charge = np.array([0.0]*system.particles.N)

    system.particles.types.append('AM')
    system.particles.types.append('AP1')
    system.particles.types.append('AP2')
    system.particles.types.append('AP3')
    system.particles.types.append('BM')
    system.particles.types.append('BP1')
    system.particles.types.append('BP2')
    system.particles.types.append('BP3')

    rigid = md.constrain.Rigid()
    rigid.body['A'] = {
                    "constituent_types": typesA,
                    "positions": positions_A,
                    "orientations": orientations
    }
    rigid.body['B'] = {
                    "constituent_types": typesB,
                    "positions": positions_B,
                    "orientations": orientations
    }

    with gsd.hoomd.open(name=job.fn('lattice.gsd'), mode='w') as f:
        f.append(system)

    # device = hoomd.device.CPU(notice_level=10) # Run on CPU
    device = hoomd.device.GPU() # Run on gpu
    simulation = hoomd.Simulation(device=device, seed=job.sp.seed)
    simulation.create_state_from_gsd(filename=job.fn('lattice.gsd'))
    
    rigid.create_bodies(simulation.state)

    np.save(job.fn("types.npy"), np.array([typesA, typesB]))
    np.save(job.fn("positions_A.npy"), positions_A)
    np.save(job.fn("positions_B.npy"), positions_B)
    np.save(job.fn("orientations.npy"), orientations)

    # Set up pair potentials
    nl = hoomd.md.nlist.Cell(buffer=0, exclusions=['body'])
    morse = md.pair.Morse(default_r_cut = job.sp.r_cut, nlist = nl)
    # lj = md.pair.LJ(default_r_cut=job.sp.r, nlist=nl)
    table = md.pair.Table(nlist = nl, default_r_cut = job.sp.rep_r_cut)

    def smooth_step(r, rmin, rmax, steepness=10):
        x = (r - rmin) / (rmax - rmin)
        return jnp.clip(1 / (1 + jnp.exp(-steepness * (x - 0.5))), 0, 1)

    def repulsive_potential(rmin, rmax, A, alpha):
        def _V(r):
            epsilon = 1e-6
            base = jnp.maximum(rmax - r, epsilon)
            # smoothing_factor = smooth_step(r, rmin, rmax)
            potential = (A / (alpha * rmax)) * base**alpha
            return jnp.where(r < rmax, potential * smooth_step(r, rmin, rmax), 0.0)
        
        return _V
    repulsive = repulsive_potential(rmin=job.sp.rep_r_min, rmax=job.sp.rep_r_max, A=job.sp.rep_A, alpha=job.sp.rep_alpha)
    tabulated_repulsive = (
        np.array(repulsive(np.linspace(0,job.sp.rep_r_cut,1001))),
        np.array(-1*vmap(grad(repulsive))(jnp.linspace(0,job.sp.rep_r_cut,1001)))
    )

    set_pair_potentials_params(job, morse, table, tabulated_repulsive, system.particles.types)

    """ Options here are the following:
    none - No shifting is performed and potentials are abruptly cut off
    shift - A constant shift is applied to the entire potential so that it is 0 at the cutoff
    xplor - A smoothing function is applied to gradually decrease both the force and potential to 0 at the cutoff when ron < rcut, and shifts the potential to 0 at the cutoff when ron >= rcut.
    """
    morse.mode = "shift"
    # lj.mode = "shift"

    integrator = md.Integrator(dt = job.sp.dt, integrate_rotational_dof=True)
    simulation.operations.integrator = integrator
    integrator.rigid = rigid

    rigid_centers_and_free = hoomd.filter.Rigid(('center', 'free'))
    md_int = md.methods.ConstantVolume(filter = rigid_centers_and_free, thermostat = md.methods.thermostats.MTTK(kT = 2.0+job.sp.kT, tau = job.sp.tau))
    integrator.methods.append(md_int)
    integrator.forces.append(morse)
    integrator.forces.append(table)

    simulation.state.thermalize_particle_momenta(filter=rigid_centers_and_free, kT=2.0+job.sp.kT)
    
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.All()
    )
    simulation.operations.computes.append(thermodynamic_properties)
    logger = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger.add(simulation, quantities = ['timestep', 'walltime'])
    logger.add(thermodynamic_properties, quantities = ['pressure','potential_energy','translational_kinetic_energy','rotational_kinetic_energy'])
    hdf5_writer = hoomd.write.HDF5Log(
        trigger=hoomd.trigger.Periodic(int(job.sp.log_period)), filename=job.fn('init_log.h5'), mode='w', logger=logger
    )
    simulation.operations.writers.append(hdf5_writer)

    # Rescale the system to achieve desired packing fraction
    concentration = job.sp.concentration
    total_N = uc.N * job.sp.Nx * job.sp.Ny * job.sp.Nz
    final_L = (total_N/concentration)**(1/3)

    inverse_volume_ramp = hoomd.variant.box.InverseVolumeRamp(
        initial_box=simulation.state.box,
        final_volume=final_L**3,
        t_start=simulation.timestep,
        t_ramp=int(job.sp.equil_step),
    )
    box_resize = hoomd.update.BoxResize(
        trigger=hoomd.trigger.Periodic(10),
        box=inverse_volume_ramp,
    )
    simulation.operations.updaters.append(box_resize)
    simulation.run(job.sp.equil_step)
    simulation.operations.updaters.remove(box_resize)
    simulation.run(1e6)

    hoomd.write.GSD.write(state=simulation.state, mode='xb', filename=job.fn('init.gsd'))


@Project.label
def dumped(job):
    if job.isfile("dump.gsd"):
        with job:
            with gsd.hoomd.open(name="dump.gsd", mode="r") as gsd_file:
                if len(gsd_file) == 600 or len(gsd_file) == 1100 or len(gsd_file) == 1600 or len(gsd_file) == 2100:
                    return True
    return False

@Project.pre.after(initialize)
@Project.post(dumped)
# @Project.operation
@Project.operation(directives={'walltime': 48, 'nranks': 1})
def equilibrate(job):
    import hoomd
    from hoomd import md
    from pair_potentials import set_pair_potentials_params


    types = np.load(job.fn("types.npy"))
    typesA = types[0]
    typesB = types[1]
    positions_A = np.load(job.fn("positions_A.npy"))
    positions_B = np.load(job.fn("positions_B.npy"))
    orientations = np.load(job.fn("orientations.npy"))

    # device = hoomd.device.CPU(notice_level=10) # Run on CPU
    device = hoomd.device.GPU() # Run on gpu
    simulation = hoomd.Simulation(device=device, seed=job.sp.seed)
    simulation.create_state_from_gsd(filename=job.fn('init.gsd'))

    rigid = md.constrain.Rigid()
    rigid.body['A'] = {
                    "constituent_types": typesA,
                    "positions": positions_A,
                    "orientations": orientations
    }
    rigid.body['B'] = {
                    "constituent_types": typesB,
                    "positions": positions_B,
                    "orientations": orientations
    }

    # Set up pair potentials
    nl = hoomd.md.nlist.Cell(buffer=0, exclusions=['body'])
    morse = md.pair.Morse(default_r_cut = job.sp.r_cut, nlist = nl)
    # lj = md.pair.LJ(default_r_cut=job.sp.r, nlist=nl)
    table = md.pair.Table(nlist = nl, default_r_cut = job.sp.rep_r_cut)

    def smooth_step(r, rmin, rmax, steepness=10):
        x = (r - rmin) / (rmax - rmin)
        return jnp.clip(1 / (1 + jnp.exp(-steepness * (x - 0.5))), 0, 1)

    def repulsive_potential(rmin, rmax, A, alpha):
        def _V(r):
            epsilon = 1e-6
            base = jnp.maximum(rmax - r, epsilon)
            # smoothing_factor = smooth_step(r, rmin, rmax)
            potential = (A / (alpha * rmax)) * base**alpha
            return jnp.where(r < rmax, potential * smooth_step(r, rmin, rmax), 0.0)
        
        return _V
    repulsive = repulsive_potential(rmin=job.sp.rep_r_min, rmax=job.sp.rep_r_max, A=job.sp.rep_A, alpha=job.sp.rep_alpha)
    tabulated_repulsive = (
        np.array(repulsive(np.linspace(0,job.sp.rep_r_cut,1001))),
        np.array(-1*vmap(grad(repulsive))(jnp.linspace(0,job.sp.rep_r_cut,1001)))
    )

    set_pair_potentials_params(job, morse, table, tabulated_repulsive, simulation.state.types["particle_types"])

    """ Options here are the following:
    none - No shifting is performed and potentials are abruptly cut off
    shift - A constant shift is applied to the entire potential so that it is 0 at the cutoff
    xplor - A smoothing function is applied to gradually decrease both the force and potential to 0 at the cutoff when ron < rcut, and shifts the potential to 0 at the cutoff when ron >= rcut.
    """
    morse.mode= "shift"
    # lj.mode = "shift"


    integrator = md.Integrator(dt = job.sp.dt, integrate_rotational_dof=True)
    simulation.operations.integrator = integrator
    integrator.rigid = rigid

    rigid_centers_and_free = hoomd.filter.Rigid(('center', 'free'))
    md_int = md.methods.ConstantVolume(filter = rigid_centers_and_free, thermostat = md.methods.thermostats.MTTK(kT = 2.0+job.sp.kT, tau = job.sp.tau))
    integrator.methods.append(md_int)
    integrator.forces.append(morse)
    integrator.forces.append(table)

    simulation.state.thermalize_particle_momenta(filter=rigid_centers_and_free, kT=2.0+job.sp.kT)

    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.All()
    )
    simulation.operations.computes.append(thermodynamic_properties)
    logger = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger.add(simulation, quantities = ['timestep', 'walltime'])
    logger.add(thermodynamic_properties, quantities = ['pressure','potential_energy','translational_kinetic_energy','rotational_kinetic_energy'])
    hdf5_writer = hoomd.write.HDF5Log(
        trigger=hoomd.trigger.Periodic(int(job.sp.log_period)), filename=job.fn('dump_log.h5'), mode='w', logger=logger
    )
    simulation.operations.writers.append(hdf5_writer)

    # pos = deprecated.dump.pos(filename = job.fn('dump.pos'), unwrap_rigid = True, period = job.sp.dump_period*10)
    # pos.set_def('A', 'sphere 1 a6a6a6')
    # pos.set_def('D', 'dipole 1 0 b31b1b f7f7f7')
    # pos.set_def('B', 'sphere 1 a6a6a6')

    gsd = hoomd.write.GSD(filename = job.fn('dump.gsd'), trigger=int(job.sp.dump_period), mode='wb')
    simulation.operations.writers.append(gsd)

    for kT in np.arange(2.0+job.sp.kT, job.sp.kT, -0.1):
        md_int.thermostat.kT = kT
        simulation.run(5e5)

    md_int.thermostat.kT = job.sp.kT
    simulation.run(job.sp.run_step)



if __name__ == '__main__':
    Project().main()

