
import numpy
from jax import vmap
import jax.numpy
import gsd.hoomd
import math

"""Lattice Initialization Functions"""

def create_lattice(unitcell, irange, jrange, krange):
    index_list = numpy.array([ [i,j,k] for i in numpy.arange(irange) for j in numpy.arange(jrange) for k in numpy.arange(krange)])
    num_unitcells = index_list.shape[0]
    N_particles = num_unitcells * unitcell.N
    
    def get_cell(i,j,k):
        cell_shift = jax.numpy.array([[i*unitcell.a1 + j*unitcell.a2 + k*unitcell.a3]])
        return unitcell.position + cell_shift
    
    unitcell_rad = numpy.max([numpy.linalg.norm(position) for position in unitcell.position])
    L1, L2, L3 = numpy.linalg.norm(unitcell.a1) * irange, numpy.linalg.norm(unitcell.a2) * jrange, numpy.linalg.norm(unitcell.a3) * krange
    
    position = numpy.array(numpy.reshape(vmap(get_cell, in_axes=(0,0,0))(index_list[:,0],index_list[:,1],index_list[:,2]),(N_particles,3)) - numpy.array([L1, L2, L3])/2)
    types = numpy.array([unitcell.get_typeid_mapping()[name] for name in unitcell.type_name]*num_unitcells).reshape((-1,))

    frame = gsd.hoomd.Frame()
    frame.particles.types = unitcell.type_name
    frame.particles.typeid = types
    frame.particles.N = N_particles
    frame.particles.position = position[0:N_particles, :]
    frame.configuration.dimensions = unitcell.dimensions
    frame.configuration.box = [L1 + 3*unitcell_rad, L2 + 3*unitcell_rad, L3 + 3*unitcell_rad, 0, 0, 0]

    frame.particles.mass = numpy.array(unitcell.mass.tolist() * num_unitcells)
    frame.particles.charge = numpy.array(unitcell.charge.tolist() * num_unitcells)
    frame.particles.diameter = numpy.array(unitcell.diameter.tolist() * num_unitcells)
    frame.particles.moment_inertia = numpy.array(unitcell.moment_inertia.tolist() * num_unitcells)
    frame.particles.orientation = numpy.array(unitcell.orientation.tolist() * num_unitcells)

    return frame

class unitcell(object):
    R""" Define a unit cell.

    Args:
        N (int): Number of particles in the unit cell.
        a1 (list): Lattice vector (3-vector).
        a2 (list): Lattice vector (3-vector).
        a3 (list): Lattice vector (3-vector). Set to [0,0,1] in 2D lattices.
        dimensions (int): Dimensionality of the lattice (2 or 3).
        position (list): List of particle positions.
        type_name (list): List of particle type names.
        mass (list): List of particle masses.
        charge (list): List of particle charges.
        diameter (list): List of particle diameters.
        moment_inertia (list): List of particle moments of inertia.
        orientation (list): List of particle orientations.

    A unit cell is a box definition (*a1*, *a2*, *a3*, *dimensions*), and particle properties for *N* particles.
    You do not need to specify all particle properties. Any property omitted will be initialized to defaults (see
    :py:func:`hoomd.data.make_snapshot`). :py:class:`hoomd.init.create_lattice` initializes the system with many
    copies of a unit cell.

    :py:class:`unitcell` is a completely generic unit cell representation. See other classes in the :py:mod:`hoomd.lattice`
    module for convenience wrappers for common lattices.

    Example::

        uc = hoomd.lattice.unitcell(N = 2,
                                    a1 = [1,0,0],
                                    a2 = [0.2,1.2,0],
                                    a3 = [-0.2,0, 1.0],
                                    dimensions = 3,
                                    position = [[0,0,0], [0.5, 0.5, 0.5]],
                                    type_name = ['A', 'B'],
                                    mass = [1.0, 2.0],
                                    charge = [0.0, 0.0],
                                    diameter = [1.0, 1.3],
                                    moment_inertia = [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                                    orientation = [[0.707, 0, 0, 0.707], [1.0, 0, 0, 0]])

    Note:
        *a1*, *a2*, *a3* must define a right handed coordinate system.

    """

    def __init__(self,
                 N,
                 a1,
                 a2,
                 a3,
                 dimensions = 3,
                 position = None,
                 type_name = None,
                 mass = None,
                 charge = None,
                 diameter = None,
                 moment_inertia = None,
                 orientation = None):

        self.N = N
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.dimensions = dimensions

        if position is None:
            self.position = numpy.array([(0,0,0)] * self.N, dtype=numpy.float64)
        else:
            self.position = numpy.asarray(position, dtype=numpy.float64)
            if len(self.position) != N:
                raise ValueError("Particle properties must have length N")

        if type_name is None:
            self.type_name = ['A'] * self.N
        else:
            self.type_name = type_name
            if len(self.type_name) != N:
                raise ValueError("Particle properties must have length N")

        if mass is None:
            self.mass = numpy.array([1.0] * self.N, dtype=numpy.float64)
        else:
            self.mass = numpy.asarray(mass, dtype=numpy.float64)
            if len(self.mass) != N:
                raise ValueError("Particle properties must have length N")

        if charge is None:
            self.charge = numpy.array([0.0] * self.N, dtype=numpy.float64)
        else:
            self.charge = numpy.asarray(charge, dtype=numpy.float64)
            if len(self.charge) != N:
                raise ValueError("Particle properties must have length N")

        if diameter is None:
            self.diameter = numpy.array([1.0] * self.N, dtype=numpy.float64)
        else:
            self.diameter = numpy.asarray(diameter, dtype=numpy.float64)
            if len(self.diameter) != N:
                raise ValueError("Particle properties must have length N")

        if moment_inertia is None:
            self.moment_inertia = numpy.array([(0,0,0)] * self.N, dtype=numpy.float64)
        else:
            self.moment_inertia = numpy.asarray(moment_inertia, dtype=numpy.float64)
            if len(self.moment_inertia) != N:
                raise ValueError("Particle properties must have length N")

        if orientation is None:
            self.orientation = numpy.array([(1,0,0,0)] * self.N, dtype=numpy.float64)
        else:
            self.orientation = numpy.asarray(orientation, dtype=numpy.float64)
            if len(self.orientation) != N:
                raise ValueError("Particle properties must have length N")

    def get_type_list(self):
        R""" Get a list of the unique type names in the unit cell.

        Returns:
            A :py:class:`list` of the unique type names present in the unit cell.
        """

        type_list = []
        for name in self.type_name:
            if not name in type_list:
                type_list.append(name)

        return type_list

    def get_typeid_mapping(self):
        R""" Get a type name to typeid mapping.

        Returns:
            A :py:class:`dict` that maps type names to integer type ids.
        """

        mapping = {}
        idx = 0

        for name in self.type_name:
            if not name in mapping:
                mapping[name] = idx
                idx = idx + 1

        return mapping
    

# Randomly place arbitrary number of particles of different types
def place_particles(types: dict, 
                    dim: int, 
                    body_radius: numpy.ndarray, 
                    mass: float, 
                    charge: float, 
                    r: float, 
                    MOI: list,
                    interbody_spacing_ratio: float = 1.0):

    diameter = 2*r
    total_num = numpy.array(list(types.values())).sum()
    grid_side_length = math.ceil(total_num**(1/dim))

    types_list = [item for type in types for item in [type]*types[type]]
    assert(total_num == len(types_list))
    type_positions = types_list + [None]*(grid_side_length**dim-total_num)
    type_positions = numpy.random.choice(type_positions,size=len(type_positions),replace=False)
    type_positions = numpy.squeeze(type_positions.reshape((-1,grid_side_length,grid_side_length)))
    type_positions
    interbody_spacing = interbody_spacing_ratio * body_radius

    positions = []
    typeid = []
    for type in types:
        for position in numpy.array(numpy.where(type_positions == type)).T * (2*body_radius+interbody_spacing) + body_radius:
            positions.append(position)
            typeid.append(type)

    positions = numpy.array(positions)

    L = numpy.max(positions) + body_radius
    positions = positions - L/2

    frame = gsd.hoomd.Frame()
    frame.particles.types = list(types.keys())
    frame.particles.typeid = typeid
    frame.particles.N = total_num
    frame.particles.position = positions[0:total_num, :]
    frame.configuration.dimensions = dim
    frame.configuration.box = [L, L, L, 0, 0, 0]

    frame.particles.mass = numpy.array([mass] * total_num)
    frame.particles.charge = numpy.array([charge] * total_num)
    frame.particles.diameter = numpy.array([diameter] * total_num)
    frame.particles.moment_inertia = numpy.array([MOI] * total_num)
    frame.particles.orientation = numpy.array([1,0,0,0] * total_num)

    return frame