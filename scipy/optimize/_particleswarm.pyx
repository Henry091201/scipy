cimport numpy as np
from libcpp cimport bool
from libc.math cimport exp, sqrt, cos, pi
from libc.stdlib cimport malloc, free

from cython.parallel import prange
import sys
import numpy as np


np.import_array()

from scipy.optimize import OptimizeResult

__all__ = ['particleswarm']

cdef tuple _update_gbest(float [:] pbest_fitnesses, float[:,:] positions, float gbest_fitness, int swarm_size):
    cdef float gbest = gbest_fitness
    cdef float[:] gbest_position = None
    cdef bint updated = False
    cdef int i
    for i in range(swarm_size):
        if pbest_fitnesses[i] < gbest:
            gbest = pbest_fitnesses[i]
            gbest_position = positions[i, :]
            updated = True

    return updated, gbest, gbest_position

cdef void _update_position(float[:,:] position, float[:,:] velocity, int swarm_size):
    cdef int i, j

    for i in range(swarm_size):
        for j in range(position.shape[1]):
            position[i, j] += velocity[i, j]

cdef float _calculate_fitness(float [:, :] positions, int particle_index, object objective_function, int dimensions, np.ndarray bounds):
    cdef float fitness = 0.0
    cdef int i
    # If the particle is outside the bounds, set the fitness to infinity
    
    fitness = objective_function(np.array(positions[particle_index, :]))

    if bounds is not None:
        for i in range(dimensions):
            if positions[particle_index][i] < bounds[i][0] or positions[particle_index][i] > bounds[i][1]:
                return float('inf')
    return fitness

cdef void _update_fitness(float fitness, float [:] pbest_fitnesses, float [:, :] pbest_fitness_positions, int particle_index,
                          float [:, :] positions, int dimensions):
    cdef int i
    if fitness < pbest_fitnesses[particle_index]:
        pbest_fitnesses[particle_index] = fitness
        # Update the position of the particle
        for i in range(dimensions):
            pbest_fitness_positions[particle_index, i] = positions[particle_index, i]

cdef float _calculate_and_update_fitness(float [:, :] positions, float [:] pbest_fitnesses, float [:, :] pbest_fitness_positions, int particle_index,
                                       object objective_function, int dimensions, np.ndarray bounds):
    cdef float fitness = _calculate_fitness(positions, particle_index, objective_function, dimensions, bounds)
    _update_fitness(fitness, pbest_fitnesses, pbest_fitness_positions, particle_index, positions, dimensions)

    return fitness

cdef int _find_best_neighbour(float [:] pbest_fitnesses, np.ndarray neighbours):
    cdef int[:] neighbours_view = neighbours
    cdef int best_neighbour = neighbours_view[0]
    cdef int i
    for i in range(neighbours_view.shape[0]):
        if pbest_fitnesses[neighbours_view[i]] < pbest_fitnesses[best_neighbour]:
            best_neighbour = neighbours_view[i]

    return best_neighbour

cdef float _cap_velocity(float vel, float max_velocity):
    # Cap the velocity if it exceeds the max velocity
    # Remember the velocity can be negative
    if vel > max_velocity:
        return max_velocity
    elif vel < -max_velocity:
        return -max_velocity
    else:
        return vel

cdef void _update_velocity(float[:, :] velocity, float[:, :] positions,
                           float[:, :] pbest_fitness_positions, float[:] pbest_fitnesses,
                           float[:] gbest_position, float w, float c1, float c2,
                           int dimensions, int swarm_size, object topology,
                           float max_velocity, State pso):
    """
    Update the velocity for each particle in the swarm based on cognitive and social components.
    """
    cdef int particle_index, best_neighbor, j
    cdef float cognitive_component, social_component, velocity_calculation
    cdef np.ndarray neighbours, best_neighbours = np.empty(swarm_size, dtype=np.int32)
    cdef int[:] best_neighbours_view = best_neighbours
    cdef np.ndarray r1 = np.random.uniform(0, 1, (swarm_size, dimensions))
    cdef np.ndarray r2 = np.random.uniform(0, 1, (swarm_size, dimensions))
    cdef double[:, :] r1_view = r1
    cdef double[:, :] r2_view = r2

    # Update velocities for each particle
    for particle_index in range(swarm_size):
        neighbours = get_neighbours(pso, particle_index, topology)
        best_neighbours_view[particle_index] = _find_best_neighbour(pbest_fitnesses, neighbours)
        best_neighbor = best_neighbours_view[particle_index]

        for j in range(dimensions):
            cognitive_component = c1 * r1_view[particle_index, j] * (pbest_fitness_positions[particle_index, j] - positions[particle_index, j])
            social_component = c2 * r2_view[particle_index, j] * (pbest_fitness_positions[best_neighbor, j] - positions[particle_index, j])
            velocity_calculation = w * velocity[particle_index, j] + cognitive_component + social_component

            # Cap the velocity if it exceeds the maximum allowed velocity
            if max_velocity > 0:
                velocity_calculation = _cap_velocity(velocity_calculation, max_velocity)

            velocity[particle_index, j] = velocity_calculation



cdef get_neighbours(State pso, int particle_index, object topology):
    cdef np.ndarray neighbours = topology(pso, particle_index).astype(np.int32)
    # if size of neighbours is 0, then raise a value error
    if neighbours.size == 0:
        raise ValueError("Topology returned an empty array. Ensure that the topology function is returning a non-empty array.")
    return neighbours

    

cdef class State:
    cdef np.ndarray velocities
    cdef np.ndarray positions 

    cdef np.ndarray bounds 

    cdef np.ndarray pbest_fitnesses
    cdef np.ndarray pbest_fitness_positions

    cdef float [:, :] velocities_view
    cdef float [:, :] positions_view
    cdef float [:] pbest_fitnesses_view
    cdef float [:, :] pbest_fitness_positions_view

    # Global best position and fitness
    cdef int max_iter
    cdef np.ndarray gbest_position 
    cdef float gbest_fitness

    # Parameters
    cdef int swarm_size
    cdef object w
    cdef float c1
    cdef float c2
    cdef int dimensions
    cdef object objective_function
    cdef object topology

    # State variables
    cdef int current_iteration

    cdef int seed

    cdef object package_result

    cdef str message
    cdef int niter_success
    cdef int niter_at_gbest

    cdef float max_velocity

    def __cinit__(self, object objective_function, int swarm_size, int dimensions, 
                int max_iter, object w, float c1, float c2, np.ndarray bounds,
                object topology, int seed, int niter_success, float max_velocity):
        # Validate the inputs to ensure they meet the expected criteria
        self.validate_inputs(
            objective_function=objective_function, swarm_size=swarm_size, 
            max_iterations=max_iter, w=w, c1=c1, c2=c2, dimensions=dimensions, 
            bounds=bounds, topology=topology, seed=seed, niter_success=niter_success, 
            max_velocity=max_velocity
        )

        # Create particle positions, velocities, and best known positions arrays
        self.velocities = np.zeros((swarm_size, dimensions), dtype=np.float32)
        self.positions = np.zeros((swarm_size, dimensions), dtype=np.float32)
        self.pbest_fitness_positions = np.zeros((swarm_size, dimensions), dtype=np.float32)
        self.pbest_fitnesses = np.zeros(swarm_size, dtype=np.float32)

        # Create initial global best position and fitness arrays
        self.gbest_position = np.zeros(dimensions, dtype=np.float32)
        self.gbest_fitness = np.inf

        # Additional state variables initialization
        self.bounds = bounds
        self.max_iter = max_iter
        self.swarm_size = swarm_size
        self.dimensions = dimensions
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.objective_function = objective_function
        self.topology = topology
        self.current_iteration = 0
        self.niter_success = niter_success
        self.niter_at_gbest = 0
        self.max_velocity = max_velocity

        # Initialize memory views for direct array manipulation
        self.velocities_view = self.velocities
        self.positions_view = self.positions
        self.pbest_fitnesses_view = self.pbest_fitnesses
        self.pbest_fitness_positions_view = self.pbest_fitness_positions

        # Seed the random number generator if a seed is provided
        if seed != -1:
            np.random.seed(seed)

    cdef void setup(self):
        """
        Setup the initial state for the particle swarm optimization by initialising positions,
        fitnesses, velocities, and updating the global best.
        """
        self.initialise_positions()  # Initialise particle positions
        self.initialise_fitnesses()  # Initialise fitness values for each particle
        self.update_gbest()          # Update global best from initial particles
        self.initialise_velocities() # Initialise particle velocities

    def get_attributes(self):
        return {
            'velocities': self.velocities,
            'positions': self.positions,
            'pbest_fitnesses': self.pbest_fitnesses,
            'pbest_fitness_positions': self.pbest_fitness_positions,
            'gbest_position': self.gbest_position,
            'gbest_fitness': self.gbest_fitness,
            'max_iter': self.max_iter,
            'swarm_size': self.swarm_size,
            'w': self.w,
            'c1': self.c1,
            'c2': self.c2,
            'dimensions': self.dimensions,
            'objective_function': self.objective_function,
            'topology': self.topology,
            'current_iteration': self.current_iteration,
            'seed': self.seed,
            'niter_success': self.niter_success,
            'niter_at_gbest': self.niter_at_gbest,
            'max_velocity': self.max_velocity}

    cpdef np.ndarray star(self, State pso, int particle_index):
        """
        Define a star topology for the swarm where every particle is connected to every other particle.

        Parameters:
        pso (State): The current state of the PSO simulation.
        particle_index (int): Index of the current particle (not used in this topology).

        Returns:
        np.ndarray: An array of indices representing all particles in the swarm.
        """
        return np.arange(self.swarm_size, dtype=np.int32)

    cpdef np.ndarray ring(self, State pso, int particle_index):
        """
        Define a ring topology for the swarm where each particle is connected to its two immediate neighbors.

        Parameters:
        pso (State): The current state of the PSO simulation.
        particle_index (int): Index of the current particle.

        Returns:
        np.ndarray: An array of indices representing the neighbors of the current particle.
        """
        cdef np.ndarray neighbours = np.empty(2, dtype=np.int32)
        neighbours[0] = (particle_index - 1) % self.swarm_size  # Wrap-around left neighbor
        neighbours[1] = (particle_index + 1) % self.swarm_size  # Wrap-around right neighbor
        return neighbours


    def __next__(self):
        """
        Perform a single iteration of the particle swarm optimisation process.

        Raises:
        StopIteration: When the maximum number of iterations is reached, or no improvement
                    is found in global best for a predefined number of iterations.
        """
        # Check if the maximum number of iterations has been reached
        if self.current_iteration >= self.max_iter:
            self.message = "Maximum number of iterations reached"
            raise StopIteration

        # Check if the number of iterations without improvement in global best exceeds limit
        if self.niter_success > 0 and self.niter_at_gbest >= self.niter_success:
            self.message = "Maximum number of iterations at global best reached"
            raise StopIteration

        # Update particle positions based on current velocities
        _update_position(self.positions_view, self.velocities_view, self.swarm_size)

        # Calculate fitness for all particles
        self.calculate_all_fitnesses()

        # Update global best from the new fitness values
        self.update_gbest()

        # Update velocities based on new particle positions and personal bests
        self.update_velocity()

        # Increment the current iteration count
        self.current_iteration += 1


    def __iter__(self):
        return self

    cdef solve(self):
        """
        Execute the particle swarm optimization process until the stopping condition is met.

        Returns:
        object: An instance of OptimizeResult containing the optimization results.
        """
        cdef int i
        for i in range(self.max_iter + 1):
            try:
                next(self)
            except StopIteration:
                break

        return self.package_result(self.message)

    def package_result(self, message):
        """
        Package the results of the optimization into a structured format.

        Parameters:
        message (str): A message describing the outcome of the optimization.

        Returns:
        object: An OptimizeResult containing the optimization outcomes.
        """
        cdef object result = OptimizeResult(
            x=self.gbest_position,
            fun=self.gbest_fitness,
            population=self.positions,
            nit=self.current_iteration,
            nfev=self.current_iteration * self.swarm_size,
            success=True,
            message=message
        )
        return result


    def validate_inputs(self, object objective_function, int swarm_size, int max_iterations, 
                        object w, float c1, float c2, int dimensions, np.ndarray bounds, 
                        object topology, int seed, int niter_success, float max_velocity):
        """
        Validate the input parameters for the particle swarm optimization configuration.

        Raises:
        ValueError: If any of the parameters do not meet the required criteria.
        """
        if not callable(objective_function):
            raise ValueError("Objective function must be callable.")
        if not callable(w) and w < 0:
            raise ValueError("Inertia weight must be greater than 0.")
        if c1 < 0 or c2 < 0:
            raise ValueError("Cognitive and social components must be greater than 0.")
        if dimensions <= 0:
            raise ValueError("Number of dimensions must be greater than 0.")
        if swarm_size < 1:
            raise ValueError("Swarm size must be greater than 0.")
        if max_iterations < 1:
            raise ValueError("Maximum number of iterations must be greater than 0.")
        if niter_success < 1 and niter_success != -1:
            raise ValueError("Number of iterations at global best must be greater than 1.")
        if max_velocity <= 0 and max_velocity != -1.0:
            raise ValueError("Maximum velocity must be greater than 0.")

    cdef void update_velocity(self):
        """
        Update the velocities of all particles in the swarm according to the chosen topology.
        """
        # Determine the appropriate topology function
        cdef object topology_local = self.resolve_topology()

        # Get current inertia from the inertia weight function or static value
        cdef float current_inertia = self.w(self) if callable(self.w) else self.w

        # Delegate to a low-level Cython function to update velocities
        _update_velocity(
            velocity=self.velocities_view, 
            positions=self.positions_view, 
            pbest_fitnesses=self.pbest_fitnesses_view,
            pbest_fitness_positions=self.pbest_fitness_positions_view, 
            gbest_position=self.gbest_position,
            w=current_inertia, c1=self.c1, c2=self.c2, swarm_size=self.swarm_size, 
            dimensions=self.dimensions,
            topology=topology_local, max_velocity=self.max_velocity, pso=self
        )

    def resolve_topology(self):
        """
        Resolve the topology to use for updating particle velocities based on the configuration.

        Returns:
        object: The topology function to be used.
        Raises:
        ValueError: If the topology is not recognized or callable.
        """
        if callable(self.topology):
            return self.topology
        elif self.topology == 'ring':
            return self.ring
        elif self.topology == 'star':
            return self.star
        else:
            raise ValueError("Invalid topology. Must be callable or one of 'ring' or 'star'.")


    cdef void initialise_positions(self):
        """
        Initialize particle positions within specified bounds for each dimension.
        If no bounds are specified, it defaults to (-5, 5) for each dimension.
        """
        cdef int i, j
        cdef np.ndarray local_bounds

        # Determine the bounds for initialization
        if self.bounds is None:
            # Default bounds if none are specified
            local_bounds = np.array([(-5, 5)] * self.dimensions, dtype=np.float32)
        else:
            local_bounds = self.bounds

        # Initialize each particle's position
        for i in range(self.swarm_size):
            for j in range(self.dimensions):
                # Generate a random position within the bounds for each dimension
                random_position = np.random.uniform(local_bounds[j][0], local_bounds[j][1])
                self.positions[i, j] = random_position


    cpdef void initialise_velocities(self):
        """
        Initialize the velocities of each particle within specified velocity bounds.
        If bounds are provided, velocities are set to a maximum of 30% of each dimension's range.
        If no bounds are given, velocities default to between -1 and 1.
        """
        cdef int i, j
        cdef float min_velocity_bound, max_velocity_bound

        # Iterate through each particle and dimension to initialize velocities
        for i in range(self.swarm_size):
            for j in range(self.dimensions):
                if self.bounds is not None:
                    # Calculate velocity bounds as 30% of the positional bounds
                    min_velocity_bound = 0.3 * self.bounds[j][0]
                    max_velocity_bound = 0.3 * self.bounds[j][1]
                    # Generate a random velocity within the calculated bounds
                    random_velocity = np.random.uniform(min_velocity_bound, max_velocity_bound)
                else:
                    # Default velocity bounds when no positional bounds are provided
                    random_velocity = np.random.uniform(-1, 1)

                # Set the particle's velocity
                self.velocities[i, j] = random_velocity


    cdef void calculate_all_fitnesses(self):
        """
        Calculate and update the fitness for all particles in the swarm.
        This method iterates through all particles, computing their fitness based on the 
        current positions and updates personal bests if the new fitness is better.
        """
        cdef int i
        # Iterate through each particle to calculate and update its fitness
        for i in range(self.swarm_size):
            _calculate_and_update_fitness(
                positions=self.positions_view,
                pbest_fitness_positions=self.pbest_fitness_positions_view,
                pbest_fitnesses=self.pbest_fitnesses_view,
                particle_index=i,
                objective_function=self.objective_function,
                dimensions=self.dimensions,
                bounds=self.bounds
            )

    cdef void initialise_fitnesses(self):
        cdef int i
        for i in range(self.swarm_size):
            self.pbest_fitnesses[i] = _calculate_fitness(self.positions, i, self.objective_function, self.dimensions, self.bounds)
            self.pbest_fitness_positions[i] = self.positions[i].copy()

    cdef void update_gbest(self):
        cdef bint updated = False
        cdef float gbest 
        cdef float[:] gbest_position

        updated, gbest, gbest_position = _update_gbest(pbest_fitnesses=self.pbest_fitnesses_view,
                                                         positions=self.positions_view, gbest_fitness=self.gbest_fitness,
                                                         swarm_size=self.swarm_size)
        if updated:
            self.gbest_fitness = gbest
            self.gbest_position = np.array(gbest_position, dtype='f')
            self.niter_at_gbest = 0
        else:
            # If no particle has a better fitness than the global best, increment the counter
            self.niter_at_gbest += 1

    cpdef int get_current_iteration(self):
        return self.current_iteration
            
    cpdef int get_swarm_size(self):
        return self.swarm_size
    cpdef int get_max_iter(self):
        return self.max_iter

    def get_velocities(self):
        return self.velocities

cdef class TestState(State):
    """
    This is a class that allows easy the Python unit tests to interface with the Cython attributes
    """
    __test__ = False

    def get_velocities(self):
        return self.velocities 
    def get_positions(self):
        return self.positions
    def get_pbest_fitnesses(self):
        return self.pbest_fitnesses
    def get_pbest_fitness_positions(self):
        return self.pbest_fitness_positions
    def get_gbest_position(self):
        return self.gbest_position
    def get_gbest_fitness(self):
        return self.gbest_fitness
    def get_max_iter(self):
        return self.max_iter
    def get_swarm_size(self):
        return self.swarm_size
    def get_w(self):
        return self.w
    def get_c1(self):
        return self.c1
    def get_c2(self):
        return self.c2
    def get_dimensions(self):
        return self.dimensions
    def get_objective_function(self):
        return self.objective_function
    def get_topology(self):
        return self.topology
    def get_current_iteration(self):
        return self.current_iteration
    def get_seed(self):
        return self.seed
    def setup_test(self):
        self.setup()

    def set_particle_position(self, int particle_index, np.ndarray position):
        self.positions[particle_index] = position

    def calculate_all_fitnesses(self) :
        return super().calculate_all_fitnesses()

    def calculate_particle_fitness(self, int particle_index):
        return _calculate_fitness(self.positions, particle_index, self.objective_function, self.dimensions, self.bounds)

    def set_particle_velocity(self, int particle_index, np.ndarray velocity):
        self.velocities[particle_index] = velocity

    def update_all_velocities(self):
        self.update_velocity()

cpdef particleswarm(
    object objective_function,
    int swarm_size,
    int dimensions,
    int max_iter=1000,
    object w=0.729,
    float c1=1.4,
    float c2=1.4,
    np.ndarray bounds=None,
    object topology='star',
    int seed=-1,
    int niter_success=-1,
    int max_velocity=-1,
):
    pso = State(
        objective_function, swarm_size, dimensions, max_iter,
        w, c1, c2, bounds, topology, seed, niter_success,
        max_velocity
    )
    pso.setup()
    return pso.solve()

def _initialise_state(object objective_function, int swarm_size,int dimensions, int max_iter=1000, float w=0.729, float c1=1.4, float c2=1.4,
                    np.ndarray bounds=None, object topology = 'star', int seed = -1, int niter_success = -1,
                    max_velocity = -1):
    """
    This returns a State object, this is used for testing.
    """
    return State(objective_function, swarm_size,dimensions, max_iter, 
                w, c1, c2, bounds, topology, seed, niter_success,
                max_velocity)
