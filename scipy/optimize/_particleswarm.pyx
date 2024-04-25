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

cpdef float rastrigin(int x, int y):
    # Rastrigin function for demonstration purposes
    cdef float rast = 20 + x ** 2 + y ** 2 - 10 * (cos(2 * pi * x) + cos(2 * pi * y))
    return rast

def rosen(x):
    """The Rosenbrock function"""
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

def sphere(x):
    """The Sphere function"""
    return np.sum(x ** 2.0)

def greiwank(x):
    """The Greiwank function"""
    d = x.shape[0]
    j = 1 + (1 / 4000) * np.sum(x ** 2.0) - np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))
    return j


def rast(x):
    # Rastrigin function for demonstration purposes
    rast = 20 + x[0] ** 2 + x[1] ** 2 - 10 * (cos(2 * pi * x[0]) + cos(2 * pi * x[1]))
    return rast

cpdef float ackley_function_2d(x, y):
    """
    Ackley function for 2-dimensional input.

    Parameters:
    x (float): X-coordinate.
    y (float): Y-coordinate.

    Returns:
    float: The value of the Ackley function for the given input coordinates.
    """
    a = 20
    b = 0.2
    c = 2 * pi

    term1 = -a * exp(-b * sqrt(0.5 * (x**2 + y**2)))
    term2 = -exp(0.5 * (cos(c * x) + cos(c * y)))

    value = term1 + term2 + a + exp(1)

    return value

cpdef hartman_6(x1, x2, x3, x4, x5, x6):
    x = np.array([x1, x2, x3, x4, x5, x6])
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 10**(-4) * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                             [2329, 4135, 8307, 3736, 1004, 9991],
                             [2348, 1451, 3522, 2883, 3047, 6650],
                             [4047, 8828, 8732, 5743, 1091, 381]])

    outer_sum = 0
    for i in range(4):
        inner_sum = 0
        for j in range(6):
            inner_sum += A[i, j] * ((x[j] - P[i, j])**2)
        outer_sum += alpha[i] * np.exp(-inner_sum)

    return -outer_sum 

"""
cpdef gbest(State pso, int particleIndex):
    cdef int swarmSize = pso.get_swarm_size()
    cdef int[:] arr = np.empty(swarmSize, dtype=np.int32)
    for i in range(swarmSize):
        arr[i] = i
    return arr
"""
cpdef np.ndarray gbest(State pso, int particleIndex):
    cdef int i
    cdef int swarmSize = pso.get_swarm_size()
    cdef np.ndarray[np.int32_t, ndim=1] arr = np.empty(swarmSize, dtype=np.int32)
    for i in range(swarmSize):
        arr[i] = i
    return arr


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

cdef void _update_position(position, velocity, int swarm_size):
    cdef int i, j

    position += velocity
    """
    for i in range(swarm_size):
        for j in range(position.shape[1]):
            position[i, j] += velocity[i, j]
            """

cdef float _calculate_fitness(float [:, :] positions, int particleIndex, object objective_function, int dimensions):
    cdef float fitness = 0.0
    cdef int i
    fitness = objective_function(np.array(positions[particleIndex, :]))
    return fitness

cdef void _update_fitness(float fitness, float [:] pbest_fitnesses, float [:, :] pbest_fitness_positions, int particleIndex,
                          float [:, :] positions, int dimensions):
    cdef int i
    if fitness < pbest_fitnesses[particleIndex]:
        pbest_fitnesses[particleIndex] = fitness
        # Update the position of the particle
        for i in range(dimensions):
            pbest_fitness_positions[particleIndex, i] = positions[particleIndex, i]

cdef float _calculate_and_update_fitness(float [:, :] positions, float [:] pbest_fitnesses, float [:, :] pbest_fitness_positions, int particleIndex,
                                       object objective_function, int dimensions):
    cdef float fitness = _calculate_fitness(positions, particleIndex, objective_function, dimensions)
    _update_fitness(fitness, pbest_fitnesses, pbest_fitness_positions, particleIndex, positions, dimensions)
    # TODO: Remember to do the bounds checking

    return fitness

cdef int _find_best_neighbour(float [:] pbest_fitnesses, np.ndarray neighbours):
    cdef int[:] neighbours_view = neighbours
    cdef int best_neighbour = neighbours_view[0]
    cdef int i
    for i in range(neighbours_view.shape[0]):
        if pbest_fitnesses[neighbours_view[i]] < pbest_fitnesses[best_neighbour]:
            best_neighbour = neighbours_view[i]

    return best_neighbour

cdef float _cap_velocity(float vel, float max_velocity) nogil:
    if vel > max_velocity:
        return max_velocity
    else:
        return vel

cdef void _update_velocity(float[:, :] velocity, float [:, :] positions, float [:, :] pbest_fitness_positions, float [:] pbest_fitnesses,
                           float [:] gbest_position, float w, float c1, float c2, int dimensions, int swarm_size, object topology,
                           float max_velocity, State pso):
    cdef int partIndex, best_neighbor, j
    cdef float cognitive_component, social_component, velocity_calculation
    cdef np.ndarray neighbours
    cdef np.ndarray best_neighbours = np.empty(swarm_size, dtype=np.int32)
    cdef int[:] best_neighbours_view = best_neighbours
    cdef np.ndarray r1 = np.random.uniform(0, 1, (swarm_size, dimensions))
    cdef np.ndarray r2 = np.random.uniform(0, 1, (swarm_size, dimensions))
    cdef double[:,:] r1_view = r1
    cdef double[:,:] r2_view = r2


    # Update the velocity in parallel
    for partIndex in range(swarm_size):
        neighbours = topology(pso, partIndex)
        best_neighbours_view[partIndex] = _find_best_neighbour(pbest_fitnesses, neighbours)
        best_neighbor = best_neighbours_view[partIndex]

        for j in range(dimensions):
            cognitive_component = c1 * r1_view[partIndex][j] * (pbest_fitness_positions[partIndex][j] - positions[partIndex][j])
            social_component = c2 * r2_view[partIndex][j] * (pbest_fitness_positions[best_neighbor][j] - positions[partIndex][j])
            velocity_calculation = w * velocity[partIndex][j] + cognitive_component + social_component

            # If the max velocity is set, cap the velocity
            if max_velocity > 0:
                velocity_calculation = _cap_velocity(velocity_calculation, max_velocity)

            velocity[partIndex][j] = velocity_calculation

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
    cdef int swarmSize
    cdef float w
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

    def __cinit__(self, object objective_function, int swarm_size, int dimensions, int max_iter, float w, float c1, float c2, np.ndarray bounds,
                 object topology, int seed, int niter_success, float max_velocity):

        self.validate_inputs(objective_function=objective_function, swarm_size=swarm_size, max_iterations=max_iter, w=w, c1=c1,
                             c2=c2, dimensions=dimensions, bounds=bounds, topology=topology, seed=seed, niter_success=niter_success, 
                             max_velocity=max_velocity)

        self.velocities = np.zeros((swarm_size, dimensions), dtype='f')
        self.positions = np.zeros((swarm_size, dimensions), dtype='f')

        self.bounds = bounds

        self.pbest_fitnesses = np.zeros(swarm_size, dtype='f')
        self.pbest_fitness_positions = np.zeros((swarm_size, dimensions), dtype='f')

        self.velocities_view = self.velocities
        self.positions_view = self.positions
        self.pbest_fitnesses_view = self.pbest_fitnesses
        self.pbest_fitness_positions_view = self.pbest_fitness_positions

        self.max_iter = max_iter
        self.gbest_position = np.zeros(dimensions, dtype='f')
        self.gbest_fitness = np.inf 

        self.swarmSize = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.dimensions = dimensions
        self.objective_function = objective_function
        self.topology = topology
    
        self.current_iteration = 0
        self.niter_success = niter_success
        self.niter_at_gbest = 0

        self.max_velocity = max_velocity

        self.seed = seed
        if seed != -1:
            np.random.seed(seed)


    cdef void setup(self):
        self.initialise_positions()
        # Initialise fitnesses
        self.initialise_fitnesses()
        # Update the global best
        self.update_gbest()
        # initialise velocities
        self.initialise_velocities()
        ##print(f"Initialised everything")
        #print(self.print_class_variables())

    def __next__(self):
        if self.current_iteration >= self.max_iter:
            self.message = "Maximum number of iterations reached"
            raise StopIteration

        if self.niter_success > 0 and self.niter_at_gbest >= self.niter_success:
            self.message = "Maximum number of iterations at global best reached"
            raise StopIteration
        # Do a single interation
        # Update the positions
        _update_position(self.positions, self.velocities, self.swarmSize)
        # self.update_all_positions()
        # Update the fitnesses
        self.calculate_all_fitnesses()
        # Update the global best
        self.update_gbest()
        # Update the velocities

        #self.update_all_velocities()
        _update_velocity(velocity=self.velocities_view, positions=self.positions_view, pbest_fitnesses=self.pbest_fitnesses_view,
                         pbest_fitness_positions=self.pbest_fitness_positions_view, gbest_position=self.gbest_position,
                         w=self.w, c1=self.c1, c2=self.c2, swarm_size=self.swarmSize, dimensions=self.dimensions,
                         topology=self.topology, max_velocity=self.max_velocity, pso=self)

        self.current_iteration += 1

    def __iter__(self):
        return self

    cdef solve(self):
        cdef int i
        for i in range(self.max_iter + 1):
            try:
                next(self)
            except StopIteration:
                break
        
        return self.package_result(self.message)

    def package_result(self, message):
        cdef object result = OptimizeResult(
        x = self.gbest_position,
        fun = self.gbest_fitness,
        population = self.positions,
        nit = self.current_iteration,
        nfev = self.current_iteration * self.swarmSize,
        success = True,
        message = message
        )
        return result

    def validate_inputs(self, object objective_function, int swarm_size, int max_iterations, float w, float c1, float c2, int dimensions, np.ndarray bounds, object topology, int seed, int niter_success, float max_velocity):
        if not callable(objective_function):
            raise ValueError("Objective function must be callable.")
        if w < 0:
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
        if not callable(topology):
            raise ValueError("Topology must be callable.")
        

    cdef void initialise_positions(self):
        cdef int i
        cdef int j

        if self.bounds is None:
            # If no bounds are given, assume the bounds are (-1, 1) for each dimension
            # TODO: This is not a good way of doing this, as this will eventually constrict the search space even if the user doesn't want it to be
            self.bounds = np.array([(-1, 1)] * self.dimensions, dtype='f')

        for i in range(self.swarmSize):
            for j in range(self.dimensions):
                # Choose a random position within the bounds for that dimension
                random_position = np.random.uniform(self.bounds[j][0], self.bounds[j][1])
                # Set the position
                self.positions[i, j] = random_position

    cdef void initialise_velocities(self):
        cdef int i
        cdef int j

        for i in range(self.swarmSize):
            for j in range(self.dimensions):
                # Choose a random velocity between -1 and 1
                random_velocity = np.random.uniform(-1, 1)
                # Set the velocity
                self.velocities[i, j] = random_velocity

    cdef float calculate_fitness(self, int particle_index):
        cdef np.ndarray position = self.positions[particle_index]
        assert len(position) == self.dimensions, "Argument 'position' must have the same length as the number of dimensions."
        cdef float fitness = self.objective_function(*position)
        return fitness

    cdef float calculate_fitness_and_update(self, int particle_index):
        cdef float fitness = self.calculate_fitness(particle_index)
        # Update the pbest fitness if the new fitness is better
        if fitness < self.pbest_fitnesses[particle_index]:
            self.pbest_fitnesses[particle_index] = fitness
            self.pbest_fitness_positions[particle_index] = self.positions[particle_index].copy()
            # If the new position is out of bounds, set the particles fitness to infinity
            # This will cause the particle to be ignored in the next iteration but still be in the swarm
            # with the potential to be reactivated
            if np.any(self.positions[particle_index] < self.bounds[:, 0]) or np.any(self.positions[particle_index] > self.bounds[:, 1]):
                self.pbest_fitnesses[particle_index] = np.inf
        return fitness

    cdef void calculate_all_fitnesses(self):
        cdef int i
        for i in range(self.swarmSize):
            _calculate_and_update_fitness(positions=self.positions_view, pbest_fitness_positions=self.pbest_fitness_positions_view,
                                          pbest_fitnesses=self.pbest_fitnesses_view, particleIndex=i, objective_function=self.objective_function,
                                          dimensions=self.dimensions)

    cdef void initialise_fitnesses(self):
        cdef int i
        for i in range(self.swarmSize):
            self.pbest_fitnesses[i] = _calculate_fitness(self.positions, i, self.objective_function, self.dimensions)
            self.pbest_fitness_positions[i] = self.positions[i].copy()

    cdef void update_veocity(self, int particle_index):
        cdef int i
        cdef float r1, r2, cognitive_component, social_component
        cdef np.ndarray velocity = self.velocities[particle_index].copy()
        cdef np.ndarray position = self.positions[particle_index].copy()
        cdef np.ndarray pbest = self.pbest_fitness_positions[particle_index].copy()

        # Find the particles neighbors that it can share information with
        neighbors = self.topology(self, particle_index)

        best_neighbor = neighbors[0]
        # Find the best neighbor
        for index in neighbors:
            if self.pbest_fitnesses[index] < self.pbest_fitnesses[best_neighbor]:
                best_neighbor = index

        for i in range(self.dimensions):
            r1 = np.random.uniform(0, 1)
            r2 = np.random.uniform(0, 1)

            cognitive_component = self.c1 * r1 * (pbest[i] - position[i])
            social_component = self.c2 * r2 * (self.pbest_fitness_positions[best_neighbor][i] - position[i])
            velocity_calculation = self.w * velocity[i] + cognitive_component + social_component

            # If the max velocity is set, cap the velocity
            if self.max_velocity > 0:
                velocity_calculation = self.cap_velocity(velocity_calculation, self.max_velocity)

            self.velocities[particle_index][i] = velocity_calculation

    cdef float cap_velocity(self, float vel, float max_velocity):
        if vel > max_velocity:
            return max_velocity
        else:
            return vel

    cdef void update_all_velocities(self):
        cdef int i
        for i in range(self.swarmSize):
            self.update_veocity(i)

    cdef void update_position(self, int particle_index):
        cdef int i
        cdef np.ndarray position = self.positions[particle_index]
        cdef np.ndarray velocity = self.velocities[particle_index]
        for i in range(self.dimensions):
            position[i] = position[i] + velocity[i]
            # The bounds are dealt with in the fitness function, so we don't need to worry about them here

    cdef void update_all_positions(self):
        cdef int i
        for i in range(self.swarmSize):
            self.update_position(i)

    cdef void update_gbest(self):
        cdef bint updated = False
        cdef float gbest 
        cdef float[:] gbest_position

        updated, gbest, gbest_position = _update_gbest(pbest_fitnesses=self.pbest_fitnesses_view,
                                                         positions=self.positions_view, gbest_fitness=self.gbest_fitness,
                                                         swarm_size=self.swarmSize)
        if updated:
            self.gbest_fitness = gbest
            self.gbest_position = np.array(gbest_position, dtype='f')
            self.niter_at_gbest = 0
        else:
            # If no particle has a better fitness than the global best, increment the counter
            self.niter_at_gbest += 1

            

    cpdef int get_swarm_size(self):
        return self.swarmSize

cpdef particleswarm(object objective_function, int swarm_size,int dimensions, int max_iter=1000, float w=0.729, float c1=2, float c2=2,
                    np.ndarray bounds=None, object topology = gbest, int seed = -1, int niter_success = -1,
                    max_velocity = -1):
    pso = State(objective_function, swarm_size,dimensions, max_iter, 
                w, c1, c2, bounds, topology, seed, niter_success,
                max_velocity)
    pso.setup()
    return pso.solve()