cimport numpy as np
from libcpp cimport bool
from libc.math cimport exp, sqrt, cos, pi
from libc.stdlib cimport malloc, free

import numpy as np

np.import_array()

from scipy.optimize import OptimizeResult

__all__ = ['particleswarm']

cpdef double rastrigin(x, y):
    # Rastrigin function for demonstration purposes
    cdef double rast = 20 + x ** 2 + y ** 2 - 10 * (cos(2 * pi * x) + cos(2 * pi * y))
    return rast

cpdef double ackley_function_2d(x, y):
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
cpdef int[:] gbest(State pso, int particleIndex):
    cdef int swarmSize = pso.get_swarm_size()
    cdef int *arr = <int *>malloc(swarmSize * sizeof(int))
    if arr == NULL:
        raise MemoryError()
    cdef int[:] result_view = <int[:swarmSize]>arr
    try:
        for i in range(swarmSize):
            arr[i] = i
        return result_view
    except:
        free(arr)
        raise

cdef void _update_position(float [:, :] position, float [:, :] velocity, int swarm_size):
    cdef int i, j
    for i in range(swarm_size):
        for j in range(position.shape[1]):
            position[i, j] += velocity[i, j]

cdef float _calculate_fitness(float [:, :] positions, int particleIndex, object objective_function, int dimensions):
    cdef float fitness = 0.0
    cdef int i
    fitness = objective_function(positions[particleIndex, 0], positions[particleIndex, 1])
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

    return fitness

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
    cdef int max_iterations
    cdef np.ndarray gbest_position 
    cdef double gbest_fitness 

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

    cdef char* message
    cdef int niter_success
    cdef int niter_at_gbest

    cdef float max_velocity

    def __cinit__(self, object objective_function, int swarm_size, int max_iterations, float w, float c1, float c2, int dimensions, np.ndarray bounds = None, object topology = gbest, int seed = -1, int niter_success = -1, float max_velocity = -1.0):
        #TODO: Look into using memoryviews for the arrays --> https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html
        self.velocities = np.zeros((swarm_size, dimensions), dtype=np.float32)
        self.positions = np.zeros((swarm_size, dimensions), dtype=np.float32)

        self.bounds = bounds

        self.pbest_fitnesses = np.zeros(swarm_size, dtype=np.float32)
        self.pbest_fitness_positions = np.zeros((swarm_size, dimensions), dtype=np.float32)

        self.velocities_view = self.velocities
        self.positions_view = self.positions
        self.pbest_fitnesses_view = self.pbest_fitnesses
        self.pbest_fitness_positions_view = self.pbest_fitness_positions

        self.max_iterations = max_iterations
        self.gbest_position = np.zeros(dimensions)
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
        if self.current_iteration >= self.max_iterations:
            self.message = "Maximum number of iterations reached"
            raise StopIteration

        if self.niter_success > 0 and self.niter_at_gbest >= self.niter_success:
            self.message = "Maximum number of iterations at global best reached"
            raise StopIteration
        # Do a single interation
        # Update the positions
        _update_position(self.positions_view, self.velocities_view, self.swarmSize)
        # self.update_all_positions()
        # Update the fitnesses
        self.calculate_all_fitnesses()
        # Update the global best
        self.update_gbest()
        # Update the velocities
        self.update_all_velocities()

        self.current_iteration += 1

    def __iter__(self):
        return self

    cdef solve(self):
        cdef int i
        for i in range(self.max_iterations + 1):
            try:
                next(self)
            except StopIteration:
                #print("Finished")
                #self.print_class_variables()
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
        #print(f"Finished writing the results")
        #print(result.keys())
        return result
        

    cdef void initialise_positions(self):
        cdef int i
        cdef int j

        if self.bounds is None:
            # If no bounds are given, assume the bounds are (-1, 1) for each dimension
            # TODO: This is not a good way of doing this, as this will eventually constrict the search space even if the user doesn't want it to be
            self.bounds = np.array([(-1, 1)] * self.dimensions)

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

    cdef double calculate_fitness(self, int particle_index):
        cdef np.ndarray position = self.positions[particle_index]
        assert len(position) == self.dimensions, "Argument 'position' must have the same length as the number of dimensions."
        cdef double fitness = self.objective_function(*position)
        return fitness

    cdef double calculate_fitness_and_update(self, int particle_index):
        cdef double fitness = self.calculate_fitness(particle_index)
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
            self.pbest_fitnesses[i] = self.calculate_fitness(i)
            self.pbest_fitness_positions[i] = self.positions[i].copy()

    cdef void update_veocity(self, int particle_index):
        cdef int i
        cdef double r1, r2, cognitive_component, social_component
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
        cdef int i 
        cdef bint updated = False
        for i in range(self.swarmSize):
            if self.pbest_fitnesses[i] < self.gbest_fitness:
                # Take copies so that we don't have to worry about the memory being overwritten
                self.gbest_fitness = self.pbest_fitnesses[i].copy()
                self.gbest_position = self.positions[i].copy()
                updated = True
        if updated:
            self.niter_at_gbest = 0
        else:
            # If no particle has a better fitness than the global best, increment the counter
            self.niter_at_gbest += 1

            

    cpdef int get_swarm_size(self):
        return self.swarmSize

cpdef particleswarm(object objective_function, int swarm_size, int max_iterations, float w, float c1, float c2,
                     int dimensions, np.ndarray bounds=None, object topology = gbest, int seed = -1, int niter_success = -1,
                    max_velocity = -1):
    pso = State(objective_function, swarm_size, max_iterations, w, c1, c2, dimensions, bounds, topology, seed, niter_success,
                max_velocity)
    pso.setup()
    return pso.solve()