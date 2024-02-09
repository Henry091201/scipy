import numpy as np
cimport numpy as np

__all__ = ['particleswarm, rastrigin, ackley_function_2d']

cdef double rastrigin(x, y):
    # Rastrigin function for demonstration purposes
    cdef double rast = 20 + x ** 2 + y ** 2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
    return rast

cdef double ackley_function_2d(x, y):
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
    c = 2 * np.pi

    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))

    value = term1 + term2 + a + np.exp(1)

    return value

cdef class State:
    cdef np.ndarray velocities 
    cdef np.ndarray positions 
    cdef np.ndarray bounds 
    cdef np.ndarray pbest_fitnesses

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

    def __cinit__(self, object objective_function, int swarm_size, int max_iterations, float w, float c1, float c2, int dimensions, np.ndarray bounds = None):
        #TODO: Look into using memoryviews for the arrays --> https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html
        self.velocities = np.zeros((swarm_size, dimensions))
        self.positions = np.zeros((swarm_size, dimensions))
        self.bounds = bounds
        self.pbest_fitnesses = np.zeros(swarm_size)

        self.max_iterations = max_iterations
        self.gbest_position = np.zeros(dimensions)
        self.gbest_fitness = np.inf 

        self.swarmSize = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.dimensions = dimensions
        self.objective_function = objective_function
    
    def print_class_variables(self):
        print(f"Velocities: {self.velocities}")
        print(f"Positions: {self.positions}")
        print(f"Bounds: {self.bounds}")
        print(f"Pbest Fitnesses: {self.pbest_fitnesses}")
        print(f"Global best position: {self.gbest_position}")
        print(f"Global best fitness: {self.gbest_fitness}")
        print(f"Swarm size: {self.swarmSize}")
        print(f"w: {self.w}")
        print(f"c1: {self.c1}")
        print(f"c2: {self.c2}")
        print(f"Dimensions: {self.dimensions}")
        print(f"Objective function: {self.objective_function}")

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
        # If the new position is out of bounds, set the particles fitness to infinity
        # This will cause the particle to be ignored in the next iteration but still be in the swarm
        # with the potential to be reactivated
        if np.any(self.positions[particle_index] < self.bounds[:, 0]) or np.any(self.positions[particle_index] > self.bounds[:, 1]):
            self.pbest_fitnesses[particle_index] = np.inf
        return fitness

    cdef void calculate_all_fitnesses(self):
        cdef int i
        for i in range(self.swarmSize):
            self.calculate_fitness_and_update(i)

    cdef void initialise_fitnesses(self):
        cdef int i
        for i in range(self.swarmSize):
            self.pbest_fitnesses[i] = self.calculate_fitness(i)

    cdef void update_veocity(self, int particle_index):
        cdef int i
        cdef double r1
        cdef double r2
        cdef double cognitive_component
        cdef double social_component
        cdef np.ndarray velocity = self.velocities[particle_index]
        cdef np.ndarray position = self.positions[particle_index]
        cdef np.ndarray pbest = self.positions[particle_index]
        cdef np.ndarray gbest_position = self.gbest_position

        for i in range(self.dimensions):
            r1 = np.random.uniform(0, 1)
            r2 = np.random.uniform(0, 1)

            cognitive_component = self.c1 * r1 * (pbest[i] - position[i])
            social_component = self.c2 * r2 * (gbest_position[i] - position[i])

            self.velocities[particle_index][i] = self.w * velocity[i] + cognitive_component + social_component

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
        for i in range(self.swarmSize):
            if self.pbest_fitnesses[i] < self.gbest_fitness:
                # Take copies so that we don't have to worry about the memory being overwritten
                self.gbest_fitness = self.pbest_fitnesses[i].copy()
                self.gbest_position = self.positions[i].copy()
    
    cdef void run(self):
        # Initialise positions
        self.initialise_positions()
        # Initialise fitnesses
        self.initialise_fitnesses()
        # Update the global best
        self.update_gbest()
        # initialise velocities
        self.initialise_velocities()
        print(f"Initialised everything")
        print(self.print_class_variables())


        # Main loop 
        cdef int i
        for i in range(self.max_iterations):
            # Update the positions
            self.update_all_positions()
            # Update the fitnesses
            self.calculate_all_fitnesses()
            # Update the global best
            self.update_gbest()
            # Update the velocities
            self.update_all_velocities()

        print(f"Finished running")
        self.print_class_variables()

def particleswarm(objective_function, swarm_size, max_iterations, w, c1, c2, dimensions, bounds=None):
    pso = State(objective_function, swarm_size, max_iterations, w, c1, c2, dimensions, bounds)
    pso.run()

