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
        print(f"Fitnesses: {self.pbest_fitnesses}")
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
                # Choose a random velocity within the bounds for that dimension
                random_velocity = np.random.uniform(-1, 1)
                # Set the velocity
                self.velocities[i, j] = random_velocity

    cdef double calculate_fitness(self, int particle_index):
        cdef np.ndarray position = self.positions[particle_index]
        assert len(position) == self.dimensions, "Argument 'position' must have the same length as the number of dimensions."
        cdef double fitness = self.objective_function(*position)
        return fitness


        

class Particle:
    def __init__(self):
        self.position = None
        self.velocity = None
        self.fitness = None
        self.pbest_position = None
        self.pbest_fitness = None

    def initialise_position(self, bounds):
        x_position = np.random.uniform(bounds[0][0], bounds[0][1])
        y_position = np.random.uniform(bounds[1][0], bounds[1][1])
        self.position = [x_position, y_position]

    def initialise_velocity(self):
        x_velocity = np.random.uniform(-1, 1)
        y_velocity = np.random.uniform(-1, 1)
        self.velocity = [x_velocity, y_velocity]

    def calculate_fitness(self, objective_function):
        self.fitness = objective_function(self.position[0], self.position[1])

    def update_velocity(self, w, c1, c2, gbest_position):
        for i in range(len(self.velocity)):
            r1 = np.random.uniform(-1, 1)
            r2 = np.random.uniform(-1, 1)

            velocity = self.velocity.copy()

            pbest = self.pbest_position.copy()
            position = self.position.copy()

            gbest_position = gbest_position.copy()

            cognitive_component = c1 * r1 * (pbest[i] - position[i])
            social_component = c2 * r2 * (gbest_position[i] - position[i])

            self.velocity[i] = w * velocity[i] + cognitive_component + social_component

    def update_position(self, bounds):
        for i in range(len(self.position)):
            self.position[i] = self.position[i] + self.velocity[i]
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]
            elif self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]

    def update_pbest(self):
        if self.fitness < self.pbest_fitness:
            self.pbest_fitness = self.fitness.copy()
            self.pbest_position = self.position.copy()


class ParticleSwarm:
    def __init__(self, objective_function, swarm_size, max_iterations, w, c1, c2, bounds):
        self.objective_function = objective_function
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds
        self.gbest_position = None
        self.gbest_fitness = 99999
        self.swarm = []

        self.plateau = 0

    def initialise_swarm(self):
        for i in range(self.swarm_size):
            particle = Particle()
            particle.initialise_position(self.bounds)
            particle.initialise_velocity()
            particle.calculate_fitness(self.objective_function)
            particle.pbest_position = particle.position
            particle.pbest_fitness = particle.fitness
            self.swarm.append(particle)

    def update_gbest(self):
        for particle in self.swarm:
            if particle.fitness < self.gbest_fitness:
                self.gbest_fitness = particle.fitness
                self.gbest_position = particle.position.copy()
                # reset plateau counter
                if self.gbest_fitness != self.objective_function(self.gbest_position[0], self.gbest_position[1]):
                    print(f"Something funky is going on: {self.gbest_fitness} != ")

                self.plateau = 0
        # if no improvement, increment plateau counter
        self.plateau += 1

    def check_termination(self):
        if self.plateau > 500:
            return True
        else:
            return False

    def run(self):
        # Do the initialisation steps
        self.initialise_swarm()
        self.update_gbest()

        # Run the iterations
        for i in range(self.max_iterations):

            for particle in self.swarm:
                particle.calculate_fitness(self.objective_function)
                particle.update_pbest()

            self.update_gbest()

            for particle in self.swarm:
                particle.update_velocity(self.w, self.c1, self.c2, self.gbest_position)
                particle.update_position(self.bounds)

            if self.gbest_fitness != self.objective_function(self.gbest_position[0], self.gbest_position[1]):
                print(f"Something funky is going on: {self.gbest_fitness} != {self.objective_function(self.gbest_position[0], self.gbest_position[1])}")

            if self.check_termination():
                print(f'Early termination at iteration {i}\n'
                      f'Best fitness: {self.gbest_fitness}\n'
                      f'Best position: {self.gbest_position}')
                break

            #print("Breakpoint")
        #print(f"Best fitness and position: {self.gbest_fitness}, {self.gbest_position}")
        return self.gbest_fitness, self.gbest_position


def particleswarm(objective_function, swarm_size, max_iterations, w, c1, c2, dimensions, bounds=None):
    pso = State(objective_function, swarm_size, max_iterations, w, c1, c2, dimensions, bounds)
    print(f"Passed the initialisation step")
    pso.print_class_variables()
    print(f"Initialising positions")
    pso.initialise_positions()
    pso.pbest_fitnesses[0] = pso.calculate_fitness(0)
    pso.pbest_fitnesses[1] = pso.calculate_fitness(1)
    pso.print_class_variables()
    print(f"Initialising velocities")
    pso.initialise_velocities()
    pso.print_class_variables()
