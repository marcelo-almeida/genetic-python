from numpy.random import randint
from numpy import argsort, unique


class Population:
    def __init__(self, genes_size: int, population_size: int, fitness_function):
        self.genes_size = genes_size
        self.population_size = population_size
        self.fitness_function = fitness_function
        self.population = None

    def generate_initial_population(self):
        self.population = randint(0, 2, size=(self.population_size, self.genes_size), dtype='b')

    def compute_fitness(self):
        unique_individuals, indexes = unique(self.population, return_inverse=True, axis=0)
        values = self.fitness_function(unique_individuals)
        values = values[indexes]
        individual = argsort(values)  # returns a list of indexes
        self.population[:] = self.population[individual]
        return values[individual]
