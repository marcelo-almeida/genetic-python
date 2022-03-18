import matplotlib.pyplot as plt
from numpy import exp, array
from numpy import mgrid

from population import Population


def fitness_function(x, y):
    return 3 * exp(-(y + 1) ** 2 - x ** 2) * (x - 1) ** 2 - (exp(-(x + 1) ** 2 - y ** 2) / 3) + exp(
        -x ** 2 - y ** 2) * (10 * x ** 3 - 2 * x + 10 * y ** 5)


# I = 2^0b[0] + 2^1b[1] + ... + 2^ib[i]
def convert_bin(x):
    cnt = array([2 ** i for i in range(x.shape[1])])
    return array([(cnt * x[i, :]).sum() for i in range(x.shape[0])])


def get_chromosome(population):
    columns = population.shape[1]
    middle = columns / 2
    max_bin = 2.0 ** middle - 1.0
    min_number = -3
    max_number = 3
    const = (max_number - min_number) / max_bin
    x = min_number + const * convert_bin(x=population[:, :int(middle)])
    y = min_number + const * convert_bin(x=population[:, int(middle):])
    return x, y


def compute(population):
    x, y = get_chromosome(population=population)
    # return - fitness_function(x=x, y=y) to calculate the min instead of max
    return fitness_function(x=x, y=y)


def plot_result(x, y):
    figure = plt.figure(figsize=(100, 100))
    axes = figure.add_subplot(111, projection='3d')
    plot_x, plot_y = mgrid[-3:3:30j, -3:3:30j]
    plot_z = fitness_function(plot_x, plot_y)
    axes.plot_wireframe(plot_x, plot_y, plot_z)
    axes.scatter(x, y, fitness_function(x, y), s=50, c='red', marker='D')
    plt.show()


def main():
    genes_size = 8
    population_size = 5
    population = Population(genes_size=genes_size, population_size=population_size, fitness_function=compute)
    population.generate_initial_population()
    population.compute_fitness()
    x, y = get_chromosome(population.population)
    plot_result(x=x, y=y)


if __name__ == '__main__':
    main()
