import numpy as np
import GA as ga

# Inputs of the equation.
equation_inputs = [4, -2, 3.5, 5, -11, -4.7]

# Number of the weights we are looking to optimize.
num_weights = 6

# Defining the population size.
sol_per_pop = 8

# The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
pop_size = (sol_per_pop, num_weights)

# Creating the initial population.
new_pop = np.random.uniform(low=-4.0, high=4.0, size=pop_size)

num_gen = 5
num_parents_mating = 4
for gen in range(num_gen):
    # Measuring the fitness of each chromosome in the population.
    fitness = ga.cal_pop_fitness(equation_inputs, new_pop)
    # Selecting the best parents in the population for mating.
    parents = ga.select_mating_pool(new_pop, fitness, num_parents_mating)
    # Generating next generation using crossover.
    offspring_crossover = ga.crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], num_weights))
    # Adding some variations to the offspring using mutation.
    offspring_mutation = ga.mutation(offspring_crossover)
    # Creating the new population based on the parents and offspring.
    new_pop[0:parents.shape[0], :] = parents
    new_pop[parents.shape[0]:, :] = offspring_mutation
