import numpy as np
np.random.seed(2)

# inputs of the equation
equation_inputs = [4, -2, 3.5, 5, -11, -4.7]
# number of the weights we are looking to optimize
num_weights = len(equation_inputs)  # 6

sol_per_pop = 8

# defining the population size
pop_size = (sol_per_pop, num_weights)   # the population will have sol_per_pop chromosome where each chromosome has num_weights genes

# creating the initial population
new_population = np.random.uniform(low=-4.0, high=4.0, size=pop_size)
print('population: {}'.format(new_population))

def cal_pop_fitness(equation_inputs, pop):
    # calculating the fitness value of each solution in the current population
    # the fitness function calculates the sum of products between each inputs and its corresponding weight
    fitness = np.sum(pop * equation_inputs, axis=1)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # select the best individuals in the current generation as parents for producing the offspring of the next generation
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)

    # the point at which the crossover takes place between two parents
    # usually, it is at the center
    crossover_point = np.uint8(offspring_size[1] / 2)

    for k in range(offspring_size[0]):
        # index of the first parent to mate
        parent1_idx = k % parents.shape[0]
        # index of the second parent to mate
        parent2_idx = (k + 1) % parents.shape[0]

        # The new offspring will have its first half of its genes taken from the first parent
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        
    return offspring

def mutation(offspring_crossover):
    # mutation changes a single gene in each offspring randomly
    for idx in range(offspring_crossover.shape[0]):
        # the random value to be added to the gene
        random_value = np.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 4] += random_value
    return offspring_crossover

num_generations = 5
num_parents_mating = 4

for generation in range(num_generations):
    # measuring the fitness of each chromosome in the population
    fitness = cal_pop_fitness(equation_inputs, new_population)
    print('fitness = {}'.format(fitness))

    # selecting the best parents in the population for mating
    parents = select_mating_pool(new_population, fitness, num_parents_mating)
    print(f'parent shape: {parents}')

    # generating the next generation using crossover
    offspring_crossover = crossover(parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights))
    print('The offspring: {}'.format(offspring_crossover))

    # adding some variations to the offspring using mutation
    offspring_mutation = mutation(offspring_crossover)
    print('The mutation: {}'.format(offspring_mutation))

    # creating the new population based on the parents and offspring
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
    print(new_population)