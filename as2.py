import matplotlib.pyplot as plt
import numpy as np

def decode_chromosome(chromosome, num_vars, bits_per_var, min_vals, max_vals):
    """
    Decodes a binary chromosome into real values for each variable.

    Args:
        chromosome: 1D numpy array of 0s and 1s
        num_vars: number of variables
        bits_per_var: number of bits per variable
        min_vals: list or array of minimum values for each variable
        max_vals: list or array of maximum values for each variable

    Returns:
        List of decoded real values (one per variable)
    """
    decoded = []
    for i in range(num_vars):
        start = i * bits_per_var
        end = start + bits_per_var
        gene_bits = chromosome[start:end]
        n = bits_per_var
        val = 0
        for j, bit in enumerate(gene_bits[::-1]):
            val += bit * (2 ** j)
        precision = (max_vals[i] - min_vals[i]) / (2**n - 1)
        decoded_val = min_vals[i] + val * precision
        decoded.append(decoded_val)
    return decoded

def fitness_proportional_selection(population, fitness, population_size):
    """
    Performs fitness-proportional (roulette wheel) selection.

    Args:
        population: 2D numpy array of individuals
        fitness: 1D numpy array of fitness values
        population_size: number of individuals to select

    Returns:
        Selected population (2D numpy array)
    """
    total_fitness = np.sum(fitness)
    if total_fitness == 0:
        probs = np.ones(population_size) / population_size
    else:
        probs = fitness / total_fitness
    selected_indices = np.random.choice(population_size, size=population_size, p=probs)
    return population[selected_indices]

def one_point_crossover(parent1, parent2, crossover_rate):
    """
    Performs one-point crossover between two parents.

    Args:
        parent1: 1D numpy array
        parent2: 1D numpy array
        crossover_rate: probability of crossover

    Returns:
        Two offspring (tuple of 1D numpy arrays)
    """
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
    else:
        child1, child2 = parent1.copy(), parent2.copy()
    return child1, child2

def bit_flip_mutation(population, mutation_rate):
    """
    Performs bit-flip mutation on a population.

    Args:
        population: 2D numpy array of individuals
        mutation_rate: probability of flipping each bit

    Returns:
        Mutated population (2D numpy array)
    """
    mutation_mask = np.random.rand(*population.shape) < mutation_rate
    return np.logical_xor(population, mutation_mask).astype(int)

def simple_ga(
    fitness_fn,
    obj_fn,
    chromosome_length=100,
    population_size=50,
    generations=100,
    crossover_rate=0.7,
    mutation_rate=0.001,
    selection_method="fitness_proportional"
):
    """
    Runs a simple genetic algorithm with fitness-proportional selection,
    one-point crossover, and bit-flip mutation.

    Args:
        fitness_fn: function to evaluate fitness of an individual
        obj_fn: function to evaluate objective value of an individual
        chromosome_length: length of each chromosome (int)
        population_size: number of individuals in the population (int)
        generations: number of generations to run (int)
        crossover_rate: probability of crossover (float)
        mutation_rate: probability of bit flip (float)

    Returns:
        best individual (1D numpy array),
        dicts of min, max, avg for fitness and objective per generation
    """
    # Initialize population randomly
    population = np.random.randint(0, 2, (population_size, chromosome_length))
    avg_fitness_per_gen = []
    min_fitness_per_gen = []
    max_fitness_per_gen = []
    min_obj_per_gen = []
    max_obj_per_gen = []
    avg_obj_per_gen = []

    for gen in range(generations):
        # Evaluate fitness and objective
        fitness = np.array([fitness_fn(ind) for ind in population])
        obj = np.array([obj_fn(ind) for ind in population])
        avg_fitness_per_gen.append(np.mean(fitness))
        min_fitness_per_gen.append(np.min(fitness))
        max_fitness_per_gen.append(np.max(fitness))
        min_obj_per_gen.append(np.min(obj))
        max_obj_per_gen.append(np.max(obj))
        avg_obj_per_gen.append(np.mean(obj))

        # Selection (always fitness proportional for generating children)
        selected = fitness_proportional_selection(population, fitness, population_size)

        # Crossover
        children = []
        for i in range(0, population_size, 2):
            parent1 = selected[i]
            parent2 = selected[(i+1) % population_size]
            child1, child2 = one_point_crossover(parent1, parent2, crossover_rate)
            children.extend([child1, child2])
        children = np.array(children[:population_size])

        # Mutation
        children = bit_flip_mutation(children, mutation_rate)

        if selection_method == "fitness_proportional":
            # Standard GA: next generation is just the children
            population = children
        elif selection_method == "chc":
            # CHC/elitist: combine parents and children, select best N
            combined = np.vstack([population, children])
            combined_fitness = np.array([fitness_fn(ind) for ind in combined])
            top_indices = np.argsort(combined_fitness)[-population_size:][::-1]
            population = combined[top_indices]
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")

    # Evaluate fitness and objective
    fitness = np.array([fitness_fn(ind) for ind in population])
    obj = np.array([obj_fn(ind) for ind in population])
    avg_fitness_per_gen.append(np.mean(fitness))
    min_fitness_per_gen.append(np.min(fitness))
    max_fitness_per_gen.append(np.max(fitness))
    min_obj_per_gen.append(np.min(obj))
    max_obj_per_gen.append(np.max(obj))
    avg_obj_per_gen.append(np.mean(obj))

    # Return best solution and fitness/objective histories
    best_idx = np.argmax([fitness_fn(ind) for ind in population])
    fitness_stats = {
        'min': min_fitness_per_gen,
        'max': max_fitness_per_gen,
        'avg': avg_fitness_per_gen
    }
    obj_stats = {
        'min': min_obj_per_gen,
        'max': max_obj_per_gen,
        'avg': avg_obj_per_gen
    }
    return population[best_idx], fitness_stats, obj_stats

# Example usage for multi-variable decoding:
def sphere_fitness(decoded_vals):
    """
    Sphere function fitness: 1 / (sum of squares + 1)
    Args:
        decoded_vals: list of real values (decoded chromosome)
    Returns:
        fitness value (float)
    """
    val = sum(x**2 for x in decoded_vals)
    return 1.0 / (val + 1)

def rosenbrock_fitness(decoded_vals):
    """
    Rosenbrock's function (De Jong F2), normalized to [0,1]:
    f(x) = 1 / (sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2] + 1)
    Global minimum at x_i = 1 for all i, f(x*) = 1
    """
    val = sum(100 * (decoded_vals[i+1] - decoded_vals[i]**2)**2 + (1 - decoded_vals[i])**2
              for i in range(len(decoded_vals) - 1))
    return 1.0 / (val + 1)

def step_fitness(decoded_vals):
    """
    Step function (De Jong F3), normalized so that the minimum possible value (-n) gives fitness 1.
    f(x) = sum_{i=1}^n floor(x_i)
    fitness = 1 / (val - min_val + 1)
    For 30 variables in [-5.12, 5.12], min_val = -30.
    """
    val = sum(np.floor(x) for x in decoded_vals)
    n = len(decoded_vals)
    min_val = -6 * n
    return 1.0 / (val - min_val + 1)

def quartic_fitness(decoded_vals):
    """
    Noisy Quartic function (De Jong F4), normalized to [0,1]:
    f(x) = 1 / (sum_{i=1}^n i * x_i^4 + noise + 1)
    Global minimum at x_i = 0 for all i, f(x*) = 1/(noise+1)
    """
    noise = np.random.rand()
    val = sum((i+1) * (x**4) for i, x in enumerate(decoded_vals)) + noise
    return 1.0 / (val + 1)

if __name__ == "__main__":
    num_vars = 3
    bits_per_var = 10
    chromosome_length = num_vars * bits_per_var
    min_vals = [-5.11] * num_vars
    max_vals = [5.12] * num_vars


    # Choose which function to use by uncommenting the desired line below:
    def fitness_fn(chromosome):
        real_vals = decode_chromosome(chromosome, num_vars, bits_per_var, min_vals, max_vals)
        return sphere_fitness(real_vals)      # De Jong F1 (Sphere)
        # return rosenbrock_fitness(real_vals)  # De Jong F2 (Rosenbrock)
        # return step_fitness(real_vals)        # De Jong F3 (Step)
        # return quartic_fitness(real_vals)     # De Jong F4 (Noisy Quartic)

    def obj_fn(chromosome):
        real_vals = decode_chromosome(chromosome, num_vars, bits_per_var, min_vals, max_vals)
        return sum(x**2 for x in real_vals)      # Sphere
        # return sum(100 * (real_vals[i+1] - real_vals[i]**2)**2 + (1 - real_vals[i])**2 for i in range(len(real_vals)-1))  # Rosenbrock
        # return sum(np.floor(x) for x in real_vals)  # Step
        # return sum((i+1) * (x**4) for i, x in enumerate(real_vals)) + 0  # Quartic (no noise for deterministic plotting)

    # Choose selection method: "fitness_proportional" or "chc"
    selection_method = "chc"  # or "chc"

    best, fitness_stats, obj_stats = simple_ga(
        fitness_fn,
        obj_fn,
        chromosome_length=chromosome_length,
        population_size=100,
        generations=200,
        crossover_rate=0.7,
        mutation_rate=0.001,
        selection_method=selection_method
    )


    # Print best decoded values
    print("Best decoded:", decode_chromosome(best, num_vars, bits_per_var, min_vals, max_vals))

    # Print best individual's chromosome, separating variables with '|'
    # Each variable is bits_per_var long
    chromo_str = "|".join(
        "".join(str(bit) for bit in best[i*bits_per_var:(i+1)*bits_per_var])
        for i in range(num_vars)
    )
    print("Best chromosome:", chromo_str)

    # Plot and save the min, max, and average fitness per generation
    plt.figure()
    plt.plot(fitness_stats['max'], label="Max Fitness")
    plt.plot(fitness_stats['avg'], label="Average Fitness")
    plt.plot(fitness_stats['min'], label="Min Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness per Generation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fitness_per_generation.png")
    plt.show()

    # Plot and save the min, max, and average objective value per generation
    plt.figure()
    plt.plot(obj_stats['max'], label="Max Objective")
    plt.plot(obj_stats['avg'], label="Average Objective")
    plt.plot(obj_stats['min'], label="Min Objective")
    plt.xlabel("Generation")
    plt.ylabel("Objective Value")
    plt.title("Objective Value per Generation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("objective_per_generation.png")
    plt.show()