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

    num_vars = 30
    bits_per_var = 8
    chromosome_length = num_vars * bits_per_var
    min_vals = [-1.27] * num_vars
    max_vals = [1.28] * num_vars

    # Number of independent GA runs
    num_runs = 30  # Change as desired
    generations = 200
    population_size = 100
    crossover_rate = 0.8
    mutation_rate = 0.01
    selection_method = "chc"  # or "fitness_proportional"

    # --- Choose function and label for output naming ---
    function_name = "quartic"  # Options: "sphere", "rosenbrock", "step", "quartic"

    def fitness_fn(chromosome):
        real_vals = decode_chromosome(chromosome, num_vars, bits_per_var, min_vals, max_vals)
        if function_name == "sphere":
            return sphere_fitness(real_vals)
        elif function_name == "rosenbrock":
            return rosenbrock_fitness(real_vals)
        elif function_name == "step":
            return step_fitness(real_vals)
        elif function_name == "quartic":
            return quartic_fitness(real_vals)
        else:
            raise ValueError(f"Unknown function_name: {function_name}")

    def obj_fn(chromosome):
        real_vals = decode_chromosome(chromosome, num_vars, bits_per_var, min_vals, max_vals)
        if function_name == "sphere":
            return sum(x**2 for x in real_vals)
        elif function_name == "rosenbrock":
            return sum(100 * (real_vals[i+1] - real_vals[i]**2)**2 + (1 - real_vals[i])**2 for i in range(len(real_vals)-1))
        elif function_name == "step":
            return sum(np.floor(x) for x in real_vals)
        elif function_name == "quartic":
            return sum((i+1) * (x**4) for i, x in enumerate(real_vals)) + 0  # No noise for deterministic plotting
        else:
            raise ValueError(f"Unknown function_name: {function_name}")


    # Arrays to accumulate stats for averaging
    fitness_min_runs = []
    fitness_max_runs = []
    fitness_avg_runs = []
    obj_min_runs = []
    obj_max_runs = []
    obj_avg_runs = []

    best_overall = None
    best_fitness = -np.inf

    # For reporting
    best_decoded_list = []  # List of best decoded values per run
    best_obj_list = []      # List of best objective values per run
    best_gen_list = []      # Generation at which best solution was first found
    reliability_tol = 1e-4  # Tolerance for considering global optimum found
    global_optimum = 0    # For Sphere function
    reliability_count = 0

    for run in range(num_runs):
        np.random.seed(None)  # Use system time or entropy for different seed each run
        best, fitness_stats, obj_stats = simple_ga(
            fitness_fn,
            obj_fn,
            chromosome_length=chromosome_length,
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            selection_method=selection_method
        )
        fitness_min_runs.append(fitness_stats['min'])
        fitness_max_runs.append(fitness_stats['max'])
        fitness_avg_runs.append(fitness_stats['avg'])
        obj_min_runs.append(obj_stats['min'])
        obj_max_runs.append(obj_stats['max'])
        obj_avg_runs.append(obj_stats['avg'])

        # Track best overall individual
        fit = fitness_fn(best)
        if fit > best_fitness:
            best_fitness = fit
            best_overall = best

        # Track best decoded and objective for this run
        best_decoded = decode_chromosome(best, num_vars, bits_per_var, min_vals, max_vals)
        best_obj = obj_fn(best)
        best_decoded_list.append(best_decoded)
        best_obj_list.append(best_obj)

        # Reliability: did we reach the global optimum (within tol)?
        if abs(best_obj - global_optimum) <= reliability_tol:
            reliability_count += 1

        # Find the first generation where min objective reaches the best_obj (within tol)
        gen_found = None
        for gen, min_obj in enumerate(obj_stats['min']):
            if abs(min_obj - best_obj) <= reliability_tol:
                gen_found = gen
                break
        if gen_found is None:
            gen_found = generations  # Not found, set to max
        best_gen_list.append(gen_found)

    # Convert to numpy arrays for easy averaging
    fitness_min_runs = np.array(fitness_min_runs)
    fitness_max_runs = np.array(fitness_max_runs)
    fitness_avg_runs = np.array(fitness_avg_runs)
    obj_min_runs = np.array(obj_min_runs)
    obj_max_runs = np.array(obj_max_runs)
    obj_avg_runs = np.array(obj_avg_runs)

    # Compute mean across runs (axis=0 is generation)
    fitness_min_mean = np.mean(fitness_min_runs, axis=0)
    fitness_max_mean = np.mean(fitness_max_runs, axis=0)
    fitness_avg_mean = np.mean(fitness_avg_runs, axis=0)
    obj_min_mean = np.mean(obj_min_runs, axis=0)
    obj_max_mean = np.mean(obj_max_runs, axis=0)
    obj_avg_mean = np.mean(obj_avg_runs, axis=0)



    # Print and save stats to file
    best_decoded_overall = decode_chromosome(best_overall, num_vars, bits_per_var, min_vals, max_vals)
    chromo_str = "|".join(
        "".join(str(bit) for bit in best_overall[i*bits_per_var:(i+1)*bits_per_var])
        for i in range(num_vars)
    )
    avg_best_decoded = np.mean(np.array(best_decoded_list), axis=0)
    avg_best_obj = np.mean(best_obj_list)
    reliability = reliability_count / num_runs
    avg_gens_to_best = np.mean(best_gen_list)

    # Print to console
    print("Best decoded (overall):", best_decoded_overall)
    print("Best chromosome (overall):", chromo_str)
    print(f"Average best decoded values over {num_runs} runs: {avg_best_decoded}")
    print(f"Average best objective value over {num_runs} runs: {avg_best_obj}")
    print(f"Reliability (fraction of runs reaching optimum): {reliability:.2f}")
    print(f"Average number of generations to reach best solution: {avg_gens_to_best:.2f}")

    # Save stats and parameters to file
    import os
    stats_dir = "stats"
    plots_dir = "plots"
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    stats_filename = os.path.join(stats_dir, f"{function_name}_{selection_method}.txt")
    with open(stats_filename, "w") as f:
        f.write(f"Function: {function_name}\n")
        f.write(f"Selection method: {selection_method}\n")
        f.write(f"num_vars: {num_vars}\n")
        f.write(f"bits_per_var: {bits_per_var}\n")
        f.write(f"chromosome_length: {chromosome_length}\n")
        f.write(f"min_vals: {min_vals}\n")
        f.write(f"max_vals: {max_vals}\n")
        f.write(f"num_runs: {num_runs}\n")
        f.write(f"generations: {generations}\n")
        f.write(f"population_size: {population_size}\n")
        f.write(f"crossover_rate: {crossover_rate}\n")
        f.write(f"mutation_rate: {mutation_rate}\n")
        f.write(f"\nBest decoded (overall): {best_decoded_overall}\n")
        f.write(f"Best chromosome (overall): {chromo_str}\n")
        f.write(f"Average best decoded values over {num_runs} runs: {avg_best_decoded}\n")
        f.write(f"Average best objective value over {num_runs} runs: {avg_best_obj}\n")
        f.write(f"Reliability (fraction of runs reaching optimum): {reliability:.2f}\n")
        f.write(f"Average number of generations to reach best solution: {avg_gens_to_best:.2f}\n")

    # Plot and save the averaged min, max, and average fitness per generation
    plt.figure()
    plt.plot(fitness_max_mean, label="Avg Max Fitness")
    plt.plot(fitness_avg_mean, label="Avg Average Fitness")
    plt.plot(fitness_min_mean, label="Avg Min Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"{function_name.title()} ({selection_method})\nAveraged Fitness per Generation over {num_runs} runs")
    plt.legend()
    plt.tight_layout()
    fitness_plot_filename = os.path.join(plots_dir, f"{function_name}_{selection_method}_fitness.png")
    plt.savefig(fitness_plot_filename)
    plt.show()

    # Plot and save the averaged min, max, and average objective value per generation
    plt.figure()
    plt.plot(obj_max_mean, label="Avg Max Objective")
    plt.plot(obj_avg_mean, label="Avg Average Objective")
    plt.plot(obj_min_mean, label="Avg Min Objective")
    plt.xlabel("Generation")
    plt.ylabel("Objective Value")
    plt.title(f"{function_name.title()} ({selection_method})\nAveraged Objective Value per Generation over {num_runs} runs")
    plt.legend()
    plt.tight_layout()
    obj_plot_filename = os.path.join(plots_dir, f"{function_name}_{selection_method}_objective.png")
    plt.savefig(obj_plot_filename)
    plt.show()