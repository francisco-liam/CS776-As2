import numpy as np
import random
from as2 import (
    sphere_fitness, rosenbrock_fitness, step_fitness, quartic_fitness
)

# Objective functions matching as2.py
def sphere_obj(x):
    return sum(val**2 for val in x)

def rosenbrock_obj(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))

def step_obj(x):
    return sum(np.floor(val) for val in x)

def quartic_obj(x):
    return sum((i+1) * (val**4) for i, val in enumerate(x))


def set_random_seed(seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

def generate_initial_solution(num_vars, min_vals, max_vals):
    return np.random.uniform(min_vals, max_vals, num_vars)

def generate_neighbor_solution(current_solution, step_size, min_vals, max_vals):
    perturbation = np.random.uniform(-step_size, step_size, len(current_solution))
    neighbor = current_solution + perturbation
    neighbor = np.clip(neighbor, min_vals, max_vals)
    return neighbor

def run_hill_climber(num_vars, min_vals, max_vals, step_size, max_no_improve, obj_function, precision=None):
    current_solution = generate_initial_solution(num_vars, min_vals, max_vals)
    if precision is not None:
        current_solution = np.round(current_solution, decimals=precision)
    current_obj = obj_function(current_solution)
    best_objs = [current_obj]
    no_improve_counter = 0
    num_iterations = 0

    # Progress reporting setup
    report_every = max(1, min(100, max_no_improve // 10))

    while no_improve_counter < max_no_improve:
        neighbor = generate_neighbor_solution(current_solution, step_size, min_vals, max_vals)
        if precision is not None:
            neighbor = np.round(neighbor, decimals=precision)
        neighbor_obj = obj_function(neighbor)
        num_iterations += 1

        if neighbor_obj <= current_obj:  # minimize objective
            if neighbor_obj < current_obj:
                no_improve_counter = 0
            current_solution = neighbor
            current_obj = neighbor_obj
        else:
            no_improve_counter += 1

        best_objs.append(current_obj)

        # Print progress every report_every iterations
        if num_iterations % report_every == 0:
            print(f"  Iter {num_iterations}: best objective so far = {current_obj:.6f} (no improvement for {no_improve_counter} steps)")

    return current_solution, current_obj, best_objs, num_iterations

if __name__ == "__main__":
    # Parameters
    num_vars = 30
    min_vals = np.array([-1.27] * num_vars)
    max_vals = np.array([1.28] * num_vars)
    step_size = 0.05
    max_no_improve = 100
    num_experiments = 30
    precision = 2
    random_seed = None

    set_random_seed(random_seed)

    # Choose objective function and name
    # Options: sphere_obj, rosenbrock_obj, step_obj, quartic_obj
    obj_function = quartic_obj  # or rosenbrock_obj, step_obj, quartic_obj
    function_name = "quartic"  # Change as needed to match obj_function

    best_solutions = []
    best_objs = []
    iterations_list = []
    best_objs_per_iter = []  # List of lists

    max_iters = 0
    for exp in range(num_experiments):
        print(f"Experiment {exp+1}/{num_experiments}...")
        sol, val, vals, iters = run_hill_climber(
            num_vars, min_vals, max_vals, step_size, max_no_improve, obj_function, precision
        )
        best_solutions.append(sol)
        best_objs.append(val)
        iterations_list.append(iters)
        best_objs_per_iter.append(vals)
        if len(vals) > max_iters:
            max_iters = len(vals)
        print(f"  Finished experiment {exp+1}: best objective = {val:.6f}, iterations = {iters}")

    # Pad all best_objs_per_iter to max_iters with their last value
    for i in range(len(best_objs_per_iter)):
        last_val = best_objs_per_iter[i][-1]
        if len(best_objs_per_iter[i]) < max_iters:
            best_objs_per_iter[i] += [last_val] * (max_iters - len(best_objs_per_iter[i]))

    avg_best_per_iter = np.mean(np.array(best_objs_per_iter), axis=0)

    # --- Save plot ---
    import os
    import matplotlib.pyplot as plt
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = os.path.join(plots_dir, f"{function_name}_hillclimber.png")

    plt.figure()
    plt.plot(avg_best_per_iter)
    plt.xlabel('Iteration')
    plt.ylabel('Average Best Objective Value')
    plt.title(f'Average Best Objective Value per Iteration ({function_name} Hill Climber)')
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

    # --- Save stats file ---
    stats_dir = "stats"
    os.makedirs(stats_dir, exist_ok=True)
    stats_filename = os.path.join(stats_dir, f"{function_name}_hillclimber_stats.txt")
    with open(stats_filename, "w") as f:
        f.write(f"Function: {function_name}\n")
        f.write("Parameters Used:\n")
        f.write(f"  num_vars: {num_vars}\n")
        f.write(f"  min_vals: {min_vals.tolist()}\n")
        f.write(f"  max_vals: {max_vals.tolist()}\n")
        f.write(f"  step_size: {step_size}\n")
        f.write(f"  max_no_improve: {max_no_improve}\n")
        f.write(f"  num_experiments: {num_experiments}\n")
        f.write(f"  precision: {precision}\n")
        f.write(f"\nAverage best solution: {np.mean(best_solutions, axis=0)}\n")
        f.write(f"Average best objective value: {np.mean(best_objs)}\n")
        f.write(f"Std dev of best objective values: {np.std(best_objs)}\n")
        f.write(f"Average number of iterations: {np.mean(iterations_list)}\n")

    # --- Print summary stats ---
    print("Average best solution:", np.mean(best_solutions, axis=0))
    print("Average best objective value:", np.mean(best_objs))
    print("Std dev of best objective values:", np.std(best_objs))
    print("Average number of iterations:", np.mean(iterations_list))
