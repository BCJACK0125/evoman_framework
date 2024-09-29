import sys
import os
import numpy as np
from datetime import datetime
import glob


# Get the path of the current .py file
current_path = os.path.dirname(os.path.abspath(__file__))

# Add the evoman_framework folder to sys.path
sys.path.append(os.path.join(current_path, ".."))

# Now you can import contents from the evoman environment
from evoman.environment import Environment
from evoman.controller import Controller

# Set headless mode to improve running speed
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = "single_island_optimization"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# Define Sigmoid activation function
def sigmoid_activation(x):
    return 1.0 / (1.0 + np.exp(-x))


# Custom controller with a simple neural network structure
class player_controller(Controller):
    def __init__(self, _n_hidden):
        self.n_hidden = [_n_hidden]  # Number of neurons in the hidden layer

    def set(self, controller, n_inputs):
        if self.n_hidden[0] > 0:
            # Extract weights and biases from controller parameters to build the neural network
            self.bias1 = controller[: self.n_hidden[0]].reshape(1, self.n_hidden[0])
            weights1_slice = n_inputs * self.n_hidden[0] + self.n_hidden[0]
            self.weights1 = controller[self.n_hidden[0] : weights1_slice].reshape(
                (n_inputs, self.n_hidden[0])
            )
            self.bias2 = controller[weights1_slice : weights1_slice + 5].reshape(1, 5)
            self.weights2 = controller[weights1_slice + 5 :].reshape(
                (self.n_hidden[0], 5)
            )

    def control(self, inputs, controller):
        # Normalize inputs
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))
        if self.n_hidden[0] > 0:
            output1 = sigmoid_activation(inputs.dot(self.weights1) + self.bias1)
            output = sigmoid_activation(output1.dot(self.weights2) + self.bias2)[0]
        else:
            bias = controller[:5].reshape(1, 5)
            weights = controller[5:].reshape((len(inputs), 5))
            output = sigmoid_activation(inputs.dot(weights) + bias)[0]

        # Determine player actions based on output
        left = 1 if output[0] > 0.5 else 0
        right = 1 if output[1] > 0.5 else 0
        jump = 1 if output[2] > 0.5 else 0
        shoot = 1 if output[3] > 0.5 else 0
        release = 1 if output[4] > 0.5 else 0

        return [left, right, jump, shoot, release]


# Initialize controller
player_controller_instance = player_controller(10)  # Example with 10 hidden neurons

# Initialize environment
enemy = [3]  # Enemy number
env = Environment(
    experiment_name=experiment_name,
    enemies=enemy,
    playermode="ai",
    player_controller=player_controller_instance,
    enemymode="static",
    level=2,
    speed="fastest",
    randomini="yes",
    visuals=False,
)

# Genetic algorithm parameters
n_inputs = env.get_num_sensors()  # Number of inputs to the controller
n_vars = (
    n_inputs * 10 + 10 + 10 * 5 + 5
)  # Total number of weights and biases in the controller
dom_u = 1
dom_l = -1
npop = 100  # Population size
gens = 50  # Total generations
mutation = 0.6  # Mutation rate
elitism_rate = 0.1  # Elitism rate


# Initialize population
def initialize_population():
    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop)
    return pop, fit_pop


# Run simulation and return fitness
def simulation(env, x):
    env.player_controller.set(x, n_inputs)  # Set controller parameters
    f, _, _, _ = env.play(pcont=x)
    return f


def simulation_test(env, x):
    env.player_controller.set(x, n_inputs)  # Set controller parameters
    f, player_life, enemy_life, _ = env.play(pcont=x)  # Return player and enemy life
    return f, player_life, enemy_life


def load_best_solution(filepath):
    return np.loadtxt(filepath)


# Evaluate population
def evaluate(x):
    return np.array([simulation(env, ind) for ind in x])


# Roulette wheel selection
def roulette_wheel_selection(pop, fit_pop):
    total_fitness = np.sum(fit_pop)
    pick = np.random.uniform(0, total_fitness)
    current = 0
    for i in range(pop.shape[0]):
        current += fit_pop[i]
        if current > pick:
            return pop[i]


# Uniform crossover
def uniform_crossover(p1, p2):
    offspring = np.zeros_like(p1)
    for i in range(len(p1)):
        if np.random.rand() < 0.5:
            offspring[i] = p1[i]
        else:
            offspring[i] = p2[i]
    return offspring


# Fixed mutation
def mutation_operation(offspring):
    for i in range(len(offspring)):
        if np.random.uniform(0, 1) <= mutation:
            offspring[i] += np.random.normal(0, 1)
    return offspring


# Elitism selection
def elitism(pop, fit_pop, elite_size):
    elite_indices = np.argsort(fit_pop)[-elite_size:]
    elite_pop = pop[elite_indices]
    elite_fitness = fit_pop[elite_indices]
    return elite_pop, elite_fitness


# Crossover and mutation operations
def crossover_and_mutation(pop):
    total_offspring = np.zeros((0, n_vars))
    for p in range(0, pop.shape[0], 2):
        p1 = pop[np.random.randint(0, pop.shape[0])]
        p2 = pop[np.random.randint(0, pop.shape[0])]

        # Uniform crossover
        offspring = uniform_crossover(p1, p2)

        # Fixed mutation operation
        offspring = mutation_operation(offspring)

        # Fitness constraint
        offspring = np.clip(offspring, dom_l, dom_u)
        total_offspring = np.vstack((total_offspring, offspring))

    return total_offspring


# Save final solution
def save_final_solution(
    pop, fit_pop, experiment_name, enemies, npop, gens, mutation, elitism_rate
):
    best_index = np.argmax(fit_pop)
    best_solution = pop[best_index]
    best_fitness = fit_pop[best_index]

    # Get current timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Filename uses the best fitness, population size, generations, enemy type, mutation rate, and elitism rate
    filename = (
        f"{experiment_name}/SI_fitness{best_fitness:.6f}_npop{npop}_gens{gens}_"
        f"enemy{enemies[0]}_mut{mutation:.2f}_elitism{elitism_rate:.2f}_{current_time}.txt"
    )

    # Save the best individual's controller weights and biases
    np.savetxt(filename, best_solution)

    # Print information about the saved file
    print(f"Best solution saved as {filename} with fitness: {best_fitness:.6f}")


# Genetic algorithm evolution process
def evolve_population(pop, fit_pop):
    # Elitism selection
    elite_size = int(elitism_rate * npop)
    elite_pop, elite_fitness = elitism(pop, fit_pop, elite_size)

    # Crossover and mutation
    offspring = crossover_and_mutation(pop)
    fit_offspring = evaluate(offspring)

    # Combine elites and offspring
    pop = np.vstack((elite_pop, offspring[: npop - elite_size]))
    fit_pop = np.hstack((elite_fitness, fit_offspring[: npop - elite_size]))

    return pop, fit_pop


# Run evolution process
def run_evolution(enemy):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"single_results_{current_time}.txt"

    with open(filename, "a") as file:
        # Write header row
        file.write(
            f"npop{npop}_gens{gens}_"
            f"enemy{enemy[0]}_mut{mutation:.2f}_elitism{elitism_rate:.2f}_{current_time}\n"
        )
        file.write("gen best mean std\n")

        pop, fit_pop = initialize_population()

        for generation in range(gens):
            pop, fit_pop = evolve_population(pop, fit_pop)

            # Record results
            best = np.max(fit_pop)  # Best value
            mean = np.mean(fit_pop)  # Mean value
            std = np.std(fit_pop)  # Standard deviation

            result = f"{generation} {best:.6f} {mean:.6f} {std:.6f}"
            print(result)

            # Write to file
            file.write(result + "\n")

        # Save the final best solution
        save_final_solution(
            pop, fit_pop, experiment_name, enemy, npop, gens, mutation, elitism_rate
        )


def run_multiple_evolutions(enemies, num_runs=10):
    for i in range(num_runs):
        print(f"Running evolution {i+1}/{num_runs}")
        run_evolution(enemies)


# Call the new function
run_multiple_evolutions(enemies=enemy, num_runs=10)
# run_evolution()


##################################################################################
# If you want to load and test the saved model, comment out run_evolution(), and uncomment the following code, modify the filename
##################################################################################


def test_loaded_model():
    # Hardcoded model file path
    filepath = f"{experiment_name}/SI_fitness90.206611_npop100_gens5_enemy8_mut0.30_elitism0.10_20240928_032146.txt"

    fitness_scores = []
    gains = []  # Used to store individual_gain for each run

    # Load the saved model
    solution = load_best_solution(filepath)

    # Run 5 times
    for i in range(5):
        fitness, player_life, enemy_life = simulation_test(env, solution)
        fitness_scores.append(fitness)

        # Calculate individual_gain
        individual_gain = player_life - enemy_life
        gains.append(individual_gain)

        print(f"Run {i+1} fitness: {fitness}, individual_gain: {individual_gain}")

    # Calculate the average fitness
    average_fitness = np.mean(fitness_scores)
    average_gain = np.mean(gains)

    print(f"\nAverage fitness over 5 runs: {average_fitness}")
    print(f"Average individual gain over 5 runs: {average_gain}")


# # Run test
# test_loaded_model()


def evaluate_models(experiment_name, enemy):
    # Get all model files
    model_files = glob.glob(f"{experiment_name}/*.txt")

    # Filter model files for the same enemy
    enemy_model_files = [file for file in model_files if f"enemy{enemy}" in file]

    best_gain = -np.inf
    best_model = None
    best_model_gains = []

    for filepath in enemy_model_files:
        fitness_scores = []
        gains = []  # Used to store individual_gain for each run

        # Load the saved model
        solution = load_best_solution(filepath)

        # Run 5 times
        for i in range(5):
            fitness, player_life, enemy_life = simulation_test(env, solution)
            fitness_scores.append(fitness)

            # Calculate individual_gain
            individual_gain = player_life - enemy_life
            gains.append(individual_gain)

            print(
                f"Model: {filepath}, Run {i+1} fitness: {fitness}, individual_gain: {individual_gain}"
            )

        # Calculate the average fitness
        average_fitness = np.mean(fitness_scores)
        average_gain = np.mean(gains)

        print(f"\nModel: {filepath}, Average fitness over 5 runs: {average_fitness}")
        print(
            f"Model: {filepath}, Average individual gain over 5 runs: {average_gain}\n"
        )

        # Find the model with the best result
        if average_gain > best_gain:
            best_gain = average_gain
            best_model = filepath
            best_model_gains = gains

    print(f"Best model: {best_model} with average individual gain: {best_gain}")
    print(f"Best model's 5 individual gains: {best_model_gains}")


# Call the new function
evaluate_models(experiment_name, enemy=enemy[0])
