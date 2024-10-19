import sys
import os
import numpy as np
from datetime import datetime
import glob
from evoman.environment import Environment
from evoman.controller import Controller

# Set headless mode to improve running speed
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = "mpde_optimization"
test_log_path = 'mpde_optimization/test_logs/'
module_path = 'mpde_optimization/models/'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

if not os.path.exists(test_log_path):
    os.makedirs(test_log_path)

if not os.path.exists(module_path):
    os.makedirs(module_path)

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

# Initialize population
def initialize_islands(enemylist):
    islands = []
    for i in range(n_islands):
        env = Environment(
            experiment_name=experiment_name,
            multiplemode="yes",
            playermode="ai",
            player_controller=player_controller_instance,
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False,
            enemies=[enemylist[i]]
        )
        pop = np.random.uniform(dom_l, dom_u, (npop // n_islands, n_vars))
        fit_pop = evaluate(pop, env)
        islands.append((pop, fit_pop, env))
    return islands

def migrate(islands):
    num_migrants = int(migration_rate * (npop // n_islands))
    for i in range(n_islands):
        # Select migrants from island i
        migrants = islands[i][0][:num_migrants]
        # Migrate to the next island (circular migration)
        target_island = (i + 1) % n_islands
        islands[target_island][0][-num_migrants:] = migrants
    return islands

# Evolve each island's population independently
def evolve_islands(islands):
    for i in range(n_islands):
        pop, fit_pop, env = islands[i]
        mutant_pop = de_mutation(pop, F)
        trial_pop = de_crossover(pop, mutant_pop, CR)
        fit_trial_pop = evaluate(trial_pop, env)
        pop, fit_pop = de_selection(pop, fit_pop, trial_pop, fit_trial_pop)
        islands[i] = (pop, fit_pop, env)
    return islands

# Run simulation and return fitness
def simulation(env, x):
    env.player_controller.set(x, n_inputs)  # Set controller parameters
    f, _, _, _ = env.play(pcont=x)
    return f

def simulation_test(env, x):
    env.player_controller.set(x, n_inputs)  # Set controller parameters
    f, player_life, enemy_life, _ = env.play(pcont=x)  # Return player and enemy life
    return f, player_life, enemy_life

# Evaluate population
def evaluate(x, env):
    return np.array([simulation(env, ind) for ind in x])

# Differential Evolution mutation
def de_mutation(pop, F):
    mutant_pop = np.zeros_like(pop)
    for i in range(npop // n_islands):
        indices = np.random.choice(npop // n_islands, 3, replace=False)
        x1, x2, x3 = pop[indices]
        mutant_pop[i] = x1 + F * (x2 - x3)
    return mutant_pop

# Differential Evolution crossover
def de_crossover(pop, mutant_pop, CR):
    trial_pop = np.zeros_like(pop)
    for i in range(npop // n_islands):
        j_rand = np.random.randint(n_vars)
        for j in range(n_vars):
            if np.random.rand() < CR or j == j_rand:
                trial_pop[i, j] = mutant_pop[i, j]
            else:
                trial_pop[i, j] = pop[i, j]
    return trial_pop

# Differential Evolution selection
def de_selection(pop, fit_pop, trial_pop, fit_trial_pop):
    new_pop = np.zeros_like(pop)
    new_fit_pop = np.zeros_like(fit_pop)
    for i in range(npop // n_islands):
        if fit_trial_pop[i] > fit_pop[i]:
            new_pop[i] = trial_pop[i]
            new_fit_pop[i] = fit_trial_pop[i]
        else:
            new_pop[i] = pop[i]
            new_fit_pop[i] = fit_pop[i]
    return new_pop, new_fit_pop

# Save final solution
def save_final_solution(
    pop, fit_pop, experiment_name, enemies, npop, gens, mutation, elitism_rate
):
    best_index = np.argmax(fit_pop)
    best_solution = pop[best_index]
    best_fitness = fit_pop[best_index]

    # Get current timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Filename uses the best fitness
    filename = (
        f"{experiment_name}/models/MPDEModel_fitness{best_fitness:.6f}_npop{npop}_gens{gens}_"
        f"enemy{enemies}_mut{mutation:.2f}_elitism{elitism_rate:.2f}_{current_time}.txt"
    )

    # Save the best individual's controller weights and biases
    np.savetxt(filename, best_solution)

    # Print information about the saved file
    print(f"Best solution saved as {filename} with fitness: {best_fitness:.6f}")

# Run evolution process
def run_evolution(enemylist):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}/test_logs/MPDE_log_{current_time}.txt"

    with open(filename, "a") as file:
        # 初始化岛屿
        islands = initialize_islands(enemylist)

        # 记录每个岛屿的敌人信息
        enemies_info = [island[2].enemies for island in islands]
        file.write(
            f"npop{npop}_gens{gens}_"
            f"enemies{enemies_info}_mut{mutation:.2f}_elitism{elitism_rate:.2f}_{current_time}\n"
        )
        file.write("gen best mean std\n")

        for generation in range(gens):
            # 进化每个岛屿
            islands = evolve_islands(islands)

            # 每隔 migration_interval 代进行迁移
            if generation % migration_interval == 0 and generation > 0:
                islands = migrate(islands)

            # 记录结果（所有岛屿的最佳值、均值和标准差）
            all_fitness = np.hstack([fit_pop for _, fit_pop, _ in islands])
            best = np.max(all_fitness)
            mean = np.mean(all_fitness)
            std = np.std(all_fitness)

            result = f"{generation} {best:.6f} {mean:.6f} {std:.6f}"
            print(result)
            file.write(result + "\n")

        # 找到所有岛屿中的最佳解决方案
        all_pop = np.vstack([pop for pop, _, _ in islands])
        all_fit_pop = np.hstack([fit_pop for _, fit_pop, _ in islands])
        save_final_solution(all_pop, all_fit_pop, experiment_name, enemies_info, npop, gens, mutation, elitism_rate)

def run_multiple_evolutions(enemylist, num_runs=10):
    for i in range(num_runs):
        print(f"Running evolution {i+1}/{num_runs}")
        run_evolution(enemylist)

# Genetic algorithm parameters
n_inputs = 20  # Number of inputs to the controller
n_vars = (
    n_inputs * 10 + 10 + 10 * 5 + 5
)  # Total number of weights and biases in the controller
dom_u = 1
dom_l = -1
npop = 100  # Population size
gens = 50  # Total generations
F = 0.8  # Differential weight
CR = 0.9  # Crossover probability
mutation = 0.2  # Mutation rate
elitism_rate = 0.2  # Elitism rate
migration_interval = 5  # Every 5 generations
migration_rate = 0.1  # 10% of the population migrates
n_islands = 4

# 运行进化过程
enemylist = [1, 2, 3, 4,5,6,7,8]  # 每个岛屿的敌人列表
run_evolution(enemylist)