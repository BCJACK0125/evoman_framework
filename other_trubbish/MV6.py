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

experiment_name = "cmaes_optimization"
test_log_path = 'cmaes_optimization/test_logs/'
module_path = 'cmaes_optimization/models/'
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

# Initialize environment
def initialize_environment():
    return Environment(
        experiment_name=experiment_name,
        multiplemode="yes",
        playermode="ai",
        player_controller=player_controller_instance,
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False,
        enemies=[1, 2, 3, 4, 5, 6, 7, 8]
    )

# Run simulation and return fitness
def simulation(env, x):
    env.player_controller.set(x, n_inputs)  # Set controller parameters
    f, _, _, _ = env.play(pcont=x)
    return f

# Evaluate population
def evaluate(pop, env):
    return np.array([simulation(env, ind) for ind in pop])

# CMA-ES algorithm
def cma_es(env, x0, sigma, gens, lam):
    n = len(x0)
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mueff = np.sum(weights) ** 2 / np.sum(weights ** 2)

    # Strategy parameter setting: Adaptation
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

    # Initialize dynamic (internal) strategy parameters and constants
    pc = np.zeros(n)
    ps = np.zeros(n)
    B = np.eye(n)
    D = np.ones(n)
    C = B @ np.diag(D ** 2) @ B.T
    invsqrtC = B @ np.diag(D ** -1) @ B.T
    eigeneval = 0
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    # Generation loop
    for gen in range(gens):
        # Generate and evaluate lambda offspring
        arz = np.random.randn(lam, n)
        arx = x0 + sigma * arz @ B @ np.diag(D)
        arfitness = evaluate(arx, env)
        arindex = np.argsort(arfitness)
        xold = x0
        x0 = (arx[arindex[:mu]].T @ weights).T  # 确保维度匹配

        # Cumulation: Update evolution paths
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (x0 - xold) / sigma
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x0 - xold) / sigma

        # Adapt covariance matrix C
        artmp = (arx[arindex[:mu]] - xold) / sigma
        C = (1 - c1 - cmu) * C + c1 * (pc[:, None] @ pc[None, :]) + cmu * artmp.T @ np.diag(weights) @ artmp

        # Adapt step size sigma
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        # Update B and D from C
        if gen - eigeneval > lam / (c1 + cmu) / n / 10:
            eigeneval = gen
            C = np.triu(C) + np.triu(C, 1).T
            D, B = np.linalg.eigh(C)
            D = np.sqrt(D)
            invsqrtC = B @ np.diag(D ** -1) @ B.T

        # Record results
        best = np.max(arfitness)  # Best value
        mean = np.mean(arfitness)  # Mean value
        std = np.std(arfitness)  # Standard deviation

        result = f"{gen} {best:.6f} {mean:.6f} {std:.6f}"
        print(result)

    return x0

# Save final solution
def save_final_solution(
    x0, experiment_name, enemies, npop, gens, sigma
):
    best_solution = x0

    # Get current timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Filename uses the best fitness, population size, generations, enemy type, mutation rate, and elitism rate
    filename = (
        f"{experiment_name}/models/CMAESModel_npop{npop}_gens{gens}_"
        f"enemy{enemies}_sigma{sigma:.2f}_{current_time}.txt"
    )

    # Save the best individual's controller weights and biases
    np.savetxt(filename, best_solution)

    # Print information about the saved file
    print(f"Best solution saved as {filename}")

# Run evolution process
def run_evolution():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}/test_logs/CMAES_log_{current_time}.txt"

    with open(filename, "a") as file:
        env = initialize_environment()
        x0 = np.random.uniform(dom_l, dom_u, n_vars)
        sigma = 0.3  # Initial step size
        lam = 4 + int(3 * np.log(n_vars))  # Population size

        # Write header row
        file.write(
            f"npop{npop}_gens{gens}_"
            f"enemies{env.enemies}_sigma{sigma:.2f}_{current_time}\n"
        )
        file.write("gen best mean std\n")

        # Run CMA-ES
        best_solution = cma_es(env, x0, sigma, gens, lam)

        # Save the final best solution
        save_final_solution(
            best_solution, experiment_name, env.enemies, npop, gens, sigma
        )

def run_multiple_evolutions(num_runs=10):
    for i in range(num_runs):
        print(f"Running evolution {i+1}/{num_runs}")
        run_evolution()

# Genetic algorithm parameters
n_inputs = 20  # Number of inputs to the controller
n_vars = (
    n_inputs * 10 + 10 + 10 * 5 + 5
)  # Total number of weights and biases in the controller
dom_u = 1
dom_l = -1
npop = 100  # Population size
gens = 50  # Total generations

# 运行进化过程
run_evolution()