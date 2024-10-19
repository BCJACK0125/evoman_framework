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

experiment_name = "pso_optimization"
test_log_path = 'pso_optimization/test_logs/'
module_path = 'pso_optimization/models/'
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

# Particle Swarm Optimization (PSO) algorithm
def pso(env, n_particles, n_vars, gens, w, c1, c2):
    # Initialize particle positions and velocities
    pop = np.random.uniform(dom_l, dom_u, (n_particles, n_vars))
    vel = np.random.uniform(-1, 1, (n_particles, n_vars))
    pbest = pop.copy()
    pbest_fitness = evaluate(pbest, env)
    gbest = pbest[np.argmax(pbest_fitness)]
    gbest_fitness = np.max(pbest_fitness)

    # Generation loop
    for gen in range(gens):
        # Update velocities and positions
        r1, r2 = np.random.rand(n_particles, n_vars), np.random.rand(n_particles, n_vars)
        vel = w * vel + c1 * r1 * (pbest - pop) + c2 * r2 * (gbest - pop)
        pop = pop + vel

        # Evaluate population
        fitness = evaluate(pop, env)

        # Update personal bests
        better_mask = fitness > pbest_fitness
        pbest[better_mask] = pop[better_mask]
        pbest_fitness[better_mask] = fitness[better_mask]

        # Update global best
        if np.max(fitness) > gbest_fitness:
            gbest = pop[np.argmax(fitness)]
            gbest_fitness = np.max(fitness)

        # Record results
        best = gbest_fitness  # Best value
        mean = np.mean(fitness)  # Mean value
        std = np.std(fitness)  # Standard deviation

        result = f"{gen} {best:.6f} {mean:.6f} {std:.6f}"
        print(result)

    return gbest

# Save final solution
def save_final_solution(
    gbest, experiment_name, enemies, npop, gens, w, c1, c2
):
    best_solution = gbest

    # Get current timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Filename uses the best fitness, population size, generations, enemy type, and PSO parameters
    filename = (
        f"{experiment_name}/models/PSOModel_npop{npop}_gens{gens}_"
        f"enemy{enemies}_w{w:.2f}_c1{c1:.2f}_c2{c2:.2f}_{current_time}.txt"
    )

    # Save the best individual's controller weights and biases
    np.savetxt(filename, best_solution)

    # Print information about the saved file
    print(f"Best solution saved as {filename}")

# Run evolution process
def run_evolution():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}/test_logs/PSO_log_{current_time}.txt"

    with open(filename, "a") as file:
        env = initialize_environment()
        n_particles = npop
        w = 0.5  # Inertia weight
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter

        # Write header row
        file.write(
            f"npop{npop}_gens{gens}_"
            f"enemies{env.enemies}_w{w:.2f}_c1{c1:.2f}_c2{c2:.2f}_{current_time}\n"
        )
        file.write("gen best mean std\n")

        # Run PSO
        best_solution = pso(env, n_particles, n_vars, gens, w, c1, c2)

        # Save the final best solution
        save_final_solution(
            best_solution, experiment_name, env.enemies, npop, gens, w, c1, c2
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