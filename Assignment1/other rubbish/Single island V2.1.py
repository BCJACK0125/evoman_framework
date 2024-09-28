import sys
import os
import numpy as np
from datetime import datetime

# 获取当前 .py 文件的路径
current_path = os.path.dirname(os.path.abspath(__file__))

# 将 evoman_framework 文件夹添加到 sys.path
sys.path.append(os.path.join(current_path, '..'))

# 现在可以导入 evoman 环境中的内容
from evoman.environment import Environment
from evoman.controller import Controller

# 设置无头模式，提升运行速度
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'single_island_optimization'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# 定义 Sigmoid 激活函数
def sigmoid_activation(x):
    return 1. / (1. + np.exp(-x))

# 自定义控制器，包含一个简单的神经网络结构
class player_controller(Controller):
    def __init__(self, _n_hidden):
        self.n_hidden = [_n_hidden]  # 隐藏层神经元数量

    def set(self, controller, n_inputs):
        if self.n_hidden[0] > 0:
            # 从控制器参数中提取权重和偏置，构建神经网络
            self.bias1 = controller[:self.n_hidden[0]].reshape(1, self.n_hidden[0])
            weights1_slice = n_inputs * self.n_hidden[0] + self.n_hidden[0]
            self.weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((n_inputs, self.n_hidden[0]))
            self.bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1, 5)
            self.weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0], 5))

    def control(self, inputs, controller):
        # 归一化输入
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))
        if self.n_hidden[0] > 0:
            output1 = sigmoid_activation(inputs.dot(self.weights1) + self.bias1)
            output = sigmoid_activation(output1.dot(self.weights2) + self.bias2)[0]
        else:
            bias = controller[:5].reshape(1, 5)
            weights = controller[5:].reshape((len(inputs), 5))
            output = sigmoid_activation(inputs.dot(weights) + bias)[0]

        # 根据输出决定玩家动作
        left = 1 if output[0] > 0.5 else 0
        right = 1 if output[1] > 0.5 else 0
        jump = 1 if output[2] > 0.5 else 0
        shoot = 1 if output[3] > 0.5 else 0
        release = 1 if output[4] > 0.5 else 0

        return [left, right, jump, shoot, release]

# 初始化控制器
player_controller_instance = player_controller(10)  # 10个隐藏神经元的示例

# 初始化环境
env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller_instance,
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# 遗传算法参数
n_inputs = env.get_num_sensors()  # 控制器输入数
n_vars = n_inputs * 10 + 10 + 10 * 5 + 5  # 控制器权重和偏置总数
dom_u = 1
dom_l = -1
npop = 100  # 种群大小
gens = 500   # 总代数
mutation = 0.3
elitism_rate = 0.1 # 精英率

# 初始化种群
# def initialize_population():
#     pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
#     fit_pop = evaluate(pop)
#     return pop, fit_pop

def initialize_population():
    # 初始化全为0的种群
    pop = np.zeros((npop, n_vars))  # 每个个体的控制器参数全为 0
    fit_pop = evaluate(pop)
    return pop, fit_pop

# 运行模拟，返回适应度
def simulation(env, x):
    env.player_controller.set(x, n_inputs)  # 设置控制器参数
    f, _, _, _ = env.play(pcont=x)
    return f

# 评估种群
def evaluate(x):
    return np.array([simulation(env, ind) for ind in x])

# 轮盘赌选择
def roulette_wheel_selection(pop, fit_pop):
    total_fitness = np.sum(fit_pop)
    pick = np.random.uniform(0, total_fitness)
    current = 0
    for i in range(pop.shape[0]):
        current += fit_pop[i]
        if current > pick:
            return pop[i]

# 均匀交叉
def uniform_crossover(p1, p2):
    offspring = np.zeros_like(p1)
    for i in range(len(p1)):
        if np.random.rand() < 0.5:
            offspring[i] = p1[i]
        else:
            offspring[i] = p2[i]
    return offspring

# 固定突变
def mutation_operation(offspring):
    for i in range(len(offspring)):
        if np.random.uniform(0, 1) <= mutation:
            offspring[i] += np.random.normal(0, 1)
    return offspring

# 精英选择
def elitism(pop, fit_pop, elite_size):
    elite_indices = np.argsort(fit_pop)[-elite_size:]
    elite_pop = pop[elite_indices]
    elite_fitness = fit_pop[elite_indices]
    return elite_pop, elite_fitness

# 交叉和突变操作
def crossover_and_mutation(pop):
    total_offspring = np.zeros((0, n_vars))
    for p in range(0, pop.shape[0], 2):
        p1 = pop[np.random.randint(0, pop.shape[0])]
        p2 = pop[np.random.randint(0, pop.shape[0])]

        # 均匀交叉
        offspring = uniform_crossover(p1, p2)

        # 固定突变操作
        offspring = mutation_operation(offspring)

        # 适应度限制
        offspring = np.clip(offspring, dom_l, dom_u)
        total_offspring = np.vstack((total_offspring, offspring))

    return total_offspring

# 遗传算法进化过程
def evolve_population(pop, fit_pop):
    # 精英选择
    elite_size = int(elitism_rate * npop)
    elite_pop, elite_fitness = elitism(pop, fit_pop, elite_size)

    # 交叉和突变
    offspring = crossover_and_mutation(pop)
    fit_offspring = evaluate(offspring)

    # 合并精英和后代
    pop = np.vstack((elite_pop, offspring[:npop - elite_size]))
    fit_pop = np.hstack((elite_fitness, fit_offspring[:npop - elite_size]))

    return pop, fit_pop

# 运行进化过程
def run_evolution():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{current_time}.txt"
    with open(filename, "a") as file:
        pop, fit_pop = initialize_population()

        for generation in range(gens):
            pop, fit_pop = evolve_population(pop, fit_pop)

            # 记录结果
            best = np.argmax(fit_pop)
            best_fitness = fit_pop[best]
            result = f"Generation {generation}: Best Fitness = {best_fitness}"
            print(result)
            file.write(result + "\n")

run_evolution()
