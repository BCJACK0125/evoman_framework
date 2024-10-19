import sys
import os
import numpy as np
from datetime import datetime
from evoman.environment import Environment
from evoman.controller import Controller

# 设置无头模式以提高运行速度
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = "de_multi_enemy_optimization"
test_log_path = 'de_multi_enemy_optimization/test_logs/'
module_path = 'de_multi_enemy_optimization/models/'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

if not os.path.exists(test_log_path):
    os.makedirs(test_log_path)

if not os.path.exists(module_path):
    os.makedirs(module_path)

# 定义Sigmoid激活函数
def sigmoid_activation(x):
    return 1.0 / (1.0 + np.exp(-x))

# 自定义控制器，具有简单的神经网络结构
class player_controller(Controller):
    def __init__(self, _n_hidden):
        self.n_hidden = [_n_hidden]  # 隐藏层神经元数量

    def set(self, controller, n_inputs):
        if self.n_hidden[0] > 0:
            # 从控制器参数中提取权重和偏置以构建神经网络
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
        # 归一化输入
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))
        if self.n_hidden[0] > 0:
            output1 = sigmoid_activation(inputs.dot(self.weights1) + self.bias1)
            output = sigmoid_activation(output1.dot(self.weights2) + self.bias2)[0]
        else:
            bias = controller[:5].reshape(1, 5)
            weights = controller[5:].reshape((len(inputs), 5))
            output = sigmoid_activation(inputs.dot(weights) + bias)[0]

        # 根据输出确定玩家动作
        left = 1 if output[0] > 0.5 else 0
        right = 1 if output[1] > 0.5 else 0
        jump = 1 if output[2] > 0.5 else 0
        shoot = 1 if output[3] > 0.5 else 0
        release = 1 if output[4] > 0.5 else 0

        return [left, right, jump, shoot, release]

# 初始化控制器
player_controller_instance = player_controller(10)  # 示例中使用10个隐藏神经元

# 初始化环境
env = Environment(
    experiment_name=experiment_name,
    multiplemode="yes",
    playermode="ai",
    player_controller=player_controller_instance,
    enemymode="static",
    level=2,
    speed="fastest",
    visuals=False,
)

# 运行模拟并返回适应度
def simulation(env, x):
    env.player_controller.set(x, n_inputs)  # 设置控制器参数
    f, _, _, _ = env.play(pcont=x)
    return f

# 评估种群
def evaluate(pop):
    return np.array([simulation(env, ind) for ind in pop])

# 初始化种群
def initialize_population():
    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop)
    return pop, fit_pop

# 差分进化变异
def de_mutation(pop, F):
    mutant_pop = np.zeros_like(pop)
    for i in range(npop):
        indices = np.random.choice(npop, 3, replace=False)
        x1, x2, x3 = pop[indices]
        mutant_pop[i] = x1 + F * (x2 - x3)
    return mutant_pop

# 差分进化交叉
def de_crossover(pop, mutant_pop, CR):
    trial_pop = np.zeros_like(pop)
    for i in range(npop):
        j_rand = np.random.randint(n_vars)
        for j in range(n_vars):
            if np.random.rand() < CR or j == j_rand:
                trial_pop[i, j] = mutant_pop[i, j]
            else:
                trial_pop[i, j] = pop[i, j]
    return trial_pop

# 差分进化选择
def de_selection(pop, fit_pop, trial_pop, fit_trial_pop):
    new_pop = np.zeros_like(pop)
    new_fit_pop = np.zeros_like(fit_pop)
    for i in range(npop):
        if fit_trial_pop[i] > fit_pop[i]:
            new_pop[i] = trial_pop[i]
            new_fit_pop[i] = fit_trial_pop[i]
        else:
            new_pop[i] = pop[i]
            new_fit_pop[i] = fit_pop[i]
    return new_pop, new_fit_pop

# 保存最终解决方案
def save_final_solution(
    pop, fit_pop, experiment_name, enemies, npop, gens, mutation, elitism_rate
):
    best_index = np.argmax(fit_pop)
    best_solution = pop[best_index]
    best_fitness = fit_pop[best_index]

    # 获取当前时间戳
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 文件名使用最佳适应度、种群大小、代数、敌人类型、变异率和精英率
    filename = (
        f"{experiment_name}/models/DEModel_fitness{best_fitness:.6f}_npop{npop}_gens{gens}_"
        f"enemy{enemies}_mut{mutation:.2f}_elitism{elitism_rate:.2f}_{current_time}.txt"
    )

    # 保存最佳个体的控制器权重和偏置
    np.savetxt(filename, best_solution)

    # 打印有关保存文件的信息
    print(f"Best solution saved as {filename} with fitness: {best_fitness:.6f}")

# 运行进化过程
def run_evolution():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}/test_logs/DE_log_{current_time}.txt"

    with open(filename, "a") as file:
        # 写入表头行
        file.write(
            f"npop{npop}_gens{gens}_"
            f"enemy{env.enemies}_mut{mutation:.2f}_elitism{elitism_rate:.2f}_{current_time}\n"
        )
        file.write("gen best mean std\n")

        pop, fit_pop = initialize_population()

        for generation in range(gens):
            # 差分进化变异和交叉
            mutant_pop = de_mutation(pop, F)
            trial_pop = de_crossover(pop, mutant_pop, CR)

            # 评估试验种群
            fit_trial_pop = evaluate(trial_pop)

            # 选择
            pop, fit_pop = de_selection(pop, fit_pop, trial_pop, fit_trial_pop)

            # 记录结果
            best = np.max(fit_pop)  # 最佳值
            mean = np.mean(fit_pop)  # 平均值
            std = np.std(fit_pop)  # 标准差

            result = f"{generation} {best:.6f} {mean:.6f} {std:.6f}"
            print(result)

            # 写入文件
            file.write(result + "\n")

        # 保存最终最佳解决方案
        save_final_solution(
            pop, fit_pop, experiment_name, env.enemies, npop, gens, mutation, elitism_rate
        )

# 加载最佳解决方案
def load_best_solution(filepath):
    return np.loadtxt(filepath)

# 模拟测试
def simulation_test(env, x):
    env.player_controller.set(x, n_inputs)  # 设置控制器参数
    f, player_life, enemy_life, _ = env.play(pcont=x)  # 返回玩家和敌人的生命值
    return f, player_life, enemy_life

# 遗传算法参数
n_inputs = 20  # 控制器的输入数量
n_vars = (
    n_inputs * 10 + 10 + 10 * 5 + 5
)  # 控制器中的权重和偏置总数
dom_u = 1
dom_l = -1
npop = 100  # 种群大小
gens = 500  # 总代数
F = 0.8  # 差分权重
CR = 0.9  # 交叉概率
mutation = 0.2  # 变异率
elitism_rate = 0.2  # 精英率

# 设置敌人
env.enemies = [1, 2, 3, 4, 5, 6, 7, 8]

# 运行进化过程
run_evolution()