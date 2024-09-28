# 选择函数：锦标赛选择

# 从种群中随机选择 tournament_size 个体，选出适应度最高的个体作为下一代的一部分。
# 交叉函数：均匀交叉

# 对于每对父代，每个基因有 50% 的概率从父代 1 或父代 2 中继承。
# 突变函数：固定突变

# 每个基因有 30% 的概率发生突变，突变时基因值加入正态分布的随机扰动。


import sys
import os
import numpy as np

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

# 使用提供的随机控制器
player_controller = Controller()

# 初始化环境
env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller,  # 使用随机控制器
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# 遗传算法参数
n_vars = 5  # 当前控制器不使用神经网络，仅产生5个动作
dom_u = 1
dom_l = -1
npop = 100  # 种群大小
gens = 500   # 总代数
mutation = 0.3
elitism_rate = 0.1  # 精英率
tournament_size = 3  # 锦标赛的大小
dynamic_mutation = False  # 固定突变率

# 初始化种群
def initialize_population():
    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop)
    return pop, fit_pop

# 运行模拟，返回适应度
def simulation(env, x):
    f, _, _, _ = env.play(pcont=x)
    return f

# 评估种群
def evaluate(x):
    return np.array([simulation(env, ind) for ind in x])

# 锦标赛选择
def tournament_selection(pop, fit_pop, tournament_size):
    selected = []
    for _ in range(npop):
        # 随机挑选 tournament_size 个体
        tournament_indices = np.random.choice(np.arange(npop), tournament_size, replace=False)
        tournament_fitness = fit_pop[tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(pop[winner_index])
    return np.array(selected)

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

    # 锦标赛选择生成新种群
    selected_pop = tournament_selection(pop, fit_pop, tournament_size)

    # 交叉和突变
    offspring = crossover_and_mutation(selected_pop)
    fit_offspring = evaluate(offspring)

    # 合并精英和后代
    pop = np.vstack((elite_pop, offspring[:npop - elite_size]))
    fit_pop = np.hstack((elite_fitness, fit_offspring[:npop - elite_size]))

    return pop, fit_pop

# 运行进化过程
def run_evolution():
    with open("results_single.txt", "a") as file:
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
