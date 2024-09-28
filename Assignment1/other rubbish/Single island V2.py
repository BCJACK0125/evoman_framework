


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
gens = 50   # 总代数
mutation_rate = 0.3 

# 选择算法类型
selection_type = "roulette"  # 可选: "tournament", "elitism", "roulette"

# 交叉算法类型
crossover_type = "two_point"  # 可选: "single_point", "two_point", "uniform"

# 突变算法类型
mutation_type = "uniform"  # 可选: "uniform", "non_uniform", "multi_point"

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

# 选择函数 - 锦标赛选择
def tournament_selection(pop, fit_pop):
    c1 = np.random.randint(0, pop.shape[0])
    c2 = np.random.randint(0, pop.shape[0])
    return pop[c1] if fit_pop[c1] > fit_pop[c2] else pop[c2]

# 选择函数 - 精英选择
def elitism_selection(pop, fit_pop):
    elite_index = np.argmax(fit_pop)
    return pop[elite_index]

# 选择函数 - 轮盘赌选择
def roulette_selection(pop, fit_pop):
    fit_sum = np.sum(fit_pop)
    pick = np.random.uniform(0, fit_sum)
    current = 0
    for i in range(len(fit_pop)):
        current += fit_pop[i]
        if current > pick:
            return pop[i]

# 交叉操作
def crossover(p1, p2):
    if crossover_type == "single_point":
        cross_point = np.random.randint(1, n_vars)
        offspring = np.concatenate((p1[:cross_point], p2[cross_point:]))
    elif crossover_type == "two_point":
        point1, point2 = sorted(np.random.randint(1, n_vars, size=2))
        offspring = np.concatenate((p1[:point1], p2[point1:point2], p1[point2:]))
    elif crossover_type == "uniform":
        offspring = np.where(np.random.rand(n_vars) > 0.5, p1, p2)
    return offspring

# 突变操作
def mutate(offspring):
    if mutation_type == "uniform":
        for i in range(len(offspring)):
            if np.random.uniform(0, 1) <= mutation_rate:
                offspring[i] += np.random.normal(0, 1)
    elif mutation_type == "non_uniform":
        tau = 1.0 / (gens**0.5)
        for i in range(len(offspring)):
            if np.random.uniform(0, 1) <= mutation_rate:
                offspring[i] += np.random.normal(0, tau * (gens - i))
    elif mutation_type == "multi_point":
        mutation_points = np.random.randint(1, n_vars, size=2)
        for point in mutation_points:
            if np.random.uniform(0, 1) <= mutation_rate:
                offspring[point] += np.random.normal(0, 1)

    offspring = np.clip(offspring, dom_l, dom_u)
    return offspring

# 交叉和突变操作
def crossover_and_mutation(pop):
    total_offspring = np.zeros((0, n_vars))
    for p in range(0, pop.shape[0], 2):
        p1 = pop[np.random.randint(0, pop.shape[0])]
        p2 = pop[np.random.randint(0, pop.shape[0])]

        # 交叉
        offspring = crossover(p1, p2)

        # 突变
        offspring = mutate(offspring)

        total_offspring = np.vstack((total_offspring, offspring))

    return total_offspring

# 选择函数入口
def select(pop, fit_pop):
    if selection_type == "tournament":
        return tournament_selection(pop, fit_pop)
    elif selection_type == "elitism":
        return elitism_selection(pop, fit_pop)
    elif selection_type == "roulette":
        return roulette_selection(pop, fit_pop)

# 遗传算法进化过程
def evolve_population(pop, fit_pop):
    offspring = crossover_and_mutation(pop)
    fit_offspring = evaluate(offspring)
    pop = np.vstack((pop, offspring))
    fit_pop = np.append(fit_pop, fit_offspring)

    # 根据适应度排序，选择最好的个体进入下一代
    sorted_indices = np.argsort(fit_pop)[-npop:]
    pop = pop[sorted_indices]
    fit_pop = fit_pop[sorted_indices]

    return pop, fit_pop

# 运行进化过程
def run_evolution():
    with open("results_single.txt", "a") as file:
        pop, fit_pop = initialize_population()
        for generation in range(gens):
            pop, fit_pop = evolve_population(pop, fit_pop)

            # 记录结果
            best = np.argmax(fit_pop)
            result = f"Generation {generation}: Best Fitness = {fit_pop[best]}"
            print(result)
            file.write(result + "\n")

run_evolution()
