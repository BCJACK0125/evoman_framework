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

experiment_name = 'multi_island_optimization'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# 使用提供的随机控制器
player_controller = Controller()

# 初始化环境
env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller,
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# 遗传算法参数
n_vars = 5  # 控制器产生5个动作
dom_u = 1
dom_l = -1
npop = 200  # 总种群大小
gens = 500   # 总代数
mutation_rate = 0.4
n_islands = 4  # 岛屿数量
migration_rate = 0.1  # 迁移比例
migration_interval = 5  # 每隔多少代迁移

# 每个岛屿的种群大小
island_pop_size = npop // n_islands

# 选择算法类型
selection_type = "roulette"  # 可选: "tournament", "elitism", "roulette"

# 交叉算法类型
crossover_type = "uniform"  # 可选: "single_point", "two_point", "uniform"

# 突变算法类型
mutation_type = "multi_point"  # 可选: "uniform", "non_uniform", "multi_point"

# 初始化多个岛屿种群
def initialize_population():
    islands = []
    for _ in range(n_islands):
        pop = np.random.uniform(dom_l, dom_u, (island_pop_size, n_vars))
        fit_pop = evaluate(pop)
        islands.append((pop, fit_pop))
    return islands

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
    for _ in range(0, pop.shape[0], 2):
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

# 迁移操作
# def migrate(islands):
#     num_im = int(island_pop_size * migration_rate)
#     migrants_list = []

#     for i in range(n_islands):
#         pop, fit_pop = islands[i]
#         best_indices = np.argsort(fit_pop)[-num_im:]
#         migrants_list.append(pop[best_indices])
#         pop = np.delete(pop, best_indices, axis=0)
#         islands[i] = (pop, fit_pop)
    
#     for i in range(n_islands):
#         next_island = (i + 1) % n_islands
#         pop, fit_pop = islands[next_island]
#         pop = np.vstack((pop, migrants_list[i]))
#         islands[next_island] = (pop, fit_pop)

def migrate(islands):
    # 计算每个岛屿需要迁移的个体数
    num_im = int(island_pop_size * migration_rate)

    # 创建一个新列表来记录所有需要迁移的个体
    migrants_list = []

    # 第一步：从每个岛屿选出最优的 num_im 个体，按适应度排序，并删除这些个体
    for i in range(n_islands):
        pop, fit_pop = islands[i]

        # 根据适应度排序，选择最优的 num_im 个体
        best_indices = np.argsort(fit_pop)[-num_im:]  # 获取最优的 num_im 个体的索引
        best_individuals = pop[best_indices]
        best_fitnesses = fit_pop[best_indices]

        # 将这些个体记录到迁移列表中
        migrants_list.append((best_individuals, best_fitnesses))

        # 删除岛屿中这些最优个体
        pop = np.delete(pop, best_indices, axis=0)
        fit_pop = np.delete(fit_pop, best_indices)

        # 更新岛屿，删除后的个体成为新的种群
        islands[i] = (pop, fit_pop)

    # 第二步：将从每个岛屿选出的个体迁移到下一个岛屿
    for i in range(n_islands):
        next_island = (i + 1) % n_islands  # 下一个岛屿的索引

        # 获取从当前岛屿迁出的个体
        migrants, migrants_fit = migrants_list[i]

        # 将迁移的个体加入下一个岛屿
        pop, fit_pop = islands[next_island]

        # 将迁移个体和原有种群结合
        pop = np.vstack((pop, migrants))
        fit_pop = np.append(fit_pop, migrants_fit)

        # 更新下一个岛屿的种群
        islands[next_island] = (pop, fit_pop)

        
  


# 岛屿内的进化
def evolve_island(pop, fit_pop):
    offspring = crossover_and_mutation(pop)
    fit_offspring = evaluate(offspring)
    pop = np.vstack((pop, offspring))
    fit_pop = np.append(fit_pop, fit_offspring)
    best_indices = np.argsort(fit_pop)[-island_pop_size:]
    pop = pop[best_indices]
    fit_pop = fit_pop[best_indices]
    return pop, fit_pop

# 运行进化过程
def run_evolution():
    with open("results_multi.txt", "a") as file:
        islands = initialize_population()
        for generation in range(gens):
            for i in range(n_islands):
                pop, fit_pop = islands[i]
                islands[i] = evolve_island(pop, fit_pop)

            if generation % migration_interval == 0 and generation > 0:
                migrate(islands)

            for i in range(n_islands):
                pop, fit_pop = islands[i]
                best = np.argmax(fit_pop)
                result = f"Island {i} - Generation {generation}: Best Fitness = {fit_pop[best]}"
                print(result)
                file.write(result + "\n")

run_evolution()
