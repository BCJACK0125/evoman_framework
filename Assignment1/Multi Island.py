import sys
import os

# 获取当前 .py 文件的路径
current_path = os.path.dirname(os.path.abspath(__file__))

# 将 evoman_framework 文件夹添加到 sys.path
sys.path.append(os.path.join(current_path, '..'))

# 现在可以导入 evoman 环境中的内容
from evoman.environment import Environment
from evoman.controller import Controller
import numpy as np


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
                  player_controller=player_controller,  # 使用随机控制器
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# 遗传算法参数
n_vars = 5  # 当前控制器不使用神经网络，仅产生5个动作
dom_u = 1
dom_l = -1
npop = 80  # 总种群大小
gens = 40   # 总代数
mutation = 0.3
n_islands = 4  # 岛屿数量
migration_rate = 0.1  # 每次迁移的比例
migration_interval = 5  # 每隔多少代进行一次迁移

# 每个岛屿的子种群大小
island_pop_size = npop // n_islands

# 初始化种群（岛屿）
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

# 选择函数（锦标赛选择）
def tournament(pop, fit_pop):
    c1 = np.random.randint(0, pop.shape[0])
    c2 = np.random.randint(0, pop.shape[0])
    return pop[c1] if fit_pop[c1] > fit_pop[c2] else pop[c2]

# 交叉和突变操作
def crossover_and_mutation(pop, mutation):
    total_offspring = np.zeros((0, n_vars))
    for p in range(0, pop.shape[0], 2):
        p1 = pop[np.random.randint(0, pop.shape[0])]
        p2 = pop[np.random.randint(0, pop.shape[0])]

        # 交叉
        cross_prop = np.random.uniform(0, 1)
        offspring = p1 * cross_prop + p2 * (1 - cross_prop)

        # 突变
        for i in range(len(offspring)):
            if np.random.uniform(0, 1) <= mutation:
                offspring[i] += np.random.normal(0, 1)

        # 适应度限制
        offspring = np.clip(offspring, dom_l, dom_u)
        total_offspring = np.vstack((total_offspring, offspring))

    return total_offspring

# 种群迁移函数
def migrate(islands):
    for i in range(n_islands):
        # 从岛屿i中随机选择部分个体，迁移到下一个岛屿
        next_island = (i + 1) % n_islands
        num_migrants = int(island_pop_size * migration_rate)
        migrants = islands[i][0][:num_migrants]

        # 将选出的个体替换到下一个岛屿
        islands[next_island][0][-num_migrants:] = migrants
        islands[next_island][1][-num_migrants:] = evaluate(migrants)

# 岛屿内的遗传算法
def evolve_island(pop, fit_pop):
    offspring = crossover_and_mutation(pop, mutation)
    fit_offspring = evaluate(offspring)
    pop = np.vstack((pop, offspring))
    fit_pop = np.append(fit_pop, fit_offspring)

    # 锦标赛选择下一代
    best = np.argmax(fit_pop)
    pop = pop[np.argsort(fit_pop)[-island_pop_size:]]  # 保留最优的子种群
    fit_pop = np.sort(fit_pop)[-island_pop_size:]

    return pop, fit_pop

# 运行进化过程
def run_evolution():
    islands = initialize_population()
    for generation in range(gens):
        # 每个岛屿独立进化
        for i in range(n_islands):
            pop, fit_pop = islands[i]
            islands[i] = evolve_island(pop, fit_pop)

        # 迁移操作
        if generation % migration_interval == 0 and generation > 0:
            migrate(islands)

        # 记录结果
        for i in range(n_islands):
            pop, fit_pop = islands[i]
            best = np.argmax(fit_pop)
            print(f"Island {i} - Generation {generation}: Best Fitness = {fit_pop[best]}")

run_evolution()