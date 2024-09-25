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
gens = 30   # 总代数
mutation = 0.2

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

# 遗传算法进化过程
def evolve_population(pop, fit_pop):
    offspring = crossover_and_mutation(pop, mutation)
    fit_offspring = evaluate(offspring)
    pop = np.vstack((pop, offspring))
    fit_pop = np.append(fit_pop, fit_offspring)

    # 锦标赛选择下一代
    best = np.argmax(fit_pop)
    pop = pop[np.argsort(fit_pop)[-npop:]]  # 保留最优的种群
    fit_pop = np.sort(fit_pop)[-npop:]

    return pop, fit_pop

# 运行进化过程
def run_evolution():
    pop, fit_pop = initialize_population()
    for generation in range(gens):
        pop, fit_pop = evolve_population(pop, fit_pop)

        # 记录结果
        best = np.argmax(fit_pop)
        print(f"Generation {generation}: Best Fitness = {fit_pop[best]}")

run_evolution()
