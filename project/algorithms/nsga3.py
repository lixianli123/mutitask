from deap import base, creator, tools, algorithms
import random
from case_config import *
from constraints import evaluate

def run_nsga3():
    # 确保 Creator 存在 (与 NSGA2 共享定义)
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    
    # 复制 NSGA2 的个体生成逻辑
    def random_ind():
        ind = []
        for r in VAR_RANGES_GEO: ind.append(random.uniform(r[0], r[1]))
        ind.append(random.uniform(0, len(VAL_FC)-0.01))
        ind.append(random.uniform(0, len(VAL_FY)-0.01))
        ind.append(random.uniform(*VAR_RANGES_MAT[0]))
        ind.append(random.uniform(*VAR_RANGES_MAT[1]))
        ind.append(random.uniform(0, len(VAL_NPB)-0.01))
        ind.append(random.uniform(0, len(VAL_NPW)-0.01))
        ind.append(random.uniform(*VAR_RANGES_MAT[2]))
        ind.append(random.uniform(*VAR_RANGES_MAT[3]))
        return creator.Individual(ind)

    # 构造边界列表
    LOW = [r[0] for r in VAR_RANGES_GEO] + \
          [0, 0] + \
          [VAR_RANGES_MAT[0][0], VAR_RANGES_MAT[1][0]] + \
          [0, 0] + \
          [VAR_RANGES_MAT[2][0], VAR_RANGES_MAT[3][0]]
          
    UP = [r[1] for r in VAR_RANGES_GEO] + \
         [len(VAL_FC)-0.01, len(VAL_FY)-0.01] + \
         [VAR_RANGES_MAT[0][1], VAR_RANGES_MAT[1][1]] + \
         [len(VAL_NPB)-0.01, len(VAL_NPW)-0.01] + \
         [VAR_RANGES_MAT[2][1], VAR_RANGES_MAT[3][1]]

    toolbox.register("individual", random_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=30.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=20.0, indpb=1.0/NDIM)
    
    # 生成参考点 (Das-Dennis)
    ref_points = tools.uniform_reference_points(nobj=3, p=NSGA3_P)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
    
    pop = toolbox.population(n=NSGA3_POP)
    
    # 初始评估
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    for ind, fit in zip(invalid_ind, map(toolbox.evaluate, invalid_ind)):
        ind.fitness.values = fit
        
    for gen in range(NSGA3_GEN):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.9, mutpb=0.1)
        for ind, fit in zip(offspring, map(toolbox.evaluate, offspring)):
            ind.fitness.values = fit
        pop = toolbox.select(pop + offspring, k=NSGA3_POP)
        
    res = []
    for ind in pop:
        f = ind.fitness.values
        res.append((f[0], -f[1], -f[2]))
    return res, pop