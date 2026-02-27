from deap import base, creator, tools, algorithms
import random
import numpy as np
from case_config import *
from constraints import evaluate

def run_nsga2():
    # 1. 设置 DEAP 环境
    # 如果已存在则不重复创建
    if not hasattr(creator, "FitnessMulti"):
        # 权重: Cost(min)=-1, Safety(max)=+1, Structural(max)=+1
        # 但在 evaluate 函数中我们返回的是 (-M, -S)，所以这里统一设为最小化 (-1, -1, -1)
        # 修正: evaluate 返回 (c, -m, -s)。我们希望最小化 c, 最小化 -m, 最小化 -s。
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    
    # 2. 个体生成器
    def random_ind():
        ind = []
        # 连续几何变量
        for r in VAR_RANGES_GEO:
            ind.append(random.uniform(r[0], r[1]))
        
        # 离散变量 (映射为连续索引)
        ind.append(random.uniform(0, len(VAL_FC)-0.01)) # fc
        ind.append(random.uniform(0, len(VAL_FY)-0.01)) # fy
        
        # 材料连续变量
        ind.append(random.uniform(*VAR_RANGES_MAT[0])) # dr
        ind.append(random.uniform(*VAR_RANGES_MAT[1])) # dp
        
        # 离散预应力
        ind.append(random.uniform(0, len(VAL_NPB)-0.01)) # npb
        ind.append(random.uniform(0, len(VAL_NPW)-0.01)) # npw
        
        # 预应力连续
        ind.append(random.uniform(*VAR_RANGES_MAT[2])) # sigma
        ind.append(random.uniform(*VAR_RANGES_MAT[3])) # hp_ratio
        
        return creator.Individual(ind)

    toolbox.register("individual", random_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
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

    # 3. 注册算子
    toolbox.register("evaluate", evaluate)
    # 模拟二进制交叉 (传入正确边界)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                     low=LOW, up=UP, eta=20.0) 
    # 多项式变异 (传入正确边界)
    toolbox.register("mutate", tools.mutPolynomialBounded, 
                     low=LOW, up=UP, eta=20.0, indpb=1.0/NDIM)
    toolbox.register("select", tools.selNSGA2)
    
    # 4. 运行主循环
    pop = toolbox.population(n=NSGA2_POP)
    
    # 初始评估
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        
    for gen in range(NSGA2_GEN):
        # 育种
        offspring = algorithms.varAnd(pop, toolbox, cxpb=NSGA2_CXPB, mutpb=NSGA2_MUTPB)
        
        # 评估
        fits = map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
            
        # 选择 (精英保留)
        pop = toolbox.select(pop + offspring, k=NSGA2_POP)
        
    # 5. 提取结果
    res = []
    for ind in pop:
        # fitness 是 (c, -m, -s)
        f = ind.fitness.values
        # 还原为 (c, m, s)
        res.append((f[0], -f[1], -f[2]))
        
    return res, pop