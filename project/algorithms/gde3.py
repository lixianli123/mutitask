import random
import numpy as np
from deap import base, creator, tools
from case_config import *
from constraints import evaluate

def run_gde3():
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    # 1. 初始化种群
    pop = []
    for _ in range(GDE3_POP):
        ind_data = []
        for r in VAR_RANGES_GEO: ind_data.append(random.uniform(r[0], r[1]))
        ind_data.append(random.uniform(0, len(VAL_FC)-0.01))
        ind_data.append(random.uniform(0, len(VAL_FY)-0.01))
        ind_data.append(random.uniform(*VAR_RANGES_MAT[0]))
        ind_data.append(random.uniform(*VAR_RANGES_MAT[1]))
        ind_data.append(random.uniform(0, len(VAL_NPB)-0.01))
        ind_data.append(random.uniform(0, len(VAL_NPW)-0.01))
        ind_data.append(random.uniform(*VAR_RANGES_MAT[2]))
        ind_data.append(random.uniform(*VAR_RANGES_MAT[3]))
        
        ind = creator.Individual(ind_data)
        ind.fitness.values = evaluate(ind)
        pop.append(ind)

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

    # 2. 进化循环
    for gen in range(GDE3_GEN):
        offspring = []
        for i in range(GDE3_POP):
            target = pop[i]
            
            # 选择 3 个不同的随机个体
            idxs = [idx for idx in range(GDE3_POP) if idx != i]
            r1, r2, r3 = random.sample(idxs, 3)
            x1, x2, x3 = pop[r1], pop[r2], pop[r3]
            
            # 差分变异 + 交叉 (DE/rand/1/bin)
            trial_ind_data = []
            j_rand = random.randint(0, NDIM-1)
            
            for j in range(NDIM):
                if random.random() < GDE3_CR or j == j_rand:
                    val = x1[j] + GDE3_F * (x2[j] - x3[j])
                    # 边界处理 (Clamping)
                    if val < LOW[j]: val = LOW[j]
                    if val > UP[j]: val = UP[j]
                    trial_ind_data.append(val)
                else:
                    trial_ind_data.append(target[j])
            
            trial = creator.Individual(trial_ind_data)
            trial.fitness.values = evaluate(trial)
            
            # GDE3 选择策略 (支配关系)
            # 1. Trial 支配 Target -> 替换
            # 2. Target 支配 Trial -> 丢弃
            # 3. 互不支配 -> 两者都保留 (稍后截断)
            
            if trial.fitness.dominates(target.fitness):
                offspring.append(trial)
            elif target.fitness.dominates(trial.fitness):
                offspring.append(target)
            else:
                offspring.append(target)
                offspring.append(trial)
        
        # 截断操作 (使用 NSGA-II 的非支配排序和拥挤度距离)
        if len(offspring) > GDE3_POP:
            pop = tools.selNSGA2(offspring, GDE3_POP)
        else:
            pop = offspring
            # 如果数量不足(罕见)，随机补充
            while len(pop) < GDE3_POP:
                 # 简单复制补充
                 pop.append(random.choice(pop))

    res = []
    for ind in pop:
        f = ind.fitness.values
        res.append((f[0], -f[1], -f[2]))
    return res, pop