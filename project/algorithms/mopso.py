import random
import numpy as np
from deap import base, creator, tools
from case_config import *
from constraints import evaluate

def run_mopso():
    # 定义粒子
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
        
    if not hasattr(creator, "Particle"):
        # 粒子具有速度、个体最优位置、个体最优适应度
        creator.create("Particle", list, fitness=creator.FitnessMulti, 
                       speed=list, best=list, bestfit=creator.FitnessMulti)

    def generate_particle():
        ind = []
        # 初始化位置
        for r in VAR_RANGES_GEO: ind.append(random.uniform(r[0], r[1]))
        ind.append(random.uniform(0, len(VAL_FC)-0.01))
        ind.append(random.uniform(0, len(VAL_FY)-0.01))
        ind.append(random.uniform(*VAR_RANGES_MAT[0]))
        ind.append(random.uniform(*VAR_RANGES_MAT[1]))
        ind.append(random.uniform(0, len(VAL_NPB)-0.01))
        ind.append(random.uniform(0, len(VAL_NPW)-0.01))
        ind.append(random.uniform(*VAR_RANGES_MAT[2]))
        ind.append(random.uniform(*VAR_RANGES_MAT[3]))
        
        part = creator.Particle(ind)
        # 初始化速度
        part.speed = [random.uniform(-1, 1) for _ in range(NDIM)]
        return part

    pop = [generate_particle() for _ in range(MOPSO_POP)]
    archive = []

    # 初始评估
    for part in pop:
        fit = evaluate(part)
        part.fitness.values = fit
        # 初始化 pbest
        part.best = list(part)
        part.bestfit.values = fit
        archive.append(part)

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

    # 归档集维护函数
    def update_archive(arch, new_parts):
        combined = arch + new_parts
        # 使用 NSGA-II 排序筛选非支配解
        non_dominated = tools.selNSGA2(combined, len(combined))
        # 实际上 selNSGA2 返回的是排序好的，前沿面在最前
        # 这里为了简化，我们假设归档集大小限制为 POP 大小
        if len(non_dominated) > MOPSO_POP:
            return tools.selNSGA2(non_dominated, MOPSO_POP)
        return non_dominated

    # 初始归档
    archive = update_archive([], pop)

    for gen in range(MOPSO_GEN):
        for part in pop:
            # 选择全局最优 gbest
            # 从归档集中随机选择一个优良个体 (Top 10%)
            if not archive:
                gbest = part.best
            else:
                top_k = max(1, int(len(archive) * 0.1))
                gbest = random.choice(archive[:top_k])
            
            # 更新速度和位置
            for i in range(NDIM):
                r1, r2 = random.random(), random.random()
                part.speed[i] = (MOPSO_W * part.speed[i] + 
                                 MOPSO_C1 * r1 * (part.best[i] - part[i]) + 
                                 MOPSO_C2 * r2 * (gbest[i] - part[i]))
                part[i] += part.speed[i]
                
                # 边界处理 (Clamping)
                if part[i] < LOW[i]: 
                    part[i] = LOW[i]
                    part.speed[i] *= -0.5 # 碰壁反弹/减速
                if part[i] > UP[i]: 
                    part[i] = UP[i]
                    part.speed[i] *= -0.5
            
            # 评估
            fit = evaluate(part)
            part.fitness.values = fit
            
            # 更新个体最优 pbest (支配关系)
            if part.fitness.dominates(part.bestfit):
                part.best = list(part)
                part.bestfit.values = fit
            elif not part.bestfit.dominates(part.fitness):
                # 互不支配时，随机更新
                if random.random() < 0.5:
                    part.best = list(part)
                    part.bestfit.values = fit
            
        # 更新归档集
        archive = update_archive(archive, pop)

    res = []
    for ind in archive:
        f = ind.fitness.values
        res.append((f[0], -f[1], -f[2]))
    return res, archive