import numpy as np


def calculate_phi(pareto_front, global_bounds):
    """
    计算 Pareto 前沿的最优 Phi 值。
    pareto_front: list of (Cost, Safety, Structural) 原始值
    global_bounds: dict, 包含所有算法聚合后的全局最优/最差边界
    """
    results = np.array(pareto_front)

    if len(results) == 0:
        return float('inf'), None

    # 1. 提取独立计算的全局理想点 (Star) 和最差点 (Nadir)
    C_star, C_nadir = global_bounds['C_star'], global_bounds['C_nadir']
    M_star, M_nadir = global_bounds['M_star'], global_bounds['M_nadir']
    S_star, S_nadir = global_bounds['S_star'], global_bounds['S_nadir']

    phis = []
    for (C, M, S) in results:
        # 防御性编程：避免极差为0导致除零错误
        range_C = C_nadir - C_star if (C_nadir - C_star) > 1e-9 else 1.0
        range_M = M_nadir - M_star if (M_nadir - M_star) > 1e-9 else 1.0
        range_S = S_star - S_nadir if (S_star - S_nadir) > 1e-9 else 1.0

        # 2. 严格按照全局极值进行归一化
        # 最小化目标 (C, M): (Value - Min) / Range
        norm_C = max(0.0, (C - C_star) / range_C)
        norm_M = max(0.0, (M - M_star) / range_M)

        # 最大化目标 (S): (Max - Value) / Range
        norm_S = max(0.0, (S_star - S) / range_S)

        # 3. 欧氏距离权重计算
        val = 0.3125 * (norm_C ** 2) + 0.3125 * (norm_M ** 2) + 0.375 * (norm_S ** 2)
        phis.append(np.sqrt(val))

    phis = np.array(phis)
    idx_min = np.argmin(phis)

    return phis[idx_min], results[idx_min]