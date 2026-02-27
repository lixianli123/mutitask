import numpy as np

def calculate_phi(pareto_front):
    """
    计算 Pareto 前沿的最优 Phi 值。
    pareto_front: list of (Cost, Safety, Structural) 原始值
    """
    results = np.array(pareto_front)
    
    if len(results) == 0:
        return None, None
        
    # 1. 寻找理想点 (C*, M*, S*) 和 纳迪尔点 (Nadir Point)
    # Cost (Min): C_star = min(C), C_nadir = max(C)
    # Safety (Max): M_star = max(M), M_nadir = min(M)
    # Structural (Max): S_star = max(S), S_nadir = min(S)
    
    C_vals = results[:, 0]
    M_vals = results[:, 1]
    S_vals = results[:, 2]
    
    C_min, C_max = np.min(C_vals), np.max(C_vals)
    M_min, M_max = np.min(M_vals), np.max(M_vals)
    S_min, S_max = np.min(S_vals), np.max(S_vals)
    
    # 2. 归一化并计算 Phi
    # 参数表要求: 
    # 最小化目标 (C): (C - Cmin) / (Cmax - Cmin)
    # 最大化目标 (S, M): (Smax - S) / (Smax - Smin)  (转为最小化)
    # 权重: 0.3125, 0.3125, 0.375
    
    phis = []
    for (C, M, S) in results:
        # Cost 归一化
        range_C = C_max - C_min
        norm_C = (C - C_min) / range_C if range_C > 1e-9 else 0.0
        
        # Safety 归一化 (Max -> Min)
        range_M = M_max - M_min
        norm_M = (M_max - M) / range_M if range_M > 1e-9 else 0.0
        
        # Structural 归一化 (Max -> Min)
        range_S = S_max - S_min
        norm_S = (S_max - S) / range_S if range_S > 1e-9 else 0.0
        
        # Phi 公式: sqrt( w1*(norm_C)^2 + w2*(norm_M)^2 + w3*(norm_S)^2 )
        # 这里的 norm_C 就是 (C - C*)/Range, 即 Cnorm - Cnorm* (因为 Cnorm* = 0)
        val = 0.3125 * norm_C**2 + 0.3125 * norm_M**2 + 0.375 * norm_S**2
        phi = np.sqrt(val)
        phis.append(phi)
        
    phis = np.array(phis)
    idx_min = np.argmin(phis)
    
    # 返回最小 Phi 值及其对应的目标函数值
    return phis[idx_min], results[idx_min]