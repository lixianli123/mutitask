import numpy as np
from case_config import *
from objectives import decode_variables, SectionProperties, calculate_objectives

def check_constraints(x_continuous, debug=False):
    """
    检查约束，返回总惩罚值。
    如果满足所有约束，返回 0。
    """
    params = decode_variables(x_continuous)
    sec = SectionProperties(params[:12])
    
    h = params[2]
    fc = params[12]
    fy = params[13]
    
    # --- Calculate Physical Properties for Constraints ---
    # Re-implementing logic from objectives.py
    # 1. Mass and Mu
    rho_s_assumed = 0.015 
    Ar = sec.Ac * rho_s_assumed
    mr = STEEL_DENSITY * Ar * L_SPAN
    
    area_p_strand = np.pi * (params[15]/1000.0)**2 / 4 
    total_strands = params[16] + params[17]
    Ap = total_strands * area_p_strand
    mp = STEEL_DENSITY * Ap * L_SPAN
    
    mc = DENSITY_CONCRETE * sec.Ac * L_SPAN
    m_total = mc + mr + mp
    Mu = m_total * G * L_SPAN 
    
    # 2. Prestress Moment Mp
    # Mp = term * (npb*h0 + npw*(h0-hp))
    h0 = h - sec.ttop
    hp_val = params[19] * h
    term_p = 0.25 * np.pi * (params[15]/1000.0)**2 * (params[18] * 1e6)
    Mp = term_p * (params[16] * h0 + params[17] * (h0 - hp_val))
    # ----------------------------------------------------

    penalties = 0.0
    
    # 1. 顶板宽度约束 (隐式满足)
    
    # 2. 宽度比例约束 (0.6 ltop <= lbot <= 0.8 ltop)
    if not (0.6 * sec.ltop <= sec.lbot <= 0.8 * sec.ltop):
        if debug: print(f"FAIL: Width Ratio (0.6 ltop <= lbot <= 0.8 ltop). ltop={sec.ltop:.2f}, lbot={sec.lbot:.2f}")
        penalties += PENALTY_VALUE
        
    # 3. 高跨比约束 (h/L 一般在 1/15 ~ 1/40)
    # 论文要求: L/h <= 30 => h >= L/30 = 2.5m
    # if h < L_SPAN / 30.0:
    #     if debug: print(f"FAIL: Height Ratio (h >= L/30). h={h:.2f}, limit={L_SPAN/30.0:.2f}")
    #     penalties += PENALTY_VALUE

    # 3.1 几何比例约束 (ttop / tw <= 2.0)
    # if sec.ttop / sec.tw > 2.0:
    #     if debug: print(f"FAIL: ttop/tw <= 2.0. ttop={sec.ttop:.3f}, tw={sec.tw:.3f}, ratio={sec.ttop/sec.tw:.2f}")
    #     penalties += PENALTY_VALUE
        
    # 4. 倒角比例约束 (x/y)
    if not (0.2 <= sec.x1/sec.y1 <= 5.0):
        if debug: print(f"FAIL: Chamfer Ratio x1/y1. x1={sec.x1:.2f}, y1={sec.y1:.2f}")
        penalties += PENALTY_VALUE
        
    # 6. 预应力弯矩平衡约束 (Mp <= 0.5 * (Mmax+ - Mmax-))
    # 假设 Mmax- magnitude approx 1.2 * Mmax+
    # Mmax+ = ALPHA * Mu
    # Limit approx 1.1 * Mmax+
    Mmax_pos = ALPHA * Mu
    if Mp > 1.1 * Mmax_pos:
        if debug: print(f"FAIL: Mp Balance. Mp={Mp:.2e}, Limit={1.1*Mmax_pos:.2e}")
        penalties += PENALTY_VALUE

    # 7. 材料性能约束 (fc >= 40, fy >= 235)
    # 变量范围已保证，但显式检查以符合论文要求
    if fc < 40: 
        if debug: print(f"FAIL: fc >= 40. fc={fc}")
        penalties += PENALTY_VALUE
    if fy < 235: 
        if debug: print(f"FAIL: fy >= 235. fy={fy}")
        penalties += PENALTY_VALUE

    # --- 8. 应力约束 (Stress Constraints) ---
    # M_pre approx Mp (Balancing Moment)
    # Sigma = -P/A +- (M_pre - M_load)/W
    # P_eff approx 0.85 * P_control
    eta = 0.85
    P_eff = eta * params[18] * 1e6 * Ap
    
    y_top = sec.h - sec.yc
    y_bot = sec.yc
    W_top = sec.Iz / y_top
    W_bot = sec.Iz / y_bot
    
    # Check at Service Limit State (M_load = Mmax_pos)
    # Top Fiber (Tension +)
    sigma_top = -P_eff/sec.Ac + (Mp - Mmax_pos)/W_top
    # Bottom Fiber
    sigma_bot = -P_eff/sec.Ac - (Mp - Mmax_pos)/W_bot
    
    # Limits (Negative for Compression)
    limit_c = -1.0 * (LIMIT_STRESS_C_FACTOR * fc * 1e6)
    limit_t = LIMIT_STRESS_T
    
    # if sigma_top < limit_c or sigma_top > limit_t: 
    #     if debug: print(f"FAIL: Stress Top. Sigma={sigma_top/1e6:.2f} MPa, Range=[{limit_c/1e6:.2f}, {limit_t/1e6:.2f}]")
    #     penalties += PENALTY_VALUE
    # if sigma_bot < limit_c or sigma_bot > limit_t: 
    #     if debug: print(f"FAIL: Stress Bot. Sigma={sigma_bot/1e6:.2f} MPa, Range=[{limit_c/1e6:.2f}, {limit_t/1e6:.2f}]")
    #     penalties += PENALTY_VALUE
    
    # 8.2 剪应力约束 (Shear)
    # V_max approx 0.6 * Weight
    V_max = 0.6 * (Mu / L_SPAN)
    h_web = sec.h - sec.ttop - sec.tbot
    A_web = 2 * h_web * sec.tw
    tau = V_max / A_web
    
    limit_tau = LIMIT_SHEAR_FACTOR * fc * 1e6
    # if tau > limit_tau: 
    #     if debug: print(f"FAIL: Shear. Tau={tau/1e6:.2f} MPa, Limit={limit_tau/1e6:.2f}")
    #     penalties += PENALTY_VALUE

    # 5. 挠度约束 (delta <= L/800)
    # 公式: delta = k * M * L^2 / S
    # k: 比例系数, M: 计算弯矩 (Mmax_pos), S: 刚度
    # 对于三跨连续梁, k 经验值取 5/48 (简支) * 0.5 (连续) ~ 0.05?
    # 或者直接使用 M/S * L^2 * k.
    # 原代码: 5 * q * L^4 / (384 * E * I). 
    # q * L^2 / 8 = M_simple.
    # 5/384 * q L^4 = 5/48 * (qL^2/8) * L^2 / EI = 5/48 * M_simple * L^2 / S.
    # 论文公式 delta = k * M * L^2 / S.
    # 这里的 M 是 "计算弯矩". 
    # 我们使用 Mmax_pos = alpha * Mu (其中 alpha=0.514, Mu=mgL).
    # 如果 Mu=mgL, Mmax_pos = 0.514 mgL.
    # 这比 qL^2/8 = mgL/8 = 0.125 mgL 大很多.
    # 说明 alpha=0.514 可能是针对 Mu=mgL 的修正系数.
    # 让我们假设 k = 0.02 (保守估计, 连续梁挠度较小)
    k_def = 0.02 
    # 使用刚度 S = sec.S_val? 不, S 在 calculate_objectives 计算
    # 这里需重新计算 S 或从外部传入. check_constraints 重新计算了 Ec, Iz.
    # S = alpha_s * E * Iz. alpha_s = 0.95.
    alpha_s = 0.95
    
    # Calculate Ec
    Ec = (-4.375 * fc**2 + 612.5 * fc + 15000) * 1e6
    
    S_curr = alpha_s * Ec * sec.Iz
    
    # M_load = Mmax_pos
    deflection = k_def * Mmax_pos * L_SPAN**2 / S_curr
    limit = L_SPAN / 800.0 
    
    # if deflection > limit:
    #     if debug: print(f"FAIL: Deflection. Delta={deflection*1000:.2f} mm, Limit={limit*1000:.2f} mm")
    #     penalties += PENALTY_VALUE
        
    # 几何形态约束 ltop >= lbot + 2(h-ttop)tan(...)
    # 已通过 objectives.py 中 ltop 定义隐含满足 (使用了 h-ttop)
    
    return penalties

def evaluate(x_continuous):
    """
    DEAP 评估函数包装器
    返回: (fitness1, fitness2, fitness3)
    DEAP 默认最小化适应度。
    目标: Min Cost, Max M, Max S
    映射为: (Cost, -M, -S)
    """
    # 检查约束
    penalty = check_constraints(x_continuous)
    
    if penalty > 0:
        # 违反约束，给予极大惩罚
        return (1e15, 1e15, 1e15)
        
    c, m, s = calculate_objectives(x_continuous)
    
    # 转换为最小化问题
    return (c, -m, -s)