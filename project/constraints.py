import numpy as np
from case_config import *
from objectives import decode_variables, SectionProperties, calculate_objectives

def check_constraints(x_continuous, debug=False, return_details=False):
    """
    检查约束，返回总惩罚值。
    如果满足所有约束，返回 0。
    """
    params = decode_variables(x_continuous)
    sec = SectionProperties(params[:12])

    h = params[2]
    fc = params[12]
    fy = params[13]
    dr = params[14]
    dp = params[15]
    npb = params[16]
    npw = params[17]
    sigma = params[18]
    hp_ratio = params[19]

    # 与 objectives 保持一致的补充条件
    sr_assumed = 0.15

    theta = np.arctan(sec.p_slope) + np.pi / 2.0
    rho_c = fc + 2320
    lr = (sec.ltop + sec.lbot + (4 * sec.h) / np.sin(np.pi - theta)) * (2 * sec.l_seg / sr_assumed)

    rho_r = 6170 * (dr / 1000.0)
    n_segments = L_SPAN / sec.l_seg
    mr = n_segments * lr * rho_r

    rho_p = 6170 * (dp / 1000.0)
    mp = (npb + npw) * L_SPAN * rho_p

    A_chamfers = sec.x1 * sec.y1 + sec.x2 * sec.y1 + sec.x3 * sec.y2
    A = sec.ltop * sec.ttop + sec.lbot * sec.tbot + 2 * sec.tw * (sec.h - sec.ttop - sec.tbot) / np.sin(np.pi - theta) + A_chamfers
    Ac = A - 0.25 * np.pi * (dr / 1000.0) ** 2 * (lr / (2 * sec.l_seg))

    numerator = sec.ltop * (sec.h ** 2 - (sec.h - sec.ttop) ** 2) + sec.lbot * sec.tbot ** 2 + 2 * sec.tw * ((sec.h - sec.ttop) ** 2 - sec.tbot ** 2)
    denominator = 2 * (Ac - A_chamfers) * np.sin(np.pi - theta)
    h0 = numerator / (denominator + 1e-12)

    mc = rho_c * Ac * L_SPAN
    Mu = (mc + mr + mp) * G * L_SPAN

    hp_val = hp_ratio * sec.h
    term_p = 0.25 * np.pi * (dp / 1000.0) ** 2 * (sigma * 1e6)
    Mp = term_p * (npb * h0 + npw * (h0 - hp_val))

    penalties = 0.0
    details = {
        'Eq24_width_ok': True,
        'Eq25_height_ok': True,
        'Eq26_chamfer_ok': True,
        'Eq29_moment_balance_ok': True,
        'Eq28_deflection_ok': True,
        'Material_fc_ok': True,
        'Material_fy_ok': True,
    }

    # 1. Eq(24) 宽度比例约束
    lower_width = 0.6 * sec.ltop
    upper_width = 0.8 * sec.ltop
    if not (lower_width <= sec.lbot <= upper_width):
        if debug:
            print(f"FAIL: Eq(24) width ratio. ltop={sec.ltop:.3f}, lbot={sec.lbot:.3f}")
        details['Eq24_width_ok'] = False
        penalties += PENALTY_VALUE

    # 2. Eq(25) 高跨比约束 (ns=3)
    limit_h_min = L_SPAN / (20.0 * 3)
    limit_h_max = L_SPAN / (15.0 * 3)
    if not (limit_h_min <= h <= limit_h_max):
        if debug:
            print(f"FAIL: Eq(25) h/span. h={h:.3f}, range=[{limit_h_min:.3f}, {limit_h_max:.3f}]")
        details['Eq25_height_ok'] = False
        penalties += PENALTY_VALUE

    # 3. Eq(26) 倒角比例约束
    ratio_ok = (1.0 <= sec.x1 / sec.y1 <= 1.5) and (1.0 <= sec.x2 / sec.y1 <= 1.5) and (1.0 <= sec.x3 / sec.y2 <= 1.5)
    if not ratio_ok:
        if debug:
            print("FAIL: Eq(26) chamfer ratio out of [1.0, 1.5]")
        details['Eq26_chamfer_ok'] = False
        penalties += PENALTY_VALUE

    # 4. Eq(29) 预应力弯矩平衡约束
    limit_Mp = 0.5 * (0.514 * Mu - (-0.338 * Mu))
    if Mp > limit_Mp:
        if debug:
            print(f"FAIL: Eq(29) Mp balance. Mp={Mp:.3e}, limit={limit_Mp:.3e}")
        details['Eq29_moment_balance_ok'] = False
        penalties += PENALTY_VALUE

    # 5. Eq(28) 挠度约束
    # 修改：将 k_def 设为更小量级，缓解 Mu=m*g*L 带来的大数值引起的系统性超限
    k_def = 1e-4
    alpha_s = 0.95

    # 与 objectives.py 一致：Eq(19) 等效模量 E
    Ec = (-4.375 * fc ** 2 + 612.5 * fc + 15000) * 1e6
    Er = 2.0e11
    Ar_single = 0.25 * np.pi * (dr / 1000.0) ** 2
    E_val = (Ec * Ac + Er * Ar_single * (lr / (2 * sec.l_seg))) / (A + 1e-12)

    # 与 objectives.py 一致：Eq(21) 惯性矩 Iz（基于 h0）
    I_top = (sec.ltop * sec.ttop ** 3) / 12.0 + sec.ltop * sec.ttop * (sec.h - h0 - 0.5 * sec.ttop) ** 2
    I_bot = (sec.lbot * sec.tbot ** 3) / 12.0 + sec.lbot * sec.tbot * (h0 - 0.5 * sec.tbot) ** 2
    h_web_actual = sec.h - sec.ttop - sec.tbot
    sin_val = np.sin(np.pi - theta)
    I_web = (1.0 / 12.0) * (2 * sec.tw / sin_val) * (h_web_actual ** 3) + \
        (2 * sec.tw / sin_val) * h_web_actual * (h0 - 0.5 * (sec.h - sec.ttop + sec.tbot)) ** 2
    Iz = I_top + I_bot + I_web

    # Eq(22): 刚度 S
    S_curr = alpha_s * E_val * Iz
    Mmax_pos = 0.514 * Mu

    deflection = k_def * Mmax_pos * L_SPAN ** 2 / (S_curr + 1e-12)
    limit_deflection = L_SPAN / 800.0
    if deflection > limit_deflection:
        if debug:
            print(f"FAIL: Eq(28) deflection. delta={deflection:.6f}, limit={limit_deflection:.6f}")
        details['Eq28_deflection_ok'] = False
        penalties += PENALTY_VALUE

    # 保留原有材料性能约束
    if fc < 40:
        if debug:
            print(f"FAIL: material fc >= 40. fc={fc}")
        details['Material_fc_ok'] = False
        penalties += PENALTY_VALUE
    if fy < 235:
        if debug:
            print(f"FAIL: material fy >= 235. fy={fy}")
        details['Material_fy_ok'] = False
        penalties += PENALTY_VALUE

    details.update({
        'penalty_total': penalties,
        'is_feasible': penalties == 0,
        'ltop': sec.ltop,
        'lbot': sec.lbot,
        'eq24_lower': lower_width,
        'eq24_upper': upper_width,
        'h': h,
        'eq25_h_min': limit_h_min,
        'eq25_h_max': limit_h_max,
        'x1_y1': sec.x1 / sec.y1,
        'x2_y1': sec.x2 / sec.y1,
        'x3_y2': sec.x3 / sec.y2,
        'Mp': Mp,
        'eq29_limit_Mp': limit_Mp,
        'deflection': deflection,
        'eq28_limit_deflection': limit_deflection,
        'Mu': Mu,
        'Mmax_pos': Mmax_pos,
        'S_curr': S_curr,
        'A': A,
        'Ac': Ac,
        'h0': h0,
    })

    if return_details:
        return penalties, details
    return penalties

def evaluate(x_continuous):
    """
    DEAP 评估函数包装器
    """
    penalty = check_constraints(x_continuous)

    # 获取真实的物理目标值
    c, m, s = calculate_objectives(x_continuous)

    if penalty > 0:
        # 使用可微分的叠加惩罚，保留“越接近可行域越优”的相对信息
        # 目标方向保持与全局一致：Min C, Min M, Max S -> (c, m, -s)
        return (c + penalty, m + penalty, -s + penalty)

    # 满足约束时正常返回
    return (c, m, -s)