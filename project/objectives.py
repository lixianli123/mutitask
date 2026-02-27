import numpy as np
from case_config import *

class SectionProperties:
    """计算箱梁截面几何特性"""
    def __init__(self, x):
        # 解包几何变量
        self.l_seg = x[0]  # l: 节段长度 (Segment Length)
        self.lbot = x[1]   # lbot: 底板宽 (m)
        self.h = x[2]      # h: 梁高
        self.ttop = x[3]   # ttop: 顶板厚
        self.tbot = x[4]   # tbot: 底板厚
        self.tw = x[5]     # tw: 腹板厚
        self.p_slope = x[6]# p: 腹板斜率比 (Web slope ratio)
        self.x1, self.x2, self.x3 = x[7], x[8], x[9]
        self.y1, self.y2 = x[10], x[11]
        
        # 导出几何尺寸
        # 腹板高度 (垂直近似)
        self.h_web = self.h - self.ttop - self.tbot
        
        # 计算顶板宽度 ltop
        # 根据几何形态约束: ltop >= lbot + 2 * (h - ttop) * tan(theta - pi/2)
        # tan(theta - pi/2) = 1/p
        # 故 ltop = lbot + 2 * (h - ttop) / p
        dx = (self.h - self.ttop) / self.p_slope
        self.ltop = self.lbot + 2 * dx
        
        # 计算面积 (Ac)
        # 顶板面积 Atop = ltop·ttop
        A_top = self.ltop * self.ttop
        # 底板面积 Abot = lbot·tbot
        A_bot = self.lbot * self.tbot
        # 腹板面积 Aweb = 2·tw·(h−ttop−tbot)
        A_webs = 2 * self.tw * self.h_web
        
        # 倒角面积 Ach = x1y1/2 + x2y1/2 + x3y2/2
        # 注意：x2 配 y1 (参数表 Line 56)
        A_chamfers = 0.5 * self.x1 * self.y1 + 0.5 * self.x2 * self.y1 + 0.5 * self.x3 * self.y2
        
        # 混凝土面积 Ac = Atop + Abot + Aweb − Ach
        self.Ac = A_top + A_bot + A_webs - A_chamfers
        
        # 计算惯性矩 (Iz) - 分块法
        # 1. 计算形心 yc (从底面起算)
        y_top = self.h - self.ttop / 2.0
        y_bot = self.tbot / 2.0
        y_web = self.tbot + self.h_web / 2.0
        
        # 简化计算 yc (忽略倒角对形心位置的影响)
        Sy = A_top * y_top + A_bot * y_bot + A_webs * y_web
        self.yc = Sy / (A_top + A_bot + A_webs)
        
        # 2. 计算 Iz
        # 顶板
        I_top = (self.ltop * self.ttop**3)/12.0 + A_top * (y_top - self.yc)**2
        # 底板
        I_bot = (self.lbot * self.tbot**3)/12.0 + A_bot * (y_bot - self.yc)**2
        # 腹板 (2个)
        I_webs = 2 * ((self.tw * self.h_web**3)/12.0 + (self.tw * self.h_web) * (y_web - self.yc)**2)
        
        self.Iz = I_top + I_bot + I_webs

def decode_variables(x_continuous):
    """
    将连续优化变量解码为物理变量 (处理离散变量映射)
    """
    x = list(x_continuous)
    
    # 映射离散变量索引
    # 12: fc
    idx_fc = int(np.clip(round(x[12]), 0, len(VAL_FC)-1))
    fc = VAL_FC[idx_fc]
    
    # 13: fy
    idx_fy = int(np.clip(round(x[13]), 0, len(VAL_FY)-1))
    fy = VAL_FY[idx_fy]
    
    # 16: npb
    idx_npb = int(np.clip(round(x[16]), 0, len(VAL_NPB)-1))
    npb = VAL_NPB[idx_npb]
    
    # 17: npw
    idx_npw = int(np.clip(round(x[17]), 0, len(VAL_NPW)-1))
    npw = VAL_NPW[idx_npw]
    
    # 其他为连续变量
    dr = x[14]
    dp = x[15]
    sigma = x[18]
    hp_ratio = x[19]
    
    # 返回解码后的完整列表
    # 前12个是几何变量，后面是解码后的物理参数
    return x[:12] + [fc, fy, dr, dp, npb, npw, sigma, hp_ratio]

def calculate_objectives(x_continuous):
    """
    计算三个目标函数
    返回: (Cost, Safety, Structural)
    注意: 优化器默认最小化，因此需要最大化的目标在返回给优化器前需取负，
    但在本函数中返回原始物理值。
    """
    params = decode_variables(x_continuous)
    
    # 提取变量
    geo_params = params[:12]
    sec = SectionProperties(geo_params)
    
    fc = params[12]
    fy = params[13]
    dr = params[14]
    dp = params[15]
    npb = params[16]
    npw = params[17]
    sigma = params[18]
    hp_ratio = params[19]
    
    # --- 1. Cost C(X) (最小化) ---
    # 单价系数
    # cc = 3fc' + 255
    cc = 3 * fc + 255
    cr = (fy * 0.1 + 15) / 11
    cp = (fy * 0.1 + 15) / 11
    
    # 质量计算
    # mr: 普通钢筋质量
    # 假设 Ar = Ac * 0.015 (1.5% 配筋率，因变量 p 现为斜率比，缺少配筋率变量)
    # nr 未定义，使用 Ar 代替
    rho_s_assumed = 0.015 
    Ar = sec.Ac * rho_s_assumed
    
    # mr = n * lr * rho_r (这里 rho_r = STEEL_DENSITY)
    # 假设 lr = L_SPAN (全长), n * lr ~ Ar * L_SPAN
    mr = STEEL_DENSITY * Ar * L_SPAN
    
    # mp: 预应力钢筋质量
    # Ap = (npb+npw) * pi * dp^2 / 4
    # dp 单位 mm -> m
    area_p_strand = np.pi * (dp/1000.0)**2 / 4 
    total_strands = npb + npw
    Ap = total_strands * area_p_strand
    # mp = (npb+npw) * L * rho_p (rho_p = STEEL_DENSITY)
    mp = STEEL_DENSITY * Ap * L_SPAN
    
    Cost = L_SPAN * sec.Ac * cc + mr * cr + mp * cp
    
    # --- 2. Safety M(X) (最大化) ---
    # 混凝土质量 mc = rho_c * Ac * L
    mc = DENSITY_CONCRETE * sec.Ac * L_SPAN
    m_total = mc + mr + mp
    
    # 极限弯矩 Mu = m * g * L
    Mu = m_total * G * L_SPAN 
    
    # 有效高度 h0 = h - ttop
    h0 = params[2] - sec.ttop
    # 预应力高度 hp = (hp/h) * h
    hp_val = hp_ratio * params[2]
    
    # 预应力弯矩 Mp
    # Mp = 0.25 * pi * dp^2 * sigma * (npb * h0 + npw * (h0 - hp))
    # dp 单位 mm -> m, sigma 单位 MPa -> Pa
    term_p = 0.25 * np.pi * (dp/1000.0)**2 * (sigma * 1e6)
    Mp = term_p * (npb * h0 + npw * (h0 - hp_val))
    
    alpha = 0.514
    # 安全目标 M = alpha * Mu - Mp (必须最大化)
    M_val = alpha * Mu - Mp
    
    # --- 3. Structural Performance S(X) (最大化) ---
    # 混凝土模量 Ec (Pa)
    # Ec = (-4.375 * fc'^2 + 612.5 * fc' + 15000) * 1e6
    Ec = (-4.375 * fc**2 + 612.5 * fc + 15000) * 1e6
    
    # 等效模量 E = (EcAc + ErAr * lr / (2l_seg)) / (Ac+Ar)
    # 注意：这里 l_seg 是 x[0] (节段长度)
    Er = 2.0e11 # 钢筋模量
    lr = L_SPAN # 假设钢筋长度为跨长
    l_seg = sec.l_seg + 1e-6 # 防止除零错误 (若算法边界设置不当导致 l_seg -> 0)
    
    E_val = (Ec * sec.Ac + Er * Ar * lr / (2 * l_seg)) / (sec.Ac + Ar + 1e-6)
    
    # 刚度 S = alpha_s * E * Iz
    alpha_s = 0.95 # 刚度折减系数
    S_val = alpha_s * E_val * sec.Iz
    
    return Cost, M_val, S_val