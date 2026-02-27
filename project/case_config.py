import numpy as np

# --- 物理常量 ---
L_SPAN = 75.0           # 跨径 (m)
B_TOP = 15.0            # 桥宽 (m)
G = 9.8                 # 重力加速度 (m/s^2)
ALPHA = 0.514           # 弯矩系数
DENSITY_CONCRETE = 2500 # 混凝土密度 (kg/m^3)
STEEL_DENSITY = 7850    # 钢材密度 (kg/m^3)

# --- 变量范围 (下界, 上界) ---
# 几何变量 (12维)
# x[0]: l (节段长度 Segment Length) [2, 5]
# x[1]: lbot (底板宽) [0.8, 1.2]
# x[2]: h (梁高) [1.2, 2.0]
# x[3]: ttop (顶板厚) [0.18, 0.25]
# x[4]: tbot (底板厚) [0.18, 0.25]
# x[5]: tw (腹板厚) [0.18, 0.25]
# x[6]: p (腹板斜率比 Web slope ratio) [3, 4]
# x[7]-x[9]: x1, x2, x3 (倒角水平) [0.15, 0.30]
# x[10]-x[11]: y1, y2 (倒角竖向) [0.07, 0.15]
VAR_RANGES_GEO = [
    (2.0, 5.0),     # l (Segment Length)
    (0.8, 1.2),     # lbot
    (1.2, 2.0),     # h
    (0.18, 0.25),   # ttop
    (0.18, 0.25),   # tbot
    (0.18, 0.25),   # tw
    (3.0, 4.0),     # p (Slope Ratio)
    (0.15, 0.30),   # x1
    (0.15, 0.30),   # x2
    (0.15, 0.30),   # x3
    (0.07, 0.15),   # y1
    (0.07, 0.15),   # y2
]

# 离散变量 (映射值)
# x[12]: fc' (混凝土抗压强度)
VAL_FC = list(range(40, 81, 5))       # 40, 45, ..., 80
# x[13]: fy
VAL_FY = [335, 400, 500]
# x[16]: npb
VAL_NPB = [14, 21, 28, 35, 42]
# x[17]: npw
VAL_NPW = [42, 49, 56, 63, 70]

# 连续材料/预应力变量
# x[14]: dr [12, 25]
# x[15]: dp [12, 16]
# x[18]: sigma [0.7*1860, 0.8*1860]
# x[19]: hp_ratio [0.25, 0.75]
VAR_RANGES_MAT = [
    (12.0, 25.0),               # dr
    (12.0, 16.0),               # dp
    (0.7 * 1860, 0.8 * 1860),   # sigma
    (0.25, 0.75)                # hp_ratio
]

NDIM = 20  # 总变量维度

# --- 优化算法参数 ---
N_RUNS = 10         # 重复运行次数
SEED = 42           # 随机种子

# NSGA-II
NSGA2_POP = 200
NSGA2_GEN = 200
NSGA2_CXPB = 0.7
NSGA2_MUTPB = 0.3

# NSGA-III
NSGA3_POP = 400     # 需为4的倍数以适应参考点
NSGA3_GEN = 400
NSGA3_P = 4         # 参考点参数 (生成15个参考点)

# GDE3
GDE3_POP = 500
GDE3_GEN = 100
GDE3_F = 0.8
GDE3_CR = 0.9

# MOPSO
MOPSO_POP = 500
MOPSO_GEN = 200
MOPSO_W = 0.2       # 惯性权重 (修正为0.2)
MOPSO_C1 = 1.5      # 个体学习因子
MOPSO_C2 = 1.5      # 全局学习因子

# 约束惩罚
PENALTY_VALUE = 1e10

# --- Stress Limits (Placeholders - Please Update) ---
LIMIT_STRESS_C_FACTOR = 0.5  # fc_allow = 0.5 * fc
LIMIT_STRESS_T = 1.0e6       # 1 MPa tension allowed
LIMIT_SHEAR_FACTOR = 0.05    # tau_allow = 0.05 * fc