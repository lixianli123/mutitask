import numpy as np
import csv
import time
import os
import random
from case_config import *
from evaluation import calculate_phi
from visualization import plot_box_phi, plot_best_run_3d, plot_best_run_surface

# 导入算法
from algorithms.nsga2 import run_nsga2
from algorithms.nsga3 import run_nsga3
from algorithms.gde3 import run_gde3
from algorithms.mopso import run_mopso

def main():
    print("=== PCS 预制拼装箱梁桥多目标优化系统启动 ===")
    print(f"跨径: {L_SPAN}m, 桥宽: {B_TOP}m")
    print(f"优化目标: Min Cost, Max Safety, Max Structural")
    print("-" * 50)
    
    # 设置随机种子
    random.seed(SEED)
    np.random.seed(SEED)
    
    algorithms = {
        'NSGA-II': run_nsga2,
        'NSGA-III': run_nsga3,
        'GDE3': run_gde3,
        'MOPSO': run_mopso
    }
    
    # 结果存储
    all_phis = {name: [] for name in algorithms}
    best_results = {name: {'phi': float('inf'), 'front': []} for name in algorithms}
    
    # CSV 初始化

    csv_filename = 'optimization_results.csv'
    with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'Run_ID', 'Cost', 'Safety', 'Structural', 'Phi_Value', 'Time_Seconds'])
    
    # 主循环：多次运行
    for run in range(1, N_RUNS + 1):
        print(f"\n>>> 开始第 {run}/{N_RUNS} 次运行...")
        
        for name, alg_func in algorithms.items():
            print(f"  正在运行 {name} ...", end='', flush=True)
            start_t = time.time()
            
            # 执行算法
            pareto_front, _ = alg_func()
            
            end_t = time.time()
            duration = end_t - start_t
            
            # 计算 Phi 指标
            best_phi, best_sol = calculate_phi(pareto_front)
            
            if best_phi is not None:
                all_phis[name].append(best_phi)
                
                # 记录该算法历史最佳结果
                if best_phi < best_results[name]['phi']:
                    best_results[name]['phi'] = best_phi
                    best_results[name]['front'] = pareto_front
                
                # 写入 CSV
                with open(csv_filename, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, run, best_sol[0], best_sol[1], best_sol[2], best_phi, duration])
                
                print(f" 完成. 用时: {duration:.1f}s, 最佳 Phi: {best_phi:.4f}")
            else:
                print(" 失败. 未找到可行解.")

        
    # 所有运行结束后，绘制 Phi 值箱线图 (1 张)
    print("\n>>> 正在生成统计图表...")
    plot_box_phi(all_phis)
    
    # 绘制每个算法的最佳运行结果 (4 * 2 = 8 张)
    for name, res in best_results.items():
        if res['front']:
            print(f"生成 {name} 最佳结果图表 (Phi={res['phi']:.4f})...")
            # 3D 散点图
            plot_best_run_3d(name, res['front'])
            # 3D 曲面图 (插值)
            plot_best_run_surface(name, res['front'])
            
    # 打印最优解统计

    print("\n=== 最终统计 ===")
    for name, phis in all_phis.items():
        if phis:
            print(f"{name}: 平均 Phi = {np.mean(phis):.4f}, 最优 Phi = {np.min(phis):.4f}")
    
    print(f"\n所有结果已保存至 {os.getcwd()}")
    print(f"- 数据: {csv_filename}")
    print(f"- 图表: boxplot_phi.png, best_3d_*.png, best_surface_*.png")


if __name__ == "__main__":
    main()