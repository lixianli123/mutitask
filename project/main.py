import numpy as np
import csv
import time
import os
import random
from case_config import *
from evaluation import calculate_phi
from constraints import check_constraints
from objectives import decode_variables, calculate_objectives
from visualization import plot_box_phi, plot_best_run_3d, plot_best_run_surface

# 导入算法
from algorithms.nsga2 import run_nsga2
from algorithms.nsga3 import run_nsga3
from algorithms.gde3 import run_gde3
from algorithms.mopso import run_mopso


DECODED_VAR_NAMES = [
    'l_seg', 'lbot', 'h', 'ttop', 'tbot', 'tw', 'p_slope',
    'x1', 'x2', 'x3', 'y1', 'y2',
    'fc', 'fy', 'dr', 'dp', 'npb', 'npw', 'sigma', 'hp_ratio'
]


def _dominates(a, b):
    """目标支配关系: Min C, Min M, Max S。"""
    ta = (a[0], a[1], -a[2])
    tb = (b[0], b[1], -b[2])
    no_worse = (ta[0] <= tb[0]) and (ta[1] <= tb[1]) and (ta[2] <= tb[2])
    strictly_better = (ta[0] < tb[0]) or (ta[1] < tb[1]) or (ta[2] < tb[2])
    return no_worse and strictly_better


def _pareto_indices(obj_list):
    """返回非支配前沿索引（第一前沿）。"""
    n = len(obj_list)
    if n == 0:
        return []
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if _dominates(obj_list[j], obj_list[i]):
                dominated[i] = True
                break
    return [i for i, d in enumerate(dominated) if not d]


def _safe_name(algo_name):
    return algo_name.lower().replace('-', '_').replace(' ', '_')


def _create_unique_output_dir(root_dir, base_name):
    os.makedirs(root_dir, exist_ok=True)
    candidate = os.path.join(root_dir, base_name)
    if not os.path.exists(candidate):
        os.makedirs(candidate)
        return candidate

    idx = 1
    while True:
        candidate = os.path.join(root_dir, f"{base_name}_{idx}")
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate
        idx += 1


def _phi_value(obj, bounds):
    c, m, s = obj
    c_star, c_nadir = bounds['C_star'], bounds['C_nadir']
    m_star, m_nadir = bounds['M_star'], bounds['M_nadir']
    s_star, s_nadir = bounds['S_star'], bounds['S_nadir']

    range_c = c_nadir - c_star if (c_nadir - c_star) > 1e-9 else 1.0
    range_m = m_nadir - m_star if (m_nadir - m_star) > 1e-9 else 1.0
    range_s = s_star - s_nadir if (s_star - s_nadir) > 1e-9 else 1.0

    norm_c = max(0.0, (c - c_star) / range_c)
    norm_m = max(0.0, (m - m_star) / range_m)
    norm_s = max(0.0, (s_star - s) / range_s)

    val = 0.3125 * (norm_c ** 2) + 0.3125 * (norm_m ** 2) + 0.375 * (norm_s ** 2)
    return float(np.sqrt(val))

def main():
    print("=== PCS 预制拼装箱梁桥多目标优化系统启动 ===")
    print(f"跨径: {L_SPAN}m, 桥宽: {B_TOP}m")
    print(f"优化目标: Min Cost, Min Safety(M), Max Structural")
    print("-" * 50)
    
    # 设置随机种子
    random.seed(SEED)
    np.random.seed(SEED)

    # 每次运行创建独立输出目录，避免文件覆盖
    project_root = os.getcwd()
    output_dir = _create_unique_output_dir(os.path.join(project_root, "outputs"), "run_results")
    os.chdir(output_dir)
    print(f"输出目录: {output_dir}")
    
    algorithms = {
        'NSGA-II': run_nsga2,
        'NSGA-III': run_nsga3,
        'GDE3': run_gde3,
        'MOPSO': run_mopso
    }

    csv_result = "optimization_results.csv"
    csv_summary = "feasibility_summary.csv"
    csv_detail = "solution_audit_details.csv"
    csv_global_pf = "pareto_front_global.csv"
    csv_best_detail = "best_phi_samples_detailed.csv"
    csv_leaderboard = "phi_leaderboard.csv"
    txt_leaderboard = "phi_leaderboard.txt"
    csv_output_index = "output_file_index.csv"

    with open(csv_detail, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Algorithm', 'Run', 'SolutionIdx', 'IsFeasible', 'PenaltyTotal',
            'Fitness1', 'Fitness2', 'Fitness3',
            'ObjCost', 'ObjMoment', 'ObjStiffness',
            'Eq24_width_ok', 'Eq25_height_ok', 'Eq26_chamfer_ok',
            'Eq29_moment_balance_ok', 'Eq28_deflection_ok',
            'Material_fc_ok', 'Material_fy_ok',
            'ltop', 'lbot', 'eq24_lower', 'eq24_upper',
            'h', 'eq25_h_min', 'eq25_h_max',
            'x1_y1', 'x2_y1', 'x3_y2',
            'Mp', 'eq29_limit_Mp',
            'deflection', 'eq28_limit_deflection',
            'Mu', 'Mmax_pos', 'S_curr', 'A', 'Ac', 'h0'
        ] + [f'decoded_{v}' for v in DECODED_VAR_NAMES])

    with open(csv_summary, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Algorithm', 'Run', 'Time(s)', 'PopulationSize', 'FrontSize',
            'FeasibleCount', 'InfeasibleCount', 'FeasibleRatio',
            'Fail_Eq24', 'Fail_Eq25', 'Fail_Eq26', 'Fail_Eq29', 'Fail_Eq28',
            'Fail_fc', 'Fail_fy'
        ])

    with open(csv_best_detail, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Algorithm', 'SourceRun', 'SourceSolutionIdx',
            'Phi', 'Cost', 'Moment', 'Stiffness'
        ] + DECODED_VAR_NAMES)

    run_cache = {name: [] for name in algorithms}
    algo_feasible_records = {name: [] for name in algorithms}
    algo_front_records = {name: [] for name in algorithms}

    # ==========================================
    # 阶段一：运行所有算法，探索解空间并收集数据
    # ==========================================
    print(">>> 阶段一：执行多目标优化算法，探索解空间...")
    for name, algo_func in algorithms.items():
        print(f"\n[开始运行算法: {name}]")

        for run in range(N_RUNS):
            start_t = time.time()
            pareto_front, population = algo_func()
            duration = time.time() - start_t

            feasible_count = 0
            pop_size = len(population) if population is not None else 0
            fail_counts = {
                'Eq24': 0, 'Eq25': 0, 'Eq26': 0,
                'Eq29': 0, 'Eq28': 0, 'fc': 0, 'fy': 0
            }
            first_infeasible_example = None

            if population is not None:
                with open(csv_detail, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    for idx, ind in enumerate(population):
                        penalty, detail = check_constraints(ind, return_details=True)
                        is_feasible = detail['is_feasible']
                        if is_feasible:
                            feasible_count += 1
                        elif first_infeasible_example is None:
                            violated = []
                            if not detail['Eq24_width_ok']:
                                violated.append('Eq24')
                            if not detail['Eq25_height_ok']:
                                violated.append('Eq25')
                            if not detail['Eq26_chamfer_ok']:
                                violated.append('Eq26')
                            if not detail['Eq29_moment_balance_ok']:
                                violated.append('Eq29')
                            if not detail['Eq28_deflection_ok']:
                                violated.append('Eq28')
                            if not detail['Material_fc_ok']:
                                violated.append('fc')
                            if not detail['Material_fy_ok']:
                                violated.append('fy')

                            first_infeasible_example = {
                                'idx': idx,
                                'violated': violated,
                                'detail': detail,
                                'decoded': decode_variables(ind)
                            }

                        if not detail['Eq24_width_ok']:
                            fail_counts['Eq24'] += 1
                        if not detail['Eq25_height_ok']:
                            fail_counts['Eq25'] += 1
                        if not detail['Eq26_chamfer_ok']:
                            fail_counts['Eq26'] += 1
                        if not detail['Eq29_moment_balance_ok']:
                            fail_counts['Eq29'] += 1
                        if not detail['Eq28_deflection_ok']:
                            fail_counts['Eq28'] += 1
                        if not detail['Material_fc_ok']:
                            fail_counts['fc'] += 1
                        if not detail['Material_fy_ok']:
                            fail_counts['fy'] += 1

                        decoded = decode_variables(ind)
                        obj_c, obj_m, obj_s = calculate_objectives(ind)
                        fit_vals = ind.fitness.values if hasattr(ind, 'fitness') and ind.fitness.valid else ('', '', '')

                        if is_feasible:
                            rec = {
                                'algorithm': name,
                                'run': run + 1,
                                'solution_idx': idx,
                                'objectives': (obj_c, obj_m, obj_s),
                                'decoded': decoded
                            }
                            algo_feasible_records[name].append(rec)

                        writer.writerow([
                            name, run + 1, idx, int(is_feasible), penalty,
                            fit_vals[0], fit_vals[1], fit_vals[2],
                            obj_c, obj_m, obj_s,
                            int(detail['Eq24_width_ok']), int(detail['Eq25_height_ok']), int(detail['Eq26_chamfer_ok']),
                            int(detail['Eq29_moment_balance_ok']), int(detail['Eq28_deflection_ok']),
                            int(detail['Material_fc_ok']), int(detail['Material_fy_ok']),
                            detail['ltop'], detail['lbot'], detail['eq24_lower'], detail['eq24_upper'],
                            detail['h'], detail['eq25_h_min'], detail['eq25_h_max'],
                            detail['x1_y1'], detail['x2_y1'], detail['x3_y2'],
                            detail['Mp'], detail['eq29_limit_Mp'],
                            detail['deflection'], detail['eq28_limit_deflection'],
                            detail['Mu'], detail['Mmax_pos'], detail['S_curr'], detail['A'], detail['Ac'], detail['h0']
                        ] + list(decoded))

            infeasible_count = pop_size - feasible_count
            feasible_ratio = (feasible_count / pop_size) if pop_size > 0 else 0.0

            def ratio_str(count, denom):
                if denom <= 0:
                    return "0.00%"
                return f"{(count / denom) * 100:.2f}%"

            front_size = len(pareto_front) if pareto_front else 0
            run_cache[name].append({
                'run': run,
                'front_size': front_size,
                'time': duration,
                'pop_size': pop_size,
                'feasible_count': feasible_count,
                'infeasible_count': infeasible_count,
                'feasible_ratio': feasible_ratio,
                'fail_counts': fail_counts
            })

            if front_size > 0:
                print(f"  - 第 {run+1}/{N_RUNS} 次运行完成. 用时: {duration:.1f}s, 前沿解: {front_size}, 可行/总数: {feasible_count}/{pop_size} ({feasible_ratio:.2%})")
            else:
                print(f"  - 第 {run+1}/{N_RUNS} 次运行失败. 未找到可行解. 可行/总数: {feasible_count}/{pop_size} ({feasible_ratio:.2%})")

            # 详细违规打印已关闭，仅保留CSV审计输出

            fc = fail_counts
            with open(csv_summary, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    name, run + 1, duration, pop_size, front_size,
                    feasible_count, infeasible_count, feasible_ratio,
                    fc['Eq24'], fc['Eq25'], fc['Eq26'], fc['Eq29'], fc['Eq28'], fc['fc'], fc['fy']
                ])

        feasible_recs = algo_feasible_records[name]
        if feasible_recs:
            pf_idx = _pareto_indices([r['objectives'] for r in feasible_recs])
            algo_front_records[name] = [feasible_recs[i] for i in pf_idx]
            print(
                f"[算法汇总: {name}] 可行解总数={len(feasible_recs)}, "
                f"算法统一前沿解数={len(algo_front_records[name])}"
            )

            algo_front_file = f"pareto_front_{_safe_name(name)}.csv"
            with open(algo_front_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Algorithm', 'Run', 'SolutionIdx', 'Cost', 'Moment', 'Stiffness'
                ] + DECODED_VAR_NAMES)
                for rec in algo_front_records[name]:
                    c, m, s = rec['objectives']
                    writer.writerow([
                        rec['algorithm'], rec['run'], rec['solution_idx'], c, m, s
                    ] + list(rec['decoded']))
            print(f"  已导出该算法前沿: {algo_front_file}")
        else:
            print(f"[算法汇总: {name}] 未找到可行解，无法生成算法前沿文件")

    # ==========================================
    # 阶段二：四算法前沿池 -> 全局唯一 Pareto Front -> 全局基准
    # ==========================================
    print("\n>>> 阶段二：四算法前沿池统一非支配排序与基准计算...")
    global_front_pool = []
    for name in algorithms:
        global_front_pool.extend(algo_front_records[name])

    if not global_front_pool:
        print("严重错误：四个算法均未得到可行前沿解！程序终止。")
        return

    global_pf_idx = _pareto_indices([r['objectives'] for r in global_front_pool])
    global_pf_records = [global_front_pool[i] for i in global_pf_idx]
    global_arr = np.array([r['objectives'] for r in global_pf_records])

    with open(csv_global_pf, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Algorithm', 'Run', 'SolutionIdx', 'Cost', 'Moment', 'Stiffness'
        ] + DECODED_VAR_NAMES)
        for rec in global_pf_records:
            c, m, s = rec['objectives']
            writer.writerow([
                rec['algorithm'], rec['run'], rec['solution_idx'], c, m, s
            ] + list(rec['decoded']))

    print(f"  四算法前沿池样本总数: {len(global_front_pool)}")
    print(f"  全局唯一 Pareto Front 数量: {len(global_pf_records)}")

    GLOBAL_BOUNDS = {
        'C_star': np.min(global_arr[:, 0]), 'C_nadir': np.max(global_arr[:, 0]),
        'M_star': np.min(global_arr[:, 1]), 'M_nadir': np.max(global_arr[:, 1]),
        'S_star': np.max(global_arr[:, 2]), 'S_nadir': np.min(global_arr[:, 2])
    }
    print(f"  全局极值基准已锁定:")
    print(f"  - Cost (C):    理想基准 {GLOBAL_BOUNDS['C_star']:.2f}, 最差基准 {GLOBAL_BOUNDS['C_nadir']:.2f}")
    print(f"  - Moment (M):  理想基准 {GLOBAL_BOUNDS['M_star']:.2f}, 最差基准 {GLOBAL_BOUNDS['M_nadir']:.2f}")
    print(f"  - Stiffness(S):理想基准 {GLOBAL_BOUNDS['S_star']:.2f}, 最差基准 {GLOBAL_BOUNDS['S_nadir']:.2f}")

    # ==========================================
    # 阶段三：使用全局统一基准计算各算法前沿的 phi
    # ==========================================
    print("\n>>> 阶段三：计算公平评价指标 phi 并输出结果...")
    all_phis = {name: [] for name in algorithms}
    best_results = {
        name: {
            'phi': float('inf'),
            'front': [],
            'best_record': None,
            'best_solution': None,
            'feasible_total': 0,
            'algo_front_size': 0,
        }
        for name in algorithms
    }

    with open(csv_result, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'FeasibleTotal', 'AlgoFrontSize', 'BestCost', 'BestMoment', 'BestStiffness', 'BestPhi'])

    for name in algorithms:
        front_records = algo_front_records[name]
        front_objs = [r['objectives'] for r in front_records]
        feasible_total = len(algo_feasible_records[name])

        if front_objs:
            phi_vals = [_phi_value(obj, GLOBAL_BOUNDS) for obj in front_objs]
            best_idx = int(np.argmin(phi_vals))
            best_phi = float(phi_vals[best_idx])
            best_rec = front_records[best_idx]
            best_sol = best_rec['objectives']

            all_phis[name].append(best_phi)
            best_results[name]['phi'] = best_phi
            best_results[name]['front'] = front_objs
            best_results[name]['best_record'] = best_rec
            best_results[name]['best_solution'] = best_sol
            best_results[name]['feasible_total'] = feasible_total
            best_results[name]['algo_front_size'] = len(front_objs)

            with open(csv_result, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    name, feasible_total, len(front_objs),
                    best_sol[0], best_sol[1], best_sol[2], best_phi
                ])

            with open(csv_best_detail, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    name, best_rec['run'], best_rec['solution_idx'],
                    best_phi, best_sol[0], best_sol[1], best_sol[2]
                ] + list(best_rec['decoded']))
        else:
            with open(csv_result, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([name, feasible_total, 0, '', '', '', ''])
            best_results[name]['feasible_total'] = feasible_total
            best_results[name]['algo_front_size'] = 0

        best_phi_val = best_results[name]['phi']
        if best_phi_val != float('inf'):
            print(f"  [{name}] 最佳 Phi 评分: {best_phi_val:.4f}")
        else:
            print(f"  [{name}] 所有的运行均未能找到可行解")

    # 阶段三补充输出：最终排行榜 + 分算法运行统计文件
    ranking_rows = []
    for name in algorithms:
        res = best_results[name]
        if res['best_solution'] is None:
            continue
        c, m, s = res['best_solution']
        rec = res['best_record']
        ranking_rows.append({
            'Algorithm': name,
            'BestPhi': res['phi'],
            'BestCost': c,
            'BestMoment': m,
            'BestStiffness': s,
            'FeasibleTotal': res['feasible_total'],
            'AlgoFrontSize': res['algo_front_size'],
            'SourceRun': rec['run'],
            'SourceSolutionIdx': rec['solution_idx'],
        })

    ranking_rows.sort(key=lambda x: x['BestPhi'])

    with open(csv_leaderboard, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Rank', 'Algorithm', 'BestPhi', 'BestCost', 'BestMoment', 'BestStiffness',
            'FeasibleTotal', 'AlgoFrontSize', 'SourceRun', 'SourceSolutionIdx'
        ])
        for rank, row in enumerate(ranking_rows, start=1):
            writer.writerow([
                rank, row['Algorithm'], row['BestPhi'], row['BestCost'], row['BestMoment'], row['BestStiffness'],
                row['FeasibleTotal'], row['AlgoFrontSize'], row['SourceRun'], row['SourceSolutionIdx']
            ])

    with open(txt_leaderboard, 'w', encoding='utf-8') as f:
        f.write('=== Phi Final Ranking ===\n')
        if not ranking_rows:
            f.write('No feasible front solution found for all algorithms.\n')
        else:
            for rank, row in enumerate(ranking_rows, start=1):
                f.write(
                    f"#{rank} {row['Algorithm']} | Phi={row['BestPhi']:.6f} | "
                    f"Cost={row['BestCost']:.3f}, Moment={row['BestMoment']:.3f}, Stiffness={row['BestStiffness']:.3f} | "
                    f"FeasibleTotal={row['FeasibleTotal']}, AlgoFrontSize={row['AlgoFrontSize']} | "
                    f"Source(run={row['SourceRun']}, idx={row['SourceSolutionIdx']})\n"
                )

    print("\n>>> 阶段三补充：Phi 排行榜")
    if not ranking_rows:
        print("  无可行前沿样本，无法生成排行榜")
    else:
        for rank, row in enumerate(ranking_rows, start=1):
            print(
                f"  #{rank} {row['Algorithm']}: Phi={row['BestPhi']:.6f}, "
                f"Front={row['AlgoFrontSize']}, Feasible={row['FeasibleTotal']}, "
                f"Source(run={row['SourceRun']}, idx={row['SourceSolutionIdx']})"
            )

    for name in algorithms:
        run_summary_file = f"run_summary_{_safe_name(name)}.csv"
        with open(run_summary_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Algorithm', 'Run', 'Time(s)', 'FrontSize', 'PopulationSize',
                'FeasibleCount', 'InfeasibleCount', 'FeasibleRatio',
                'Fail_Eq24', 'Fail_Eq25', 'Fail_Eq26', 'Fail_Eq29', 'Fail_Eq28', 'Fail_fc', 'Fail_fy'
            ])
            for row in run_cache[name]:
                fc = row['fail_counts']
                writer.writerow([
                    name, row['run'] + 1, row['time'], row['front_size'], row['pop_size'],
                    row['feasible_count'], row['infeasible_count'], row['feasible_ratio'],
                    fc['Eq24'], fc['Eq25'], fc['Eq26'], fc['Eq29'], fc['Eq28'], fc['fc'], fc['fy']
                ])

    # ==========================================
    # 阶段四：调用可视化制图
    # ==========================================
    print("\n>>> 阶段四：正在生成统计图表...")
    plot_box_phi(all_phis)

    for name, res in best_results.items():
        if res['front']:
            print(f"生成 {name} 最佳结果图表 (Phi={res['phi']:.4f})...")
            plot_best_run_3d(name, res['front'])
            plot_best_run_surface(name, res['front'])
    print(f"\n所有结果已保存至 {os.getcwd()}")
    print(f"- 数据(最佳Phi): {csv_result}")
    print(f"- 数据(最佳样本详情): {csv_best_detail}")
    print(f"- 数据(Phi排行榜): {csv_leaderboard}")
    print(f"- 报告(Phi排行榜文本): {txt_leaderboard}")
    print(f"- 数据(可行性汇总): {csv_summary}")
    print(f"- 数据(逐解核查): {csv_detail}")
    print(f"- 数据(全局前沿): {csv_global_pf}")
    for name in algorithms:
        print(f"- 数据({name}运行汇总): run_summary_{_safe_name(name)}.csv")
        print(f"- 数据({name}前沿): pareto_front_{_safe_name(name)}.csv")
    print(f"- 图表: boxplot_phi.png, best_3d_*.png, best_surface_*.png")

    with open(csv_output_index, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'File'])
        writer.writerow(['BestPhiSummary', csv_result])
        writer.writerow(['BestPhiDetailedSample', csv_best_detail])
        writer.writerow(['PhiLeaderboardCsv', csv_leaderboard])
        writer.writerow(['PhiLeaderboardText', txt_leaderboard])
        writer.writerow(['FeasibilitySummary', csv_summary])
        writer.writerow(['SolutionAuditDetails', csv_detail])
        writer.writerow(['GlobalParetoFront', csv_global_pf])
        for name in algorithms:
            writer.writerow([f'{name}ParetoFront', f"pareto_front_{_safe_name(name)}.csv"])
            writer.writerow([f'{name}RunSummary', f"run_summary_{_safe_name(name)}.csv"])


if __name__ == "__main__":
    main()