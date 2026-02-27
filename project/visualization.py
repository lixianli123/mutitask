import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# 设置中文字体 (尝试通用字体，若乱码请自行调整)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False

def plot_best_run_3d(alg_name, front):
    """绘制单个算法最佳运行的 3D 散点图"""
    if len(front) == 0: return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    data = np.array(front)
    # Cost, Safety, Structural
    ax.scatter(data[:,0], data[:,1], data[:,2], c='b', marker='o', s=30, alpha=0.7)
    
    ax.set_xlabel('Cost (Minimize)')
    ax.set_ylabel('Safety (Maximize)')
    ax.set_zlabel('Structural (Maximize)')
    plt.title(f'{alg_name} - Best Run Pareto Front (3D)')
    
    filename = f'best_3d_{alg_name}.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"图表已保存: {filename}")

def plot_best_run_surface(alg_name, front):
    """
    绘制单个算法最佳运行的 3D 插值曲面图 (Surface Plot)
    使用三角剖分插值 (Trisurf) 拟合 Pareto 前沿曲面
    """
    if len(front) < 3: return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    data = np.array(front)
    x = data[:, 0]  # Cost
    y = data[:, 1]  # Safety
    z = data[:, 2]  # Structural
    
    # 使用 plot_trisurf 进行三角剖分插值拟合
    surf = ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none', alpha=0.8, shade=True)
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Structural')
    
    ax.set_xlabel('Cost (Minimize)')
    ax.set_ylabel('Safety (Maximize)')
    ax.set_zlabel('Structural (Maximize)')
    plt.title(f'{alg_name} - Best Run Pareto Surface (Interpolated)')
    
    # 调整视角以获得更好的观察效果
    ax.view_init(elev=30, azim=45)
    
    filename = f'best_surface_{alg_name}.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"图表已保存: {filename}")

def plot_box_phi(phi_results):
    """绘制 Phi 值箱线图"""
    data = []
    labels = []
    # 确保顺序一致
    for alg in ['NSGA-II', 'NSGA-III', 'GDE3', 'MOPSO']:
        if alg in phi_results:
            data.append(phi_results[alg])
            labels.append(alg)
        
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, patch_artist=True)
    plt.ylabel('Phi Value (Lower is Better)')
    plt.title('Algorithm Performance Comparison (10 Runs)')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    filename = 'boxplot_phi.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"图表已保存: {filename}")