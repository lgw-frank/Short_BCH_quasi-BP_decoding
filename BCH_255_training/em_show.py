# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:14:57 2026

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 演示EM更新公式的直观理解
np.random.seed(42)

# 生成混合数据
n_samples = 1000
true_mus = [-1, 1]
true_sigma = 0.8
true_pi = 0.5

# 生成数据
components = np.random.choice([0, 1], size=n_samples, p=[true_pi, 1-true_pi])
y = np.zeros(n_samples)
for i, comp in enumerate(components):
    y[i] = np.random.normal(true_mus[comp], true_sigma)

# 初始猜测
mu1, mu2 = -0.5, 0.5  # 初始均值（不准确）
sigma2 = 1.0  # 初始方差

print("=== EM更新公式的逐步演示 ===\n")

# 进行一次EM迭代
# E-step: 计算责任
prob1 = 0.5 * norm.pdf(y, mu1, np.sqrt(sigma2))
prob2 = 0.5 * norm.pdf(y, mu2, np.sqrt(sigma2))
gamma1 = prob1 / (prob1 + prob2)
gamma2 = 1 - gamma1

# 可视化责任
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：数据点及其责任
axes[0].scatter(y, gamma1, c=gamma1, cmap='coolwarm', alpha=0.6, s=20)
axes[0].axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
axes[0].set_xlabel('y')
axes[0].set_ylabel('责任 γ₁ (属于分量1的概率)')
axes[0].set_title('E-Step: 计算每个点的责任')
axes[0].grid(True, alpha=0.3)

# 右图：加权平均的直观演示
# 展示如何用加权平均计算新均值
n1 = np.sum(gamma1)
n2 = np.sum(gamma2)
mu1_new = np.sum(gamma1 * y) / n1
mu2_new = np.sum(gamma2 * y) / n2

# 绘制原始均值和加权后的新均值
x_range = np.linspace(-3, 3, 200)
axes[1].hist(y, bins=40, density=True, alpha=0.5, label='数据分布')
axes[1].axvline(x=mu1, color='blue', linestyle='--', label=f'μ₁旧 = {mu1:.3f}', alpha=0.7)
axes[1].axvline(x=mu2, color='red', linestyle='--', label=f'μ₂旧 = {mu2:.3f}', alpha=0.7)
axes[1].axvline(x=mu1_new, color='blue', linewidth=2, label=f'μ₁新 = {mu1_new:.3f}')
axes[1].axvline(x=mu2_new, color='red', linewidth=2, label=f'μ₂新 = {mu2_new:.3f}')
axes[1].set_xlabel('y')
axes[1].set_ylabel('密度')
axes[1].set_title('M-Step: 加权平均更新均值')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("更新公式详解：")
print(f"\n分量1 (左峰):")
print(f"  总责任 N₁ = Σ γ₁ᵢ = {n1:.1f}")
print(f"  加权和 Σ(γ₁ᵢ·yᵢ) = {np.sum(gamma1 * y):.3f}")
print(f"  新均值 μ₁ = {np.sum(gamma1 * y):.3f} / {n1:.1f} = {mu1_new:.3f}")

print(f"\n分量2 (右峰):")
print(f"  总责任 N₂ = Σ γ₂ᵢ = {n2:.1f}")
print(f"  加权和 Σ(γ₂ᵢ·yᵢ) = {np.sum(gamma2 * y):.3f}")
print(f"  新均值 μ₂ = {np.sum(gamma2 * y):.3f} / {n2:.1f} = {mu2_new:.3f}")

# 方差更新
weighted_sse = np.sum(gamma1 * (y - mu1_new)**2 + gamma2 * (y - mu2_new)**2)
sigma2_new = weighted_sse / n_samples
print(f"\n方差更新:")
print(f"  加权平方和 = {weighted_sse:.3f}")
print(f"  新方差 σ² = {weighted_sse:.3f} / {n_samples} = {sigma2_new:.3f}")