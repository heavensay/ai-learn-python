"""
绘制以下图形：
训练集离散点
线性回归
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['STHeiti']  # Windows系统推荐 SimHei，Mac系统可改为 'STHeiti'

xpoints = np.array([50,60,70,80,90])
ypoints = np.array([150,180,200,230,260])
# plt.show()

# 1. 定义 x 的取值范围（例如从 -10 到 20，生成 100 个点）
x = np.linspace(45, 100, 100)

# 2. 根据公式计算对应的 y 值
y = 2.7 * x + 15

# 3. 创建画布
plt.figure(figsize=(8, 6))

# zorder确保散点显示在最上层
plt.scatter(xpoints,ypoints, color='red', s=50, label='面积-价格离散点', zorder=3)
plt.plot(x, y, label=r'$y = 2.7x + 15$', color='blue', linewidth=2)

# 4. 绘制参考辅助线（X轴与Y轴的原点虚线）
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

# 5. 添加标题、坐标轴标签、图例和网格
plt.title('最优模型 $y = 2.7x + 15$', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', rotation=0, labelpad=25, fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)

# 6. 显示图表
plt.show()