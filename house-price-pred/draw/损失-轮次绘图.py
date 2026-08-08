import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. 配置中文显示与负号显示
# --------------------------------------------------
plt.rcParams['font.sans-serif'] = ['STHeiti']  # Windows系统推荐 SimHei，Mac 可改为 'STHeiti'
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# --------------------------------------------------
# 2. 提取表格中的数据
# --------------------------------------------------
epochs = [0, 1, 2, 1000, 2000, 3000, 4000, 5000, 5083]
losses = [21540.000000, 15657.791862, 11404.313610, 23.612032, 4.421096, 3.097977, 3.006755, 3.000466, 3.000373]

# --------------------------------------------------
# 3. 创建画布（左右两张图并排展示）
# --------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- 图 1：常规线性刻度 (Linear Scale) ---
ax1.plot(epochs, losses, marker='o', color='#1f77b4', linewidth=2, markersize=5, label='Loss 变化曲线')
ax1.set_title('训练损失下降曲线 (线性刻度)', fontsize=13)
ax1.set_xlabel('Epoch (训练轮次)', fontsize=11)
ax1.set_ylabel('Loss (损失值)', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(fontsize=10)

# 标注终点 Loss
ax1.annotate(f'最终 Loss: {losses[-1]:.4f}',
             xy=(epochs[-1], losses[-1]),
             xytext=(epochs[-1]-1500, losses[-1]+3000),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
             fontsize=10)

# --- 图 2：Y轴对数刻度 (Log Scale - 强烈推荐用于观测收敛细节) ---
ax2.plot(epochs, losses, marker='s', color='#d62728', linewidth=2, markersize=5, label='Loss (Log Scale)')
ax2.set_yscale('log')  # 将 Y 轴设置为对数刻度
ax2.set_title('训练损失下降曲线 (Y轴对数刻度 - 细节观察)', fontsize=13)
ax2.set_xlabel('Epoch (训练轮次)', fontsize=11)
ax2.set_ylabel('Loss (对数刻度)', fontsize=11)
ax2.grid(True, which="both", linestyle=':', alpha=0.6)  # 开启细网格
ax2.legend(fontsize=10)

# 自动调整布局并显示
plt.tight_layout()
plt.show()