import math

# ==============================================================================
# 零基础教学：机器学习全流程演示 (一元线性回归与梯度下降)
# 
# 包含了完整的 6 个步骤：
# 【步骤 1】数据收集 (训练集 & 两个测试集)
# 【步骤 2】数据预处理与简化标准化
# 【步骤 3】模型结构设计
# 【步骤 4】梯度下降训练 (控制台与注释同步算法公式、映射关系与计算细节)
# 【步骤 5】模型评估 (逐点残差、各项评估指标与阈值对照)
# 【步骤 6】部署决策 (根据上线标准判定是否通过)
# ==============================================================================


# ==============================================================================
# 步骤 1: 数据收集 (Data Collection)
# ==============================================================================
# 【教学说明】:
# 机器学习的第一步是准备数据。我们需要训练集用来学习规律，
# 以及测试集用来验证模型泛化能力（一个符合真实规律，一个模拟异常噪声）。

# 训练数据集：真实的房屋面积与价格
train_data = {
    "name": "训练集A_标准房价数据",
    "x": [50.0, 60.0, 70.0, 80.0, 90.0],
    "y": [150.0, 180.0, 200.0, 230.0, 260.0]
}

# 测试数据集 1：符合线性规律的数据 (预期：测试通过)
test_data_pass = {
    "name": "测试集1_正常分布数据 (预期通过)",
    "x": [55.0, 65.0, 75.0, 85.0],
    "y": [165.0, 190.0, 215.0, 245.0]
}

# 测试数据集 2：偏离线性规律的异常/噪声数据 (预期：测试不通过)
test_data_fail = {
    "name": "测试集2_异常噪音数据 (预期不通过)",
    "x": [55.0, 65.0, 75.0, 85.0],
    "y": [100.0, 300.0, 120.0, 400.0]  # 价格严重偏离正常逻辑
}

print("=" * 70)
print("【步骤 1】数据收集 (Data Collection)")
print("=" * 70)
print("【教学说明】:")
print("  机器学习的第一步是准备数据。我们需要训练集用来学习规律，")
print("  以及测试集用来验证模型泛化能力（一个符合真实规律，一个模拟异常噪声）。")
print(f"  - 训练集: {train_data['name']}")
print(f"    特征 面积(㎡) x = {train_data['x']}")
print(f"    标签 价格(万) y = {train_data['y']}")
print(f"  - 测试集 1 (高品质): {test_data_pass['name']}")
print(f"  - 测试集 2 (高噪音): {test_data_fail['name']}")


# ==============================================================================
# 步骤 2: 数据预处理与简化标准化 (Data Preprocessing)
# ==============================================================================
# 【教学说明】:
# 为什么要做标准化/特征缩放？
# 因为原始房屋面积 x (50~90) 的数值远大于权重 w，直接计算会导致梯度过大甚至引发数值爆炸。
# 此处采用简化放缩处理：将面积除以 100，使其放缩到 [0.5, 0.9] 的合理小数值区间。
# 【使用的预处理函数】: simplify_scale(x) -> x_scaled = x / 100.0

def simplify_scale(x_list):
    """简化的特征缩放函数：x_scaled = x / 100.0"""
    return [x / 100.0 for x in x_list]

def restore_w(w_scaled):
    """将缩放数据训练出的 w 还原为真实物理量纲下的权重: w_real = w_scaled / 100.0"""
    return w_scaled / 100.0

print("\n" + "=" * 70)
print("【步骤 2】数据预处理与特征缩放 (Data Preprocessing)")
print("=" * 70)
print("【教学说明】:")
print("  为什么要做标准化/特征缩放？")
print("  因为原始房屋面积 x (50~90) 的数值远大于权重 w，直接计算会导致梯度过大甚至引发数值爆炸。")
print("  此处采用简化放缩处理：将面积除以 100，使其放缩到 [0.5, 0.9] 的合理小数值区间。")
print("  【使用的预处理函数】: simplify_scale(x) -> x_scaled = x / 100.0")

x_train_scaled = simplify_scale(train_data['x'])
print(f"  原始训练特征 x : {train_data['x']}")
print(f"  缩放训练特征 x_scaled : {x_train_scaled}")


# ==============================================================================
# 步骤 3: 模型设计 (Model Design)
# ==============================================================================
# 【教学说明】:
# 构建一元线性回归方程，包含两个核心可学习参数：权重 w (Weight) 和 偏置 b (Bias)。
# 【前向传播计算公式】: y_hat = w * x + b (一元线性函数)

class LinearModel:
    """
    一元线性回归模型结构
    【使用的前向传播函数】: predict(x) -> y_pred = w * x + b
    """
    def __init__(self):
        # 权重 w 与偏置 b 初始化为 0.0
        self.w = 0.0
        self.b = 0.0

    def predict(self, x_input):
        return self.w * x_input + self.b

print("\n" + "=" * 70)
print("【步骤 3】模型结构设计与参数初始化 (Model Design)")
print("=" * 70)
print("【教学说明】:")
print("  构建一元线性回归方程，包含两个核心可学习参数：权重 w (Weight) 和 偏置 b (Bias)。")
print("  【前向传播计算公式】: y_hat = w * x + b (一元线性函数)")

model = LinearModel()
print(f"  初始化模型参数: 初始权重 w = {model.w}, 初始偏置 b = {model.b}")


# ==============================================================================
# 步骤 4: 训练模型 (Train Model - 梯度下降)
# ==============================================================================
# 【教学说明】:
# 使用核心算法：批量梯度下降法 (Batch Gradient Descent)。
# 通过不断计算预测值与真实值的误差，求导得到梯度，并沿负梯度方向更新参数。
#
# 【关键函数与算法数学公式映射】:
#   1. 前向传播预测函数: y_hat_i = w * x_scaled_i + b
#   2. 损失函数 (MSE 均方误差): Loss = (1 / 2N) * Σ (y_hat_i - y_i)^2
#   3. 权重梯度求导函数 (dw): dw = (1 / N) * Σ (y_hat_i - y_i) * x_scaled_i
#   4. 偏置梯度求导函数 (db): db = (1 / N) * Σ (y_hat_i - y_i)
#   5. 参数更新迭代公式: w = w - lr * dw  |  b = b - lr * db
#
# 【训练阈值与终止条件说明】:
#   - 早期停止收敛阈值 (tol = 1e-6): 当两次迭代 Loss 变化量小于该值时，判定梯度接近0、已收敛。
#   - 最大迭代轮数上限 (epochs = 10000): 防止无限循环的保底机制。

def train_gradient_descent(model, x_scaled, y_train, lr=0.1, epochs=10000, tol=1e-6):
    n = len(x_scaled)
    prev_loss = float('inf')

    print("\n" + "=" * 70)
    print("【步骤 4】启动梯度下降训练 (Train Model - Gradient Descent)")
    print("=" * 70)
    print("【教学说明】:")
    print("  使用核心算法：批量梯度下降法 (Batch Gradient Descent)。")
    print("  通过不断计算预测值与真实值的误差，求导得到梯度，并沿负梯度方向更新参数。")
    print("\n【关键函数与算法数学公式映射】:")
    print("  1. 前向传播预测函数: y_hat_i = w * x_scaled_i + b")
    print("  2. 损失函数 (MSE 均方误差): Loss = (1 / 2N) * Σ (y_hat_i - y_i)^2")
    print("  3. 权重梯度求导函数 (dw): dw = (1 / N) * Σ (y_hat_i - y_i) * x_scaled_i")
    print("  4. 偏置梯度求导函数 (db): db = (1 / N) * Σ (y_hat_i - y_i)")
    print("  5. 参数更新迭代公式: w = w - lr * dw  |  b = b - lr * db")
    print("\n【训练阈值与终止条件说明】:")
    print(f"  - 早期停止收敛阈值 (tol = {tol}): 当两次迭代 Loss 变化量小于该值时，判定梯度接近0、已收敛。")
    print(f"  - 最大迭代轮数上限 (epochs = {epochs}): 防止无限循环的保底机制。")
    print("-" * 70)
    print(f"【超参数设置】: 学习率 lr = {lr}, 最大轮数 epochs = {epochs}, 提前停止阈值 tol = {tol}")
    print("-" * 70)
    print("Epoch 轮次 |    Loss 损失   |  w (缩放域)  |    b 偏置    | 梯度 dw (∂J/∂w) | 梯度 db (∂J/∂b)")
    print("-" * 70)

    for epoch in range(epochs):
        # 1. 前向传播计算预测值: y_hat_i = w * x_scaled_i + b
        y_pred = [model.predict(xi) for xi in x_scaled]

        # 2. 计算均方误差损失 Loss (MSE): Loss = (1 / 2N) * Σ (y_hat_i - y_i)^2
        loss = sum((yp - yi) ** 2 for yp, yi in zip(y_pred, y_train)) / (2.0 * n)

        # 3. 计算偏导数梯度:
        # dw = (1 / N) * Σ (y_hat_i - y_i) * x_scaled_i
        # db = (1 / N) * Σ (y_hat_i - y_i)
        dw = sum((yp - yi) * xi for yp, yi, xi in zip(y_pred, y_train, x_scaled)) / n
        db = sum(yp - yi for yp, yi in zip(y_pred, y_train)) / n

        # 4. 打印代表性节点 (第0, 1, 2次及后续每1000轮)
        if epoch in (0, 1, 2) or (epoch % 1000 == 0 and epoch > 0):
            print(f"{epoch:8d} | {loss:12.6f} | {model.w:11.4f} | {model.b:11.4f} | {dw:15.4f} | {db:15.4f}")

        # 5. 阈值判断：检查 Loss 是否收敛 (|prev_loss - loss| < tol)
        if abs(prev_loss - loss) < tol:
            # 打印代表性节点(最后一次训练)
            if not (epoch in (0, 1, 2) or (epoch % 1000 == 0 and epoch > 0)):
                print(f"{epoch:8d} | {loss:12.6f} | {model.w:11.4f} | {model.b:11.4f} | {dw:15.4f} | {db:15.4f}")

            print("-" * 70)
            print(f"✓【收敛阈值触发】: 第 {epoch} 轮 Loss 变化量 < {tol}，梯度接近 0，训练提前完成！")
            print(f"最终收敛 Loss = {loss:.6f}, 收敛梯度 dw = {dw:.6f}, db = {db:.6f}")
            break

        prev_loss = loss

        # 6. 参数更新: w = w - lr * dw, b = b - lr * db
        model.w -= lr * dw
        model.b -= lr * db

    # 将 w 还原回真实量纲 (元/㎡ 转为 万元/㎡): w_real = w_scaled / 100.0
    w_real = restore_w(model.w)
    b_real = model.b
    print("-" * 70)
    print(f"【训练完成】还原至真实物理量纲的拟合方程: y = {w_real:.4f} * x + ({b_real:.4f})")

    return w_real, b_real


# 运行训练过程
w_final, b_final = train_gradient_descent(model, x_train_scaled, train_data['y'])


# ==============================================================================
# 步骤 5 & 6: 评估模型与部署决策 (Evaluate & Deploy)
# ==============================================================================
# 【步骤 5 教学说明】:
# 计算逐点预测残差 (Residual = y_true - y_pred)，并根据数学公式推导三大评估指标：
# - R² (决定系数): 1 - (SSE / SST)，越接近 1 说明模型拟合越好。
# - RMSE (均方根误差): √(SSE / N)，对大误差/离群点惩罚更重。
# - MAE (平均绝对误差): (1/N) * Σ|残差|，衡量平均物理尺寸偏差。
#
# 【步骤 6 教学说明】:
# 对比评估指标与工程设定的硬性上线门槛 (Thresholds)，判定是否能部署至生产环境。

def evaluate_and_deploy(dataset_info, w, b):
    x_raw = dataset_info['x']
    y_raw = dataset_info['y']
    name = dataset_info['name']
    n = len(x_raw)

    # 1. 逐点预测与残差计算 (Residual = y_true - y_pred)
    y_pred = [w * x + b for x in x_raw]
    residuals = [y_true - yp for y_true, yp in zip(y_raw, y_pred)]
    sq_residuals = [r ** 2 for r in residuals]

    # 2. 指标计算函数
    sse = sum(sq_residuals)                             # 残差平方和 SSE
    y_mean = sum(y_raw) / n                            # 真实值均值 y_mean
    sst = sum((y - y_mean) ** 2 for y in y_raw)          # 总平方和 SST
    ssr = sum((yp - y_mean) ** 2 for yp in y_pred)       # 回归平方和 SSR

    r_squared = 1.0 - (sse / sst) if sst != 0 else 0.0   # R² 决定系数 = 1 - (SSE/SST)
    rmse = math.sqrt(sse / n)                           # RMSE 均方根误差 = √(SSE/N)
    mae = sum(abs(r) for r in residuals) / n             # MAE 平均绝对误差 = (1/N) * Σ|残差|

    # ------------------ 打印详细评估报告 ------------------
    print("\n" + "=" * 70)
    print(f"【步骤 5】评估模型 (Model Evaluation) - 数据集: 【{name}】")
    print("=" * 70)
    print("【教学说明】:")
    print("  计算逐点预测残差 (Residual = y_true - y_pred)，并根据数学公式推导三大评估指标：")
    print("  - R² (决定系数): 1 - (SSE / SST)，越接近 1 说明模型拟合越好。")
    print("  - RMSE (均方根误差): √(SSE / N)，对大误差/离群点惩罚更重。")
    print("  - MAE (平均绝对误差): (1/N) * Σ|残差|，衡量平均物理尺寸偏差。")

    print("\n--------------------------------------------------")
    print("逐点预测验证明细表")
    print("--------------------------------------------------")
    print("    面积       真实价格     预测价格       残差       残差平方")
    for x_val, y_true, yp, r, sq_r in zip(x_raw, y_raw, y_pred, residuals, sq_residuals):
        print(f"    {x_val:<8.1f}   {y_true:<8.1f}   {yp:<8.2f}   {r:<8.2f}   {sq_r:<8.2f}")

    print(f"\n残差平方和 SSE = {sse:.2f}")
    print(f"均方误差 MSE = {sse / n:.6f}")

    print("\n--------------------------------------------------")
    print("模型核心评估报告")
    print("--------------------------------------------------")
    print(f"预测数组 y_pred = {[round(p, 2) for p in y_pred]}")
    print(f"真实均值 y_mean = {y_mean:.6f} | 总平方和 SST = {sst:.6f} | 残差平方和 SSE = {sse:.6f}")

    print("\n【拟合优度】")
    print(f"  R² (决定系数) = {r_squared:.6f}")
    print("  解释: R² 越接近 1，说明模型对数据变化的解释能力越强、拟合越好。")

    print("\n【误差指标】")
    print(f"  RMSE (均方根误差) = {rmse:.4f}")
    print(f"  MAE  (平均绝对误差) = {mae:.4f}")
    print("  解释: RMSE 对离群大误差惩罚更重；MAE 衡量平均偏差大小，越小越好。")

    print("\n【残差分布图形化呈现】")
    print("     x值        残差                 可视化 (字符画)")
    for x_val, r in zip(x_raw, residuals):
        bar_len = int(abs(r) / 2)  # 缩放比例供展示
        if r >= 0:
            bar = "+" + "█" * bar_len
        else:
            bar = "-" + "░" * bar_len
        print(f"     {x_val:<8.1f}   {r:<6.2f} {bar}")

    # ------------------ 上线决策与阈值标准判定 ------------------
    # 判定标准：R² >= 0.90 且 RMSE <= 15.0
    r2_threshold = 0.90
    rmse_threshold = 15.0

    is_passed = (r_squared >= r2_threshold) and (rmse <= rmse_threshold)

    print("\n--------------------------------------------------")
    print("【步骤 6】部署决策 (Deployment Decision)")
    print("--------------------------------------------------")
    print("【教学说明】:")
    print("  对比评估指标与工程设定的硬性上线门槛 (Thresholds)，判定是否能部署至生产环境。")
    print(f"上线判定硬性标准: R² 必须 >= {r2_threshold} 且 RMSE 必须 <= {rmse_threshold}")
    print(f"当前测试指标结果: R² = {r_squared:.4f}, RMSE = {rmse:.4f}")

    if is_passed:
        print("\n  ✓【评估结论】: [通过 ✅]")
        print("  -> 执行动作: 满足上线标准，模型成功部署至生产环境 (Deploy)！")
    else:
        print("\n  ✕【评估结论】: [未通过 ❌]")
        print("  -> 执行动作: 未满足上线标准，触发【反馈循环】，拒绝部署！")
        print(f"  -> 未通过原因: ", end="")
        if r_squared < r2_threshold:
            print(f"R² ({r_squared:.4f}) 未达到标准 ({r2_threshold})；", end="")
        if rmse > rmse_threshold:
            print(f"RMSE ({rmse:.4f}) 超过允许上限 ({rmse_threshold})；", end="")
        print()


# 评估测试集 1 (预期通过)
evaluate_and_deploy(test_data_pass, w_final, b_final)

# 评估测试集 2 (预期不通过)
evaluate_and_deploy(test_data_fail, w_final, b_final)