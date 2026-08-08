# ==================== 1. 准备数据 ====================
x = [50, 60, 70, 80, 90]
y = [150, 180, 200, 230, 260]
n = len(x)

# ==================== 2. 定义损失函数 J(w, b) ====================
def compute_loss(w, b, x, y):
    """
    计算均方误差损失 J(w, b)
    J(w, b) = (1/n) * Σ(y_i - (w*x_i + b))²
    """
    total_loss = 0.0
    for i in range(n):
        prediction = w * x[i] + b
        error = y[i] - prediction
        total_loss += error ** 2
    return total_loss / n  # ✅ 补上除以 n，变为标准 MSE

# ==================== 3. 定义梯度计算函数 ====================
def compute_gradients(w, b, x, y):
    """
    计算 J(w, b) 对 w 和 b 的偏导数
    ∂J/∂w = (-2/n) * Σ((y_i - (w*x_i + b)) * x_i)
    ∂J/∂b = (-2/n) * Σ(y_i - (w*x_i + b))
    """
    dw = 0.0
    db = 0.0
    for i in range(n):
        prediction = w * x[i] + b
        error = y[i] - prediction
        dw += error * x[i]
        db += error
    dw = (-2.0 / n) * dw
    db = (-2.0 / n) * db
    return dw, db

# ==================== 4. 梯度下降优化 ====================
# 初始化：lr学习率，epochs迭代次数
def gradient_descent(x, y, lr=0.0001, epochs=200000, tol=1e-8):
    """
    使用梯度下降法最小化 J(w, b)
    """
    w = 0.0
    b = 0.0
    prev_loss = float('inf')

    for epoch in range(epochs):
        loss = compute_loss(w, b, x, y)

        if abs(prev_loss - loss) < tol:
            print(f"✅ 第 {epoch} 轮收敛")
            break
        prev_loss = loss

        dw, db = compute_gradients(w, b, x, y)
        w -= lr * dw
        b -= lr * db

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1:5d} | Loss: {loss:.6f} | w: {w:.4f} | b: {b:.4f}")

    return w, b, loss

# ==================== 5. 执行训练 ====================
print("=" * 50)
print("纯 Python 实现最小二乘法（梯度下降）")
print("=" * 50)
w_final, b_final, final_loss = gradient_descent(x, y)

print("\n" + "=" * 50)
print("训练结果")
print("=" * 50)
print(f"最优权重 w = {w_final:.4f}")
print(f"最优偏差 b = {b_final:.4f}")
print(f"最终损失 J(w,b) = {final_loss:.6f}")
print(f"拟合直线: y = {w_final:.4f}x + ({b_final:.4f})")

# ==================== 6. 验证：计算每个点的预测值和残差 ====================
print("\n" + "=" * 50)
print("预测验证")
print("=" * 50)
print(f"{'面积':>6} {'真实价格':>8} {'预测价格':>8} {'残差':>8} {'残差平方':>10}")
total_residual_sq = 0.0
for i in range(n):
    pred = w_final * x[i] + b_final
    residual = y[i] - pred
    residual_sq = residual ** 2
    total_residual_sq += residual_sq
    print(f"{x[i]:>6} {y[i]:>8} {pred:>8.2f} {residual:>8.2f} {residual_sq:>10.2f}")

print(f"\n残差平方和 = {total_residual_sq:.2f}")
print(f"均方误差 J(w,b) = {total_residual_sq / n:.6f}")