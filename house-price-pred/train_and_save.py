import json
import pickle

# 1. 假定经过训练得到的真实量纲参数 (y = w * x + b)
w_real = 2.6953
b_real = 15.2280

# 2. 序列化导出权重文件
weights = {"w": w_real, "b": b_real}

with open("model.bin", "wb") as f:
    pickle.dump(weights, f)

print("✅ 模型权重已导出为: model.bin")