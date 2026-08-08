# 🏡 Housing Price Linear Model (Toy Demo)

这是一个基于纯 Python 实现的一元线性回归房价预测开源模型。

## 🚀 快速开始 (Quickstart)

```python
from pipeline import load_pipeline

# 1. 加载模型
pipe = load_pipeline()

# 2. 预测 80 平方米房屋价格
price = pipe.predict(80.0)
print(f"预测价格: {price} 万元")
```

## 介绍

~~~
├── config.json              # [配置] 模型结构、版本与评价指标
├── model.bin                # [权重] 训练出来的 w 与 b 序列化二进制文件
├── pipeline.py              # [SDK] 供使用者一行代码调用的 API 接口
├── train_and_save.py        # [源码] 训练与导出权重的完整可复现脚本
├── app.py                   # [服务] 基于 FastAPI 的在线 RESTful 接口
├── draw                     # [绘图] 此项目中用到的相关图形
└── README.md                # [文档] 包含模型说明、Benchmark 和 QuickStart 示例
~~~