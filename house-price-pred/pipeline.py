import json
import pickle
import os

class HousingPricePipeline:
    """开源模型的推理 SDK 封装"""
    def __init__(self, model_dir="./"):
        config_path = os.path.join(model_dir, "config.json")
        weights_path = os.path.join(model_dir, "model.bin")

        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # 加载权重
        with open(weights_path, "rb") as f:
            weights = pickle.load(f)
            self.w = weights["w"]
            self.b = weights["b"]

    def predict(self, area_sqm: float) -> float:
        """输入房屋面积(㎡)，返回预测价格(万元)"""
        if area_sqm <= 0:
            raise ValueError("房屋面积必须大于0")

        # 前向传播推断
        price = self.w * area_sqm + self.b
        return round(price, 2)

# 便捷加载函数
def load_pipeline(model_dir="./"):
    return HousingPricePipeline(model_dir)

# 本地测试 SDK
if __name__ == "__main__":
    pipe = load_pipeline()
    print("测试 75 ㎡ 房屋预测价格:", pipe.predict(75.0), "万元")