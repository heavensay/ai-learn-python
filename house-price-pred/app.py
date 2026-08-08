# from fastapi import FastAPI
# from pipeline import load_pipeline
#
# app = FastAPI(title="Housing Price Prediction API")
# pipe = load_pipeline()
#
# @app.get("/")
# def read_root():
#     return {"status": "online", "model": pipe.config["model_type"]}
#
# @app.post("/predict")
# def predict(area: float):
#     price = pipe.predict(area)
#     return {"area_sqm": area, "predicted_price_wan": price}