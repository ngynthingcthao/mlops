from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# --- Load model ---
model = joblib.load(r"./best_model.pkl")

# --- Định nghĩa Pydantic BaseModel cho input ---
class AutismInput(BaseModel):
    A1_Score: int
    A2_Score: int
    A3_Score: int
    A4_Score: int
    A5_Score: int
    A6_Score: int
    A7_Score: int
    A8_Score: int
    A9_Score: int
    A10_Score: int
    age: float
    gender: int             
    ethnicity: int          
    jaundice: int           
    autism: int             
    country_of_res: int     
    used_app_before: int    
    relation: int           

# --- Khởi tạo app ---
app = FastAPI(title="Autism Prediction API")

# --- Endpoint gốc / ---
@app.get("/")
def read_root():
    return {
        "project": "Autism Prediction API",
        "description": "API dự đoán nguy cơ tự kỷ",
        "author": "Thao"
    }

# --- Endpoint /predict ---
@app.post("/predict")
def predict(data: AutismInput):
    features = np.array([
        data.A1_Score, data.A2_Score, data.A3_Score, data.A4_Score, data.A5_Score,
        data.A6_Score, data.A7_Score, data.A8_Score, data.A9_Score, data.A10_Score,
        data.age, data.gender, data.ethnicity, data.jaundice, data.autism,
        data.country_of_res, data.used_app_before, data.relation
    ], dtype=float).reshape(1, -1)

    # Dự đoán
    pred = model.predict(features)[0]
    try:
        proba = model.predict_proba(features)[:,1][0]
    except Exception:
        proba = None

    meaning = "Có nguy cơ tự kỷ" if pred == 1 else "Không có nguy cơ tự kỷ"
    response = {"prediction": meaning}
    return response
