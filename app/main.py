from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

class WorkoutInput(BaseModel):
    age: int = Field(..., ge=16, le=100)
    bmi: float = Field(..., ge=10, le=60)
    experience_years: float = Field(..., ge=0, le=50)
    weight_lifted_kg: float = Field(..., ge=1, le=500)
    sets: int = Field(..., ge=1, le=20)
    reps: int = Field(..., ge=1, le=100)
    rest_days: int = Field(..., ge=0, le=14)
    sleep_hours: float = Field(..., ge=0, le=24)
    warm_up_done: int = Field(..., ge=0, le=1)
    past_injury: int = Field(..., ge=0, le=1)

class PredictionOutput(BaseModel):
    injury_risk_probability: float
    risk_level: str
    recommendation: str

def get_risk_level(prob):
    if prob < 0.3: return "LOW"
    elif prob < 0.6: return "MEDIUM"
    else: return "HIGH"

def get_recommendation(risk, data):
    if risk == "LOW": return "Safe to train!"
    elif risk == "MEDIUM": return "Moderate risk. Reduce weight and warm up."
    tips = []
    if data.weight_lifted_kg > 100: tips.append("reduce weight")
    if data.rest_days < 2: tips.append("rest 2 more days")
    if data.warm_up_done == 0: tips.append("do a warm-up")
    if data.sleep_hours < 7: tips.append("sleep more")
    if data.past_injury == 1: tips.append("see a physio")
    return "HIGH RISK! Please: " + ", ".join(tips) + "."

@app.get("/")
def home(): return {"message": "API running! Visit /docs"}

@app.get("/health")
def health(): return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: WorkoutInput):
    f = np.array([[data.age, data.bmi, data.experience_years, data.weight_lifted_kg, data.sets, data.reps, data.rest_days, data.sleep_hours, data.warm_up_done, data.past_injury]])
    fs = scaler.transform(f)
    prob = float(model.predict_proba(fs)[0][1])
    risk = get_risk_level(prob)
    return PredictionOutput(injury_risk_probability=round(prob,3), risk_level=risk, recommendation=get_recommendation(risk, data))