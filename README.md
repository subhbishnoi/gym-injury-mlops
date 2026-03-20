# Gym Injury Risk Prediction System

## Problem Statement
Millions of gym-goers get injured every year due to excessive load and poor recovery. This project predicts injury risk BEFORE it happens.

## How It Works
- User enters workout details
- ML model analyzes the data
- Returns LOW / MEDIUM / HIGH risk
- Gives safety recommendations

## Tech Stack
- ML Model: Random Forest (scikit-learn)
- API: FastAPI
- Testing: pytest
- Container: Docker
- CI/CD: GitHub Actions

## Run The Project

pip install fastapi uvicorn scikit-learn joblib numpy pandas

python model/train_model.py

uvicorn app.main:app --reload

## Example
Input: age=21, weight=130kg, rest_days=0, no warmup

Output: HIGH RISK - reduce weight, rest 2 days, do warmup

## Future Work
- Wearable device integration
- Real-time prediction
- Deep learning model