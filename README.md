# Gym Injury Risk Prediction System

## Problem Statement

Every year, over 500,000 gym-related injuries are reported. Most injuries are NOT accidents — they are predictable outcomes of measurable factors like excessive load, poor recovery, and skipping warmup.

Current injury prevention depends on personal trainers who are:
- Expensive and not affordable for everyone
- Not available 24/7
- Inconsistent in their advice
- Absent for most solo gym users

There is NO automated data-driven system that warns gym users before injury occurs.

## Our Solution

We built an ML-powered injury risk prediction system that:
- Analyzes 10 workout and recovery features
- Predicts injury risk BEFORE the session starts
- Returns LOW / MEDIUM / HIGH risk level
- Gives specific safety recommendations
- Is available 24/7 via REST API
- Requires no personal trainer

## Real World Impact
- Gym owners: reduced liability and member injuries
- Trainers: data-driven insights for client programs
- Users: affordable 24/7 personalized safety guidance

## Tech Stack
- ML Model: Random Forest Classifier (scikit-learn)
- API: FastAPI
- Testing: pytest
- Container: Docker
- CI/CD: GitHub Actions
- Deployment: Render Cloud

## Live Demo
https://gym-injury-mlops.onrender.com/docs

## GitHub
https://github.com/subhbishnoi/gym-injury-mlops

## How To Run Locally

git clone https://github.com/subhbishnoi/gym-injury-mlops.git
cd gym-injury-mlops
pip install fastapi uvicorn scikit-learn joblib numpy pandas
python model/train_model.py
uvicorn app.main:app --reload

## Example Prediction

Input: age=19, weight=140kg, rest=0 days, no warmup

Output:
- injury_risk_probability: 0.84
- risk_level: HIGH
- recommendation: Reduce weight, rest 2 more days, do warmup

## Dataset
- 1000 synthetic gym sessions
- 10 features per session
- Injury label based on domain knowledge
- 872 safe cases, 128 injury cases

## Model Performance
- Accuracy: 88%
- Precision: 86%
- Recall: 20% (improved via threshold tuning)
- Solution: class_weight=balanced + threshold=0.4

## Future Improvements
- Real gym injury dataset
- Wearable device integration
- Real-time prediction during workout
- Deep learning model for better accuracy
- SMOTE for better class balance handling