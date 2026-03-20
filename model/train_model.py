import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

print("Starting model training...")

np.random.seed(42)
n = 1000

data = {
    "age":              np.random.randint(18, 60, n),
    "bmi":              np.random.uniform(18, 35, n),
    "experience_years": np.random.uniform(0, 10, n),
    "weight_lifted_kg": np.random.uniform(20, 150, n),
    "sets":             np.random.randint(1, 6, n),
    "reps":             np.random.randint(5, 20, n),
    "rest_days":        np.random.randint(0, 7, n),
    "sleep_hours":      np.random.uniform(4, 9, n),
    "warm_up_done":     np.random.randint(0, 2, n),
    "past_injury":      np.random.randint(0, 2, n),
}

df = pd.DataFrame(data)

df["injury"] = (
    (df["weight_lifted_kg"] > 100) &
    (df["rest_days"] < 2) &
    (df["warm_up_done"] == 0)
).astype(int)

noise_idx = np.random.choice(n, size=80, replace=False)
df.loc[noise_idx, "injury"] = 1 - df.loc[noise_idx, "injury"]

print(f"Total sessions : {n}")
print(f"Injury cases   : {df['injury'].sum()}")
print(f"Safe cases     : {(df['injury']==0).sum()}")

X = df.drop("injury", axis=1)
y = df["injury"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("\nModel Performance:")
print(classification_report(y_test, y_pred))

os.makedirs("model", exist_ok=True)
joblib.dump(model,  "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ model.pkl  saved!")
print("✅ scaler.pkl saved!")
print("\nML part DONE. Ready for MLOps!")