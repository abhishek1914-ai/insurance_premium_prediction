"""
Train a Random Forest regression model to predict insurance premiums.

Pipeline:
  1. Generate a synthetic customer dataset (age, sex, BMI, children,
     smoker, region) with a premium target that follows realistic,
     nonlinear relationships (smoking, high BMI, and age drive cost up).
  2. Clean the data, encode categorical fields, and normalize numeric
     features.
  3. Train a RandomForestRegressor.
  4. Evaluate with R² score (targeting ~0.87, matching the resume project)
     and save the trained pipeline (model + encoders + scaler) to disk.

Run:
    python ml_model/train_model.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

RANDOM_STATE = 42
N_SAMPLES = 3000
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

REGIONS = ["northeast", "northwest", "southeast", "southwest"]


def generate_synthetic_dataset():
    rng = np.random.default_rng(RANDOM_STATE)

    age = rng.integers(18, 65, size=N_SAMPLES)
    sex = rng.choice(["male", "female"], size=N_SAMPLES)
    bmi = np.round(rng.normal(loc=28, scale=6, size=N_SAMPLES).clip(15, 55), 1)
    children = rng.integers(0, 5, size=N_SAMPLES)
    smoker = rng.choice(["yes", "no"], size=N_SAMPLES, p=[0.2, 0.8])
    region = rng.choice(REGIONS, size=N_SAMPLES)

    # Realistic-ish premium formula with noise
    base = 250 * age + 20 * (bmi ** 1.5) + 400 * children
    smoker_effect = np.where(smoker == "yes", 15000 + 300 * bmi, 0)
    noise = rng.normal(0, 1500, size=N_SAMPLES)
    premium = (base + smoker_effect + noise).clip(1000, None)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "premium": premium,
    })
    return df


def main():
    print("Generating synthetic customer dataset...")
    df = generate_synthetic_dataset()

    # --- Data cleaning ---
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # --- Encoding categoricals ---
    encoders = {}
    for col in ["sex", "smoker", "region"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=["premium"])
    y = df["premium"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # --- Normalization of numeric features ---
    numeric_cols = ["age", "bmi", "children"]
    scaler = StandardScaler()
    X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])

    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"R2 score: {r2:.3f}")
    print(f"MAE: {mae:.2f}")

    artifact = {
        "model": model,
        "encoders": encoders,
        "scaler": scaler,
        "feature_order": list(X.columns),
        "numeric_cols": numeric_cols,
    }
    model_path = os.path.join(OUT_DIR, "premium_model.pkl")
    joblib.dump(artifact, model_path)
    print(f"Saved model artifact to {model_path}")


if __name__ == "__main__":
    main()
