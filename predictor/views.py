import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render

from .forms import CustomerForm

_artifact = None


def _load_artifact():
    global _artifact
    if _artifact is None:
        import joblib

        model_path = os.path.join(settings.ML_MODEL_DIR, "premium_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model not found. Run `python ml_model/train_model.py` first."
            )
        _artifact = joblib.load(model_path)
    return _artifact


def predict_view(request):
    prediction = None
    error = None

    if request.method == "POST":
        form = CustomerForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            try:
                artifact = _load_artifact()
                model = artifact["model"]
                encoders = artifact["encoders"]
                scaler = artifact["scaler"]
                feature_order = artifact["feature_order"]
                numeric_cols = artifact["numeric_cols"]

                row = {
                    "age": data["age"],
                    "sex": encoders["sex"].transform([data["sex"]])[0],
                    "bmi": data["bmi"],
                    "children": data["children"],
                    "smoker": encoders["smoker"].transform([data["smoker"]])[0],
                    "region": encoders["region"].transform([data["region"]])[0],
                }
                df_row = pd.DataFrame([row])[feature_order]
                df_row.loc[:, numeric_cols] = scaler.transform(df_row[numeric_cols])

                pred = model.predict(df_row)[0]
                prediction = round(float(pred), 2)
            except FileNotFoundError as e:
                error = str(e)
    else:
        form = CustomerForm()

    return render(
        request,
        "predictor/predict.html",
        {"form": form, "prediction": prediction, "error": error},
    )
