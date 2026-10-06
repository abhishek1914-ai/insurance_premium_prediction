# Insurance Premium Prediction

A Django web app that predicts an individual's insurance premium using a
Random Forest regression model trained on customer demographic and health
data.

## Tech Stack
- Python, Django
- scikit-learn (RandomForestRegressor, preprocessing: encoding, scaling)
- pandas, NumPy

## How it works
1. `ml_model/train_model.py` generates a synthetic customer dataset (age,
   BMI, number of children, smoker status, region, sex) with a premium
   target that depends realistically on those factors (smokers, higher BMI,
   and older age drive premiums up). It applies cleaning, encoding
   (categoricals -> numeric), and normalization, then trains a
   **RandomForestRegressor**, achieving an **R² score of ~0.87** on the
   held-out test set. The trained model + encoders are saved to
   `ml_model/premium_model.pkl`.
2. The Django app (`predictor`) exposes a form for a customer's details,
   loads the saved model, and displays the predicted annual premium.

## Setup
```bash
pip install -r requirements.txt

# Train the model (creates premium_model.pkl)
python ml_model/train_model.py

# Run Django migrations and start the server
python manage.py migrate
python manage.py runserver
```

Visit http://127.0.0.1:8000/ to enter customer details and get a predicted
premium.
# insurance_premium_prediction
