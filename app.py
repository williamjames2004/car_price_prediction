from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.pipeline import Pipeline


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "car_price_model.pkl"

FEATURES = [
    "symboling",
    "fueltype",
    "aspiration",
    "carbody",
    "drivewheel",
    "wheelbase",
    "carlength",
    "carwidth",
    "curbweight",
    "enginetype",
    "cylindernumber",
    "enginesize",
    "boreratio",
    "horsepower",
    "citympg",
    "highwaympg",
]

NUMERIC_FIELDS = [
    "symboling",
    "wheelbase",
    "carlength",
    "carwidth",
    "curbweight",
    "enginesize",
    "boreratio",
    "horsepower",
    "citympg",
    "highwaympg",
]

CATEGORICAL_OPTIONS = {
    "fueltype": ["gas", "diesel"],
    "aspiration": ["std", "turbo"],
    "carbody": [
        "convertible",
        "hardtop",
        "hatchback",
        "sedan",
        "wagon",
    ],
    "drivewheel": ["fwd", "rwd", "4wd"],
    "enginetype": [
        "dohc",
        "dohcv",
        "l",
        "ohc",
        "ohcf",
        "ohcv",
        "rotor",
    ],
    "cylindernumber": [
        "two",
        "three",
        "four",
        "five",
        "six",
        "eight",
        "twelve",
    ],
}

DEFAULT_VALUES = {
    "symboling": "0",
    "fueltype": "gas",
    "aspiration": "std",
    "carbody": "sedan",
    "drivewheel": "fwd",
    "wheelbase": "94.5",
    "carlength": "171.2",
    "carwidth": "65.5",
    "curbweight": "2500",
    "enginetype": "ohc",
    "cylindernumber": "four",
    "enginesize": "130",
    "boreratio": "3.47",
    "horsepower": "100",
    "citympg": "25",
    "highwaympg": "30",
}

model = None

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

    if not isinstance(model, Pipeline):
        raise TypeError(
            "The saved model must be the preprocessing and SVR Pipeline "
            "created by model.py."
        )


def get_form_values(form):
    values = DEFAULT_VALUES.copy()

    for field in FEATURES:
        submitted_value = form.get(field)

        if submitted_value is not None:
            values[field] = submitted_value.strip()

    return values


def validate_and_prepare(values):
    prepared = {}

    for field in NUMERIC_FIELDS:
        raw_value = values.get(field, "").strip()

        if not raw_value:
            raise ValueError(f"Please enter a value for {field}.")

        try:
            number = float(raw_value)
        except ValueError:
            raise ValueError(f"{field} must be a valid number.")

        if not pd.notna(number):
            raise ValueError(f"{field} must be a finite number.")

        if field != "symboling" and number < 0:
            raise ValueError(f"{field} cannot be negative.")

        if field == "symboling" and not number.is_integer():
            raise ValueError("Symboling must be a whole number.")

        prepared[field] = int(number) if field == "symboling" else number

    for field, allowed_values in CATEGORICAL_OPTIONS.items():
        value = values.get(field, "").strip().lower()

        if value not in allowed_values:
            raise ValueError(f"Please select a valid {field}.")

        prepared[field] = value

    return pd.DataFrame(
        [{field: prepared[field] for field in FEATURES}],
        columns=FEATURES,
    )


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        prediction=None,
        error=None,
        submitted=DEFAULT_VALUES,
    )


@app.route("/predict", methods=["POST"])
def predict():
    submitted = get_form_values(request.form)

    if model is None:
        return render_template(
            "index.html",
            prediction=None,
            error=(
                "The trained model file was not found. "
                "Run model.py to generate car_price_model.pkl, "
                "then restart the application."
            ),
            submitted=submitted,
        ), 500

    try:
        car_data = validate_and_prepare(submitted)
        predicted_price = float(model.predict(car_data)[0])

        if not pd.notna(predicted_price):
            raise ValueError("The model returned an invalid prediction.")

        return render_template(
            "index.html",
            prediction=predicted_price,
            error=None,
            submitted=submitted,
        )

    except (ValueError, TypeError) as exc:
        return render_template(
            "index.html",
            prediction=None,
            error=str(exc),
            submitted=submitted,
        ), 400

    except Exception:
        app.logger.exception("Car price prediction failed.")

        return render_template(
            "index.html",
            prediction=None,
            error=(
                "Prediction failed. Please check your input values "
                "and verify that the saved model matches model.py."
            ),
            submitted=submitted,
        ), 500


if __name__ == "__main__":
    app.run(debug=True)