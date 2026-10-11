
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("data.csv")

# Selected input features
features = [
    "symboling", "fueltype", "aspiration", "carbody",
    "drivewheel", "wheelbase", "carlength", "carwidth",
    "curbweight", "enginetype", "cylindernumber",
    "enginesize", "boreratio", "horsepower",
    "citympg", "highwaympg"
]

target = "price"

# Prepare data
X = data[features]
y = pd.to_numeric(data[target], errors="coerce")

# Remove rows with missing target values
valid_rows = y.notna()
X = X[valid_rows]
y = y[valid_rows]

# Numerical and categorical columns
numerical_features = [
    "symboling", "wheelbase", "carlength", "carwidth",
    "curbweight", "enginesize", "boreratio", "horsepower",
    "citympg", "highwaympg"
]

categorical_features = [
    "fueltype", "aspiration", "carbody",
    "drivewheel", "enginetype", "cylindernumber"
]

# Preprocessing pipelines
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# SVR model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", SVR(kernel="rbf", C=100, epsilon=0.1))
])

# Evaluate if enough data is available
if len(X) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Model Evaluation")
    print("MAE:", round(mean_absolute_error(y_test, predictions), 2))

    if y_test.nunique() > 1:
        print("R2 Score:", round(r2_score(y_test, predictions), 4))
else:
    print("Dataset is too small for reliable evaluation.")

# Train final model on all available data
model.fit(X, y)

# Save trained model
joblib.dump(model, "car_price_model.pkl")

print("Car price model trained and saved successfully!")