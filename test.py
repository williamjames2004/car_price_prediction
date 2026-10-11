
import pandas as pd
import joblib

# Load trained model
model = joblib.load("car_price_model.pkl")

# Get car details
symboling = int(input("Symboling: "))
fueltype = input("Fuel Type (gas/diesel): ").strip().lower()
aspiration = input("Aspiration (std/turbo): ").strip().lower()
carbody = input("Car Body: ").strip().lower()
drivewheel = input("Drive Wheel (fwd/rwd/4wd): ").strip().lower()

wheelbase = float(input("Wheelbase: "))
carlength = float(input("Car Length: "))
carwidth = float(input("Car Width: "))
curbweight = float(input("Curb Weight: "))

enginetype = input("Engine Type: ").strip().lower()
cylindernumber = input("Cylinder Number: ").strip().lower()

enginesize = float(input("Engine Size: "))
boreratio = float(input("Bore Ratio: "))
horsepower = float(input("Horsepower: "))
citympg = float(input("City MPG: "))
highwaympg = float(input("Highway MPG: "))

# Prepare input data
car = pd.DataFrame([{
    "symboling": symboling,
    "fueltype": fueltype,
    "aspiration": aspiration,
    "carbody": carbody,
    "drivewheel": drivewheel,
    "wheelbase": wheelbase,
    "carlength": carlength,
    "carwidth": carwidth,
    "curbweight": curbweight,
    "enginetype": enginetype,
    "cylindernumber": cylindernumber,
    "enginesize": enginesize,
    "boreratio": boreratio,
    "horsepower": horsepower,
    "citympg": citympg,
    "highwaympg": highwaympg
}])

# Predict car price
prediction = model.predict(car)[0]

print("\nPredicted Car Price:", round(prediction, 2))