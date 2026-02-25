import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("car_data.csv")

# One-hot encode categorical columns
data = pd.get_dummies(
    data,
    columns=["Car_Name", "Fuel_Type", "Transmission", "Owner_Type"]
)

# Features and target
X = data.drop("Price", axis=1)
y = data["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save as dictionary (IMPORTANT)
model_data = {
    "model": model,
    "columns": X.columns
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("Model trained and saved correctly!")
