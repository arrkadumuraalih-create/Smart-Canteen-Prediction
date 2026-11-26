
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load dataset
data = pd.read_csv("canteen_data.csv")

print("Dataset Loaded Successfully!")
print(data.head())

# Features and target
X = data[
    ["food_item", "day", "students_present", "is_exam", "is_holiday"]
]

y = data["units_sold"]

# Preprocessing
categorical_features = ["food_item", "day"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

# Random Forest model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# Predictions
predictions = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = mse ** 0.5

r2 = r2_score(y_test, predictions)

print("\n------ MODEL EVALUATION ------")

print("Mean Absolute Error:", round(mae, 2))

print("Mean Squared Error:", round(mse, 2))

print("Root Mean Squared Error:", round(rmse, 2))

print("R2 Score:", round(r2, 2))

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel Saved Successfully!")

# Graph 1: Actual vs Predicted
plt.figure(figsize=(8, 5))

plt.plot(
    y_test.values,
    label="Actual Demand",
    marker="o"
)

plt.plot(
    predictions,
    label="Predicted Demand",
    marker="x"
)

plt.title("Actual vs Predicted Food Demand")
plt.xlabel("Test Records")
plt.ylabel("Number of Plates")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("static/prediction_graph.png")
plt.show()

# Graph 2: Food Item Demand
food_demand = data.groupby("food_item")["units_sold"].mean()

plt.figure(figsize=(8, 5))

food_demand.plot(kind="bar")

plt.title("Average Demand by Food Item")
plt.xlabel("Food Items")
plt.ylabel("Average Plates Sold")

plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()

plt.savefig("static/food_demand_graph.png")
plt.show()
