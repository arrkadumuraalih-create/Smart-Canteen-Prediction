
from flask import Flask, render_template, request, send_file
import pickle
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# Project file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
DATA_FILE = os.path.join(BASE_DIR, "canteen_data.csv")
HISTORY_FILE = os.path.join(BASE_DIR, "prediction_history.csv")

# Load trained model
with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

# Load dataset
data = pd.read_csv(DATA_FILE)

# Dashboard statistics
total_records = len(data)
average_demand = round(data["units_sold"].mean())
highest_demand = int(data["units_sold"].max())
lowest_demand = int(data["units_sold"].min())

# Food recommendation
food_demand = data.groupby("food_item")["units_sold"].mean()
recommended_food = food_demand.idxmax()
recommended_demand = round(food_demand.max())


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    surplus = 0
    shortage = 0
    prepared_quantity = None
    error_message = None

    if request.method == "POST":
        try:
            food_item = request.form["food_item"]
            day = request.form["day"]
            students_present = int(request.form["students_present"])
            is_exam = int(request.form["is_exam"])
            is_holiday = int(request.form["is_holiday"])
            prepared_quantity = int(request.form["prepared_quantity"])

            if students_present < 0 or prepared_quantity < 0:
                raise ValueError

            # Prepare input for prediction
            input_data = pd.DataFrame([{
                "food_item": food_item,
                "day": day,
                "students_present": students_present,
                "is_exam": is_exam,
                "is_holiday": is_holiday
            }])

            # Predict food demand
            result = model.predict(input_data)[0]
            prediction = max(0, round(result))

            # Calculate estimated surplus and shortage
            surplus = max(0, prepared_quantity - prediction)
            shortage = max(0, prediction - prepared_quantity)

            # Save prediction history
            new_record = pd.DataFrame([{
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "food_item": food_item,
                "day": day,
                "students_present": students_present,
                "predicted_demand": prediction,
                "prepared_quantity": prepared_quantity,
                "surplus": surplus,
                "shortage": shortage
            }])

            new_record.to_csv(
                HISTORY_FILE,
                mode="a",
                header=not os.path.exists(HISTORY_FILE),
                index=False
            )

        except (ValueError, KeyError):
            error_message = "Please enter valid values. Numbers cannot be negative."

    # Load recent prediction history
    recent_history = []

    if os.path.exists(HISTORY_FILE):
        history_data = pd.read_csv(HISTORY_FILE)
        recent_history = (
            history_data.tail(10)
            .iloc[::-1]
            .to_dict("records")
        )

    return render_template(
        "index.html",
        prediction=prediction,
        surplus=surplus,
        shortage=shortage,
        prepared_quantity=prepared_quantity,
        total_records=total_records,
        average_demand=average_demand,
        highest_demand=highest_demand,
        lowest_demand=lowest_demand,
        recommended_food=recommended_food,
        recommended_demand=recommended_demand,
        recent_history=recent_history,
        error_message=error_message
    )


# Download prediction history
@app.route("/download")
def download_history():

    if os.path.exists(HISTORY_FILE):
        return send_file(
            HISTORY_FILE,
            as_attachment=True,
            download_name="canteen_prediction_history.csv"
        )

    return "No prediction history available yet. Make a prediction first."


if __name__ == "__main__":
    app.run(debug=True)
