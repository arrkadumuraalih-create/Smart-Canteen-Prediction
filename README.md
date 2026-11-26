
# Smart Canteen - AI-Based Food Demand Prediction System

## 1. Project Overview

Smart Canteen is a machine learning-based web application
that predicts food demand in a college canteen.

The system helps canteen management estimate the required
food quantity based on student attendance, food item,
day of the week, examination days and holidays.

It aims to support better food planning and reduce
unnecessary food preparation.

## 2. Problem Statement

Traditional canteens often prepare food without accurately
estimating daily demand. This can lead to excess food
preparation or food shortages.

This project uses machine learning to estimate food demand
and support better preparation decisions.

## 3. Objectives

- Predict daily food demand.
- Estimate required food quantity.
- Calculate estimated surplus and possible shortage.
- Analyze food item demand.
- Maintain prediction history.
- Download prediction records.
- Support food waste reduction.

## 4. Technologies Used

- Python
- Flask
- HTML
- CSS
- Pandas
- Scikit-learn
- Matplotlib
- Random Forest Regression

## 5. Machine Learning Algorithm

Random Forest Regression is used to predict food demand.

The model uses the following input features:

- Food item
- Day
- Number of students present
- Examination day
- Holiday

The target variable is units_sold.

## 6. Main Features

1. Food Demand Prediction
2. Dashboard Statistics
3. Food Recommendation
4. Actual vs Predicted Graph
5. Food Item Demand Analysis
6. Surplus and Shortage Calculation
7. Prediction History
8. CSV Download

## 7. Project Structure

Smart_Canteen_Prediction/
- app.py
- train_model.py
- canteen_data.csv
- model.pkl
- prediction_history.csv
- templates/index.html
- static/style.css
- static/prediction_graph.png
- static/food_demand_graph.png

## 8. Expected Outcome

The application provides estimated food demand,
dashboard statistics and visual analysis to help
canteen management make informed food preparation decisions.

## 9. Future Enhancements

- Live student attendance integration
- Real-time canteen sales tracking
- Advanced demand forecasting
- Administrator login
- Database integration
- Online deployment

## 10. Conclusion

The Smart Canteen project demonstrates how machine
learning can support food demand prediction and
improve canteen planning.
