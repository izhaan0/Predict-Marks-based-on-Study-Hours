"""  
📌 How it works:

load_model() → loads the trained Linear Regression model from models/linear_regression_model.pkl.

predict_score() → takes study hours as input, reshapes them, and predicts marks.

Main script → asks the user for study hours and prints the predicted marks.
"""

import pickle
import pandas as pd
import numpy as np
import os
from joblib import load

def load_model(model_path="./models/student_marks_model.pkl"):
    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found at {model_path}")
        return None
    try:
        model = load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict_score(hours , model):
    """  
    
        Predict the marks based on study hours using the loaded model.
                
    """

    hours_df = pd.DataFrame([[hours]], columns=["Hours"])
    predicted_score = model.predict(hours_df)
    return predicted_score

# Example usage
if __name__ == "__main__":
    
    model = load_model()
    if model is not None:
        try:
            hours = float(input("Enter the number of study hours: "))
            predicted_marks = predict_score(hours, model)
            print(f"\n📘 Predicted Marks for {hours} hours of study: {predicted_marks[0]:.2f}")
        except ValueError:
            print("⚠️ Invalid input. Please enter a valid number for study hours.")
    