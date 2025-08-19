"""
📌 What this does:

Uses your preprocessing pipeline

Trains a Linear Regression model

Prints evaluation metrics

Saves the trained model in ./models/student_marks_model.pkl
    
"""

import numpy as np
import os 
import pickle
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error
from preprocess import load_data, check_nulls, clean_data, get_feature_target

def train_model(file_path, model_save_path = './models/student_marks_model.pkl'):
    """
    Train a Linear Regression model to train the marks based on study hours.

    """
    # Step 1: Load and preprocess data
    df = load_data(file_path)
    check_nulls(df)
    df = clean_data(df , strategy='drop')
    X,y = get_feature_target(df)
    
    # Step 2: Split the data into training and testing sets
    X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    # Step 3 : Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train , y_train)
    
    # Step 4 : Evaluate the model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred) 
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    
    print("\n📊Model Evaluation Results")
    print(f"R-squared: {r2:.3f}")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    
    # Step 5: Save the trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved at {model_save_path}")

    return model
    
    # Example Usage
    
if __name__ == "__main__":
    file_path = "./data/score_updated.csv"
    model_path = "./models/student_marks_model.pkl"
    print("Training model...")
    print("This may take a few minutes depending on your system.")
    print("Please wait...")
    trained_model = train_model(file_path , model_path)
    
    
    