"""
📌 Student Marks Predictor Web App with Visualization
"""

import streamlit as st
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sys
import os

# Add src/ folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from predict import load_model, predict_score

# Streamlit UI
st.set_page_config(page_title="📘 Student Marks Predictor", layout="centered")
st.title("📘 Students Marks Predictor")
st.write("Predict your marks based on your study hours")

# Load the trained model
model = load_model()
if model is not None:
    # User Input 
    hours = st.slider("Select Study Hours", min_value=0.0, max_value=10.0, step=0.5)
    
    if st.button("Predict Marks"):
        predicted_marks = predict_score(hours, model)
        st.success(f"🎯 Predicted Marks for {hours} hours of study: {predicted_marks[0]:.2f}")
        
        # Plot regression line 
        x_line = np.linspace(0, 10, 100)
        y_line = [predict_score(h, model)[0] for h in x_line]
        
        plt.figure(figsize=(8,5))
        plt.plot(x_line, y_line, color='red', label='Regression Line')
        plt.scatter(hours, predicted_marks, color='blue', s=100, label='Your Input')
        plt.xlabel('Study Hours')
        plt.ylabel('Predicted Marks')
        plt.title('Study Hours vs Predicted Marks')
        plt.legend()
        st.pyplot(plt)
else:
    st.error("⚠️ Model could not be Loaded. Please train")