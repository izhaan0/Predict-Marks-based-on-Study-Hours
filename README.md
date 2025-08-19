📘 Student Marks Predictor

📌 Project Overview

This project predicts **student marks based on the number of study hours** using a **Linear Regression model**.
It’s a beginner-friendly **Machine Learning regression problem** that demonstrates:

* Data preprocessing
* Model training & evaluation
* Deployment with Streamlit (optional)

---

🚀 Features

* Predict marks (%) by entering study hours
* Visualization of study hours vs marks
* Linear Regression model implementation
* Simple and interactive **web app** using Streamlit

---

🛠 Tech Stack

Language: Python 🐍
  Libraries:

  * `numpy` – numerical operations
  * `pandas` – dataset handling
  * `matplotlib`, `seaborn` – data visualization
  * `scikit-learn` – Linear Regression & evaluation
  * `joblib` / `pickle` – model saving
  * `streamlit` – web app (for deployment)

---

📂 Project Structure

```
student-marks-predictor/
│
├── data/
│   └── student_scores.csv         # Dataset
│
├── notebooks/
│   └── EDA.ipynb                  # Exploratory data analysis
│
├── src/
│   ├── preprocess.py              # Data preprocessing
│   ├── train_model.py             # Train Linear Regression model and evaluation of model
│   └── predict.py                 # Prediction function
│
├── app/
│   └── app.py                     # Streamlit web application
│
├── models/
│   └── linear_regression.pkl      # Saved trained model
│
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```


📊 Workflow

1. Load Data → `pandas`
2. EDA → Visualize Hours vs Marks
3. Train Model → `LinearRegression` from scikit-learn
4. Evaluate Model → MAE, MSE, R² Score
5. Save Model → `joblib`
6. Deploy App → User enters hours → Predict marks



🔮 Example Prediction

Input: `5 hours of study`
Output: `Predicted Marks ≈ 55%`



▶️ How to Run

🔧 1. Clone Repo

```bash
git clone https://github.com/your-username/student-marks-predictor.git
cd student-marks-predictor
```

📦 2. Install Dependencies

```bash
pip install -r requirements.txt
```

🏋️ 3. Train Model

```bash
python src/train_model.py
```

🌐 4. Run Web App

```bash
streamlit run app/app.py
```



📈 Results

* Regression line fitting study hours vs marks
* R² Score close to **1.0** for small datasets
* Predictions within a reasonable error margin


🤝 Contributing

Contributions are welcome!

* Fork the repo
* Create a new branch
* Make changes and commit
* Submit a pull request


📜 License

This project is licensed under the **MIT License**.
