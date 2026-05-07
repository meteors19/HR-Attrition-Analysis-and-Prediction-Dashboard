### HR-Attrition-Analysis-and-Prediction-Dashboard
# 📋 Overview
HR Attrition Analysis & Prediction is a Machine Learning-powered web application designed to predict employee attrition and help HR teams make data-driven retention decisions. The system analyzes the IBM HR Analytics dataset, identifies key attrition drivers through EDA, trains and compares multiple ML models, and deploys an interactive Streamlit dashboard with real-time employee risk scoring. This project helps HR professionals proactively identify high-risk employees and take timely action to reduce turnover costs.

# 📁 Dataset
The project uses the IBM HR Analytics Employee Attrition & Performance dataset:

Source: Kaggle – IBM HR Analytics Dataset
Records: 1,470 employees
Features: 35 columns including demographics, job details, satisfaction scores, and compensation

Key Columns:

Attrition — Target variable (Yes/No)
Department — Employee's department
OverTime — Whether employee works overtime
JobSatisfaction — Satisfaction rating (1–4)
MonthlyIncome — Monthly salary
YearsAtCompany — Tenure in years


# 🧠 Features

Resume Upload & Employee Data Input
Exploratory Data Analysis (EDA) with 9 visualizations
Multi-model Training & Comparison
Real-time Attrition Risk Scoring
Interactive Prediction Dashboard
Feature Importance Analysis
HR Recommendations Engine
Confusion Matrix & ROC Curve Evaluation


# 📊 Technologies Used
Programming & Frameworks

  1) Python 3.x
  2) Streamlit

Libraries

  1) Pandas
  2) NumPy
  3) Scikit-learn
  4) Matplotlib
  5) Seaborn
  6) Plotly
  7) Pickle

Machine Learning

  1) Random Forest Classifier
  2) Gradient Boosting Classifier
  3) Logistic Regression

Visualization

  1) Matplotlib
  2) Seaborn
  3) Plotly (Interactive Charts)


# 📈 Machine Learning & Analysis Techniques
The project uses:

Exploratory Data Analysis (EDA)
Label Encoding for Categorical Variables
Standard Scaling for Numerical Features
Train-Test Split (80/20) with Stratification
Cross-Validation (5-Fold)
Feature Importance Ranking
ROC-AUC Scoring
Confusion Matrix Analysis


# 📊 Exploratory Analysis
Performed using Pandas, Matplotlib, and Seaborn.
Key Insights:

1) Employees working overtime are 3x more likely to leave
2) Sales department has the highest attrition rate among all departments
3) Low job satisfaction (rating 1–2) significantly increases attrition risk
4) Younger employees (age 25–35) show the highest attrition concentration
5) Lower monthly income is strongly correlated with higher attrition


# 📈 Performance
Model                  Accuracy     ROC-AUC          CV Accuracy
Random Forest           ~84.35%        ~0.789       86.14% ± 0.92%
Gradient Boosting       ~84.01%        ~0.7969      85.38% ± 1.98%
Logistic Regression     ~87.41%        ~0.8057      86.91% ± 1.03%

⚠️ Note: Your actual numbers will differ — replace the above with the real output printed by hr_attrition_full.py when you run it.


# 🧪 Tools & Libraries

1) Python
2) Streamlit
3) Scikit-learn
4) Pandas
5) NumPy
6) Matplotlib
7) Seaborn
8) Plotly
9) Pickle


# 📊 Dashboard Features
The interactive Streamlit dashboard includes:

Tab 1 — Overview Dashboard: KPI metrics, department attrition rates, overtime analysis, age distribution
Tab 2 — Prediction Tool: Input employee details → get real-time attrition risk score with gauge chart + HR recommendations
Tab 3 — Feature Insights: Top 15 attrition drivers ranked by importance, income vs attrition boxplot


🚀 How to Run
bash# 1. Clone the repository
git clone https://github.com/yourusername/hr-attrition-prediction

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly openpyxl

# 3. Add dataset
# Place WA_Fn-UseC_-HR-Employee-Attrition.csv inside the data/ folder

# 4. Run the ML pipeline first (generates model files)
python hr_attrition_full.py

# 5. Launch the Streamlit app
streamlit run app.py

# 📁 Project Structure
hr-attrition-prediction/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── model/
│   ├── best_model.pkl
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
├── notebooks/
│   ├── plot1_department_attrition.png
│   ├── plot2_overtime_attrition.png
│   └── ... (9 plots total)
├── hr_attrition_full.py
├── app.py
└── README.md

# 🚀 Future Enhancements

Real-time HR system integration
Multi-company dataset support
SHAP values for model explainability
Email alert system for high-risk employees
BERT-based text analysis for exit interview data
Department-level attrition forecasting
