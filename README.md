# 📊 Telco Customer Churn Prediction

An end-to-end Machine Learning project that analyzes customer behavior and predicts customer churn using Python, Scikit-learn, and XGBoost, with an interactive Streamlit dashboard.

## 🚀 Project Overview

Customer churn is a major challenge for telecommunications companies. Understanding why customers leave can help businesses improve customer retention and make better decisions.

This project uses historical customer data to identify churn patterns, analyze important factors influencing customer decisions, and build machine learning models to predict whether a customer is likely to leave.

## 🎯 Project Objectives

- Perform Exploratory Data Analysis (EDA) to understand customer behavior.
- Clean and preprocess customer data.
- Identify key factors associated with customer churn.
- Build and compare multiple Machine Learning models.
- Evaluate model performance using appropriate classification metrics.
- Develop an interactive dashboard for customer churn prediction.

## 📂 Dataset

**Dataset:** Telco Customer Churn

**File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

- Total records: 7,043 customers
- Total features: 21 columns
- Target variable: `Churn`
- Classification type: Binary Classification

The target variable contains two classes:
- `Yes`: Customer churned.
- `No`: Customer did not churn.

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand customer characteristics and identify patterns associated with churn.

Key findings:

- Customers with month-to-month contracts have higher churn rates.
- Customers with shorter tenure are more likely to leave.
- Customers with higher monthly charges show greater churn tendencies.
- Fiber optic internet customers represent an important segment for churn analysis.
- Customers without technical support tend to have higher churn rates.

## ⚙️ Data Preprocessing

The following preprocessing techniques were applied:

- Handling missing and inconsistent values.
- Encoding categorical variables.
- Feature engineering.
- Feature scaling using StandardScaler.
- Train-test split with stratification.
- Preparing numerical and categorical features for model training.

## 🤖 Machine Learning Models

Three classification algorithms were trained and evaluated:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

### Model Evaluation

The models were evaluated on the test dataset using:

- Accuracy
- Precision
- Recall
- F1-score

### Test Set Performance

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 0.66 | 0.56 | 0.60 |
| Random Forest | 76.37% | 0.55 | 0.62 | 0.58 |
| XGBoost | 80.48% | 0.67 | 0.53 | 0.59 |

### Cross-Validation Results

| Model | Accuracy | F1-score |
|---|---:|---:|
| Logistic Regression | 0.802 | 0.592 |
| Random Forest | 0.776 | 0.604 |
| XGBoost | 0.805 | 0.590 |

**Model Selection:** Random Forest was retained as the final candidate in the project notebook, with the highest cross-validation F1-score. Logistic Regression achieved the highest test-set F1-score among the reported models.

## 📈 Streamlit Dashboard

The project includes an interactive Streamlit application called **ChurnIQ**.

Dashboard features:

- Customer churn prediction.
- Interactive filtering.
- Customer segmentation.
- Churn-related KPI cards.
- Data visualization using Plotly.
- Customer behavior analysis.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Model Serialization | Joblib |
| Dashboard | Streamlit |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

## 📁 Project Structure

```text
project-1/
│
├── .gitignore
├── README.md
├── Customer_Churn.ipynb
│
└── elco_Churn_Project/
    ├── app.py
    ├── churn_pipeline.joblib
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## 💻 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Chheangdaravichea/Telco-Customer-Churn-Prediction.git
```

### 2. Navigate to the project folder

```bash
cd Telco-Customer-Churn-Prediction
```

### 3. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn xgboost joblib plotly
```

### 4. Run the Streamlit application

```bash
cd elco_Churn_Project
streamlit run app.py
```

The application will open in your browser.

## 🔮 Future Improvements

- Improve model performance through hyperparameter tuning.
- Perform more advanced feature engineering.
- Explore additional classification algorithms.
- Improve model interpretability using SHAP.
- Deploy the application publicly.
- Develop more advanced customer retention insights.

## 👨‍💻 Author

**Chheang Daravichea**

Computer Science Student | Data Analytics | Data Science | Machine Learning

GitHub: [@Chheangdaravichea](https://github.com/Chheangdaravichea)

---

⭐ If you find this project interesting, feel free to explore the repository!