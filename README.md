# Loan Approval Prediction System

A Machine Learning-based Loan Approval Prediction System developed using Python, Scikit-learn, XGBoost, and Streamlit. The system predicts whether a loan application is likely to be approved based on applicant information such as income, credit history, education, employment status, family details, and property location.

The project follows the complete Machine Learning lifecycle, including data preprocessing, exploratory data analysis, feature engineering, model development, evaluation, and deployment.

---

## Project Objective

The objective of this project is to develop a predictive system that determines whether a loan application should be approved or rejected based on applicant information.

The system aims to assist financial institutions in making faster, data-driven, and consistent lending decisions.

---

## Problem Statement

Financial institutions process a large number of loan applications daily. Manual evaluation of applications is time-consuming and may result in inconsistent decisions.

This project applies Machine Learning techniques to automate loan approval prediction based on historical applicant data.

---

## Dataset

Dataset Source: Kaggle Loan Prediction Dataset

https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

### Dataset Information

The dataset contains applicant information including:

- Gender
- Marital Status
- Dependents
- Education
- Self Employment Status
- Applicant Income
- Co-Applicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
- Loan Status

### Dataset Size

- Total Records: 614
- Total Features: 13
- Target Variable: Loan_Status

### Target Classes

- Y – Loan Approved
- N – Loan Rejected

---

## Machine Learning Concepts Used

### Classification

The project predicts a categorical outcome:

- Approved
- Rejected

### Feature Selection

Important features were selected based on correlation analysis and domain knowledge.

### Data Balancing

The target variable distribution was analyzed before model training.

### Model Evaluation

Performance was measured using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

---

## Machine Learning Lifecycle

### Phase 1: Problem Understanding

#### Business Problem

Predict whether a loan application should be approved based on applicant information.

#### Objectives

- Reduce manual effort
- Improve decision-making consistency
- Increase processing efficiency

---

### Phase 2: Data Collection

Tasks Performed:

- Downloaded dataset from Kaggle
- Studied dataset attributes
- Identified target variable

---

### Phase 3: Data Preprocessing

Tasks Performed:

#### Missing Value Handling

- Mode imputation for categorical features
- Median imputation for numerical features

#### Duplicate Removal

- Checked and removed duplicate records

#### Encoding

Converted categorical variables into numerical values.

Examples:

| Feature | Encoding |
|----------|----------|
| Male | 1 |
| Female | 0 |
| Yes | 1 |
| No | 0 |

#### Data Transformation

Applied preprocessing techniques where required.

---

### Phase 4: Exploratory Data Analysis

Analysis Performed:

#### Univariate Analysis

- Applicant Income Distribution
- Loan Amount Distribution
- Credit History Distribution

#### Bivariate Analysis

- Income vs Loan Status
- Credit History vs Loan Status
- Education vs Loan Status

#### Correlation Analysis

Generated a correlation matrix and heatmap.

#### Visualizations

- Histograms
- Boxplots
- Scatter Plots
- Heatmaps

---

### Phase 5: Feature Engineering

New features were created to improve model performance.

#### Total Income

```python
TotalIncome = ApplicantIncome + CoapplicantIncome
```

#### Loan Income Ratio

```python
LoanIncomeRatio = LoanAmount / TotalIncome
```

#### EMI

```python
EMI = LoanAmount / Loan_Amount_Term
```

---

### Phase 6: Model Building

Three Machine Learning models were trained and evaluated.

#### Logistic Regression

Used as a baseline classification model.

#### Random Forest Classifier

Used for improved predictive performance and feature importance analysis.

#### XGBoost Classifier

Used for gradient boosting-based classification.

---

### Phase 7: Model Evaluation

#### Logistic Regression

| Metric | Score |
|----------|----------|
| Accuracy | 86.18% |
| Precision | 84.00% |
| Recall | 98.82% |
| F1 Score | 90.81% |
| ROC-AUC | 80.09% |

#### Random Forest

| Metric | Score |
|----------|----------|
| Accuracy | 86.18% |
| Precision | 87.78% |
| Recall | 92.94% |
| F1 Score | 90.29% |
| ROC-AUC | 85.12% |

#### XGBoost

| Metric | Score |
|----------|----------|
| Accuracy | 82.11% |
| Precision | 85.39% |
| Recall | 89.41% |
| F1 Score | 87.35% |
| ROC-AUC | 82.60% |

---

## Best Model

Based on overall classification performance:

**Selected Model: Random Forest Classifier**

---

## Deployment

The final model is deployed using Streamlit.

The application accepts:

- Personal Information
- Employment Information
- Financial Information
- Property Details
- Loan Details

The application provides:

- Loan Approval Prediction
- Approval Confidence
- EMI Estimation
- Loan Eligibility Insights

---

## Project Structure

```text
Loan_Approval_Prediction/
│
├── Dataset/
│   ├── train_u6lujuX_CVtuZ9i.csv
│   ├── cleaned_loan_data.csv
│   └── feature_engineered_data.csv
│
├── Model/
│   └── loan_approval_model.pkl
│
├── EDA_Charts/
│
├── loan_prediction.py
├── 02_data_preprocessing.py
├── 03_eda.py
├── 04_feature_engineering.py
├── 05_model_training.py
├── app.py
│
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Loan-Approval-Prediction-System.git
```

### Navigate to Project Folder

```bash
cd Loan-Approval-Prediction-System
```

### Install Dependencies

```bash
py -m pip install pandas numpy scikit-learn matplotlib xgboost streamlit joblib
```

### Run Application

```bash
py -m streamlit run app.py
```

---

## Sample Input

| Parameter | Example |
|------------|------------|
| Age | 26 |
| Employment Type | Salaried Employee |
| Annual Income | ₹600,000 |
| CIBIL Score | 750 |
| Property Location | Urban |
| Loan Amount | ₹300,000 |

---

## Key Features

- End-to-End Machine Learning Pipeline
- Data Preprocessing and Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Multiple Classification Models
- Model Comparison and Evaluation
- Interactive Streamlit Application
- Real-Time Loan Approval Prediction
- Loan Eligibility Estimation
- EMI Calculation

---

## Future Scope

Potential improvements include:

- Integration with Banking APIs
- Advanced Credit Risk Assessment
- Explainable AI Techniques (SHAP, LIME)
- Cloud Deployment (AWS, Azure)
- MLOps Pipeline Implementation
- Loan Recommendation System

---

## Author

Mohd Anzaf Ali

B.Tech Computer Science and Engineering  
UCER, Prayagraj

---

## License

This project has been developed for academic and educational purposes.