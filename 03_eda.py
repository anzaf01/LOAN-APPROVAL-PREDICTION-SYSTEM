import pandas as pd
import matplotlib.pyplot as plt

print("=" * 50)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 50)

# Load Cleaned Dataset
df = pd.read_csv("Dataset/cleaned_loan_data.csv")

# Create output folder
import os
os.makedirs("EDA_Charts", exist_ok=True)

# ==========================================
# Dataset Overview
# ==========================================

print("\nDataset Shape:")
print(df.shape)

print("\nStatistical Summary:")
print(df.describe())

# ==========================================
# 1. Loan Status Distribution
# ==========================================

plt.figure(figsize=(6,4))
df["Loan_Status"].value_counts().plot(kind="bar")
plt.title("Loan Approval Distribution")
plt.xlabel("Loan Status (0=Rejected, 1=Approved)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("EDA_Charts/loan_status_distribution.png")
plt.close()

# ==========================================
# 2. Applicant Income Histogram
# ==========================================

plt.figure(figsize=(6,4))
plt.hist(df["ApplicantIncome"], bins=20)
plt.title("Applicant Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("EDA_Charts/applicant_income_histogram.png")
plt.close()

# ==========================================
# 3. Loan Amount Histogram
# ==========================================

plt.figure(figsize=(6,4))
plt.hist(df["LoanAmount"], bins=20)
plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("EDA_Charts/loan_amount_histogram.png")
plt.close()

# ==========================================
# 4. Applicant Income Boxplot
# ==========================================

plt.figure(figsize=(6,4))
plt.boxplot(df["ApplicantIncome"])
plt.title("Applicant Income Boxplot")
plt.tight_layout()
plt.savefig("EDA_Charts/applicant_income_boxplot.png")
plt.close()

# ==========================================
# 5. Loan Amount Boxplot
# ==========================================

plt.figure(figsize=(6,4))
plt.boxplot(df["LoanAmount"])
plt.title("Loan Amount Boxplot")
plt.tight_layout()
plt.savefig("EDA_Charts/loan_amount_boxplot.png")
plt.close()

# ==========================================
# 6. Scatter Plot
# ==========================================

plt.figure(figsize=(6,4))
plt.scatter(df["ApplicantIncome"], df["LoanAmount"])
plt.title("Income vs Loan Amount")
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.tight_layout()
plt.savefig("EDA_Charts/income_vs_loanamount.png")
plt.close()

# ==========================================
# Correlation Matrix
# ==========================================

corr_matrix = df.corr()

plt.figure(figsize=(10,8))
plt.imshow(corr_matrix)
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)),
           corr_matrix.columns,
           rotation=90)
plt.yticks(range(len(corr_matrix.columns)),
           corr_matrix.columns)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("EDA_Charts/correlation_heatmap.png")
plt.close()

# ==========================================
# Top Correlations with Loan Status
# ==========================================

print("\nCorrelation with Loan Status:")
print(corr_matrix["Loan_Status"].sort_values(ascending=False))

print("\nEDA Charts Saved Successfully!")

print("\nCharts Location:")
print("EDA_Charts/")

print("\nEDA COMPLETED SUCCESSFULLY")