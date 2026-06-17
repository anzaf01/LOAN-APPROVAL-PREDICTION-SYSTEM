import pandas as pd

print("=" * 50)
print("FEATURE ENGINEERING")
print("=" * 50)

# Load cleaned dataset
df = pd.read_csv("Dataset/cleaned_loan_data.csv")

# ==========================================
# Create New Features
# ==========================================

# Total Income
df["TotalIncome"] = (
    df["ApplicantIncome"] +
    df["CoapplicantIncome"]
)

# Loan Income Ratio
df["LoanIncomeRatio"] = (
    df["LoanAmount"] /
    (df["TotalIncome"] + 1)
)

# EMI Approximation
df["EMI"] = (
    df["LoanAmount"] /
    df["Loan_Amount_Term"]
)

# ==========================================
# Check New Features
# ==========================================

print("\nNew Features Created:")
print([
    "TotalIncome",
    "LoanIncomeRatio",
    "EMI"
])

print("\nFirst 5 Rows:")
print(
    df[
        [
            "TotalIncome",
            "LoanIncomeRatio",
            "EMI"
        ]
    ].head()
)

# ==========================================
# Save Dataset
# ==========================================

df.to_csv(
    "Dataset/feature_engineered_data.csv",
    index=False
)

print("\nFeature Engineered Dataset Saved!")
print(
    "Dataset/feature_engineered_data.csv"
)

print("\nFEATURE ENGINEERING COMPLETED")