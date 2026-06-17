import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("=" * 50)
print("DATA PREPROCESSING")
print("=" * 50)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("Dataset/train.csv")

print("\nDataset Loaded Successfully!")

# ==========================================
# Remove Unnecessary Column
# ==========================================

df.drop("Loan_ID", axis=1, inplace=True)

print("\nLoan_ID column removed.")

# ==========================================
# Check Missing Values Before Cleaning
# ==========================================

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# ==========================================
# Handle Missing Values
# ==========================================

# Categorical Columns
categorical_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Self_Employed"
]

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Numerical Columns
numerical_cols = [
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing values handled successfully.")

# ==========================================
# Verify Missing Values
# ==========================================

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# ==========================================
# Remove Duplicate Records
# ==========================================

duplicates = df.duplicated().sum()

print("\nDuplicate Records Found:", duplicates)

if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicates removed.")
else:
    print("No duplicate records found.")

# ==========================================
# Encode Categorical Features
# ==========================================

encoder = LabelEncoder()

categorical_features = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in categorical_features:
    df[col] = encoder.fit_transform(df[col])

print("\nCategorical variables encoded successfully.")

# ==========================================
# Dataset Information
# ==========================================

print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

# ==========================================
# Save Cleaned Dataset
# ==========================================

df.to_csv("Dataset/cleaned_loan_data.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("File saved as: Dataset/cleaned_loan_data.csv")

print("\n" + "=" * 50)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 50)