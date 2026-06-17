import pandas as pd

print("=" * 50)
print("LOAN APPROVAL PREDICTION SYSTEM")
print("=" * 50)

# Load Dataset
try:
    df = pd.read_csv("Dataset/train.csv")  # Change filename if needed

    print("\nDataset Loaded Successfully!")

    # Dataset Shape
    print("\nDataset Shape:")
    print(df.shape)

    # Columns
    print("\nColumns:")
    print(df.columns.tolist())

    # First 5 Rows
    print("\nFirst 5 Rows:")
    print(df.head())

    # Missing Values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Duplicate Records
    print("\nDuplicate Records:")
    print(df.duplicated().sum())

    # Target Variable Distribution
    if "Loan_Status" in df.columns:
        print("\nLoan Status Distribution:")
        print(df["Loan_Status"].value_counts())

except FileNotFoundError:
    print("\nERROR: Dataset file not found.")
    print("Place your CSV file inside the Dataset folder.")
    print("Example:")
    print("Loan_Approval_Prediction/Dataset/train.csv")

except Exception as e:
    print("\nAn error occurred:")
    print(e)