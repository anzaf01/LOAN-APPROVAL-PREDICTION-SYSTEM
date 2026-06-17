import joblib

model = joblib.load("Model/loan_approval_model.pkl")

print("Number of features expected:", model.n_features_in_)