import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Model/loan_approval_model.pkl")

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Approval Prediction System")
st.markdown("### AI-Based Loan Eligibility Assessment")

st.info(
    "Model trained on Kaggle Loan Prediction Dataset. "
    "Loan amount is entered in rupees but converted internally to thousands for prediction."
)

st.divider()

# ================= PERSONAL INFO =================

st.header("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:
    applicant_name = st.text_input("Applicant Name")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=25
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    married = st.selectbox(
        "Marital Status",
        ["Married", "Single"]
    )

    family_members = st.selectbox(
        "Number of Family Members",
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

    education = st.selectbox(
        "Highest Education",
        [
            "10th Pass",
            "12th Pass",
            "Diploma",
            "Bachelor's Degree",
            "Master's Degree",
            "MBA",
            "M.Tech",
            "PhD"
        ]
    )

st.divider()

# ================= EMPLOYMENT INFO =================

st.header("💼 Employment Information")

col1, col2 = st.columns(2)

with col1:
    employment_type = st.selectbox(
        "Employment Type",
        [
            "Salaried Employee",
            "Business Owner",
            "Freelancer",
            "Self Employed"
        ]
    )

with col2:
    annual_income = st.number_input(
        "Annual Income (₹)",
        min_value=50000.0,
        max_value=1000000.0,
        value=300000.0,
        step=50000.0
    )

st.divider()

# ================= FINANCIAL INFO =================

st.header("💳 Financial Information")

col1, col2 = st.columns(2)

with col1:
    cibil_score = st.number_input(
        "CIBIL Score",
        min_value=300,
        max_value=900,
        value=750
    )

with col2:
    coapplicant_income = st.number_input(
        "Co-Applicant Annual Income (₹)",
        min_value=0.0,
        max_value=500000.0,
        value=0.0,
        step=50000.0
    )

st.divider()

# ================= PROPERTY DETAILS =================

st.header("🏠 Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    property_area = st.selectbox(
        "Property Location",
        ["Rural", "Semiurban", "Urban"]
    )

with col2:
    property_value = st.number_input(
        "Property / Land Value (₹)",
        min_value=0.0,
        step=100000.0
    )

with col3:
    land_area = st.number_input(
        "Land Area (sq.ft)",
        min_value=0.0,
        step=100.0
    )

st.divider()

# ================= LOAN DETAILS =================

st.header("💰 Loan Details")

col1, col2, col3 = st.columns(3)

with col1:
    loan_amount = st.number_input(
        "Required Loan Amount (₹)",
        min_value=9000.0,
        max_value=700000.0,
        value=100000.0,
        step=10000.0
    )

with col2:
    loan_term = st.number_input(
        "Loan Term (Months)",
        min_value=12.0,
        max_value=480.0,
        value=360.0,
        step=12.0
    )

with col3:
    loan_purpose = st.selectbox(
        "Loan Purpose",
        [
            "Home Loan",
            "Vehicle Loan",
            "Education Loan",
            "Business Loan",
            "Personal Loan"
        ]
    )

st.divider()

# ================= PREDICTION =================

if st.button("🔍 Check Loan Eligibility"):

    gender_encoded = 1 if gender == "Male" else 0
    married_encoded = 1 if married == "Married" else 0

    if family_members <= 2:
        dependents = 0
    elif family_members <= 4:
        dependents = 1
    elif family_members <= 6:
        dependents = 2
    else:
        dependents = 3

    education_encoded = (
        0 if education in [
            "Bachelor's Degree",
            "Master's Degree",
            "MBA",
            "M.Tech",
            "PhD"
        ]
        else 1
    )

    self_employed_encoded = (
        0 if employment_type == "Salaried Employee"
        else 1
    )

    applicant_income = annual_income / 12
    coapplicant_income_monthly = coapplicant_income / 12

    credit_history = 1 if cibil_score >= 700 else 0

    property_map = {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    }

    property_encoded = property_map[property_area]

    # IMPORTANT:
    # Kaggle dataset stores LoanAmount in thousands.
    # Example: 320 means ₹3,20,000.
    loan_amount_model = loan_amount / 1000

    total_income = applicant_income + coapplicant_income_monthly

    loan_income_ratio = loan_amount_model / (total_income + 1)

    emi = loan_amount_model / loan_term

    input_data = pd.DataFrame([[
        gender_encoded,
        married_encoded,
        dependents,
        education_encoded,
        self_employed_encoded,
        applicant_income,
        coapplicant_income_monthly,
        loan_amount_model,
        loan_term,
        credit_history,
        property_encoded,
        total_income,
        loan_income_ratio,
        emi
    ]])

    prediction = model.predict(input_data)[0]

    try:
        approval_probability = model.predict_proba(input_data)[0][1]
    except Exception:
        approval_probability = None

    # Estimated eligibility in rupees
    eligible_amount = total_income * 36

    if cibil_score >= 800:
        eligible_amount *= 1.20
    elif cibil_score >= 750:
        eligible_amount *= 1.10
    elif cibil_score < 650:
        eligible_amount *= 0.80

    if property_value > 0:
        property_based_limit = property_value * 0.80
        eligible_amount = min(eligible_amount, property_based_limit)

    monthly_interest = 0.09 / 12

    emi_amount = (
        loan_amount
        * monthly_interest
        * ((1 + monthly_interest) ** loan_term)
    ) / (
        ((1 + monthly_interest) ** loan_term) - 1
    )

    st.divider()
    st.header("📊 Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Estimated Eligible Amount",
            f"₹{eligible_amount:,.0f}"
        )

    with col2:
        st.metric(
            "Estimated EMI",
            f"₹{emi_amount:,.0f}/month"
        )

    with col3:
        if approval_probability is not None:
            st.metric(
                "Approval Confidence",
                f"{approval_probability * 100:.1f}%"
            )

    if prediction == 1:

        st.success("✅ Congratulations! Loan is likely to be Approved.")

        if loan_amount <= eligible_amount:
            st.info("Requested loan amount is within estimated eligibility.")

        if cibil_score >= 750:
            st.info("Strong credit profile detected.")

    else:

        st.error("❌ Loan is likely to be Rejected.")

        st.subheader("🔍 Possible Reasons")

        reasons = []

        if cibil_score < 700:
            reasons.append("CIBIL score is below the preferred lending range.")

        if annual_income < 300000:
            reasons.append("Annual income may be low for loan approval.")

        if loan_amount > eligible_amount:
            reasons.append("Requested loan amount is higher than estimated eligibility.")

        if employment_type in ["Freelancer", "Self Employed"]:
            reasons.append("Income stability may require stronger documentation.")

        if reasons:
            for reason in reasons:
                st.write(f"• {reason}")
        else:
            st.write("• The ML model found the application profile risky based on training data.")

        st.subheader("💡 Suggestions")

        if cibil_score < 700:
            st.write("• Improve your CIBIL score above 700.")

        if annual_income < 300000:
            st.write("• Increase income proof or add a co-applicant.")

        if loan_amount > eligible_amount:
            st.write("• Reduce the requested loan amount.")

        if employment_type in ["Freelancer", "Self Employed"]:
            st.write("• Provide ITR, bank statements, and income proof.")

        st.subheader("💰 Maximum Loan You May Get")

        st.info(
            f"Based on the provided details, estimated maximum eligible loan amount is ₹{eligible_amount:,.0f}."
        )

        if loan_amount > eligible_amount:
            reduction = loan_amount - eligible_amount
            st.warning(
                f"Reduce your requested amount by approximately ₹{reduction:,.0f}."
            )

    # st.subheader("📋 Application Summary")

    # summary = {
    #     "Applicant": applicant_name,
    #     "Age": age,
    #     "Employment Type": employment_type,
    #     "Annual Income": f"₹{annual_income:,.0f}",
    #     "Co-Applicant Annual Income": f"₹{coapplicant_income:,.0f}",
    #     "CIBIL Score": cibil_score,
    #     "Property Location": property_area,
    #     "Property Value": f"₹{property_value:,.0f}",
    #     "Land Area": f"{land_area:,.0f} sq.ft",
    #     "Requested Loan Amount": f"₹{loan_amount:,.0f}",
    #     "Loan Amount Sent to Model": f"{loan_amount_model:.0f}",
    #     "Estimated Eligible Amount": f"₹{eligible_amount:,.0f}",
    #     "Estimated EMI": f"₹{emi_amount:,.0f}/month",
    #     "Loan Purpose": loan_purpose
    # }

    # st.json(summary)