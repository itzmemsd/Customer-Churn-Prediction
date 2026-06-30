"""
==========================================================
Customer Churn Prediction System
Streamlit Application

Part 1
==========================================================
"""

import streamlit as st
import pandas as pd
import joblib

# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(

    page_title="Customer Churn Prediction",

    page_icon="📊",

    layout="wide"

)

# ======================================================
# Load Model
# ======================================================

@st.cache_resource

def load_model():

    return joblib.load(

        "models/best_model.pkl"

    )

model = load_model()

# ======================================================
# Title
# ======================================================

st.title("📊 Customer Churn Prediction System")

st.markdown(
"""
Predict whether a customer is likely to leave the company
using Machine Learning.
"""
)

st.divider()

# ======================================================
# Sidebar
# ======================================================

st.sidebar.title("About")

st.sidebar.info(
"""
MBA AI & DS Mini Project

Customer Churn Prediction System

Machine Learning Algorithms

• Logistic Regression

• Decision Tree

• Random Forest

• Support Vector Machine
"""
)

# ======================================================
# Input Form
# ======================================================

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(

        "Gender",

        ["Male", "Female"]

    )

    senior = st.selectbox(

        "Senior Citizen",

        [0,1]

    )

    partner = st.selectbox(

        "Partner",

        ["Yes","No"]

    )

    dependents = st.selectbox(

        "Dependents",

        ["Yes","No"]

    )

    tenure = st.slider(

        "Tenure (Months)",

        0,

        72,

        12

    )

    phone = st.selectbox(

        "Phone Service",

        ["Yes","No"]

    )

    multiple = st.selectbox(

        "Multiple Lines",

        [

            "No",

            "Yes",

            "No phone service"

        ]

    )

with col2:

    internet = st.selectbox(

        "Internet Service",

        [

            "DSL",

            "Fiber optic",

            "No"

        ]

    )

    security = st.selectbox(

        "Online Security",

        [

            "Yes",

            "No",

            "No internet service"

        ]

    )

    backup = st.selectbox(

        "Online Backup",

        [

            "Yes",

            "No",

            "No internet service"

        ]

    )

    protection = st.selectbox(

        "Device Protection",

        [

            "Yes",

            "No",

            "No internet service"

        ]

    )

    support = st.selectbox(

        "Tech Support",

        [

            "Yes",

            "No",

            "No internet service"

        ]

    )

    tv = st.selectbox(

        "Streaming TV",

        [

            "Yes",

            "No",

            "No internet service"

        ]

    )

    movies = st.selectbox(

        "Streaming Movies",

        [

            "Yes",

            "No",

            "No internet service"

        ]

    )
    # ======================================================
# Remaining Customer Details
# ======================================================

col3, col4 = st.columns(2)

with col3:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        [
            "Yes",
            "No"
        ]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col4:

    monthly = st.number_input(
        "Monthly Charges",
        min_value=18.25,
        max_value=120.00,
        value=70.00,
        step=0.50
    )

    total = st.number_input(
        "Total Charges",
        min_value=0.00,
        value=850.00,
        step=10.00
    )

# ======================================================
# Prediction Button
# ======================================================

st.divider()

predict = st.button(
    "Predict Customer Churn",
    use_container_width=True
)

# ======================================================
# Prepare Input
# ======================================================

if predict:

    customer = pd.DataFrame({

        "gender":[gender],

        "SeniorCitizen":[senior],

        "Partner":[partner],

        "Dependents":[dependents],

        "tenure":[tenure],

        "PhoneService":[phone],

        "MultipleLines":[multiple],

        "InternetService":[internet],

        "OnlineSecurity":[security],

        "OnlineBackup":[backup],

        "DeviceProtection":[protection],

        "TechSupport":[support],

        "StreamingTV":[tv],

        "StreamingMovies":[movies],

        "Contract":[contract],

        "PaperlessBilling":[paperless],

        "PaymentMethod":[payment],

        "MonthlyCharges":[monthly],

        "TotalCharges":[total]

    })

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0]

    churn_probability = probability[1] * 100
        # ======================================================
    # Display Prediction
    # ======================================================

    st.divider()

    st.subheader("Prediction Result")

    if prediction == "Yes":

        st.error("⚠ Customer is likely to churn.")

    else:

        st.success("✅ Customer is likely to remain with the company.")

    # ======================================================
    # Probability
    # ======================================================

    st.metric(

        label="Churn Probability",

        value=f"{churn_probability:.2f}%"

    )

    # ======================================================
    # Risk Level
    # ======================================================

    if churn_probability < 30:

        risk = "🟢 Low Risk"

    elif churn_probability < 70:

        risk = "🟡 Medium Risk"

    else:

        risk = "🔴 High Risk"

    st.info(f"Risk Level : {risk}")

    # ======================================================
    # Business Recommendations
    # ======================================================

    st.subheader("Business Recommendation")

    if churn_probability >= 70:

        st.warning(
            """
• Offer personalized discounts

• Contact the customer immediately

• Provide loyalty rewards

• Assign a dedicated customer support executive

• Improve customer engagement
            """
        )

    elif churn_probability >= 30:

        st.info(
            """
• Send promotional offers

• Recommend suitable service plans

• Increase customer interaction

• Monitor customer satisfaction
            """
        )

    else:

        st.success(
            """
• Customer retention is good

• Continue regular engagement

• Recommend premium services

• Maintain existing service quality
            """
        )

# ======================================================
# Footer
# ======================================================

st.divider()

st.caption(
    "Customer Churn Prediction System | Mini Project | Developed using Streamlit & Scikit-learn"
)