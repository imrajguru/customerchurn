import streamlit as st
import pickle
import numpy as np

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ----------------------------
# Custom Styling
# ----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-size: 16px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("📌 About Project")
st.sidebar.info("""
This application predicts whether a customer is likely to churn.

Built using Machine Learning techniques 
for business risk analysis.
""")

st.sidebar.markdown("### 🛠 Tech Stack")
st.sidebar.write("• Python")
st.sidebar.write("• Scikit-learn")
st.sidebar.write("• Streamlit")
st.sidebar.write("• Data Preprocessing")

# ----------------------------
# Main Title
# ----------------------------
st.title("📊 Customer Churn Prediction")
st.write("Enter customer details below to predict churn risk.")

# ----------------------------
# Input Fields
# ----------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", 
                               ["Electronic check", "Mailed check", 
                                "Bank transfer", "Credit card"])

# ----------------------------
# Simple Encoding (Example)
# ----------------------------
gender = 1 if gender == "Male" else 0
senior_citizen = 1 if senior_citizen == "Yes" else 0

contract_mapping = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

internet_mapping = {
    "No": 0,
    "DSL": 1,
    "Fiber optic": 2
}

payment_mapping = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer": 2,
    "Credit card": 3
}

contract = contract_mapping[contract]
internet_service = internet_mapping[internet_service]
payment_method = payment_mapping[payment_method]

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("🔍 Predict Churn Risk"):

    input_data = np.array([[gender, senior_citizen, tenure,
                            monthly_charges, total_charges,
                            contract, internet_service,
                            payment_method]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    confidence = np.max(probability) * 100

    st.markdown("---")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Customer Churn")
        st.write(f"Confidence: **{confidence:.2f}%**")
    else:
        st.success("✅ Low Risk Customer")
        st.write(f"Confidence: **{confidence:.2f}%**")

    st.progress(int(confidence))

st.markdown("---")
st.caption("Developed by Rajguru Thevar | ML Portfolio Project")