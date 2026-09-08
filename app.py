import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("credit_risk_random_forest.pkl")


model = load_model()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #dddddd;
    margin-bottom: 20px;
}

.result {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">💳 Credit Scoring Model</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Credit Risk Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("👤 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    income = st.number_input(
        "Annual Income ($)",
        min_value=1000,
        max_value=1000000,
        value=60000,
        step=1000
    )

    home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )

    employment_length = st.number_input(
        "Employment Length (Years)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0
    )


with col2:

    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        ]
    )

    loan_grade = st.selectbox(
        "Loan Grade",
        ["A", "B", "C", "D", "E", "F", "G"]
    )

    loan_amount = st.number_input(
        "Loan Amount ($)",
        min_value=500,
        max_value=100000,
        value=10000,
        step=500
    )

    interest_rate = st.number_input(
        "Loan Interest Rate (%)",
        min_value=1.0,
        max_value=30.0,
        value=11.5,
        step=0.1
    )


with col3:

    loan_percent_income = st.number_input(
        "Loan Percent of Income",
        min_value=0.01,
        max_value=1.0,
        value=0.17,
        step=0.01
    )

    previous_default = st.selectbox(
        "Previous Default on File",
        ["N", "Y"]
    )

    credit_history_length = st.number_input(
        "Credit History Length (Years)",
        min_value=0,
        max_value=50,
        value=6,
        step=1
    )


st.divider()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

income_to_loan_ratio = income / loan_amount

credit_history_ratio = credit_history_length / age


if loan_percent_income <= 0.20:
    loan_burden = "Low"

elif loan_percent_income <= 0.40:
    loan_burden = "Medium"

elif loan_percent_income <= 0.60:
    loan_burden = "High"

else:
    loan_burden = "Very High"


if age <= 25:
    age_group = "Young"

elif age <= 35:
    age_group = "Adult"

elif age <= 50:
    age_group = "Middle-aged"

else:
    age_group = "Senior"


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.subheader("🔍 Credit Risk Prediction")

predict_button = st.button(
    "🚀 Predict Credit Risk",
    use_container_width=True
)


if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "person_age": [age],
        "person_income": [income],
        "person_home_ownership": [home_ownership],
        "person_emp_length": [employment_length],
        "loan_intent": [loan_intent],
        "loan_grade": [loan_grade],
        "loan_amnt": [loan_amount],
        "loan_int_rate": [interest_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_default_on_file": [previous_default],
        "cb_person_cred_hist_length": [credit_history_length],

        # Engineered features
        "income_to_loan_ratio": [income_to_loan_ratio],
        "loan_burden": [loan_burden],
        "age_group": [age_group],
        "credit_history_ratio": [credit_history_ratio]
    })


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    non_default_probability = probabilities[0] * 100
    default_probability = probabilities[1] * 100


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.divider()

    if prediction == 0:

        st.success(
            "🟢 LOW CREDIT RISK — NON-DEFAULT"
        )

        st.markdown(
            '<div class="result">'
            'Credit Risk: LOW'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.error(
            "🔴 HIGH CREDIT RISK — DEFAULT"
        )

        st.markdown(
            '<div class="result">'
            'Credit Risk: HIGH'
            '</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # PROBABILITY METRICS
    # --------------------------------------------------------

    st.subheader("📊 Prediction Probabilities")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            "Non-Default Probability",
            f"{non_default_probability:.2f}%"
        )

    with metric2:
        st.metric(
            "Default Probability",
            f"{default_probability:.2f}%"
        )


    # --------------------------------------------------------
    # PROBABILITY BAR
    # --------------------------------------------------------

    probability_df = pd.DataFrame({
        "Outcome": ["Non-Default", "Default"],
        "Probability": [
            non_default_probability,
            default_probability
        ]
    })

    st.bar_chart(
        probability_df.set_index("Outcome")
    )


    # --------------------------------------------------------
    # APPLICANT ANALYSIS
    # --------------------------------------------------------

    st.subheader("📋 Applicant Analysis")

    analysis_col1, analysis_col2, analysis_col3 = st.columns(3)

    with analysis_col1:
        st.metric(
            "Annual Income",
            f"${income:,.0f}"
        )

    with analysis_col2:
        st.metric(
            "Loan Amount",
            f"${loan_amount:,.0f}"
        )

    with analysis_col3:
        st.metric(
            "Loan Burden",
            loan_burden
        )


    # --------------------------------------------------------
    # FINANCIAL RATIOS
    # --------------------------------------------------------

    st.subheader("💰 Financial Indicators")

    ratio1, ratio2 = st.columns(2)

    with ratio1:
        st.metric(
            "Income / Loan Ratio",
            f"{income_to_loan_ratio:.2f}"
        )

    with ratio2:
        st.metric(
            "Credit History / Age",
            f"{credit_history_ratio:.2f}"
        )


    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    st.subheader("💡 Model Recommendation")

    if prediction == 0:

        st.info(
            "The model predicts that this applicant has a "
            "lower probability of loan default based on the "
            "financial and credit information provided."
        )

    else:

        st.warning(
            "The model predicts a higher probability of loan "
            "default. The applicant's financial information "
            "should be reviewed carefully before making a "
            "credit decision."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Credit Scoring Model | CodeAlpha Machine Learning Internship"
)