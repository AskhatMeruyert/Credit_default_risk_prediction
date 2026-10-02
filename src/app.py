import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Default Risk Prediction",
    page_icon="💳",
    layout="wide"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "credit_risk_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


artifact = load_model()

model = artifact["model"]
threshold = artifact["threshold"]
feature_names = artifact["features"]


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("💳 Credit Default Risk Prediction")

st.write(
    "Estimate the probability of serious financial delinquency "
    "based on the client's financial profile."
)

st.caption(
    f"Decision threshold: {threshold:.3f}"
)

st.divider()


# --------------------------------------------------
# Client information
# --------------------------------------------------

st.subheader("Client information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=21,
        max_value=109,
        value=40
    )

    monthly_income = st.number_input(
        "Monthly income",
        min_value=0.0,
        value=5000.0,
        step=500.0
    )

    debt_ratio = st.number_input(
        "Debt ratio",
        min_value=0.0,
        value=0.35,
        step=0.05,
        format="%.2f"
    )


with col2:

    utilization = st.number_input(
        "Revolving utilization",
        min_value=0.0,
        value=0.30,
        step=0.05,
        format="%.2f",
        help="Utilization of unsecured credit lines."
    )

    open_credit_lines = st.number_input(
        "Open credit lines and loans",
        min_value=0,
        value=8
    )

    real_estate_loans = st.number_input(
        "Real estate loans or lines",
        min_value=0,
        value=1
    )


with col3:

    late_30_59 = st.number_input(
        "30–59 days past due",
        min_value=0,
        value=0
    )

    late_60_89 = st.number_input(
        "60–89 days past due",
        min_value=0,
        value=0
    )

    late_90 = st.number_input(
        "90+ days late",
        min_value=0,
        value=0
    )


dependents = st.number_input(
    "Number of dependents",
    min_value=0.0,
    value=0.0,
    step=1.0
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button(
    "Predict default risk",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------
    # Feature engineering
    # --------------------------------------------------

    monthly_income_missing = 0
    dependents_missing = 0

    delinquency_special = int(
        late_30_59 >= 96
        or late_60_89 >= 96
        or late_90 >= 96
    )

    utilization_extreme = int(
        utilization > 10
    )

    monthly_income_zero = int(
        monthly_income == 0
    )

    input_data = pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines": [utilization],
        "age": [age],
        "NumberOfTime30-59DaysPastDueNotWorse": [late_30_59],
        "DebtRatio": [debt_ratio],
        "MonthlyIncome": [monthly_income],
        "NumberOfOpenCreditLinesAndLoans": [open_credit_lines],
        "NumberOfTimes90DaysLate": [late_90],
        "NumberRealEstateLoansOrLines": [real_estate_loans],
        "NumberOfTime60-89DaysPastDueNotWorse": [late_60_89],
        "NumberOfDependents": [dependents],
        "MonthlyIncome_missing": [monthly_income_missing],
        "NumberOfDependents_missing": [dependents_missing],
        "delinquency_special": [delinquency_special],
        "utilization_extreme": [utilization_extreme],
        "MonthlyIncome_zero": [monthly_income_zero]
    })

    # Ensure exact training feature order
    input_data = input_data[feature_names]

    # --------------------------------------------------
    # Model prediction
    # --------------------------------------------------

    probability = model.predict_proba(input_data)[0, 1]

    prediction = int(
        probability >= threshold
    )


    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    st.subheader("Prediction result")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Default probability",
        f"{probability:.1%}"
    )

    col2.metric(
        "Decision threshold",
        f"{threshold:.1%}"
    )

    col3.metric(
        "Prediction",
        "High risk" if prediction == 1 else "Low risk"
    )


    # --------------------------------------------------
    # Risk visualization
    # --------------------------------------------------

    st.write("Risk probability")

    st.progress(
        min(float(probability), 1.0)
    )


    # --------------------------------------------------
    # Prediction explanation
    # --------------------------------------------------

    if prediction == 1:

        st.error(
            "⚠️ Elevated default risk detected. "
            "The predicted probability exceeds the model's "
            "decision threshold."
        )

    else:

        st.success(
            "✅ Lower default risk detected. "
            "The predicted probability is below the model's "
            "decision threshold."
        )

    st.caption(
        "The probability represents the model's estimate of "
        "serious financial delinquency within the next two years."
    )


    # --------------------------------------------------
    # Model input
    # --------------------------------------------------

    with st.expander("Model input"):

        st.dataframe(
            input_data.T.rename(columns={0: "Value"}),
            use_container_width=True
        )


# --------------------------------------------------
# Model information
# --------------------------------------------------

st.divider()

with st.expander("About the model"):

    st.markdown(
        f"""
### HistGradientBoosting Classifier

The model estimates the probability that a client will experience
**serious financial delinquency within the next two years**.

**Decision threshold:** `{threshold:.3f}`

#### Final test performance

| Metric | Score |
|---|---:|
| ROC-AUC | **0.868** |
| PR-AUC | **0.403** |
| Precision | **0.39** |
| Recall | **0.52** |
| F1-score | **0.45** |

The classification threshold was selected on a separate validation
set by maximizing the F1-score and was subsequently evaluated on
the untouched test set.

The model uses financial and credit-history characteristics,
including credit utilization, delinquency history, debt ratio,
income, age and the number of active credit lines.
        """
    )


# --------------------------------------------------
# Disclaimer
# --------------------------------------------------

st.divider()

st.warning(
    "This application is an educational machine learning project. "
    "Model predictions should not be used as the sole basis for "
    "real-world lending or financial decisions."
)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.caption(
    "Machine Learning portfolio project • "
    "HistGradientBoostingClassifier"
)