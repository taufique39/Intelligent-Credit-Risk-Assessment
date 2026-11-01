"""
Intelligent Credit Risk Assessment Using Explainable AI
Interactive Streamlit Demonstration Dashboard

Authors: Major Project Team
Session: 2026-27 | B.Tech CSE
Benchmark Dataset: Statlog (German Credit Data) - UCI Repository
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.explainability import setup_lime_explainer, PARENT_CATEGORICAL_FEATURES

# =====================================================================
# PAGE CONFIGURATION & THEME
# =====================================================================
st.set_page_config(
    page_title="Intelligent Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File Paths
SPLITS_DIR = BASE_DIR / "data" / "splits"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
SHAP_IMPORTANCE_CSV = REPORTS_DIR / "shap_global_importance.csv"

MODEL_FILES = {
    "Logistic Regression (Baseline)": MODELS_DIR / "logistic_regression.joblib",
    "Random Forest": MODELS_DIR / "random_forest.joblib",
    "XGBoost (Primary XAI Model)": MODELS_DIR / "xgboost.joblib",
    "LightGBM": MODELS_DIR / "lightgbm.joblib"
}


# =====================================================================
# CACHED RESOURCE LOADERS
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_model_pipeline(model_name: str):
    """Loads a serialized scikit-learn Pipeline containing preprocessor + classifier."""
    path = MODEL_FILES.get(model_name)
    if not path or not path.exists():
        st.error(f"Model file not found at: {path}")
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def get_xgb_tree_explainer():
    """Initializes and caches the SHAP TreeExplainer for the XGBoost pipeline."""
    xgb_pipe = load_model_pipeline("XGBoost (Primary XAI Model)")
    if xgb_pipe is None:
        return None, None
    clf = xgb_pipe.named_steps["classifier"]
    preprocessor = xgb_pipe.named_steps["preprocessor"]
    explainer = shap.TreeExplainer(clf)
    return explainer, preprocessor


@st.cache_resource(show_spinner=False)
def get_xgb_lime_explainer():
    """Initializes and caches the LIME TabularExplainer for the XGBoost pipeline."""
    xgb_pipe = load_model_pipeline("XGBoost (Primary XAI Model)")
    train_path = SPLITS_DIR / "X_train_raw.csv"
    if xgb_pipe is None or not train_path.exists():
        return None, None, None
    X_train_raw = pd.read_csv(train_path, index_col=0)
    explainer, _, lime_predict_proba, cat_encoders = setup_lime_explainer(xgb_pipe, X_train_raw)
    return explainer, lime_predict_proba, cat_encoders


@st.cache_data(show_spinner=False)
def load_comparison_data():
    """Loads model performance comparison metrics."""
    if COMPARISON_CSV.exists():
        return pd.read_csv(COMPARISON_CSV)
    return None


@st.cache_data(show_spinner=False)
def load_global_importance_data():
    """Loads aggregated SHAP global feature importance rankings."""
    if SHAP_IMPORTANCE_CSV.exists():
        return pd.read_csv(SHAP_IMPORTANCE_CSV)
    return None


# =====================================================================
# FEATURE MAPPINGS & DESCRIPTIONS
# =====================================================================
CHECKING_STATUS_OPTIONS = {
    "Overdrawn / Deficit Balance (< 0 DM) [A11]": "A11",
    "Low to Moderate Balance (0 to < 200 DM) [A12]": "A12",
    "Healthy Balance (>= 200 DM / Salary Account >= 1 yr) [A13]": "A13",
    "No Checking Account on File [A14]": "A14"
}

CREDIT_HISTORY_OPTIONS = {
    "No Prior Credits Taken / All Loans Repaid Duly [A30]": "A30",
    "All Credits at This Bank Repaid Duly [A31]": "A31",
    "Existing Credits Repaid Duly Till Now [A32]": "A32",
    "Past Delays in Paying Off Credits [A33]": "A33",
    "Critical Account / Other Credits Existing Elsewhere [A34]": "A34"
}

PURPOSE_OPTIONS = {
    "New Car Purchase [A40]": "A40",
    "Used Car Purchase [A41]": "A41",
    "Furniture / Equipment [A42]": "A42",
    "Radio / Television / Electronics [A43]": "A43",
    "Domestic Household Appliances [A44]": "A44",
    "Repairs & Maintenance [A45]": "A45",
    "Education [A46]": "A46",
    "Retraining / Professional Course [A48]": "A48",
    "Business Investment / Expansion [A49]": "A49",
    "Other Loan Purposes [A410]": "A410"
}

SAVINGS_STATUS_OPTIONS = {
    "Low Liquid Savings (< 100 DM) [A61]": "A61",
    "Moderate Liquid Savings (100 to < 500 DM) [A62]": "A62",
    "Substantial Liquid Savings (500 to < 1000 DM) [A63]": "A63",
    "High Liquid Savings Buffer (>= 1000 DM) [A64]": "A64",
    "Unknown / No Savings Account on File [A65]": "A65"
}

EMPLOYMENT_OPTIONS = {
    "Unemployed [A71]": "A71",
    "Short Tenure (< 1 Year) [A72]": "A72",
    "1 to 4 Years [A73]": "A73",
    "4 to 7 Years [A74]": "A74",
    "Long Tenure (>= 7 Years) [A75]": "A75"
}

PERSONAL_STATUS_OPTIONS = {
    "Male: Divorced / Separated [A91]": "A91",
    "Female: Divorced / Separated / Married [A92]": "A92",
    "Male: Single [A93]": "A93",
    "Male: Married / Widowed [A94]": "A94"
}

OTHER_PARTIES_OPTIONS = {
    "None (Individual Applicant) [A101]": "A101",
    "Co-applicant [A102]": "A102",
    "Guarantor [A103]": "A103"
}

PROPERTY_MAGNITUDE_OPTIONS = {
    "Real Estate (Home / Land) [A121]": "A121",
    "Building Society Savings / Life Insurance [A122]": "A122",
    "Car or Other Property [A123]": "A123",
    "Unknown / No Collateral or Property [A124]": "A124"
}

OTHER_PLANS_OPTIONS = {
    "Bank Installment Plan [A141]": "A141",
    "Store Installment Plan [A142]": "A142",
    "None (No Other Installment Plans) [A143]": "A143"
}

HOUSING_OPTIONS = {
    "Rent [A151]": "A151",
    "Own Home [A152]": "A152",
    "Free Accommodation (Employer/Family) [A153]": "A153"
}

JOB_OPTIONS = {
    "Unemployed / Unskilled (Non-resident) [A171]": "A171",
    "Unskilled (Resident) [A172]": "A172",
    "Skilled Employee / Official [A173]": "A173",
    "Management / Self-employed / Highly Qualified Officer [A174]": "A174"
}

DEPENDENTS_OPTIONS = {
    "1 Dependent [1]": 1,
    "2 or More Dependents [2]": 2
}

TELEPHONE_OPTIONS = {
    "None / Not Registered [A191]": "A191",
    "Yes, Registered Under Customer Name [A192]": "A192"
}

FOREIGN_WORKER_OPTIONS = {
    "Foreign Worker: Yes [A201]": "A201",
    "Foreign Worker: No [A202]": "A202"
}


# =====================================================================
# HEADER & GOVERNANCE NOTICE
# =====================================================================
st.title("💳 Intelligent Credit Risk Assessment Using Explainable AI")
st.caption(
    "A Major Project System for Transparent Loan Default Risk Prediction and Auditable Decision Rationales "
    "using Machine Learning (XGBoost, LightGBM, Random Forest, Logistic Regression) paired with SHAP & LIME."
)

st.info(
    "⚖️ **Academic Prototype Notice & Governance Disclaimer:**\n"
    "This system is an academic engineering prototype trained on the historical German Credit Data benchmark. "
    "Predictions, scores, and explanations represent statistical model correlations. They **do not constitute legal or commercial lending decisions**, "
    "**do not establish causation**, and **do not prove regulatory compliance or demographic fairness** (e.g., under US ECOA or EU GDPR)."
)

tab_pred, tab_perf, tab_global_xai = st.tabs([
    "🔍 1. Applicant Risk Prediction",
    "📊 2. Model Performance & Benchmarking",
    "🌐 3. Global Explainability (SHAP & Governance)"
])


# =====================================================================
# TAB 1: APPLICANT RISK PREDICTION & LOCAL XAI
# =====================================================================
with tab_pred:
    st.subheader("Interactive Applicant Profile & Risk Assessment")
    st.markdown("Configure an applicant profile across the 20 attributes to generate an instant prediction and auditable feature attribution.")

    with st.sidebar:
        st.header("⚙️ Evaluation Engine")
        selected_model_name = st.selectbox(
            "Select Classification Model:",
            list(MODEL_FILES.keys()),
            index=2  # Default to XGBoost
        )
        st.markdown(
            "*(Pipelines contain pre-fitted StandardScaler and OneHotEncoder; raw inputs are ingested directly without external transformation.)*"
        )
        st.divider()
        st.markdown("**Decision Threshold:** `0.50` (Default mathematical cutoff)")
        st.caption("Class 0 = Good / Non-Risky\nClass 1 = Bad / Risky / Default")

    # Form with 5 Structured Sections
    with st.form("applicant_form"):
        # 1. Loan Request Details
        st.markdown("#### 📝 1. Loan Request Details")
        lcol1, lcol2, lcol3 = st.columns(3)
        with lcol1:
            credit_amount = st.number_input(
                "Loan Amount (Credit Amount in DM)",
                min_value=250,
                max_value=20000,
                value=3500,
                step=100,
                help="Requested loan principal in Deutsche Mark (historical benchmark currency)."
            )
            yearly_income_info = st.number_input(
                "Estimated Annual Income (Informational Only)",
                min_value=0,
                value=30000,
                step=1000,
                help="Benchmark Note: The German Credit dataset does not ingest raw income. Model evaluation relies on the Installment Rate (% band) below. This field is for reference only and does not affect the prediction."
            )
            st.caption("ℹ️ *Annual Income is purely informational; it is NOT passed to the model.*")
        with lcol2:
            duration = st.slider(
                "Loan Repayment Duration (Months)",
                min_value=4,
                max_value=72,
                value=24,
                step=1,
                help="Duration of the loan repayment period in months."
            )
            sel_purpose = st.selectbox(
                "Loan Purpose",
                list(PURPOSE_OPTIONS.keys()),
                index=0,
                help="Primary purpose or intended use of the requested credit facility."
            )
        with lcol3:
            installment_rate = st.slider(
                "Installment Rate (% Band of Disposable Income)",
                min_value=1,
                max_value=4,
                value=3,
                help="Installment commitment as a percentage band of borrower's disposable income (1: <20%, 2: 20-25%, 3: 25-35%, 4: >35%)."
            )

        st.divider()

        # 2. Banking & Credit History
        st.markdown("#### 🏦 2. Banking & Credit History")
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            sel_checking = st.selectbox(
                "Checking Account Status / Category",
                list(CHECKING_STATUS_OPTIONS.keys()),
                index=0,
                help="Status tier of borrower's existing checking account (not an exact balance amount)."
            )
            sel_history = st.selectbox(
                "Previous Credit Track Record",
                list(CREDIT_HISTORY_OPTIONS.keys()),
                index=2,
                help="Past repayment track record across existing and prior loans."
            )
            existing_credits = st.slider(
                "Existing Credits at This Bank",
                min_value=1,
                max_value=4,
                value=1,
                help="Number of active loan accounts currently open at this institution."
            )
        with bcol2:
            sel_savings = st.selectbox(
                "Savings Account / Liquid Assets Buffer",
                list(SAVINGS_STATUS_OPTIONS.keys()),
                index=0,
                help="Total liquid savings and bond holdings tier."
            )
            sel_plans = st.selectbox(
                "Other Installment Obligations",
                list(OTHER_PLANS_OPTIONS.keys()),
                index=2,
                help="Installment plans ongoing with other banks or department stores."
            )

        st.divider()

        # 3. Employment & Property
        st.markdown("#### 💼 3. Employment & Property")
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            sel_job = st.selectbox(
                "Employment / Job Classification",
                list(JOB_OPTIONS.keys()),
                index=2,
                help="Borrower's professional qualification level and employment classification."
            )
            sel_employment = st.selectbox(
                "Present Employment Tenure",
                list(EMPLOYMENT_OPTIONS.keys()),
                index=2,
                help="Duration of employment with current employer / business."
            )
        with ecol2:
            sel_property = st.selectbox(
                "Most Valuable Property / Collateral",
                list(PROPERTY_MAGNITUDE_OPTIONS.keys()),
                index=2,
                help="Borrower's highest-value asset category available as tangible security."
            )
            sel_housing = st.selectbox(
                "Housing Arrangement",
                list(HOUSING_OPTIONS.keys()),
                index=1,
                help="Residential accommodation status (own home, rent, or free)."
            )

        st.divider()

        # 4. Applicant Details
        st.markdown("#### 👤 4. Applicant Details")
        acol1, acol2, acol3 = st.columns(3)
        with acol1:
            age = st.slider(
                "Applicant Age (Years)",
                min_value=18,
                max_value=75,
                value=32,
                help="Age of applicant in years."
            )
        with acol2:
            sel_personal = st.selectbox(
                "Sex & Marital Status",
                list(PERSONAL_STATUS_OPTIONS.keys()),
                index=2,
                help="Demographic personal status as recorded in the German Credit benchmark."
            )
        with acol3:
            sel_parties = st.selectbox(
                "Co-applicants / Guarantors",
                list(OTHER_PARTIES_OPTIONS.keys()),
                index=0,
                help="Presence of a second liable borrower or external credit guarantor."
            )

        # 5. Additional Credit Attributes (Expander)
        with st.expander("⚙️ 5. Additional Credit Attributes (Institutional & Demographic Factors)", key="expander_additional_attributes"):
            st.caption("These secondary attributes are part of the trained 20-feature schema. Default values are pre-selected.")
            xcol1, xcol2 = st.columns(2)
            with xcol1:
                residence_since = st.slider(
                    "Present Residence Duration (Years)",
                    min_value=1,
                    max_value=4,
                    value=2,
                    help="Years applicant has lived at their current residential address."
                )
                sel_dependents = st.selectbox(
                    "Number of Financial Dependents",
                    list(DEPENDENTS_OPTIONS.keys()),
                    index=0,
                    help="Number of persons dependent on applicant for financial maintenance."
                )
            with xcol2:
                sel_telephone = st.selectbox(
                    "Telephone Registration on File",
                    list(TELEPHONE_OPTIONS.keys()),
                    index=1,
                    help="Telephone account registered under applicant name."
                )
                sel_foreign = st.selectbox(
                    "Foreign Worker Status",
                    list(FOREIGN_WORKER_OPTIONS.keys()),
                    index=0,
                    help="Regulatory foreign worker classification."
                )

        submit_btn = st.form_submit_button("🚀 Run Risk Evaluation & Local Explainability", use_container_width=True)

    # Construct Raw Dataframe
    raw_input_dict = {
        "checking_status": CHECKING_STATUS_OPTIONS[sel_checking],
        "duration": int(duration),
        "credit_history": CREDIT_HISTORY_OPTIONS[sel_history],
        "purpose": PURPOSE_OPTIONS[sel_purpose],
        "credit_amount": int(credit_amount),
        "savings_status": SAVINGS_STATUS_OPTIONS[sel_savings],
        "employment": EMPLOYMENT_OPTIONS[sel_employment],
        "installment_rate": int(installment_rate),
        "personal_status": PERSONAL_STATUS_OPTIONS[sel_personal],
        "other_parties": OTHER_PARTIES_OPTIONS[sel_parties],
        "residence_since": int(residence_since),
        "property_magnitude": PROPERTY_MAGNITUDE_OPTIONS[sel_property],
        "age": int(age),
        "other_payment_plans": OTHER_PLANS_OPTIONS[sel_plans],
        "housing": HOUSING_OPTIONS[sel_housing],
        "existing_credits": int(existing_credits),
        "job": JOB_OPTIONS[sel_job],
        "num_dependents": int(DEPENDENTS_OPTIONS[sel_dependents]),
        "telephone": TELEPHONE_OPTIONS[sel_telephone],
        "foreign_worker": FOREIGN_WORKER_OPTIONS[sel_foreign]
    }

    # Persist evaluation state so the results & XAI selector stay visible on radio clicks
    if submit_btn:
        st.session_state["has_evaluated"] = True
        st.session_state["evaluated_input_dict"] = raw_input_dict

    if st.session_state.get("has_evaluated", False):
        evaluated_dict = st.session_state.get("evaluated_input_dict", raw_input_dict)
        eval_df = pd.DataFrame([evaluated_dict])
        pipeline = load_model_pipeline(selected_model_name)
        if pipeline is not None:
            # 1. Validation of Class Ordering
            classes = list(pipeline.classes_)
            if classes != [0, 1]:
                st.error(f"Unexpected model class ordering: {classes}. Expected [0, 1].")
            else:
                probs = pipeline.predict_proba(eval_df)[0]
                p_good = float(probs[0])
                p_risky = float(probs[1])
                pred_label = int(pipeline.predict(eval_df)[0])

                st.divider()
                st.markdown("### 📋 Risk Assessment Outcome")

                res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1.2])

                with res_col1:
                    if pred_label == 1:
                        st.error("### ⚠️ Classification: RISKY APPLICANT")
                        st.markdown(
                            f"**Recommended Action:** Refer for Senior Credit Committee Review or Request Additional Collateral.\n"
                            f"*(Default probability exceeds default 0.50 cutoff)*"
                        )
                    else:
                        st.success("### ✅ Classification: NON-RISKY APPLICANT")
                        st.markdown(
                            f"**Recommended Action:** Standard Loan Processing & Document Verification.\n"
                            f"*(Default probability within acceptable baseline)*"
                        )

                with res_col2:
                    st.metric(
                        label="Predicted Default Risk P(Risky = 1)",
                        value=f"{p_risky:.1%}",
                        delta=f"{(p_risky - 0.50):+.1%} vs 0.50 threshold",
                        delta_color="inverse"
                    )
                    st.metric(
                        label="Predicted Solvency P(Non-Risky = 0)",
                        value=f"{p_good:.1%}"
                    )

                with res_col3:
                    st.markdown("**Probability Gauge:**")
                    st.progress(p_risky, text=f"Risk Score: {p_risky:.1%}")
                    st.caption(
                        f"Active Model: **{selected_model_name}**\n"
                        f"Threshold: **0.50 (Uncalibrated mathematical cutoff)**"
                    )

                # =================================================================
                # LOCAL EXPLAINABILITY SECTION (STRICTLY FOR CURRENT APPLICANT)
                # =================================================================
                st.markdown("---")
                st.markdown("### 🔬 Local Explainability for Current Applicant")

                if "XGBoost" in selected_model_name:
                    st.markdown(
                        f"**Model Explaining:** `XGBoost Pipeline` (Auditable Local Attribution for this exact applicant)\n"
                        f"Select your preferred Explainable AI framework below to audit the applicant's risk factors in real time."
                    )

                    xai_choice = st.radio(
                        "Select Local Explanation Framework:",
                        [
                            "🌳 SHAP (TreeExplainer — Additive Log-Odds)",
                            "🍋 LIME (Local Surrogate — Interpretable Decision Rules)",
                            "📊 Both (Side-by-Side Comparison)"
                        ],
                        horizontal=True,
                        key="xai_framework_selector"
                    )

                    def render_shap_local():
                        explainer, preprocessor = get_xgb_tree_explainer()
                        if explainer is None or preprocessor is None:
                            st.error("SHAP explainer could not be loaded.")
                            return

                        try:
                            # Transform the single current applicant row
                            trans_row = preprocessor.transform(eval_df)
                            shap_obj = explainer(trans_row)

                            encoded_names = [
                                c.replace("num__", "").replace("cat__", "")
                                for c in preprocessor.get_feature_names_out()
                            ]

                            instance_shap = shap_obj.values[0]
                            base_val = float(shap_obj.base_values[0])
                            sum_val = float(base_val + instance_shap.sum())

                            # Top 10 contributing features
                            top_indices = np.argsort(np.abs(instance_shap))[::-1][:10]
                            top_features = [encoded_names[i] for i in top_indices]
                            top_weights = [instance_shap[i] for i in top_indices]

                            fig, ax = plt.subplots(figsize=(8.5, 4.6))
                            colors = ["#d9534f" if w > 0 else "#27ae60" for w in top_weights[::-1]]
                            bars = ax.barh(top_features[::-1], top_weights[::-1], color=colors, edgecolor="black", height=0.6)

                            ax.axvline(0, color="black", linestyle="--", alpha=0.7)
                            ax.set_title(
                                f"SHAP Local Drivers for Current Applicant\n(Base Log-Odds: {base_val:.2f} ➔ Final: {sum_val:.2f})",
                                fontsize=10.5,
                                fontweight="bold"
                            )
                            ax.set_xlabel("SHAP Value (Impact on Log-Odds of Default)", fontsize=9)
                            ax.grid(axis="x", linestyle="--", alpha=0.5)

                            for bar in bars:
                                w = bar.get_width()
                                align = "left" if w >= 0 else "right"
                                ax.annotate(
                                    f"{w:+.3f}",
                                    xy=(w, bar.get_y() + bar.get_height() / 2),
                                    xytext=(4 if w >= 0 else -4, 0),
                                    textcoords="offset points",
                                    va="center",
                                    ha=align,
                                    fontsize=8,
                                    fontweight="bold"
                                )

                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                            st.caption(
                                "💡 **SHAP Interpretation:** Features extending right (red) increase default log-odds. "
                                "Features extending left (green) act as protective factors reducing default risk."
                            )
                        except Exception as e:
                            st.error(f"Error computing SHAP local explanation: {e}")

                    def render_lime_local():
                        try:
                            lime_explainer, lime_predict_proba, cat_encoders = get_xgb_lime_explainer()
                            if lime_explainer is None or lime_predict_proba is None or cat_encoders is None:
                                st.error("LIME explainer could not be loaded. Please ensure data/splits/X_train_raw.csv exists.")
                                return

                            # Encode categorical strings to integer indices for LIME perturbation
                            raw_row = evaluated_dict.copy()
                            for col in PARENT_CATEGORICAL_FEATURES:
                                cats = cat_encoders[col]
                                val = raw_row[col]
                                raw_row[col] = cats.index(val) if val in cats else 0
                            instance_vals = pd.Series(raw_row)[lime_explainer.feature_names].values.astype(float)

                            with st.spinner("Generating LIME local surrogate explanation..."):
                                exp = lime_explainer.explain_instance(
                                    instance_vals,
                                    lime_predict_proba,
                                    num_features=8,
                                    labels=[1]
                                )

                            lime_list = exp.as_list(label=1)
                            rules = [r for r, w in lime_list][::-1]
                            weights = [w for r, w in lime_list][::-1]
                            colors = ["#d9534f" if w > 0 else "#27ae60" for w in weights]

                            fig_lime, ax_lime = plt.subplots(figsize=(8.5, 4.6))
                            bars_lime = ax_lime.barh(rules, weights, color=colors, edgecolor="black", height=0.6)
                            ax_lime.axvline(0, color="black", linestyle="--", alpha=0.7)
                            ax_lime.set_title(
                                f"LIME Local Explanation for Current Applicant\n(Target: Risky = 1 | Predicted Risk: {p_risky:.1%})",
                                fontsize=10.5,
                                fontweight="bold"
                            )
                            ax_lime.set_xlabel("LIME Local Surrogate Weight (Marginal Contribution to P(Risky=1))", fontsize=9)
                            ax_lime.grid(axis="x", linestyle="--", alpha=0.5)

                            for bar in bars_lime:
                                w = bar.get_width()
                                align = "left" if w >= 0 else "right"
                                ax_lime.annotate(
                                    f"{w:+.3f}",
                                    xy=(w, bar.get_y() + bar.get_height() / 2),
                                    xytext=(4 if w >= 0 else -4, 0),
                                    textcoords="offset points",
                                    va="center",
                                    ha=align,
                                    fontsize=8,
                                    fontweight="bold"
                                )

                            plt.tight_layout()
                            st.pyplot(fig_lime)
                            plt.close(fig_lime)

                            st.caption(
                                "💡 **LIME Interpretation:** LIME fits a local linear surrogate model around this applicant. "
                                "Red rules indicate conditions increasing default probability; green rules indicate protective conditions."
                            )
                        except Exception as e:
                            st.error(f"Error computing LIME local explanation: {e}")

                    if "SHAP" in xai_choice:
                        render_shap_local()
                    elif "LIME" in xai_choice:
                        render_lime_local()
                    else:
                        c_shap, c_lime = st.columns(2)
                        with c_shap:
                            st.markdown("#### 🌳 SHAP Explanation")
                            render_shap_local()
                        with c_lime:
                            st.markdown("#### 🍋 LIME Explanation")
                            render_lime_local()

                else:
                    st.warning(
                        f"ℹ️ **Local Explanation Model Notice:**\n"
                        f"The active prediction above was generated by **{selected_model_name}**. "
                        f"The dedicated local explainers (SHAP TreeExplainer & LIME Tabular) are configured for the **XGBoost (Primary XAI Model)** ensemble.\n\n"
                        f"To view live local SHAP and LIME feature attributions for this applicant profile, please select **XGBoost (Primary XAI Model)** from the sidebar."
                    )


# =====================================================================
# TAB 2: MODEL PERFORMANCE & BENCHMARKING
# =====================================================================
with tab_perf:
    st.subheader("Model Performance & Independent Test Evaluation")
    st.markdown(
        "Performance benchmarks evaluated across 5-fold Stratified Cross-Validation on training data (n=800) "
        "and verified once on the held-out test split (n=200). Models were fitted inside zero-leakage scikit-learn Pipelines."
    )

    metrics_df = load_comparison_data()
    if metrics_df is not None:
        st.markdown("#### 📊 Comparative Evaluation Matrix")
        st.dataframe(
            metrics_df.style.format({
                "CV ROC-AUC (mean)": "{:.4f}",
                "CV ROC-AUC (std)": "{:.4f}",
                "Test Accuracy": "{:.2%}",
                "Test ROC-AUC": "{:.4f}",
                "Test PR-AUC": "{:.4f}",
                "Test F1 (Risky)": "{:.4f}",
                "Test Recall / TPR": "{:.2%}",
                "Test Precision (Risky)": "{:.2%}",
                "Test FPR": "{:.2%}"
            }).highlight_max(subset=["Test ROC-AUC", "Test PR-AUC", "Test Recall / TPR"], color="#d4edda")
              .highlight_min(subset=["Test FPR"], color="#d4edda"),
            use_container_width=True
        )
    else:
        st.error("Model comparison metrics CSV not found in reports directory.")

    st.markdown("---")
    st.markdown("#### 📈 Visual Diagnostic Plots")

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        roc_path = FIGURES_DIR / "roc_curves.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="Receiver Operating Characteristic (ROC) Curves on Held-Out Test Set", use_container_width=True)

    with plot_col2:
        pr_path = FIGURES_DIR / "precision_recall_curves.png"
        if pr_path.exists():
            st.image(str(pr_path), caption="Precision-Recall Curves against Baseline Positive Prevalence (0.30)", use_container_width=True)

    cm_path = FIGURES_DIR / "confusion_matrices.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion Matrices across All Four Candidate Models (Held-Out Test Set, n=200)", use_container_width=True)


# =====================================================================
# TAB 3: GLOBAL EXPLAINABILITY (SHAP & GOVERNANCE)
# =====================================================================
with tab_global_xai:
    st.subheader("Global Model Explainability & Feature Rankings")
    st.markdown(
        "Global feature importance reveals which attributes have the greatest systematic influence on credit risk scores "
        "across the entire applicant population."
    )

    imp_df = load_global_importance_data()
    if imp_df is not None:
        gx_col1, gx_col2 = st.columns([1, 1.3])
        with gx_col1:
            st.markdown("#### 🏆 Aggregated Parent-Feature Importance")
            st.markdown(
                "*(Computed by summing absolute SHAP values across all one-hot levels per parent feature, then averaging across test borrowers)*"
            )
            st.dataframe(
                imp_df.style.format({
                    "Aggregated_Mean_Abs_SHAP": "{:.4f}"
                }).bar(subset=["Aggregated_Mean_Abs_SHAP"], color="#3498db"),
                use_container_width=True,
                height=450
            )

        with gx_col2:
            bar_path = FIGURES_DIR / "shap_feature_importance_bar.png"
            if bar_path.exists():
                st.image(str(bar_path), caption="Aggregated SHAP Feature Importance Ranking for XGBoost", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🐝 SHAP Summary (Beeswarm) Distribution")
    st.markdown(
        "The beeswarm plot illustrates feature value distribution and directional impact on default log-odds. "
        "Each dot represents a test borrower. Red dots denote high feature values; blue dots denote low values."
    )
    beeswarm_path = FIGURES_DIR / "shap_summary_beeswarm.png"
    if beeswarm_path.exists():
        st.image(str(beeswarm_path), caption="SHAP Summary (Beeswarm) Plot for XGBoost", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🍋 Historical Benchmark LIME Case Studies")
    with st.expander("📑 View Programmatically Audited Test-Set Case Studies (Applicant #522 & #772)", expanded=False):
        st.markdown(
            "Below are the verified local surrogate explanations generated during Step 6 offline auditing for two actual test-set applicants:"
        )
        lcol1, lcol2 = st.columns(2)
        with lcol1:
            lime_tp_path = FIGURES_DIR / "lime_true_positive_app522.png"
            if lime_tp_path.exists():
                st.image(str(lime_tp_path), caption="True Positive (High Risk): Applicant #522 (P=91.8%)", use_container_width=True)
        with lcol2:
            lime_tn_path = FIGURES_DIR / "lime_true_negative_app772.png"
            if lime_tn_path.exists():
                st.image(str(lime_tn_path), caption="True Negative (Low Risk): Applicant #772 (P=0.8%)", use_container_width=True)

    st.markdown("---")
    st.markdown("#### ⚖️ Explainable AI Governance & Fair Lending Standards")
    st.markdown("""
    1. **Non-Causality:** Post-hoc explanations (SHAP and LIME) measure empirical associations inside the model. They do not demonstrate real-world causality.
    2. **Regulatory Auditing:** Frameworks like the US Equal Credit Opportunity Act (ECOA) and EU GDPR 'Right to Explanation' require credit providers to justify adverse lending decisions.
    3. **Detecting Algorithmic Bias:** Sensitive attributes (such as `personal_status` and `foreign_worker`) were audited through SHAP. Transparent attribution allows institutions to detect disparate impact before deploying models into production.
    """)

# Footer
st.divider()
st.caption(
    "Intelligent Credit Risk Assessment System | B.Tech CSE Major Project 2026-27 | Built with Python, Scikit-Learn, XGBoost, SHAP, LIME & Streamlit."
)
