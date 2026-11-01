# Intelligent Credit Risk Assessment Using Explainable AI (XAI)

An end-to-end, auditable machine learning research framework for credit risk evaluation, combining high-performing gradient-boosted decision trees and linear baselines with post-hoc explainability frameworks (**SHAP** and **LIME**). Features an interactive academic research decision-support application built with **Streamlit**, designed for credit risk assessment, regulatory interpretability research, and model auditing demonstrations.

---

## 📌 Project Overview & Key Features

Modern credit risk modeling increasingly explores complex, non-linear machine learning architectures that deliver superior predictive discrimination compared to traditional credit scorecards. However, regulatory frameworks such as the **US Equal Credit Opportunity Act (ECOA) / Regulation B** and **EU General Data Protection Regulation (GDPR) Article 22** require financial institutions to provide clear, adverse-action explanations for automated or algorithmic credit decisions.

This project investigates the balance between predictive performance and regulatory interpretability:

- **Multi-Model Benchmark:** Evaluates four diverse machine learning architectures: Logistic Regression (interpretable linear baseline), Random Forest, XGBoost (primary explainability engine), and LightGBM.
- **Strict Leakage Prevention:** Built on scikit-learn `Pipeline` and `ColumnTransformer` constructs; feature scalers and categorical encoders are fitted strictly inside each cross-validation fold.
- **Dual-Layer Explainability:**
  - **Global Interpretability (SHAP TreeExplainer):** Aggregates 61 one-hot encoded levels back to the 20 original parent financial attributes to expose systematic population-level risk drivers.
  - **Local Interpretability (SHAP & LIME):** Provides applicant-specific attributions in real time, contrasting game-theoretic Shapley log-odds contributions against intuitive LIME decision rules.
- **Interactive Three-Tab Decision-Support Application:**
  1. *Applicant Risk Prediction & Real-Time Local XAI:* Streamlined loan underwriting form with probability gauge and dynamic SHAP/LIME visual attributions.
  2. *Model Performance & Benchmarking:* Comparative ROC curves, Precision-Recall curves, confusion matrices, and 5-fold cross-validation metrics.
  3. *Global Explainability & Governance:* Population-level feature importance rankings, SHAP beeswarm distributions, historical benchmark case studies, and fair lending guidelines.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Language & Environment** | Python 3.10+ / 3.11 / 3.12 / 3.13 |
| **Data Processing** | NumPy, Pandas, SciPy |
| **Machine Learning** | Scikit-Learn, XGBoost, LightGBM, Joblib |
| **Explainable AI (XAI)** | SHAP (TreeExplainer, LinearExplainer), LIME (LimeTabularExplainer) |
| **Visualization** | Matplotlib, Seaborn |
| **Web Application** | Streamlit |
| **Deployment & Cloud** | Streamlit Community Cloud, GitHub, Debian Linux (`libgomp1`) |

---

## 🏛️ Project Architecture

```text
Intelligent-Credit-Risk-Assessment/
├── app.py                   # Interactive Streamlit decision-support application
├── requirements.txt         # Core Python package dependencies
├── packages.txt             # Linux system runtime dependencies (libgomp1)
├── .gitignore               # Version control exclusion rules
├── README.md                # Project documentation and architecture guide
│
├── data/
│   ├── raw/                 # Benchmark Statlog German Credit dataset (german.data)
│   └── splits/              # Stratified 80/20 train/test splits (raw & transformed)
│       ├── X_train_raw.csv
│       ├── X_test_raw.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── X_train_transformed.csv
│       └── X_test_transformed.csv
│
├── models/                  # Serialized scikit-learn pipelines (.joblib)
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── lightgbm.joblib
│   └── preprocessor.joblib
│
├── reports/                 # Empirical evaluation & audit outputs
│   ├── model_comparison.csv
│   ├── model_evaluation_report.txt
│   ├── shap_global_importance.csv
│   ├── xai_explanation_report.txt
│   └── figures/             # High-resolution ROC, PR, CM, SHAP & LIME plots
│
└── src/                     # Modular backend pipeline source code
    ├── data_loader.py       # Automated dataset ingestion & schema mapping
    ├── eda.py               # Exploratory data analysis & statistical distributions
    ├── preprocessing.py     # Stratified splitting & ColumnTransformer preprocessing
    ├── train.py             # 5-fold cross-validation & model evaluation pipeline
    └── explainability.py    # SHAP & LIME explainability engines & report generator
```

---

## 📊 Dataset Description

The system uses the historical **Statlog (German Credit Data)** benchmark dataset from the **UCI Machine Learning Repository**:

- **Total Observations:** 1,000 historical credit applicants.
- **Target Variable (`credit_risk`):** Categorizes applicants into credit risk classifications:
  - `0 = Good Credit Risk` (700 instances / 70.0% prevalence)
  - `1 = Bad Credit Risk` (300 instances / 30.0% prevalence)
- **Currency:** Denominated in **Deutsche Mark (DM)**.
- **Attributes:** 20 original parent features (7 continuous/discrete numerical, 13 categorical):
  - *Financial & Banking:* Checking account status (`checking_status`), savings balance (`savings_status`), credit repayment history (`credit_history`), existing credits at this bank (`existing_credits`).
  - *Loan Details:* Duration in months (`duration`), requested amount in DM (`credit_amount`), installment rate as % of disposable income (`installment_rate`), intended loan purpose (`purpose`).
  - *Personal & Socio-Demographic:* Age (`age`), employment tenure (`employment`), housing status (`housing`), property magnitude (`property_magnitude`), personal status / gender (`personal_status`), foreign worker status (`foreign_worker`), dependents (`num_dependents`), telephone (`telephone`).

> **Methodological Note on Target Terminology:** The benchmark labels reflect a composite assessment of borrower creditworthiness and credit risk assigned by financial evaluators; they represent good vs. bad credit risk classifications rather than direct observations of permanent loan defaults.

---

## 📈 Model Comparison & Benchmark Results

All candidate models were evaluated using **5-Fold Stratified Cross-Validation** on the training split ($n=800$) and independently evaluated once on the strictly held-out test split ($n=200$, 140 Good / 60 Bad). All transformations were performed within zero-leakage scikit-learn Pipelines.

### Comprehensive Performance Benchmark

| Model Candidate | 5-Fold CV ROC-AUC (mean ± std) | Test Accuracy | Test ROC-AUC | Test PR-AUC | Test Precision (Risky) | Test Recall / TPR (Risky) | Test F1 (Risky) | Test FPR | Confusion Matrix (TN, FP, FN, TP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.7706 ± 0.0535 | **78.00%** | **0.8040** | **0.6351** | 66.67% | **53.33%** | **0.5926** | 11.43% | TN=124, FP=16, FN=28, TP=32 |
| **XGBoost (Primary XAI Engine)** | **0.7889 ± 0.0396** | 76.50% | 0.7890 | 0.6334 | 65.85% | 45.00% | 0.5347 | 10.00% | TN=126, FP=14, FN=33, TP=27 |
| **LightGBM** | 0.7869 ± 0.0496 | **78.00%** | 0.7883 | 0.5930 | 68.18% | 50.00% | 0.5769 | 10.00% | TN=126, FP=14, FN=30, TP=30 |
| **Random Forest** | 0.7888 ± 0.0422 | 77.50% | 0.7879 | 0.6249 | **74.19%** | 38.33% | 0.5055 | **5.71%** | TN=132, FP=8, FN=37, TP=23 |

### Key Benchmark Insights
- **XGBoost Generalization:** Shows high stability across cross-validation and test evaluation:
  $$\Delta_{\text{ROC-AUC}} = \text{Test ROC-AUC } (0.7890) - \text{Mean CV ROC-AUC } (0.7889) = +0.0001$$
- **Linear Benchmark:** Logistic Regression achieved the highest test ROC-AUC ($0.8040$) and bad credit risk recall ($53.33\%$), serving as a strong parametric baseline.
- **Conservative Screening:** Random Forest achieved the highest precision ($74.19\%$) and lowest false alarm rate ($\text{FPR} = 5.71\%$, with only 8 false positives out of 140 non-risky applicants).

---

## 🔬 Explainable AI (XAI) Implementation

### 1. SHAP (Shapley Additive exPlanations)
- **Engine:** `shap.TreeExplainer` applied to the native booster of the fitted `XGBoost` pipeline.
- **Output Space:** Evaluated in log-odds margin space for Class 1 (Bad Credit Risk). Positive SHAP values increase the log-odds of being classified as bad credit risk; negative values act as protective factors reducing the risk score.
- **Mathematical Identity:** The exact relation $\hat{P}(\text{Bad Credit Risk}=1) = \sigma\left(\phi_0 + \sum \phi_i\right)$ matches pipeline `predict_proba` with a max error of $1.79 \times 10^{-7}$.
- **Aggregation:** Sums absolute Shapley contributions across all one-hot levels per applicant before averaging across the test population:
  $$\text{Importance}(F) = \frac{1}{N} \sum_{i=1}^{N} \sum_{j \in \text{levels}(F)} |\phi_{i,j}|$$
- **Top Population Risk Drivers:**
  1. `checking_status` (Mean $|\text{SHAP}| = 0.8079$) — Overdrawn balances (`A11`) strongly increase risk scores; no checking account (`A14`) strongly protects.
  2. `duration` (Mean $|\text{SHAP}| = 0.3837$) — Longer loan tenures monotonically increase risk scores.
  3. `credit_history` (Mean $|\text{SHAP}| = 0.3456$) — Past delinquency history increases risk scores.
  4. `purpose` (Mean $|\text{SHAP}| = 0.3395$) — Specific loan usage types influence repayment likelihood.
  5. `credit_amount` (Mean $|\text{SHAP}| = 0.3050$) — High requested principal amounts elevate risk profile.

### 2. LIME (Local Interpretable Model-agnostic Explanations)
- **Engine:** `LimeTabularExplainer` operating directly on the 20 raw feature representations.
- **Pipeline Coupling:** A custom decoding wrapper maps discrete integer samples back to categorical strings, executing the full scikit-learn pipeline for perturbed instance evaluations.
- **Surrogate Weights:** Interpreted as local surrogate linear weights approximating the model's decision surface within an $\epsilon$-neighborhood, providing intuitive threshold rules (e.g., `duration > 24.00`, `checking_status=A11`).

### 3. Ethical & Governance Framework
1. **Non-Causality:** SHAP and LIME values quantify statistical associations within the trained model; they do not imply real-world causation (e.g., artificially reducing loan duration does not guarantee borrower creditworthiness).
2. **Fairness Auditing:** Explainability serves as an auditing tool to detect empirical reliance on sensitive attributes (such as `personal_status` or `foreign_worker`), facilitating fair-lending reviews prior to deployment.

---

## 💻 Local Installation & Execution

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- Git installed.

### Step-by-Step Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/Intelligent-Credit-Risk-Assessment.git
   cd Intelligent-Credit-Risk-Assessment
   ```

2. **Create and Activate a Virtual Environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the Interactive Decision-Support Application:**
   ```bash
   streamlit run app.py
   ```
   The application will automatically open in your default browser at `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud Deployment

This repository is pre-configured for one-click deployment on **Streamlit Community Cloud**:

1. **Push Repository to GitHub:**
   Ensure `app.py`, `requirements.txt`, `packages.txt`, `models/`, `data/splits/`, `reports/`, and `src/` are committed and pushed to your GitHub repository.
2. **Access Streamlit Cloud:**
   Log in to [share.streamlit.io](https://share.streamlit.io/) using your GitHub account.
3. **Deploy New App:**
   - Click **New app**.
   - Select your repository: `<your-username>/Intelligent-Credit-Risk-Assessment`.
   - Set **Branch:** `main` (or `master`).
   - Set **Main file path:** `app.py`.
   - Under **Advanced settings**, ensure Python version is set to **3.11** (or 3.10 / 3.12).
4. **Deploy:**
   Click **Deploy!** Streamlit Cloud will automatically:
   - Install Debian system dependencies from `packages.txt` (`libgomp1` for LightGBM/XGBoost).
   - Install Python libraries from `requirements.txt`.
   - Host and serve the interactive decision-support application globally.

---

## 📄 License & Academic Citation

Developed as an Academic Major Capstone Project in Computer Science and Engineering (2026–27). Benchmark data courtesy of the UCI Machine Learning Repository (Statlog German Credit Data).
