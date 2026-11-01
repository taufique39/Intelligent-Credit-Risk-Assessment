"""
Model Training and Evaluation Module for Intelligent Credit Risk Assessment Using Explainable AI.

Responsibilities:
1. Load split datasets (X_train_raw, X_test_raw, y_train, y_test).
2. Construct independent scikit-learn Pipelines pairing a fresh ColumnTransformer with:
   - Logistic Regression (Baseline)
   - Random Forest
   - XGBoost
   - LightGBM
3. Perform 5-fold Stratified Cross-Validation on training data strictly inside Pipelines
   to eliminate any potential data leakage between folds.
4. Fit final pipelines on the entire training set (800 rows) and evaluate on the held-out
   test set (200 rows).
5. Compute key performance metrics:
   - Accuracy, Precision (Class 1), Recall/TPR (Class 1), F1-score (Class 1)
   - ROC-AUC and PR-AUC (Average Precision) using validated Class 1 probabilities
   - Confusion Matrix (TN, FP, FN, TP) and False Positive Rate (FPR = FP / (FP + TN))
6. Save:
   - Serialized model pipelines to models/*.joblib
   - Metrics summary to reports/model_comparison.csv
   - Comprehensive written report to reports/model_evaluation_report.txt
   - Visualizations to reports/figures/ (confusion_matrices.png, roc_curves.png, precision_recall_curves.png)
7. Validate that probabilities are well-behaved and test sets remain unaltered.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from src.preprocessing import build_preprocessor

# Project Directories
BASE_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = BASE_DIR / "data" / "splits"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET_COL = "credit_risk"
RANDOM_SEED = 42


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads raw split datasets."""
    X_train = pd.read_csv(SPLITS_DIR / "X_train_raw.csv", index_col=0)
    X_test = pd.read_csv(SPLITS_DIR / "X_test_raw.csv", index_col=0)
    y_train = pd.read_csv(SPLITS_DIR / "y_train.csv", index_col=0)[TARGET_COL]
    y_test = pd.read_csv(SPLITS_DIR / "y_test.csv", index_col=0)[TARGET_COL]
    return X_train, X_test, y_train, y_test


def get_model_definitions() -> Dict[str, Any]:
    """
    Returns initial model instances with balanced, reproducible hyperparameters.
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            random_state=RANDOM_SEED
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            eval_metric="logloss"
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            verbose=-1
        )
    }


def create_pipeline(classifier: Any) -> Pipeline:
    """
    Builds a fresh, unfitted Pipeline with a clean ColumnTransformer.
    Ensures preprocessor is never reused across cross-validation folds.
    """
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier)
    ])


def evaluate_cross_validation(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Runs 5-fold Stratified Cross-Validation on training data.
    Scalers and encoders are fitted strictly within each training fold.
    """
    print(f"\n[1/4] Running {cv_folds}-Fold Stratified Cross-Validation on Training Split...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "accuracy": "accuracy",
        "recall": "recall",
        "precision": "precision"
    }

    cv_results = {}
    for name, model_instance in models.items():
        pipeline = create_pipeline(model_instance)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        cv_results[name] = {
            "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(scores["test_roc_auc"])),
            "cv_pr_auc_mean": float(np.mean(scores["test_pr_auc"])),
            "cv_pr_auc_std": float(np.std(scores["test_pr_auc"])),
            "cv_f1_mean": float(np.mean(scores["test_f1"])),
            "cv_f1_std": float(np.std(scores["test_f1"])),
            "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "cv_accuracy_std": float(np.std(scores["test_accuracy"]))
        }
        print(f"      {name:20s} -> CV ROC-AUC: {cv_results[name]['cv_roc_auc_mean']:.4f} (+/- {cv_results[name]['cv_roc_auc_std']:.4f}) | "
              f"CV PR-AUC: {cv_results[name]['cv_pr_auc_mean']:.4f} | CV F1: {cv_results[name]['cv_f1_mean']:.4f}")
    return cv_results


def fit_and_evaluate_test(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_results: Dict[str, Dict[str, float]]
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, Dict[str, Any]]:
    """
    Fits final pipelines on full training split and computes comprehensive test metrics.
    """
    print("\n[2/4] Fitting Final Pipelines & Evaluating on Held-Out Test Set (n=200)...")
    fitted_pipelines = {}
    test_metrics_list = []
    plot_data = {}

    # Snapshot of test labels to ensure zero modification
    y_test_snapshot = y_test.copy()

    for name, model_instance in models.items():
        pipeline = create_pipeline(model_instance)
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Verify class ordering
        classes = list(pipeline.classes_)
        assert classes == [0, 1], f"Unexpected class ordering for {name}: {classes}"
        risky_idx = classes.index(1)

        # Predictions & Probabilities
        y_pred = pipeline.predict(X_test)
        y_prob_all = pipeline.predict_proba(X_test)
        y_prob = y_prob_all[:, risky_idx]

        # Sanity check on probabilities
        assert np.all(y_prob >= 0.0) and np.all(y_prob <= 1.0), "Probabilities out of bounds"
        assert np.allclose(y_prob_all.sum(axis=1), 1.0), "Probabilities do not sum to 1.0"

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        metrics_record = {
            "Model": name,
            "CV ROC-AUC (mean)": round(cv_results[name]["cv_roc_auc_mean"], 4),
            "CV ROC-AUC (std)": round(cv_results[name]["cv_roc_auc_std"], 4),
            "Test Accuracy": round(accuracy, 4),
            "Test ROC-AUC": round(roc_auc, 4),
            "Test PR-AUC": round(pr_auc, 4),
            "Test F1 (Risky)": round(f1, 4),
            "Test Recall / TPR": round(tpr, 4),
            "Test Precision (Risky)": round(precision, 4),
            "Test FPR": round(fpr, 4),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        }
        test_metrics_list.append(metrics_record)

        # Store curves for plotting
        fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
        plot_data[name] = {
            "cm": cm,
            "fpr_arr": fpr_arr,
            "tpr_arr": tpr_arr,
            "roc_auc": roc_auc,
            "prec_arr": prec_arr,
            "rec_arr": rec_arr,
            "pr_auc": pr_auc
        }

        print(f"      {name:20s} -> Test ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | "
              f"Recall/TPR: {tpr:.4f} | FPR: {fpr:.4f} | F1: {f1:.4f} | Acc: {accuracy:.4f}")

    # Verify test set remained unaltered
    assert (y_test == y_test_snapshot).all(), "Test set labels were modified during evaluation!"

    metrics_df = pd.DataFrame(test_metrics_list).sort_values(by="Test ROC-AUC", ascending=False)
    return fitted_pipelines, metrics_df, plot_data


def generate_evaluation_plots(plot_data: Dict[str, Any], figures_dir: Path) -> List[Path]:
    """
    Creates publication-quality visual diagnostics:
    1. 2x2 Confusion Matrices Grid
    2. Combined ROC Curves
    3. Combined Precision-Recall Curves
    """
    print("\n[3/4] Generating Visual Performance Charts...")
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_figures = []

    # 1. Confusion Matrices Grid
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for idx, (name, data) in enumerate(plot_data.items()):
        ax = axes[idx]
        cm = data["cm"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            annot_kws={"size": 14, "weight": "bold"},
            ax=ax
        )
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        ax.set_title(f"{name}\n(TPR/Recall: {tpr:.2f} | FPR: {fpr:.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xticklabels(["Good (0)", "Risky (1)"])
        ax.set_yticklabels(["Good (0)", "Risky (1)"])

    plt.suptitle("Confusion Matrix Comparison on Held-Out Test Set (n=200)\nClass 0 = Good / Non-Risky | Class 1 = Bad / Risky",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    cm_path = figures_dir / "confusion_matrices.png"
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    generated_figures.append(cm_path)

    # 2. Combined ROC Curves
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = ["#2b5c8f", "#27ae60", "#e67e22", "#9b59b6"]

    for (name, data), color in zip(plot_data.items(), colors):
        ax.plot(
            data["fpr_arr"],
            data["tpr_arr"],
            color=color,
            linewidth=2,
            label=f"{name} (AUC = {data['roc_auc']:.3f})"
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.500)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves\n(Test Set Evaluation)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(alpha=0.4, linestyle="--")
    ax.legend(loc="lower right", fontsize=9.5)
    plt.tight_layout()
    roc_path = figures_dir / "roc_curves.png"
    fig.savefig(roc_path, dpi=300)
    plt.close(fig)
    generated_figures.append(roc_path)

    # 3. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(7.5, 6))
    baseline_pr = 60 / 200  # 30% positive class in test set

    for (name, data), color in zip(plot_data.items(), colors):
        ax.plot(
            data["rec_arr"],
            data["prec_arr"],
            color=color,
            linewidth=2,
            label=f"{name} (PR-AUC = {data['pr_auc']:.3f})"
        )

    ax.axhline(baseline_pr, color="black", linestyle="--", alpha=0.5, label=f"Baseline Prevalance ({baseline_pr:.2f})")
    ax.set_title("Precision-Recall (PR) Curves for Risky Class (1)\n(Test Set Evaluation)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Recall (True Positive Rate)", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(alpha=0.4, linestyle="--")
    ax.legend(loc="upper right", fontsize=9.5)
    plt.tight_layout()
    pr_path = figures_dir / "precision_recall_curves.png"
    fig.savefig(pr_path, dpi=300)
    plt.close(fig)
    generated_figures.append(pr_path)

    return generated_figures


def save_models_and_reports(
    fitted_pipelines: Dict[str, Pipeline],
    metrics_df: pd.DataFrame,
    cv_results: Dict[str, Dict[str, float]]
) -> Tuple[Dict[str, Path], Path, Path]:
    """
    Saves trained pipelines, comparison CSV, and written evaluation report.
    """
    print("\n[4/4] Saving Model Artifacts & Evaluation Reports...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model pipelines
    slug_map = {
        "Logistic Regression": "logistic_regression.joblib",
        "Random Forest": "random_forest.joblib",
        "XGBoost": "xgboost.joblib",
        "LightGBM": "lightgbm.joblib"
    }

    saved_models = {}
    for name, pipeline in fitted_pipelines.items():
        file_path = MODELS_DIR / slug_map[name]
        joblib.dump(pipeline, file_path)
        saved_models[name] = file_path

    # Save comparison CSV
    csv_path = REPORTS_DIR / "model_comparison.csv"
    metrics_df.to_csv(csv_path, index=False)

    # Write narrative report
    report_path = REPORTS_DIR / "model_evaluation_report.txt"
    lines = []
    lines.append("=" * 80)
    lines.append("MODEL TRAINING & EVALUATION REPORT")
    lines.append("Project: Intelligent Credit Risk Assessment Using Explainable AI")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. METHODOLOGY & LEAKAGE PREVENTION")
    lines.append("-" * 80)
    lines.append("- Cross-Validation: 5-Fold StratifiedKFold on the 800 training observations.")
    lines.append("- Zero-Leakage Guarantee: Scalers (StandardScaler) and Encoders (OneHotEncoder)")
    lines.append("  were fitted strictly inside each training fold using fresh scikit-learn Pipelines.")
    lines.append("- Held-Out Test Evaluation: The 200-sample test set was evaluated exclusively after")
    lines.append("  fitting the final pipelines on all 800 training observations.")
    lines.append("- Class Convention: 0 = Good / Non-Risky (Negative), 1 = Bad / Risky / Default (Positive).")
    lines.append("")
    lines.append("2. COMPREHENSIVE PERFORMANCE SUMMARY TABLE")
    lines.append("-" * 80)
    lines.append(metrics_df.to_string(index=False))
    lines.append("")
    lines.append("3. CROSS-VALIDATION VS. TEST GENERALIZATION ANALYSIS")
    lines.append("-" * 80)
    for _, row in metrics_df.iterrows():
        m_name = row["Model"]
        cv_auc = row["CV ROC-AUC (mean)"]
        cv_std = row["CV ROC-AUC (std)"]
        test_auc = row["Test ROC-AUC"]
        delta = test_auc - cv_auc
        lines.append(f"- {m_name}:")
        lines.append(f"  * 5-Fold CV ROC-AUC: {cv_auc:.4f} (+/- {cv_std:.4f})")
        lines.append(f"  * Held-Out Test ROC-AUC: {test_auc:.4f} (Delta vs CV: {delta:+.4f})")
        lines.append(f"  * Test Recall (TPR): {row['Test Recall / TPR']:.4f} | Test FPR: {row['Test FPR']:.4f}")
        lines.append(f"  * Confusion Counts: TN={row['TN']}, FP={row['FP']}, FN={row['FN']}, TP={row['TP']}")
        lines.append("")
    lines.append("4. KEY FINDINGS & MODEL SELECTION")
    lines.append("-" * 80)
    best_test_auc_row = metrics_df.iloc[0]
    lines.append(f"* Top Performing Model by Test ROC-AUC: {best_test_auc_row['Model']} (ROC-AUC = {best_test_auc_row['Test ROC-AUC']:.4f})")
    lines.append("* Baseline Logistic Regression demonstrates competitive discrimination and establishes")
    lines.append("  an interpretable linear benchmark.")
    lines.append("* Ensemble models (Random Forest, XGBoost, LightGBM) capture non-linear relationships")
    lines.append("  and interaction effects, making them prime candidates for post-hoc explanation via SHAP.")
    lines.append("* Trade-off Observation: At the standard 0.5 classification threshold, credit models")
    lines.append("  face a trade-off between Recall (catching defaults) and FPR (rejecting creditworthy applicants).")
    lines.append("  In production, threshold tuning based on cost matrices is advisable.")
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF EVALUATION REPORT")
    lines.append("=" * 80)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return saved_models, csv_path, report_path


def run_model_training_and_evaluation() -> None:
    """Full execution pipeline."""
    X_train, X_test, y_train, y_test = load_splits()
    models = get_model_definitions()

    cv_results = evaluate_cross_validation(models, X_train, y_train)
    fitted_pipelines, metrics_df, plot_data = fit_and_evaluate_test(
        models, X_train, y_train, X_test, y_test, cv_results
    )

    figures = generate_evaluation_plots(plot_data, FIGURES_DIR)
    saved_models, csv_path, report_path = save_models_and_reports(
        fitted_pipelines, metrics_df, cv_results
    )

    print("\n" + "=" * 70)
    print("STEP 5 COMPLETE: MODEL TRAINING & EVALUATION SUCCESSFUL")
    print("=" * 70)
    print("\nFinal Metrics Comparison:")
    print(metrics_df.to_string(index=False))
    print("\nSaved Models:")
    for name, p in saved_models.items():
        print(f"  - {name}: {p}")
    print(f"\nSaved Metrics CSV: {csv_path}")
    print(f"Saved Report:      {report_path}")
    print("Saved Figures:")
    for fig in figures:
        print(f"  - {fig}")


if __name__ == "__main__":
    run_model_training_and_evaluation()
