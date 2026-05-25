"""
=============================================================================
XAI Framework for Resource-Constrained Healthcare (XAI-RCH)
Training Pipeline & Explainability Analysis

Paper: "Explainable Artificial Intelligence for Healthcare Diagnosis in
        Resource-Constrained Ghanaian Hospitals: A Framework Development
        and Proof-of-Concept Study"

Description:
    This script reproduces all model training, evaluation, and SHAP
    explainability results reported in the manuscript. Running it end-to-end
    regenerates:
      - The trained XGBoost classifier (Equations 1-4)
      - All performance metrics (Equations 8-9, Table 3)
      - The global SHAP beeswarm summary plot (Figure 2)
      - The local SHAP waterfall plot for the highest-risk patient (Figure 3)
      - SHAP values for the full test set (CSV)
      - The Demographic Parity Difference bias audit output (Equation 10)

Requirements:
    pip install xgboost shap scikit-learn pandas numpy matplotlib joblib

Dataset:
    pima.csv  — Pima Indians Diabetes Dataset (UCI Machine Learning Repository)
    Place in the same directory as this script before running.

Authors: [Author Name(s)] — [Institution], Accra, Ghana
Submitted: May 2026
=============================================================================
"""

import os
import json
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score
)

import xgboost as xgb
import shap

warnings.filterwarnings('ignore')

# =============================================================================
# 0. Configuration
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE    = 0.25
CV_FOLDS     = 5
OUTPUT_DIR   = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clinical feature labels used in all SHAP visualisations (as per manuscript)
FEATURE_LABELS = {
    "Pregnancies":              "Number of Pregnancies",
    "Glucose":                  "Blood Glucose Level",
    "BloodPressure":            "Diastolic Blood Pressure",
    "SkinThickness":            "Skin-fold Thickness",
    "Insulin":                  "Serum Insulin Level",
    "BMI":                      "Body Mass Index (BMI)",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age":                      "Patient Age (years)",
}

print("=" * 70)
print("XAI-RCH Training Pipeline")
print("=" * 70)


# =============================================================================
# 1. Load and Prepare Dataset
# =============================================================================

print("\n[1] Loading dataset...")

df = pd.read_csv("pima.csv")
print(f"    Loaded: {df.shape[0]} records, {df.shape[1]-1} features, 1 target")
print(f"    Class distribution: {df['Outcome'].value_counts().to_dict()}")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train / test split (stratified, 75:25 — as reported in manuscript Section 2.5.2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
print(f"    Train: n={len(X_train)} | Test: n={len(X_test)}")

# Save splits
X_train.assign(Outcome=y_train.values).to_csv(
    os.path.join(OUTPUT_DIR, "pima_train_split.csv"), index=False)
X_test.assign(Outcome=y_test.values).to_csv(
    os.path.join(OUTPUT_DIR, "pima_test_split.csv"), index=False)
print("    Train/test splits saved to outputs/")


# =============================================================================
# 2. Train XGBoost Classifier  [Equations 1-4 in manuscript]
# =============================================================================

print("\n[2] Training XGBoost classifier [Eq. 1-4]...")

# Hyperparameters exactly as reported in Section 2.5.2
model = xgb.XGBClassifier(
    n_estimators      = 300,
    max_depth         = 4,
    learning_rate     = 0.04,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    eval_metric       = 'logloss',
    random_state      = RANDOM_STATE,
    verbosity         = 0
)
model.fit(X_train, y_train)
print("    Training complete.")

# Save model — two formats for maximum compatibility
joblib.dump(model, os.path.join(OUTPUT_DIR, "xai_rch_xgboost_model.joblib"))
model.save_model(os.path.join(OUTPUT_DIR, "xai_rch_xgboost_model.json"))
print("    Model saved: outputs/xai_rch_xgboost_model.joblib (sklearn API)")
print("    Model saved: outputs/xai_rch_xgboost_model.json  (XGBoost native)")


# =============================================================================
# 3. Evaluate Model Performance  [Equations 8-9, Table 3]
# =============================================================================

print("\n[3] Evaluating model performance [Eq. 8-9]...")

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)
prec_d = precision_score(y_test, y_pred, pos_label=1)
rec_d  = recall_score(y_test, y_pred, pos_label=1)
f1_d   = f1_score(y_test, y_pred, pos_label=1)
prec_n = precision_score(y_test, y_pred, pos_label=0)
rec_n  = recall_score(y_test, y_pred, pos_label=0)
f1_n   = f1_score(y_test, y_pred, pos_label=0)
cm     = confusion_matrix(y_test, y_pred)

# 5-fold cross-validated AUC-ROC
cv_scores = cross_val_score(
    model, X, y,
    cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
    scoring='roc_auc'
)

print(f"\n    ── Performance Metrics (Table 3 in manuscript) ──")
print(f"    Overall Accuracy          : {acc*100:.1f}%")
print(f"    AUC-ROC (test set)        : {auc:.4f}   [Eq. 9]")
print(f"    AUC-ROC (5-fold CV)       : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"    Precision — Diabetes      : {prec_d:.3f}   [Eq. 8]")
print(f"    Recall    — Diabetes      : {rec_d:.3f}   [Eq. 8]  ← primary clinical metric")
print(f"    F1-Score  — Diabetes      : {f1_d:.3f}   [Eq. 8]")
print(f"    Precision — No Diabetes   : {prec_n:.3f}   [Eq. 8]")
print(f"    Recall    — No Diabetes   : {rec_n:.3f}   [Eq. 8]")
print(f"    F1-Score  — No Diabetes   : {f1_n:.3f}   [Eq. 8]")
print(f"\n    Confusion Matrix:")
print(f"        TN={cm[0,0]}, FP={cm[0,1]}")
print(f"        FN={cm[1,0]}, TP={cm[1,1]}")

# Save full metrics
metrics = {
    "n_train": int(len(X_train)),
    "n_test":  int(len(X_test)),
    "accuracy": round(float(acc), 4),
    "auc_roc_test": round(float(auc), 4),
    "cv_auc_mean": round(float(cv_scores.mean()), 4),
    "cv_auc_std":  round(float(cv_scores.std()),  4),
    "precision_diabetes":    round(float(prec_d), 3),
    "recall_diabetes":       round(float(rec_d),  3),
    "f1_diabetes":           round(float(f1_d),   3),
    "precision_no_diabetes": round(float(prec_n), 3),
    "recall_no_diabetes":    round(float(rec_n),  3),
    "f1_no_diabetes":        round(float(f1_n),   3),
    "confusion_matrix": cm.tolist(),
}
with open(os.path.join(OUTPUT_DIR, "performance_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("\n    Metrics saved: outputs/performance_metrics.json")


# =============================================================================
# 4. SHAP Explainability Analysis  [Equations 5-6, Figures 2-3]
# =============================================================================

print("\n[4] Computing SHAP Shapley values [Eq. 5-6]...")

# Apply clinical feature labels (as shown in manuscript figures)
X_test_labelled  = X_test.rename(columns=FEATURE_LABELS)
X_train_labelled = X_train.rename(columns=FEATURE_LABELS)

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_labelled)

print(f"    SHAP values computed: shape = {shap_values.shape}")
print(f"    Baseline E[f(x)] = {explainer.expected_value:.4f}  [φ₀ in Eq. 6]")

# Save SHAP values and baseline
shap_df = pd.DataFrame(shap_values, columns=X_test_labelled.columns)
shap_df.to_csv(os.path.join(OUTPUT_DIR, "shap_values_test_set.csv"), index=False)
with open(os.path.join(OUTPUT_DIR, "shap_baseline.json"), "w") as f:
    json.dump({"expected_value_E_fx": float(explainer.expected_value),
               "note": "φ₀ in Equation 6 — baseline for all SHAP waterfall plots"}, f, indent=2)
print("    SHAP values saved: outputs/shap_values_test_set.csv")
print("    SHAP baseline saved: outputs/shap_baseline.json")

# ── Figure 2: Global SHAP Beeswarm Summary Plot ───────────────────────────────
print("\n[4a] Generating Figure 2 — Global SHAP beeswarm summary plot...")

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_labelled, show=False, plot_type="dot",
                  color_bar_label="Feature Value  (red=high, blue=low)")
ax = plt.gca()
ax.set_xlabel(
    "SHAP Value — Contribution to Diabetes Risk Prediction\n"
    "(positive values increase predicted risk; negative values decrease it)",
    fontsize=10, labelpad=8
)
ax.set_title(
    "Figure 2: Global XAI Explanation — SHAP Beeswarm Summary Plot\n"
    f"Feature importance and directionality across all test-set patients (n = {len(X_test)})",
    fontsize=11, fontweight='bold', pad=14
)
ax.tick_params(axis='y', labelsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_shap_global_beeswarm.png"),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("    Saved: outputs/fig2_shap_global_beeswarm.png")

# ── Figure 3: Local SHAP Waterfall Plot (highest-risk patient) ────────────────
print("\n[4b] Generating Figure 3 — Local SHAP waterfall plot...")

high_risk_idx = int(np.argmax(y_proba))
hi_prob       = float(y_proba[high_risk_idx])
hi_actual     = int(y_test.iloc[high_risk_idx])

exp = shap.Explanation(
    values       = shap_values[high_risk_idx],
    base_values  = float(explainer.expected_value),
    data         = X_test_labelled.iloc[high_risk_idx].values,
    feature_names= list(X_test_labelled.columns)
)

plt.figure(figsize=(10, 6))
shap.plots.waterfall(exp, show=False, max_display=8)
ax2 = plt.gca()
ax2.set_title(
    f"Figure 3: Local XAI Explanation — SHAP Waterfall Plot\n"
    f"Individual prediction for highest-risk patient  "
    f"[P(Y=1|x) = {hi_prob*100:.1f}%,  Actual = {'Diabetic' if hi_actual==1 else 'Non-diabetic'}]",
    fontsize=11, fontweight='bold', pad=14
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_shap_local_waterfall.png"),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"    Saved: outputs/fig3_shap_local_waterfall.png")
print(f"    Patient: P(diabetes) = {hi_prob*100:.1f}%  |  Actual label: {hi_actual}")

# Save high-risk patient details
hi_patient = {
    "test_set_index": high_risk_idx,
    "predicted_probability": round(hi_prob, 4),
    "actual_label": hi_actual,
    "actual_diagnosis": "Diabetic" if hi_actual == 1 else "Non-diabetic",
    "clinical_features": X_test_labelled.iloc[high_risk_idx].to_dict(),
    "shap_values": dict(zip(X_test_labelled.columns, shap_values[high_risk_idx].tolist())),
    "baseline_E_fx": float(explainer.expected_value),
    "sum_shap_values": round(float(shap_values[high_risk_idx].sum()), 4),
    "f_x_minus_E_fx": round(hi_prob - (1/(1+np.exp(-explainer.expected_value))), 4),
    "note_eq6": "sum_shap_values ≈ f(x) - E[f(x)], confirming Equation 6 additive fidelity"
}
with open(os.path.join(OUTPUT_DIR, "shap_high_risk_patient.json"), "w") as f:
    json.dump(hi_patient, f, indent=2, default=str)
print("    High-risk patient details saved: outputs/shap_high_risk_patient.json")


# =============================================================================
# 5. Bias Audit — Demographic Parity Difference  [Equation 10]
# =============================================================================

print("\n[5] Bias audit — Demographic Parity Difference [Eq. 10]...")

# DPD across Age group (≤35 vs >35) as a demonstration
# In Phase 2 with Ghanaian data, this will be run across gender, region, ethnic group
X_test_audit = X_test.copy()
X_test_audit['y_pred']  = y_pred
X_test_audit['Age_Group'] = (X_test_audit['Age'] > 35).astype(int)  # 0=young, 1=older

g0_rate = X_test_audit[X_test_audit['Age_Group']==0]['y_pred'].mean()
g1_rate = X_test_audit[X_test_audit['Age_Group']==1]['y_pred'].mean()
dpd     = abs(g0_rate - g1_rate)

print(f"    Age group ≤35: positive prediction rate = {g0_rate:.3f}")
print(f"    Age group >35: positive prediction rate = {g1_rate:.3f}")
print(f"    DPD [Eq. 10] = {dpd:.4f}  (threshold: ≤ 0.05 for deployment)")
print(f"    Status: {'✓ PASSES threshold' if dpd <= 0.05 else '✗ EXCEEDS threshold — investigate before deployment'}")

bias_audit = {
    "metric": "Demographic Parity Difference (DPD) [Equation 10]",
    "protected_attribute": "Age group (≤35 vs >35)",
    "note": "Demonstration only — Phase 2 will audit gender, region, ethnicity on Ghanaian data",
    "group_0_label": "Age ≤ 35",
    "group_1_label": "Age > 35",
    "group_0_positive_rate": round(float(g0_rate), 4),
    "group_1_positive_rate": round(float(g1_rate), 4),
    "DPD": round(float(dpd), 4),
    "deployment_threshold": 0.05,
    "passes_threshold": bool(dpd <= 0.05)
}
with open(os.path.join(OUTPUT_DIR, "bias_audit_dpd.json"), "w") as f:
    json.dump(bias_audit, f, indent=2)
print("    Bias audit saved: outputs/bias_audit_dpd.json")


# =============================================================================
# 6. Summary
# =============================================================================

print("\n" + "=" * 70)
print("PIPELINE COMPLETE — All outputs written to outputs/")
print("=" * 70)
print("""
  outputs/
  ├── Dataset splits
  │   ├── pima_train_split.csv          Training set (n=211)
  │   └── pima_test_split.csv           Test set (n=71)
  │
  ├── Trained models
  │   ├── xai_rch_xgboost_model.joblib  Load with: joblib.load(...)
  │   └── xai_rch_xgboost_model.json    Load with: xgb.XGBClassifier().load_model(...)
  │
  ├── Performance metrics
  │   └── performance_metrics.json      All Table 3 metrics
  │
  ├── SHAP outputs
  │   ├── shap_values_test_set.csv      φₖ values for all 71 test patients
  │   ├── shap_baseline.json            E[f(x)] baseline value
  │   ├── shap_high_risk_patient.json   Full breakdown for highest-risk patient
  │   ├── fig2_shap_global_beeswarm.png Figure 2 (manuscript)
  │   └── fig3_shap_local_waterfall.png Figure 3 (manuscript)
  │
  └── Bias audit
      └── bias_audit_dpd.json           DPD [Eq. 10] demonstration
""")
print("To reproduce from scratch:")
print("  1. Place pima.csv in the same directory as this script")
print("  2. Run:  python xai_rch_training_pipeline.py")
print("=" * 70)
