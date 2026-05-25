XAI-RCH Research Package
=========================
Paper: "Explainable Artificial Intelligence for Healthcare Diagnosis in
        Resource-Constrained Ghanaian Hospitals: A Framework Development
        and Proof-of-Concept Study"

CONTENTS
--------
01_dataset/
    pima.csv                      Full cleaned Pima Indians Diabetes Dataset (n=282)
    pima_train_split.csv          Training partition (n=211, stratified 75:25)
    pima_test_split.csv           Test partition (n=71, stratified 75:25)
    DATA_SOURCE.txt               Dataset provenance and citation

02_model/
    xai_rch_xgboost_model.joblib  Trained XGBoost model (sklearn API)
    xai_rch_xgboost_model.json    Trained XGBoost model (native XGBoost format)
    MODEL_INFO.txt                Hyperparameters and training configuration

03_shap_outputs/
    shap_values_test_set.csv      Shapley values φₖ for all 71 test patients [Eq. 5-6]
    shap_baseline.json            Baseline E[f(x)] used in waterfall plots [Eq. 6]
    shap_high_risk_patient.json   Full SHAP breakdown for highest-risk patient [Fig. 3]

04_metrics/
    performance_metrics.json      All Table 3 metrics (accuracy, AUC, precision, recall, F1, CM)
    bias_audit_dpd.json           Demographic Parity Difference audit [Eq. 10]

05_figures/
    fig_framework.png             Figure 1 — XAI-RCH Framework structural diagram
    fig2_shap_global_beeswarm.png Figure 2 — Global SHAP beeswarm summary plot
    fig3_shap_local_waterfall.png Figure 3 — Local SHAP waterfall plot

06_code/
    xai_rch_training_pipeline.py  Full reproducible training and analysis script

REPRODUCTION
------------
1. Install dependencies:
   pip install xgboost shap scikit-learn pandas numpy matplotlib joblib

2. Place pima.csv in the same directory as xai_rch_training_pipeline.py

3. Run:
   python xai_rch_training_pipeline.py

   This regenerates all models, metrics, SHAP values, and figures from scratch.

CITATION
--------
If you use this code or data in your research, please cite:
  Bashiru (2026). Explainable Artificial Intelligence for Healthcare
  Diagnosis in Resource-Constrained Ghanaian Hospitals: A Framework Development
  and Proof-of-Concept Study. BMC Medical Informatics and Decision Making .

Dataset citation:
  Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S.
  (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.
  Proceedings of the Annual Symposium on Computer Application in Medical Care, 261-265.
  UCI Repository: https://archive.ics.uci.edu/dataset/34/diabetes
