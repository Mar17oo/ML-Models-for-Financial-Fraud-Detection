from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Random Forest uses the UNSCALED features (trees don't care about scale).
# Rebuild unscaled arrays with Hour included, matching the scaled feature set.
def run_random_forest_models(X_train, y_train, X_val, y_val):
    feature_cols_rf = list(X_train.columns)   # includes V1-V28, Time, Amount, Hour
    X_train_rf = X_train[feature_cols_rf].values
    X_val_rf   = X_val[feature_cols_rf].values

    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        max_depth=None, n_jobs=-1, random_state=42)
    rf.fit(X_train_rf, y_train)

    # Evaluate (note: pass UNSCALED val data to this model)
    proba_rf = rf.predict_proba(X_val_rf)[:, 1]
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                roc_auc_score, average_precision_score, confusion_matrix)
    pred_rf = (proba_rf >= 0.5).astype(int)
    print("--- Random Forest + class weights (val) ---")
    print("Confusion matrix [[TN FP][FN TP]]:")
    print(confusion_matrix(y_val, pred_rf))
    print(f"Precision: {precision_score(y_val, pred_rf):.3f}")
    print(f"Recall:    {recall_score(y_val, pred_rf):.3f}")
    print(f"F1:        {f1_score(y_val, pred_rf):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_val, proba_rf):.3f}")
    print(f"PR-AUC:    {average_precision_score(y_val, proba_rf):.3f}")

    # Feature importances — top 10
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1][:10]
    print("\nTop 10 features by importance:")
    for i in order:
        print(f"  {feature_cols_rf[i]:8s}: {importances[i]:.4f}")

    return {"rf": rf, "proba_rf": proba_rf, "feature_cols_rf": feature_cols_rf}