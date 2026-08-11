import pandas as pd
import numpy as np #pip install numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix, precision_recall_curve)
from imblearn.over_sampling import SMOTE   # pip install imbalanced-learn

from supportVectorMachine_Model import run_svm_models


# 1. Load and inspect
df = pd.read_csv("data/creditcard.csv")
print("Observations:", df.shape[0])
print("Features:", df.shape[1] - 1)          # minus the Class column
print("Missing values:", df.isnull().sum().sum())
print(df["Class"].value_counts())
print("Fraud rate:", df["Class"].mean())      # expect ~0.001727

# 2. Separate features and label
X = df.drop(columns=["Class"])
y = df["Class"]

# 3. Stratified split: 60% train, 15% val, 25% test
#    First carve off 25% test, then split the remaining 75% into 60/15.
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=42)
    # 0.20 of the remaining 75% = 15% of the whole

# 4. Report fraud count and percentage in each split (required by brief)
for name, yy in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    print(f"{name}: n={len(yy)}, frauds={yy.sum()}, rate={yy.mean():.4%}")


# Work on the TRAINING set only — we inspect train, never peek at test.
train = X_train.copy()
train["Class"] = y_train.values
train["Hour"] = (train["Time"] / 3600) % 24
train["HourBin"] = train["Hour"].astype(int)   # 0-23 integer buckets

# Fraud rate per hour, plus how many transactions fall in each hour
by_hour = train.groupby("HourBin").agg(
    n=("Class", "size"),
    frauds=("Class", "sum"),
).reset_index()
by_hour["fraud_rate_pct"] = 100 * by_hour["frauds"] / by_hour["n"]

print(by_hour.to_string(index=False))

# Quick visual
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(by_hour["HourBin"], by_hour["frauds"])
ax[0].set_title("Fraud count by hour"); ax[0].set_xlabel("Hour of day"); ax[0].set_ylabel("Frauds")
ax[1].bar(by_hour["HourBin"], by_hour["fraud_rate_pct"])
ax[1].set_title("Fraud rate (%) by hour"); ax[1].set_xlabel("Hour of day"); ax[1].set_ylabel("Fraud rate %")
plt.tight_layout(); plt.savefig("fraud_by_hour.png", dpi=120); plt.show()


# 5. Feature engineering + scaling for modeling
# Add Hour to X_train/X_val/X_test BEFORE scaling, so the scaler and every
# split see the exact same columns (mismatched columns here is what causes
# NaN in the scaled arrays or a shape mismatch downstream).
for d in (X_train, X_val, X_test):
    d["Hour"] = (d["Time"] / 3600) % 24

feature_cols = list(X_train.columns)          # now includes Hour
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[feature_cols])   # fit on TRAIN only
X_val_scaled   = scaler.transform(X_val[feature_cols])
X_test_scaled  = scaler.transform(X_test[feature_cols])

print("NaN in X_train_scaled:", np.isnan(X_train_scaled).sum())
print("Shape:", X_train_scaled.shape, "| Columns scaled:", len(feature_cols))


# Helper: evaluate a fitted model on the validation set at the default 0.5 threshold
def evaluate(model, X, y_true, label):
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    print(f"\n--- {label} ---")
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(confusion_matrix(y_true, pred))
    print(f"Precision: {precision_score(y_true, pred):.3f}")
    print(f"Recall:    {recall_score(y_true, pred):.3f}")
    print(f"F1:        {f1_score(y_true, pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, proba):.3f}")
    print(f"PR-AUC:    {average_precision_score(y_true, proba):.3f}")
    return proba

# ---- Strategy A: class weights ----
# The model is told a fraud mistake costs ~580x a normal one (balanced = n/(2*class_count)).
logreg_cw = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
logreg_cw.fit(X_train_scaled, y_train)
proba_cw = evaluate(logreg_cw, X_val_scaled, y_val, "LogReg + class weights (val)")

# ---- Strategy B: SMOTE on TRAINING ONLY ----
# Synthetic frauds are generated ONLY inside the training set, after the split.
# Validation stays untouched real data — we never evaluate on synthetic points.
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_scaled, y_train)
print("After SMOTE, training class counts:", np.bincount(y_train_sm))

logreg_sm = LogisticRegression(max_iter=1000, random_state=42)
logreg_sm.fit(X_train_sm, y_train_sm)
proba_sm = evaluate(logreg_sm, X_val_scaled, y_val, "LogReg + SMOTE (val)")

# ---- Strategy C: SVMs (Linear + RBF), from supportVectorMachine_Model.py ----
svm_results = run_svm_models(X_train_scaled, y_train, X_val_scaled, y_val, evaluate)

# ---- Threshold tuning for the winning model (class weights) ----
# Use the class-weights model's validation probabilities (the winner)
proba_val = logreg_cw.predict_proba(X_val_scaled)[:, 1]

# Precision and recall at every possible threshold
precisions, recalls, thresholds = precision_recall_curve(y_val, proba_val)

# For each threshold, also compute F1 so we can find a balanced point
# (precision_recall_curve returns one more precision/recall than thresholds)
f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]

print(f"Best-F1 threshold on validation: {best_threshold:.3f}")
print(f"  precision={precisions[best_idx]:.3f}  recall={recalls[best_idx]:.3f}  F1={f1s[best_idx]:.3f}")

# Show a few thresholds so you can SEE the trade-off, not just the optimum
print("\nThreshold | Precision | Recall")
for t in [0.10, 0.30, 0.50, 0.70, 0.90, 0.99]:
    pred_t = (proba_val >= t).astype(int)
    tp = ((pred_t == 1) & (y_val == 1)).sum()
    fp = ((pred_t == 1) & (y_val == 0)).sum()
    fn = ((pred_t == 0) & (y_val == 1)).sum()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    print(f"   {t:.2f}   |   {prec:.3f}   |  {rec:.3f}")

# Precision-Recall curve figure for the report
plt.figure(figsize=(6, 5))
plt.plot(recalls, precisions)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall curve (LogReg, validation)")
plt.grid(True, alpha=0.3)
plt.savefig("pr_curve_logreg.png", dpi=120, bbox_inches="tight")
plt.show()