from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import numpy as np


def run_svm_models(X_train_scaled, y_train, X_val_scaled, y_val, evaluate):
    """Train the Linear and RBF SVM strategies and evaluate them on the validation set.

    Takes the already-scaled train/val splits and the `evaluate` helper from
    index.py so this module shares the exact same data/metrics instead of
    redefining them.
    """
    # ---- Linear SVM on full training data, with class weights (our chosen imbalance fix) ----
    # LinearSVC is fast but doesn't output probabilities natively, so we wrap it
    # in CalibratedClassifierCV to get predict_proba for threshold tuning + PR-AUC.
    lin_svm = LinearSVC(class_weight="balanced", C=1.0, max_iter=5000, random_state=42)
    lin_svm_cal = CalibratedClassifierCV(lin_svm, method="sigmoid", cv=3)
    lin_svm_cal.fit(X_train_scaled, y_train)
    proba_linsvm = evaluate(lin_svm_cal, X_val_scaled, y_val, "Linear SVM + class weights (val)")

    # ---- RBF-kernel SVM on a STRATIFIED SUBSET (stated clearly, per the brief) ----
    # RBF on 170k rows is very slow, so we train on ~20k rows preserving the fraud rate.
    X_sub, _, y_sub, _ = train_test_split(
        X_train_scaled, y_train, train_size=20000, stratify=y_train, random_state=42)
    print("RBF training subset class counts:", np.bincount(y_sub))

    rbf_svm = SVC(kernel="rbf", class_weight="balanced", C=1.0, gamma="scale",
                  probability=True, random_state=42)
    rbf_svm.fit(X_sub, y_sub)
    proba_rbfsvm = evaluate(rbf_svm, X_val_scaled, y_val, "RBF SVM (20k subset) + class weights (val)")

    return {
        "lin_svm": lin_svm_cal, "proba_linsvm": proba_linsvm,
        "rbf_svm": rbf_svm, "proba_rbfsvm": proba_rbfsvm,
    }
