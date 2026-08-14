import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia las alertas informativas de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)


def run_neural_network(X_train_scaled, y_train, X_val_scaled, y_val):
    tf.random.set_seed(42)

    # Class weights: tell the network fraud errors cost more (same idea as before).
    n_neg, n_pos = np.bincount(y_train)
    total = n_neg + n_pos
    class_weight = {0: total / (2 * n_neg), 1: total / (2 * n_pos)}
    print("Class weights:", class_weight)

    # Small architecture with dropout — deliberately modest to limit overfitting.
    model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),   # outputs a probability
    ])

    model.compile(optimizer="adam",
                loss="binary_crossentropy",
                metrics=[keras.metrics.AUC(name="pr_auc", curve="PR")])

    # Early stopping: watch validation PR-AUC, stop when it stops improving,
    # and restore the best weights (prevents over-training).
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_pr_auc", mode="max", patience=5, restore_best_weights=True)

    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=50, batch_size=2048,
        class_weight=class_weight,
        callbacks=[early_stop], verbose=1)

    # Evaluate on validation
    proba_nn = model.predict(X_val_scaled).ravel()
    pred_nn = (proba_nn >= 0.5).astype(int)
    print("\n--- Neural Network + class weights (val) ---")
    print("Confusion matrix [[TN FP][FN TP]]:")
    print(confusion_matrix(y_val, pred_nn))
    print(f"Precision: {precision_score(y_val, pred_nn):.3f}")
    print(f"Recall:    {recall_score(y_val, pred_nn):.3f}")
    print(f"F1:        {f1_score(y_val, pred_nn):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_val, proba_nn):.3f}")
    print(f"PR-AUC:    {average_precision_score(y_val, proba_nn):.3f}")