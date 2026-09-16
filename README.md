# Credit Card Fraud Detection

Comparing four machine learning models to detect fraudulent credit card transactions in a heavily imbalanced dataset, where only 0.17% of transactions are fraud.

**Main result:** A Random Forest reached a test PR-AUC of **0.839**. At the chosen operating threshold it flagged only **2 false positives out of 71,079 legitimate transactions** while catching 82 of 123 frauds.

---

## The problem

Fraud detection is hard because fraud is rare. In this dataset, 492 out of 284,807 transactions are fraud. A model that predicts "not fraud" every single time would be 99.8% accurate and completely useless.

There is also a business trade-off. Missing a fraud costs the bank money. Flagging a real customer costs the bank trust and support staff time. A good model has to let you choose where to sit between those two costs.

## The data

- **Source:** [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle)
- **Size:** 284,807 transactions over 48 hours, European cardholders, September 2013
- **Features:** 30 columns. V1–V28 are anonymised PCA components. `Time` and `Amount` are the only raw features.
- **Target:** `Class` (1 = fraud, 0 = legitimate)
- **Fraud rate:** 0.1727%
- **Missing values:** none

The full dataset was used, not the reduced sample.

## What I did

### 1. Splitting

The data was split 60% train / 15% validation / 25% test, stratified on the fraud label so every split has the same fraud rate.

I considered a chronological split instead, since that is closer to how the model would work in production (train on the past, predict the future). The problem is that 492 frauds spread unevenly across 48 hours leaves some time windows with almost no fraud, which makes precision and recall unstable.

I chose the stratified split, then **re-ran the whole pipeline under the rejected chronological split as a check**. Random Forest scored PR-AUC 0.810 there versus 0.839 under stratification. The conclusion did not change.

### 2. Feature engineering

`Time` was converted into `Hour` (0–23) to look for daily patterns. Exploring the training set only, fraud turned out to be far from evenly spread through the day:

- Baseline fraud rate: 0.17%
- At 02:00: about 1.9%
- At 04:00: about 1.2%

That is 7 to 11 times more likely overnight. The reason is that legitimate traffic drops at night (around 2,000 transactions at 02:00 versus 10,000 at midday), so fraud makes up a bigger share.

Using raw fraud *counts* would have been misleading. Hour 11:00 has 33 fraud cases but a rate of only 0.32%, simply because it is one of the busiest hours. The rate is the right measure, not the count.

`Time` itself was then dropped, since removing it barely moved Random Forest PR-AUC (0.832 without versus 0.828 with).

### 3. Handling the imbalance

I compared two approaches on Logistic Regression:

| Strategy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Class weights | 0.065 | 0.892 | 0.122 | 0.981 | 0.712 |
| SMOTE | 0.060 | 0.892 | 0.112 | 0.978 | 0.711 |

Class weights won narrowly on every metric. They are also simpler, faster, and leave the training data untouched. SMOTE would have expanded 295 real frauds into 170,589 rows by inventing 170,294 synthetic ones, which added little real signal in 30-dimensional PCA space.

Class weights were used for all four models.

### 4. Models

| Model | Configuration |
|---|---|
| Logistic Regression | L2 regularisation, C = 1, scaled features |
| Linear SVM | Scaled features, probability calibration, C = 10 (grid search) |
| Random Forest | 200 trees, unscaled features |
| Neural Network | Keras, two hidden layers (32 and 16 units), ReLU, dropout 0.3, early stopping |

Random Forest used unscaled features on purpose: tree models do not care about scale, and scaling would only make feature importances harder to read.

**The scaler was fitted on the training set only** and then applied to validation and test. This prevents information from the test set leaking into training.

## Choosing the metric

PR-AUC is the primary metric, not accuracy and not ROC-AUC.

With a fraud rate of 0.17%, accuracy is meaningless. ROC-AUC is more subtle: it can look excellent while the model still performs badly on the rare class.

This is not a theoretical worry, and the results prove it. **The two metrics rank the models differently:**

- By ROC-AUC, Logistic Regression looks best (0.976 on test)
- By PR-AUC, Random Forest wins (0.839 on test)

That disagreement is itself the evidence that PR-AUC is the right choice here.

For reference, a random classifier gets a PR-AUC equal to the fraud rate: **0.0017**.

## Results

### Model comparison (validation, threshold 0.5)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.902** | 0.743 | **0.815** | 0.935 | **0.832** |
| Logistic Regression | 0.065 | **0.892** | 0.122 | **0.981** | 0.712 |
| Linear SVM | 0.872 | 0.554 | 0.678 | 0.979 | 0.699 |
| Neural Network | 0.077 | 0.851 | 0.142 | 0.942 | 0.675 |

### Threshold tuning (Random Forest, validation)

| Threshold | Precision | Recall |
|---|---|---|
| 0.10 | 0.674 | 0.838 |
| 0.30 | 0.833 | 0.811 |
| 0.365 (best F1) | 0.896 | 0.811 |
| 0.50 | 0.902 | 0.743 |
| **0.75 (chosen)** | **0.981** | 0.689 |
| 0.90 | 1.000 | 0.541 |

At threshold 0.5, the model raises 944 false alarms to catch 66 frauds. In a real bank, that means 944 legitimate customers get their card questioned to find 66 criminals. That is expensive.

**I chose 0.75 for operational reasons, not statistical ones.** The statistically optimal threshold by F1 is 0.365. I picked 0.75 because false positives drive support costs and customer churn, and I state that as a business decision rather than dressing it up as the mathematically best answer.

### Final test results

| Metric | Value |
|---|---|
| PR-AUC | 0.839 |
| Bootstrap 95% CI (1,000 resamples) | [0.768, 0.904] |
| Validation PR-AUC | 0.832 |
| Precision @ 0.75 | 0.976 |
| Recall @ 0.75 | 0.667 |
| False positives | 2 out of 71,079 |
| Frauds caught | 82 out of 123 |

The gap between validation (0.832) and test (0.839) is small, which means the model generalises and did not overfit. The confidence interval is fairly wide because the test set contains only 123 frauds.

**The test set was never touched until the final evaluation.** All hyperparameters and thresholds were chosen using training and validation only.

## What the model cannot tell you

Random Forest ranks V14, V10, V4 and V17 as the most predictive features. Logistic Regression's largest coefficients are the same four. Two very different models agreeing suggests the signal is real.

But V1–V28 are anonymised PCA components and the original recipe was never released. So I can say *which* components predict fraud. I cannot say *what real-world quantity* each one represents. Feature importance shows where the signal is, not the behaviour behind it. That limits any business insight a bank could draw from this model.

## Limitations

- **Only 48 hours of data from 2013.** A model trained on this could drift badly on current transactions.
- **No customer history and no merchant data.** Production fraud systems rely heavily on both.
- **No cost matrix.** The threshold was chosen on operational reasoning, not on real money figures. With actual fraud losses and support costs, 0.75 might not be the right number.
- **Amount is used as a feature but not as a cost.** F1 penalises a missed £4 fraud exactly as much as a missed £3,000 one.
- **The overnight fraud spike may be an artefact.** With only two daily cycles in the data, it could reflect two bursts rather than a real pattern.

## Future work

- Test on a larger dataset covering more than a year, with multiple customers and merchants
- Quantify the monetary cost of missed fraud to set the threshold properly
- Try distribution-aware SMOTE variants (Borderline-SMOTE, ADASYN) against the class-weights baseline

## Repository contents

```
index.py               Main pipeline: loading, splitting, preprocessing, training, evaluation
randomForestModel.py   Random Forest model and threshold analysis
```

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib tensorflow

# Download creditcard.csv from the Kaggle link above and place it in the project folder

python index.py
```

The dataset is not included in this repository because of its size. Download it from Kaggle.

## Built with

Python · scikit-learn · Keras/TensorFlow · pandas · NumPy · matplotlib

---

Coursework project for CF969-7-SP, MSc Applications of Artificial Intelligence, University of Essex, August 2026.
