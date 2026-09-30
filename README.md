# Customer Churn Prediction

An end-to-end machine learning pipeline that predicts customer churn for a telecom company, using the Telco Customer Churn dataset. The notebook covers data cleaning, exploratory analysis, encoding, model training, evaluation, and predictions on new sample customers.

## Results

Evaluated on a stratified 80/20 split (1,409 test customers, `random_state=42`):

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | **0.7984** | **0.6406** | **0.5481** | **0.5908** |
| Random Forest | 0.7857 | 0.6200 | 0.4973 | 0.5519 |

Logistic Regression edges out Random Forest on every metric. Recall is around 50-55% for both, so roughly half of actual churners are missed. Since only 26.5% of customers churn, there is room to improve with class weighting, threshold tuning, or resampling.

## Key Findings

- **Contract type matters:** month-to-month customers churn at 42.7%, versus 11.3% (one year) and 2.8% (two year).
- **Fiber optic customers churn more:** 41.9%, compared with 19.0% for DSL and 7.4% for no internet.
- **Shorter tenure and higher monthly charges** are associated with churn.
- **Top Random Forest features:** TotalCharges, MonthlyCharges, tenure, Contract, OnlineSecurity.

<p align="center">
  <img src="images/churn_by_contract.png" width="45%" alt="Churn by contract type">
  <img src="images/feature_importance.png" width="45%" alt="Top 10 feature importances">
</p>

## Project Structure

```
churn-prediction/
├── Churn_Prediction.ipynb   # Full analysis and modeling
├── requirements.txt         # Python dependencies
├── data/README.md           # How to get the dataset
├── images/                  # Figures used in this README
├── LICENSE
└── README.md
```

## Getting Started

```bash
git clone https://github.com/<your-username>/churn-prediction.git
cd churn-prediction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset as described in [`data/README.md`](data/README.md), then:

```bash
jupyter notebook Churn_Prediction.ipynb
```

## Methodology

1. **Data understanding:** shape, dtypes, column dictionary.
2. **Preprocessing:** convert `TotalCharges` to numeric, impute the blank values with `MonthlyCharges` (new customers), drop `customerID`.
3. **EDA:** churn distribution, churn by contract, internet service, tenure, monthly charges, and demographics.
4. **Encoding:** binary columns to 0/1, label encoding for other categoricals, target `Yes/No` to 1/0.
5. **Modeling:** stratified train/test split; Logistic Regression (standardized features) and Random Forest (100 trees, max depth 15).
6. **Evaluation:** accuracy, precision, recall, F1, confusion matrix, feature importance.
7. **Application:** churn probability for three hand-made sample customers.

## Limitations and Next Steps

- No cross-validation or hyperparameter tuning yet.
- Class imbalance is not addressed (try `class_weight="balanced"`, SMOTE, or a lower decision threshold to raise recall).
- Label encoding is applied to nominal features; one-hot encoding would suit linear models better.
- Try gradient boosting (XGBoost / LightGBM) and add SHAP for explainability.

## Tech Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter.

## License

Released under the [MIT License](LICENSE).
