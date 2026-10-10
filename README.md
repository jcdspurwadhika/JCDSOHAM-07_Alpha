# Customer Churn Prediction for E-Commerce
*Repository: JCDSOHAM-07_Alpha*

Identify customers likely to churn early enough to act, and spend the retention budget where it pays off.

[Live App](https://ecommerce-churn-prediction-dodj5namudes7b9gjcrdgc.streamlit.app/) · [Business Dashboard](https://public.tableau.com/views/CustomerChurn--E-Commerce/Overview) · [Analysis Notebook](ecommerce_churn_project.ipynb)

## Business Problem

Losing a customer costs far more than keeping one. Offering retention incentives to everyone is expensive, and offering them to nobody means silent revenue loss. The retention team needs to know **who is about to leave**, and **who is worth the offer**.

In the data, 16.58% of customers churned. Assuming a customer lifetime value (CLV) of $200 and a retention offer cost of $20, one missed churner costs about 10x as much as one wasted offer. That ratio drives every modelling decision below.

## Result

On a held-out test set of 1,015 customers:

| Metric | Value |
|---|---|
| Recall | 92.9% |
| Precision | 83.0% |
| F3 | 0.9176 |

| Strategy | Estimated cost of errors |
|---|---|
| Offer retention to everyone | $16,940 |
| Targeted by this model | $3,040 |
| **Saving** | **$13,900 (82.1%)** |

Cost of errors = missed churners x $200 + wasted offers x $20. The estimate assumes every offer successfully retains the customer, so it is an upper bound on the benefit. Real savings depend on offer acceptance rates, which should be measured with an A/B test.

## Approach

**Data.** 5,630 customers, 18 features covering tenure, order behaviour, satisfaction, complaints, payment and device preferences, and demographics. Cleaning covered inconsistent category labels, structural zeros and missing values.

**Leakage control.** The train/test split happens first (80/20, stratified). Imputation is fit on the training set only and wrapped in a `ColumnTransformer` + `Pipeline`, so no test information reaches training.

**Class imbalance.** Handled with class weighting rather than SMOTE (XGBoost via balanced sample weights). I also tested threshold tuning with out-of-fold cross-validation and rejected it: on the test set it lowered F3 (0.9176 to 0.9019) and raised cost ($3,040 to $3,640), so the default 0.5 threshold stays.

**Metric.** F3, derived from the cost ratio (sqrt(10) is about 3.16, rounded to 3), which weights recall 9x more than precision. Precision, recall and F1 are reported as supporting metrics.

**Models.** Logistic Regression, Decision Tree, Random Forest, XGBoost and SVM (RBF) were compared with 5-fold cross-validation. XGBoost won and was tuned with `RandomizedSearchCV` (60 iterations).

**Interpretation.** Gain importance and SHAP agree that tenure and complaints are the strongest drivers: customers in their first 1 month churn at about 51%, and customers who complained churn at about 31% versus 11% for those who did not.

## Risk Tiers

Model scores are not calibrated probabilities, because class weighting inflates them. So scores are mapped to tiers, each validated against the observed churn rate on the test set.

| Tier | Score | Customers | Observed churn | Action |
|---|---|---|---|---|
| High | > 0.7 | 155 | 93.55% | Prioritise retention offer |
| Medium | 0.3 - 0.7 | 90 | 15.56% | Optional, monitor |
| Low | < 0.3 | 770 | 1.17% | No intervention |

## Application

A Streamlit app for the Customer Retention / CRM team:

- **Individual prediction:** check one customer's risk, e.g. while handling a complaint.
- **Batch prediction:** upload a customer CSV, get a ranked priority list, and simulate the offer budget (cost, expected churners covered, net benefit) with a slider. Results can be downloaded as CSV.
- **About the model:** performance, how to read scores, and limitations.

A Tableau dashboard presents the business view: churn drivers, segment breakdowns, and risk-tier prioritisation with the cost saving.

## Run Locally

```bash
git clone https://github.com/jcdspurwadhika/JCDSOHAM-07_Alpha
cd JCDSOHAM-07_Alpha/app
pip install -r requirements.txt
python -m streamlit run app.py
```

Pinned versions: Streamlit 1.60.0, scikit-learn 1.9.0, XGBoost 3.3.0, pandas 3.0.3, NumPy 2.4.6, joblib 1.5.3. The saved model must load under the same scikit-learn and XGBoost versions.

## Repository Structure

```
.
├── ecommerce_churn_project.ipynb   # EDA, modelling, evaluation, cost-benefit
├── Ecommerce_Dataset.csv           # source data
└── app/
    ├── app.py                      # Streamlit application
    ├── xgboost_churn_model_tuned.pkl
    ├── app_meta.pkl                # feature schema, medians, category options
    └── requirements.txt
```

## Limitations

- The data source does not define "churn", so the model treats the label as given.
- CLV and retention cost are industry benchmarks in USD, not company data; replace them with internal figures before using the savings number for decisions.
- Evaluation uses 1,015 test customers, so estimates carry uncertainty; performance on new data may differ.
- The model is trained on one historical snapshot. It needs monitoring for drift and periodic retraining.
- Intended as a prioritisation aid, not an automated decision.

## Next Steps

- Measure the real offer acceptance rate with an A/B test and update the cost model.
- Calibrate scores so they can be read as probabilities.
- Add drift monitoring and a retraining schedule.

## Author

**Emir Abdallah** · [LinkedIn](https://www.linkedin.com/in/emir-abdallah-62543815b/) · [GitHub](https://github.com/EmirAbdallah)
