# 🏦 CreditWise — Loan Approval Prediction System

An end-to-end machine learning project that predicts loan approval from applicant details. It covers EDA, preprocessing, feature engineering, model comparison (Logistic Regression, KNN, Naive Bayes) and deployment as an interactive Streamlit web app.

## 🚀 Live Demo

https://credit-wise.streamlit.app/

## 🧐 Project Overview

The goal is to automate the first-pass loan eligibility decision from the details an applicant provides online. Three classification models are trained on the same preprocessed data and compared, and the user can switch between them in the app to see each model's prediction and confidence.

## 📊 Dataset

- **Size:** 1,000 applicant records, 18 input features + target (`Loan_Approved`: Yes/No)
- **Numeric features:** Applicant/Coapplicant income, Age, Dependents, Credit Score, Existing Loans, DTI Ratio, Savings, Collateral Value, Loan Amount, Loan Term
- **Categorical features:** Employment Status, Marital Status, Loan Purpose, Property Area, Education Level, Gender, Employer Category
- **Class balance:** approved applicants are the minority class (~30% of the test set), so accuracy alone is not enough. Precision, recall and F1 are reported for the "Approved" class.
- **Note:** this is an educational dataset with bounded feature ranges (e.g. Credit Score 550–799, DTI 0.10–0.60), not real bank data.

## 🧪 Pipeline

1. **Missing values:** ~5% of each column was missing. Numeric columns were imputed with the mean and categorical columns with the most frequent value.
2. **EDA:** class distribution, feature distributions, box plots by approval status, and a correlation heatmap. The strongest signals are **Credit Score (r = +0.45)** and **DTI Ratio (r = −0.44)**; most other features have a weak linear relationship with approval.
3. **Encoding:** Label Encoding for Education Level and the target; One-Hot Encoding (`drop="first"`) for the other six categorical columns.
4. **Split and scaling:** 80/20 train-test split (`random_state=42`); `StandardScaler` fitted on the training set only.
5. **Feature engineering:** added `Credit_Score_sq` and `DTI_Ratio_sq` to allow non-linear effects; the raw Credit Score and DTI columns are replaced by the squared versions.
6. **Final feature vector:** 27 features (10 numeric/label-encoded + 15 one-hot + 2 squared).
7. **Models:** Logistic Regression, KNN (k = 9) and Gaussian Naive Bayes.
8. **Deployment:** models and transformers saved with pickle and served through Streamlit Community Cloud.

## 📈 Model Performance

Evaluated on a held-out test set of 200 applicants (61 approved, 139 not approved). A model that always predicts "not approved" would score ~69.5% accuracy, so that is the baseline to beat. Precision, recall and F1 are for the "Approved" class.

**Final models (with engineered features)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** | **87.5%** | 0.79 | 0.80 | 0.80 |
| Naive Bayes | 86.5% | 0.78 | 0.77 | 0.78 |
| KNN (k = 9) | 77.0% | 0.67 | 0.49 | 0.57 |

**Before feature engineering (raw Credit Score and DTI)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 86.5% | 0.78 | 0.77 | 0.78 |
| Naive Bayes | 86.5% | 0.80 | 0.74 | 0.77 |
| KNN (k = 9) | 76.0% | 0.66 | 0.44 | 0.53 |

**Takeaways**
- Logistic Regression is the best model overall and is also the most interpretable.
- KNN performs poorly, mainly on recall: only two features carry strong signal, so the other 25 dilute the distance metric.
- Feature engineering gave a small gain for Logistic Regression (F1 0.78 → 0.80). On a 200-sample test set, a 1% accuracy difference is 2 predictions, so small gaps should not be over-interpreted.

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Libraries:** scikit-learn 1.8.0, Pandas, NumPy, Matplotlib, Seaborn
- **Frontend:** Streamlit
- **Deployment:** GitHub + Streamlit Community Cloud

## 📂 Project Structure

```
├── app.py                  # Streamlit application
├── credit_wise.ipynb       # EDA, preprocessing, feature engineering, training, evaluation
├── loan_approval_data.csv  # Dataset
├── logistic_model.pkl      # Trained models
├── knn_model.pkl
├── nb_model.pkl
├── ohe.pkl                 # One-Hot Encoder
├── le_education.pkl        # Label encoders
├── le_loan_status.pkl
├── scaler.pkl              # StandardScaler
└── requirements.txt
```

## 📖 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/Nikitasoni22/Credit-wise.git
   cd Credit-wise
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

> The saved `.pkl` files only load reliably with the same scikit-learn version they were trained with (1.8.0), so keep the version pinned in `requirements.txt`.

## ⚠️ Limitations & Future Work

- **Preprocessing order:** missing values were imputed before the train-test split. Imputers should be fitted on the training set only, ideally inside a scikit-learn `Pipeline`.
- **Evaluation:** a single hold-out split was used, with no cross-validation or hyperparameter tuning. Next step: stratified k-fold CV, tuned K / regularization, and ROC-AUC.
- **Feature engineering:** over the 550–799 credit score range, the squared feature is almost linear in the raw score, so it adds little non-linearity. Keeping both raw and squared features, and testing with and without them, would be a cleaner experiment.
- **Fairness:** Gender, Marital Status and Age are used as inputs. A production credit model should exclude protected attributes and audit approval rates across groups.
- **Calibration:** the app's "confidence" is the model's `predict_proba`, which is not guaranteed to be a calibrated probability (especially for Naive Bayes and KNN).
- **Input ranges:** the model was trained on a limited range of values (e.g. Loan Term 12–84 months, Credit Score 550–799). Inputs outside that range are extrapolation and should not be trusted.
- **Model persistence:** pickle files should only be loaded from trusted sources. Safer formats such as ONNX or skops are worth considering.
- **More models:** try tree-based models (Random Forest, Gradient Boosting) with SHAP explanations.

## 🖼️ Screenshots

<img width="932" height="524" alt="image" src="https://github.com/user-attachments/assets/4e97003f-4d77-430f-84eb-98fdf849ed5b" />
<img width="1918" height="927" alt="Screenshot 2026-04-05 173224" src="https://github.com/user-attachments/assets/794be096-61f6-4017-99a2-a5b900e816a3" />

---

Developed by **Nikita Soni** ([@Nikitasoni22](https://github.com/Nikitasoni22)) as a Machine Learning minor project.
