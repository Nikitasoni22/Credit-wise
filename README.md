# 🏦 AI-Powered Loan Approval Prediction System

An interactive machine learning web application that predicts loan eligibility using multiple classification algorithms. This project demonstrates a full end-to-end ML pipeline, from Exploratory Data Analysis (EDA) and feature engineering to cloud deployment.

## 🚀 Live Demo
https://credit-wise.streamlit.app/

## 🧐 Project Overview
The goal of this project is to automate the loan eligibility process based on customer details provided during the online application. By using historical data, we've built a robust system that balances accuracy with interpretability.

### 🧪 Key Features & Engineering
- **Multi-Model Support:** Compare results between **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Naive Bayes**.
- **Advanced Feature Engineering:** Implemented polynomial features (squared Credit Score and DTI ratio) to capture non-linear risks.
- **27-Feature Vector:** Integrated One-Hot Encoding and Label Encoding to handle a mix of categorical and numerical data.
- **Real-time Inference:** A responsive UI built with Streamlit for instant risk analysis.

## 📊 Model Performance
After extensive testing and feature engineering, the models achieved the following performance metrics:

| Model | Accuracy | Precision | Recall |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | 82% | 0.81 | 0.84 |
| **KNN** | 79% | 0.78 | 0.80 |
| **Naive Bayes** | 85% | 0.89 | 0.82 |

*(Note: Naive Bayes showed high precision but is more sensitive to outliers in the engineered feature space.)*

## 🛠️ Tech Stack
- **Language:** Python 3.11
- **Libraries:** Scikit-Learn (v1.8.0), Pandas, Numpy, Pickle
- **Frontend:** Streamlit
- **Deployment:** GitHub & Streamlit Community Cloud

## 📂 Project Structure
- `app.py`: The main Streamlit application logic.
- `credit_wise.ipynb`: Jupyter Notebook containing EDA, Preprocessing, and Training.
- `*.pkl`: Serialized models and transformers (Scaler, OHE, Label Encoders).
- `requirements.txt`: List of dependencies for cloud environment setup.

## 📖 How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/Nikitasoni22/Loan-Approval-ML-Project.git](https://github.com/Nikitasoni22/Loan-Approval-ML-Project.git)
   ```
2.   Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3.  Run the app:
   ```bash
   streamlit run app.py
   ```

   Developed by Nikitasoni22 as a Minor Project in Machine Learning.

  **Screenshots**
  <img width="932" height="524" alt="image" src="https://github.com/user-attachments/assets/4e97003f-4d77-430f-84eb-98fdf849ed5b" />
  <img width="1918" height="927" alt="Screenshot 2026-04-05 173224" src="https://github.com/user-attachments/assets/794be096-61f6-4017-99a2-a5b900e816a3" />

