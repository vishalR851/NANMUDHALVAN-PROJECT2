# -*- coding: utf-8 -*-
"""NM PROJECT.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Jb87APWawphcDjGovWl_sMra7Du9w-h-
"""

!pip install streamlit scikit-learn pandas xgboost shap matplotlib pandas-profiling streamlit-pandas-profiling pyngrok

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import shap
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(layout="wide")
st.title("📉 Customer Churn Prediction App (Custom Dataset)")

# Load dataset
uploaded_file = st.file_uploader("Upload your churn dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Data Preview:")
    st.dataframe(df.head())

    # Drop irrelevant columns
    drop_cols = ['RowNumber', 'CustomerId', 'Surname']
    df.drop(columns=drop_cols, inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    for col in ['Geography', 'Gender']:
        df[col] = le.fit_transform(df[col])

    # Split features and target
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # Train/test split + scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model training
    st.subheader("🔁 Model Training & Comparison")
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'Support Vector Machine (SVM)': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors (KNN)': KNeighborsClassifier()
    }

    model_results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        # Calculate different evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

        model_results[name] = {
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall,
            'ROC AUC': roc_auc
        }

    # Convert model results to DataFrame
    result_df = pd.DataFrame(model_results).T.sort_values(by='Accuracy', ascending=False)
    st.write("📈 Model Evaluation Comparison:")
    st.dataframe(result_df)

    # Best model selection based on accuracy
    best_model_name = result_df.index[0]
    best_model = models[best_model_name]
    st.success(f"✅ Best Model: {best_model_name} with accuracy {result_df.loc[best_model_name, 'Accuracy']:.2f}")

    # SHAP Analysis (Optional)
    st.subheader("🔍 SHAP Feature Importance")
    if best_model_name == 'XGBoost':  # SHAP works best with XGBoost
        explainer = shap.Explainer(best_model)
        shap_values = explainer(X_test_scaled)
        shap.summary_plot(shap_values, X_test)
        st.pyplot()

!ngrok config add-authtoken 2u72IeZvUkS7VVpI32hd0zOWZH4_6CbaRrLLKayoy8dtQFZVs

!pkill streamlit
!streamlit run app.py &>/content/logs.txt &

from pyngrok import ngrok
public_url = ngrok.connect(addr="8501", proto="http")
print("Streamlit URL:", public_url)

with open("requirements.txt", "w") as f:
    f.write("streamlit\npandas\nnumpy\nscikit-learn\nplotly")
from google.colab import files
files.download("requirements.txt")