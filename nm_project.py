import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import shap

# Set up Streamlit page configuration (This must be the first command in your app)
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# Custom CSS for styling the Streamlit app
st.markdown("""
    <style>
        body {
            background-color: #F0F8FF;
            font-family: 'Arial', sans-serif;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #4B0082;
        }
        .header {
            font-size: 30px;
            font-weight: bold;
            color: #000080;
        }
        .sidebar .sidebar-content {
            background-color: #E6E6FA;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 20px;
            border-radius: 12px;
            padding: 10px 24px;
            margin-top: 20px;
        }
        .stTextInput>div>div>input {
            font-size: 18px;
        }
        .stDataFrame {
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Title with emoji
st.markdown('<p class="title">📉Customer Churn Prediction using Machine Learning</p>', unsafe_allow_html=True)

# Sidebar with custom header
st.sidebar.header("🛠️ Application Menu")
option = st.sidebar.selectbox("Select the section", ["Upload Dataset", "Model Evaluation", "SHAP Explainability"])

# Upload Dataset Section
if option == "Upload Dataset":
    st.markdown('<p class="header">📊 Upload Your Dataset</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your churn dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Raw Data")
        st.write(df.head())

        # Data Preprocessing
        drop_cols = ['RowNumber', 'CustomerId', 'Surname']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

        # Label Encoding for categorical features
        le = LabelEncoder()
        for col in ['Geography', 'Gender']:
            if col in df.columns:
                df[col] = le.fit_transform(df[col])

        # Correlation heatmap
        st.subheader("📊 Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        st.pyplot(fig)

# Model Evaluation Section
elif option == "Model Evaluation":
    st.markdown('<p class="header">🏆 Model Performance Comparison</p>', unsafe_allow_html=True)

    # Upload dataset section
    uploaded_file = st.file_uploader("Upload your churn dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Preprocessing
        drop_cols = ['RowNumber', 'CustomerId', 'Surname']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

        le = LabelEncoder()
        for col in ['Geography', 'Gender']:
            df[col] = le.fit_transform(df[col])

        # Prepare features and target
        X = df.drop('Exited', axis=1)
        y = df['Exited']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=500),
            'XGBoost': XGBClassifier(eval_metric='logloss', n_estimators=50),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier()
        }

        # Model training and evaluation
        model_results = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
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

        # Display results in a dataframe
        result_df = pd.DataFrame(model_results).T.round(3).sort_values(by="Accuracy", ascending=False)
        st.dataframe(result_df)

# SHAP Explainability Section
elif option == "SHAP Explainability":
    st.markdown('<p class="header">🔍 SHAP Explainability for XGBoost</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your churn dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Preprocessing
        drop_cols = ['RowNumber', 'CustomerId', 'Surname']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

        le = LabelEncoder()
        for col in ['Geography', 'Gender']:
            df[col] = le.fit_transform(df[col])

        # Prepare features and target
        X = df.drop('Exited', axis=1)
        y = df['Exited']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train XGBoost model
        xgb_model = XGBClassifier(eval_metric='logloss')
        xgb_model.fit(X_train_scaled, y_train)

        # SHAP explainability for XGBoost
        explainer = shap.Explainer(xgb_model, X_train_scaled)
        shap_values = explainer(X_test_scaled[:100])

        # Beeswarm plot
        fig = plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig)

        # Bar plot
        fig = plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig)
