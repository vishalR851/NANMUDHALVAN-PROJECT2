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
import seaborn as sns

# Set up Streamlit page configuration
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# Sidebar: Title and file uploader
st.sidebar.title("Customer Churn Prediction")
uploaded_file = st.sidebar.file_uploader("Choose a dataset", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Sidebar: Dataset preview
    st.sidebar.subheader("Dataset Preview")
    st.sidebar.write(df.head())

    # Drop unnecessary columns
    drop_cols = ['RowNumber', 'CustomerId', 'Surname']
    df.drop(columns=drop_cols, inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    for col in ['Geography', 'Gender']:
        df[col] = le.fit_transform(df[col])

    # Display the correlation heatmap in the main area
    st.subheader("🔍 Feature Correlation Heatmap")
    plt.figure(figsize=(12,8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    st.pyplot()

    # Prepare data for modeling
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model dictionary
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'XGBoost': XGBClassifier(eval_metric='logloss'),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier()
    }

    # Model results
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

    # Sidebar: Model evaluation results
    st.sidebar.subheader("📊 Model Evaluation Results")
    result_df = pd.DataFrame(model_results).T.sort_values(by='Accuracy', ascending=False)
    result_df = result_df.round(3)
    st.sidebar.write(result_df)

    # SHAP explainability for XGBoost
    st.subheader("🔍 SHAP Explainability for XGBoost")
    with st.spinner("Generating SHAP plots..."):
        explainer = shap.Explainer(models['XGBoost'], X_train_scaled)
        shap_values = explainer(X_test_scaled[:100])

        # Beeswarm plot
        st.pyplot(shap.summary_plot(shap_values, X_test_scaled[:100], plot_type="dot"))
