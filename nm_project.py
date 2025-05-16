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

# Streamlit App Config
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📉 Customer Churn Prediction using Machine Learning")

# Sidebar Menu
st.sidebar.header("🛠 Application Menu")
option = st.sidebar.selectbox("Select the section", [
    "Data Overview",
    "Preprocessing Overview",
    "Model Evaluation",
    "SHAP Explainability",
    "Manual Prediction",
    "Dataset Prediction"
])

uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.warning("Please upload a dataset to get started!")

# ----------------------- Data Preprocessing ------------------------
def preprocess_data(df):
    df_cleaned = df.drop(columns=[col for col in ['RowNumber', 'CustomerId', 'Surname'] if col in df.columns])
    le = LabelEncoder()
    for col in ['Geography', 'Gender']:
        if col in df_cleaned.columns:
            df_cleaned[col] = le.fit_transform(df_cleaned[col])
    return df_cleaned

# ----------------------- Model Training Functions ------------------------
@st.cache_resource
def train_all_models(X_train_scaled, y_train):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=500),
        'XGBoost': XGBClassifier(eval_metric='logloss', n_estimators=50),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier()
    }
    for model in models.values():
        model.fit(X_train_scaled, y_train)
    return models

@st.cache_resource
def train_xgboost_model(X_train_scaled, y_train):
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train_scaled, y_train)
    return model

@st.cache_resource
def compute_shap_values(_model, X_train_scaled, X_test_scaled):
    explainer = shap.Explainer(_model)
    shap_values = explainer(X_test_scaled[:50])
    return shap_values

# ----------------------- Data Overview ------------------------
if option == "Data Overview" and uploaded_file is not None:
    st.header("📊 Dataset Overview")

    st.subheader("📂 Raw Data ")
    st.write(df.head())

    st.subheader("📊 Feature Correlation Heatmap")
    df_cleaned = preprocess_data(df) 
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig)

# ----------------------- Preprocessing Overview ------------------------
elif option == "Preprocessing Overview" and uploaded_file is not None:
    st.header("🧹 Data Preprocessing Overview")

    st.subheader("📂 Before Transformation (Raw Data)")
    st.write(df.head())

    df_cleaned = preprocess_data(df)
    st.subheader("✅ After Preprocessing (Label Encoding + Column Removal)")
    st.write(df_cleaned.head())

    feature_names = df_cleaned.drop('Exited', axis=1).columns
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned.drop('Exited', axis=1)), columns=feature_names)

    st.subheader("📐 After Scaling (StandardScaler Applied)")
    st.write(X_scaled.head())

# ----------------------- Model Evaluation ------------------------
elif option == "Model Evaluation" and uploaded_file is not None:
    st.header("🏆 Model Performance Comparison")

    df_cleaned = preprocess_data(df)
    X = df_cleaned.drop('Exited', axis=1)
    y = df_cleaned['Exited']
    feature_names = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

    models = train_all_models(X_train_scaled, y_train)
    model_results = {}
    for name, model in models.items():
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

    result_df = pd.DataFrame(model_results).T.round(3).sort_values(by="Accuracy", ascending=False)
    st.dataframe(result_df)

# ----------------------- SHAP Explainability ------------------------
elif option == "SHAP Explainability" and uploaded_file is not None:
    st.header("🔍 SHAP Explainability for XGBoost")

    df_cleaned = preprocess_data(df)
    X = df_cleaned.drop('Exited', axis=1)
    y = df_cleaned['Exited']
    feature_names = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

    xgb_model = train_xgboost_model(X_train_scaled, y_train)
    shap_values = compute_shap_values(xgb_model, X_train_scaled, X_test_scaled)

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, features=X_test_scaled[:50], feature_names=feature_names, show=False)
    st.pyplot(fig)

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features=X_test_scaled[:50], feature_names=feature_names, plot_type="bar", show=False)
    st.pyplot(fig)

# ----------------------- Manual Prediction ------------------------
elif option == "Manual Prediction" and uploaded_file is not None:
    st.header("📝 Manual Customer Churn Prediction")

    df_cleaned = preprocess_data(df)
    feature_names = df_cleaned.drop('Exited', axis=1).columns
    X = df_cleaned.drop('Exited', axis=1)
    y = df_cleaned['Exited']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    xgb_model = train_xgboost_model(X_scaled, y)

    st.subheader("🔹 Input Values for One Customer")
    user_input = {}
    for feature in feature_names:
        if feature in ['Geography', 'Gender']:
            options = sorted(df[feature].unique())
            choice = st.selectbox(f"Select {feature}", options, key=feature)
            label_map = {label: idx for idx, label in enumerate(options)}
            user_input[feature] = label_map[choice]
        else:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            default_val = float(df[feature].median())
            user_input[feature] = st.number_input(f"Input {feature}", min_value=min_val, max_value=max_val, value=default_val, key=feature)

    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)

    if st.button("Predict Churn (Single Customer)"):
        prediction = xgb_model.predict(input_scaled)[0]
        prediction_proba = xgb_model.predict_proba(input_scaled)[0][1]
        st.write(f"### Prediction: {'Customer will churn 🚪' if prediction == 1 else 'Customer will NOT churn ✅'}")
        st.write(f"### Churn Probability: {prediction_proba:.2f}")

# ----------------------- Dataset Prediction ------------------------
elif option == "Dataset Prediction" and uploaded_file is not None:
    st.header("📁 Churn Prediction for Customers from Dataset")

    df_cleaned = preprocess_data(df)
    X = df_cleaned.drop('Exited', axis=1)
    y = df_cleaned['Exited']
    feature_names = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    xgb_model = train_xgboost_model(X_scaled, y)

    sample_df = X.sample(n=10, random_state=42)
    sample_scaled = scaler.transform(sample_df)
    sample_preds = xgb_model.predict(sample_scaled)
    sample_probs = xgb_model.predict_proba(sample_scaled)[:, 1]

    sample_results = sample_df.copy()
    sample_results['Prediction'] = sample_preds
    sample_results['Churn Probability'] = sample_probs.round(3)
    sample_results['Prediction Label'] = sample_results['Prediction'].map({0: 'Not Churn', 1: 'Churn'})

    st.subheader("🔟 Predictions for 10 Random Customers")
    st.dataframe(sample_results)
