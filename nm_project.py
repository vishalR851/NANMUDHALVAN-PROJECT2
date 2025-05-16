import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap

# Page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📉 Customer Churn Prediction using Machine Learning")

# Sidebar
st.sidebar.header("📦 Customer Prediction")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Data Preprocessing
def preprocess_data(df):
    df_cleaned = df.drop(columns=[col for col in ['RowNumber', 'CustomerId', 'Surname'] if col in df.columns])
    le = LabelEncoder()
    for col in ['Geography', 'Gender']:
        if col in df_cleaned.columns:
            df_cleaned[col] = le.fit_transform(df_cleaned[col])
    return df_cleaned

@st.cache_resource
def train_model(X_scaled, y):
    model = XGBClassifier(eval_metric='logloss', n_estimators=50)
    model.fit(X_scaled, y)
    return model

@st.cache_resource
def compute_shap_text_values(_model, X_sample):
    explainer = shap.Explainer(_model)
    shap_values = explainer(X_sample)
    return shap_values

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_cleaned = preprocess_data(df)
    
    X = df_cleaned.drop('Exited', axis=1)
    y = df_cleaned['Exited']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = train_model(X_scaled, y)

    st.header("🔎 Customer Prediction Results")

    # Random 10 sample customers
    sample_df = X.sample(n=10, random_state=42)
    sample_scaled = scaler.transform(sample_df)
    sample_preds = model.predict(sample_scaled)
    sample_probs = model.predict_proba(sample_scaled)[:, 1]

    sample_results = sample_df.copy()
    sample_results['Prediction'] = sample_preds
    sample_results['Churn Probability'] = sample_probs.round(3)
    sample_results['Prediction Label'] = sample_results['Prediction'].map({0: 'Not Churn', 1: 'Churn'})

    # SHAP values for top factors (text only)
    shap_values = compute_shap_text_values(model, sample_scaled)

    # Display with text explanation
    for i in range(len(sample_results)):
        st.subheader(f"Customer #{i+1}")
        st.write(sample_results.iloc[i][:-3])  # Show customer features
        st.write(f"**Prediction:** {sample_results.iloc[i]['Prediction Label']}")
        st.write(f"**Churn Probability:** {sample_results.iloc[i]['Churn Probability']}")
        
        # Get top 3 influencing features as text
        shap_scores = list(zip(X.columns, shap_values[i].values))
        top_factors = sorted(shap_scores, key=lambda x: abs(x[1]), reverse=True)[:3]
        factor_text = ", ".join([f"{name} ({'+' if val > 0 else '-'}{abs(val):.2f})" for name, val in top_factors])
        st.info(f"Top factors influencing this prediction: {factor_text}")
        st.markdown("---")
else:
    st.warning("Please upload a dataset to view predictions.")
