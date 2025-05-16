import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("📉 Customer Churn Prediction using Machine Learning")

# Sidebar menu
section = st.sidebar.selectbox(
    "Select the section",
    ["Over View", "Data Preprocessing", "Model Evaluation", "Manual Prediction"]
)

# Upload Dataset
uploaded_file = st.sidebar.file_uploader("Upload your churn dataset (CSV)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    st.sidebar.warning("Please upload a dataset to proceed.")
    st.stop()

if section == "Over View":
    st.header("📊 Dataset Overview")
    st.subheader("Raw Data")
    st.dataframe(data.head())

    st.subheader("Feature Correlation Heatmap")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif section == "Data Preprocessing":
    st.header("⚙️ Data Preprocessing")

    # Drop unnecessary columns
    data_prep = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

    # Encode categorical variables
    le_gender = LabelEncoder()
    data_prep["Gender"] = le_gender.fit_transform(data_prep["Gender"])

    le_geo = LabelEncoder()
    data_prep["Geography"] = le_geo.fit_transform(data_prep["Geography"])

    st.subheader("Data after Encoding")
    st.dataframe(data_prep.head())

    # Scale numerical features
    scaler = StandardScaler()
    features_to_scale = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    data_prep[features_to_scale] = scaler.fit_transform(data_prep[features_to_scale])

    st.subheader("Data after Scaling")
    st.dataframe(data_prep.head())

elif section == "Model Evaluation":
    st.header("📈 Model Training and Evaluation")

    # Preprocessing (same as above)
    data_prep = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
    le_gender = LabelEncoder()
    data_prep["Gender"] = le_gender.fit_transform(data_prep["Gender"])
    le_geo = LabelEncoder()
    data_prep["Geography"] = le_geo.fit_transform(data_prep["Geography"])

    scaler = StandardScaler()
    features_to_scale = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    data_prep[features_to_scale] = scaler.fit_transform(data_prep[features_to_scale])

    # Split dataset
    X = data_prep.drop("Exited", axis=1)
    y = data_prep["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

elif section == "Manual Prediction":
    st.header("🔮 Predict Customer Churn for New Input")

    # We need the trained model and encoders from the Model Evaluation section
    # So, re-run the preprocessing and model training here for simplicity
    data_prep = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
    le_gender = LabelEncoder()
    data_prep["Gender"] = le_gender.fit_transform(data_prep["Gender"])
    le_geo = LabelEncoder()
    data_prep["Geography"] = le_geo.fit_transform(data_prep["Geography"])

    scaler = StandardScaler()
    features_to_scale = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    data_prep[features_to_scale] = scaler.fit_transform(data_prep[features_to_scale])

    X = data_prep.drop("Exited", axis=1)
    y = data_prep["Exited"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Input fields
    Geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    Gender = st.selectbox("Gender", ["Female", "Male"])
    CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    Tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
    Balance = st.number_input("Balance", min_value=0.0, value=10000.0)
    NumOfProducts = st.selectbox("Number of Products", [1, 2, 3, 4])
    HasCrCard = st.selectbox("Has Credit Card", [0, 1])
    IsActiveMember = st.selectbox("Is Active Member", [0, 1])
    EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

    input_dict = {
        "Geography": [Geography],
        "Gender": [Gender],
        "CreditScore": [CreditScore],
        "Age": [Age],
        "Tenure": [Tenure],
        "Balance": [Balance],
        "NumOfProducts": [NumOfProducts],
        "HasCrCard": [HasCrCard],
        "IsActiveMember": [IsActiveMember],
        "EstimatedSalary": [EstimatedSalary]
    }
    input_df = pd.DataFrame(input_dict)

    # Encode input categorical data
    input_df["Gender"] = le_gender.transform(input_df["Gender"])
    input_df["Geography"] = le_geo.transform(input_df["Geography"])

    # Scale numeric features
    input_df[features_to_scale] = scaler.transform(input_df[features_to_scale])

    if st.button("Predict Churn"):
        pred = model.predict(input_df)
        result = "Yes, the customer will churn." if pred[0] == 1 else "No, the customer will not churn."
        st.success(result)
