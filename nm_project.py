

# Load dataset
df = pd.read_csv('/content/drive/MyDrive/DATASETS/churn.csv') 

# Drop irrelevant columns
drop_cols = ['RowNumber', 'CustomerId', 'Surname']
df.drop(columns=drop_cols, inplace=True)

# Encode categorical features
le = LabelEncoder()
for col in ['Geography', 'Gender']:
    df[col] = le.fit_transform(df[col])

# Correlation heatmap
plt.figure(figsize=(12,8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Split data
X = df.drop('Exited', axis=1)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'XGBoost': XGBClassifier(eval_metric='logloss'),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier()
}

# Train and evaluate
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

# Display results
result_df = pd.DataFrame(model_results).T.sort_values(by='Accuracy', ascending=False)
result_df = result_df.round(3)
print("Model Performance Comparison:")
print(result_df)

# ----------------------------
# SHAP Analysis for XGBoost
# ----------------------------
print("\nGenerating SHAP Explainability Visualizations...")

# Use TreeExplainer for XGBoost
explainer = shap.Explainer(models['XGBoost'], X_train_scaled)
shap_values = explainer(X_test_scaled[:100])  # Sampling 100 rows for speed

# Beeswarm Plot - shows feature impact for each prediction
shap.plots.beeswarm(shap_values)

# Optional: Bar Plot - shows average importance of each feature
shap.plots.bar(shap_values)
