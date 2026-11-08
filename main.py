

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
import shap
import matplotlib.pyplot as plt
import os

# Download the dataset using Kaggle API. We used WA_Fn-UseC_-Telco-Customer-Churn.csv
print("Downloading dataset from Kaggle...")
os.system('kaggle datasets download -d blastchar/telco-customer-churn --unzip')

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

df = df.drop('customerID', axis=1)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X_train.columns if col not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ])

baseline_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

baseline_model.fit(X_train, y_train)
y_pred_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]

def top_10_percent_recall(y_true, y_pred_proba):
    y_true_arr = np.array(y_true)
    top_10_cutoff = int(0.1 * len(y_true_arr))
    
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    top_10_indices = sorted_indices[:top_10_cutoff]
    
    true_positives = y_true_arr[top_10_indices].sum()
    total_positives = y_true_arr.sum()
    
    return true_positives / total_positives

roc_auc = roc_auc_score(y_test, y_pred_proba_baseline)
pr_auc = average_precision_score(y_test, y_pred_proba_baseline)
top_10_recall = top_10_percent_recall(y_test, y_pred_proba_baseline)

print("--- Baseline: Logistic Regression ---")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"Recall in Top 10%: {top_10_recall:.4f}")


rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor), # Reusing the exact same preprocessing pipeline
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

rf_model.fit(X_train, y_train)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
pr_auc_rf = average_precision_score(y_test, y_pred_proba_rf)
top_10_recall_rf = top_10_percent_recall(y_test, y_pred_proba_rf)

print("\n--- Main Model: Random Forest ---")
print(f"ROC-AUC: {roc_auc_rf:.4f}")
print(f"PR-AUC: {pr_auc_rf:.4f}")
print(f"Recall in Top 10%: {top_10_recall_rf:.4f}")

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__max_depth': [3, 4, 5]
}

print("\nTraining Tuned Gradient Boosting... (this might take a minute)")
gb_search = RandomizedSearchCV(
    gb_pipeline, 
    param_distributions=param_grid, 
    n_iter=5, # Testing 5 random combinations to keep it fast
    scoring='roc_auc', 
    cv=3, 
    random_state=42, 
    n_jobs=-1
)

gb_search.fit(X_train, y_train)

y_pred_proba_gb = gb_search.predict_proba(X_test)[:, 1]

roc_auc_gb = roc_auc_score(y_test, y_pred_proba_gb)
pr_auc_gb = average_precision_score(y_test, y_pred_proba_gb)
top_10_recall_gb = top_10_percent_recall(y_test, y_pred_proba_gb)

print("--- Improvement: Tuned Gradient Boosting ---")
print(f"Best parameters found: {gb_search.best_params_}")
print(f"ROC-AUC: {roc_auc_gb:.4f}")
print(f"PR-AUC: {pr_auc_gb:.4f}")
print(f"Recall in Top 10%: {top_10_recall_gb:.4f}")

best_pipeline = gb_search.best_estimator_
classifier = best_pipeline.named_steps['classifier']
preprocessor = best_pipeline.named_steps['preprocessor']

X_test_transformed = preprocessor.transform(X_test)

if hasattr(X_test_transformed, 'toarray'):
    X_test_transformed = X_test_transformed.toarray()
    
feature_names = preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_test_transformed)

plt.figure(figsize=(10, 6))
plt.title("Global Feature Importance (SHAP)")
shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
plt.tight_layout()
plt.show()

profiles_to_explain = [0, 10, 20]


for idx in profiles_to_explain:
    print(f"\n--- Customer Profile Index {idx} ---")
    print(f"Actual Churn: {y_test.iloc[idx]} | Predicted Probability: {y_pred_proba_gb[idx]:.4f}")
    plt.figure()
    shap.waterfall_plot(shap.Explanation(values=shap_values[idx], 
                                         base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
                                         data=X_test_transformed[idx], 
                                         feature_names=feature_names),
                        show=False)
    plt.title(f"SHAP Explanation for Customer {idx}")
    plt.tight_layout()
    plt.show()