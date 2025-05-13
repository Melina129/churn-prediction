# Required libraries
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read CSV file
df = pd.read_csv('churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 1. Convert 'TotalCharges' column to numeric (errors become NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 2. Drop missing rows (especially for TotalCharges)
df.dropna(inplace=True)

# 3. Drop 'customerID' column
df.drop('customerID', axis=1, inplace=True)

# 4. Convert 'Churn' column to 0/1
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# 📌 New Feature: TotalRevenue
df['TotalRevenue'] = df['MonthlyCharges'] * df['tenure']

# Separate numerical and categorical columns
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.drop('Churn')
categorical_features = df.select_dtypes(include='object').columns

print("Numerical columns:", list(numerical_features))
print("Categorical columns:", list(categorical_features))

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
print("New column count:", df_encoded.shape[1])

# Standardize numerical features
scaler = StandardScaler()
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

# Split features and target variable
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model (balanced & C=0.5)
log_model_balanced = LogisticRegression(class_weight='balanced', C=0.5, max_iter=1000)
log_model_balanced.fit(X_train, y_train)
y_pred_balanced = log_model_balanced.predict(X_test)

# Results
print("Balanced Model - Accuracy:", accuracy_score(y_test, y_pred_balanced))
print("Balanced Model - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_balanced))
print("Balanced Model - Classification Report:\n", classification_report(y_test, y_pred_balanced))

# Explanation
print("\n🔹 Accuracy:")
print("  Shows how many of the total examples the model correctly predicted.")
print("  Can be misleading with imbalanced datasets.")

print("\n🔹 Precision:")
print("  Of the examples predicted as 'churn', how many are actually churn?")
print("  Formula: True churn predictions / All churn predictions")
print("  Important to reduce false alarms.")

print("\n🔹 Recall:")
print("  Of the actual churned customers, how many were captured?")
print("  Formula: True churn predictions / All actual churn cases")
print("  Critical for preventing customer loss.")

print("\n🔹 F1-Score:")
print("  Harmonic mean of Precision and Recall.")
print("  Better summarizes model quality on imbalanced datasets.")

print("\n🔹 Confusion Matrix:")
print("  TP: Actual churn and correctly predicted")
print("  FP: Not churn but predicted as churn")
print("  TN: Not churn and correctly predicted")
print("  FN: Actual churn but missed (wrong prediction)")



