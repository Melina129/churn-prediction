# Gerekli kütüphaneler
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# 1. Veri Yükleme ve Temizlik
df = pd.read_csv('churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# 2. Yeni Özellik: TotalRevenue
df['TotalRevenue'] = df['MonthlyCharges'] * df['tenure']

# 3. Feature Engineering
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.drop('Churn')
categorical_features = df.select_dtypes(include='object').columns

df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

scaler = StandardScaler()
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

# 4. Train/Test Split
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. RandomizedSearchCV ile Random Forest Optimizasyonu
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced'],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    scoring='f1',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# 6. En iyi model ile değerlendirme
best_rf = random_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

print("Optimized Random Forest - Accuracy:", accuracy_score(y_test, y_pred_best))
print("Optimized Random Forest - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("Optimized Random Forest - Classification Report:\n", classification_report(y_test, y_pred_best))
print("Best Parameters:\n", random_search.best_params_)
