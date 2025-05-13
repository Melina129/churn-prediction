# Gerekli kütüphaneler
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Veri Yükleme
df = pd.read_csv('churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Veri Temizleme
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# 3. Yeni Özellik
df['TotalRevenue'] = df['MonthlyCharges'] * df['tenure']

# 4. Sayısal / Kategorik Ayırma
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.drop('Churn')
categorical_features = df.select_dtypes(include='object').columns

# 5. One-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# 6. Sayısal sütunları ölçekle
scaler = StandardScaler()
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

# 7. X ve y tanımla
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# 8. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 9. Hiperparametre Aralığı
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.25, 0.5, 1.0],
    'scale_pos_weight': [1, 2, 3]
}

# 10. Model Nesnesi
xgb_base = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

# 11. Randomized Search
xgb_random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=20,
    scoring='f1',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# 12. Eğitim
xgb_random_search.fit(X_train, y_train)

# 13. Tahmin
best_xgb = xgb_random_search.best_estimator_
y_pred_best_xgb = best_xgb.predict(X_test)

# 14. Değerlendirme
print("✅ Optimized XGBoost - Accuracy:", accuracy_score(y_test, y_pred_best_xgb))
print("✅ Optimized XGBoost - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best_xgb))
print("✅ Optimized XGBoost - Classification Report:\n", classification_report(y_test, y_pred_best_xgb))
print("✅ Best Parameters:\n", xgb_random_search.best_params_)
