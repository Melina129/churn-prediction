# Gerekli kütüphaneler
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# CSV dosyasını okuma
df = pd.read_csv('churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 1. TotalCharges sütununu sayıya çevir (hataları NaN yap)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 2. Eksik satırları sil (özellikle TotalCharges için)
df.dropna(inplace=True)

# 3. customerID sütununu düşür
df.drop('customerID', axis=1, inplace=True)

# 4. Churn sütununu 0/1'e çevir
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# 📌 Yeni Özellik: TotalRevenue
df['TotalRevenue'] = df['MonthlyCharges'] * df['tenure']

# Sayısal ve kategorik sütunları ayır
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.drop('Churn')
categorical_features = df.select_dtypes(include='object').columns

print("Sayısal sütunlar:", list(numerical_features))
print("Kategorik sütunlar:", list(categorical_features))

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
print("Yeni sütun sayısı:", df_encoded.shape[1])

# Sayısal verileri standardize et
scaler = StandardScaler()
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

# Özellikler ve hedef değişkeni ayır
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model (balanced & C=0.5)
log_model_balanced = LogisticRegression(class_weight='balanced', C=0.5, max_iter=1000)
log_model_balanced.fit(X_train, y_train)
y_pred_balanced = log_model_balanced.predict(X_test)

# Sonuçlar
print("Balanced Model - Accuracy:", accuracy_score(y_test, y_pred_balanced))
print("Balanced Model - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_balanced))
print("Balanced Model - Classification Report:\n", classification_report(y_test, y_pred_balanced))

#Yorum
print("\n🔹 Accuracy (Doğruluk):")
print("  Modelin tüm örnekler içinde ne kadarını doğru tahmin ettiğini gösterir.")
print("  Dengesiz veri setlerinde yanıltıcı olabilir.")

print("\n🔹 Precision (Kesinlik):")
print("  Modelin 'churn' dediği örneklerin kaçı gerçekten churn?")
print("  Yani: Doğru churn tahmini / Tüm churn tahmini")
print("  Yanlış alarmları azaltmak için önemlidir.")

print("\n🔹 Recall (Duyarlılık):")
print("  Gerçekten churn eden müşterilerin kaçı yakalandı?")
print("  Yani: Doğru churn tahmini / Tüm gerçek churn")
print("  Müşteri kaybını önlemede kritiktir.")

print("\n🔹 F1-Score:")
print("  Precision ve Recall'un dengeli ortalamasıdır.")
print("  Dengesiz veri setlerinde model kalitesini daha iyi özetler.")

print("\n🔹 Confusion Matrix:")
print("  TP: Gerçek churn ve doğru tahmin")
print("  FP: Churn değil ama churn diye tahmin")
print("  TN: Gerçek churn değil ve doğru tahmin")
print("  FN: Gerçek churn ama kaçırılmış (yanlış tahmin)")








