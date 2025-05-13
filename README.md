 Churn Prediction Project 📊

## Project Description
This project focuses on predicting customer churn using supervised machine learning techniques such as Logistic Regression, Random Forest, and XGBoost. The goal is to identify customers likely to leave a subscription-based service, enabling proactive customer retention strategies.

---

## Technologies & Libraries Used
- Python
- pandas
- numpy
- scikit-learn
- XGBoost
- StandardScaler
- RandomizedSearchCV

---

## Project Structure

churn-prediction/
├── lr_churn.py
├── rf_churn.py
├── XGboost_churn.py
└── customer_data.csv

---

## Model Performance (Example Results)
| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 86%      | 82%       | 78%    | 80%      |
| Random Forest       | 90%      | 88%       | 84%    | 86%      |
| XGBoost             | 92%      | 89%       | 87%    | 88%      |

*Note: These are example scores. Actual results may vary depending on hyperparameters and data split.*
---

## How to Use
```bash
# Install dependencies
pip install -r requirements.txt

# Run model scripts
python lr_churn.py
python rf_churn.py
python XGboost_churn.py

---

## Author

**[@Melina129](https://github.com/Melina129)**





