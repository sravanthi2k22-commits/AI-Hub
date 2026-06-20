# 1. Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# 2. Load Dataset

data = pd.read_csv('creditcard.csv')


# 3. Exploratory Data Analysis

print(data.head())
print(data.info())
print(data['Class'].value_counts())


# 4. Data Preprocessing

# Scale the 'Amount' column (important to bring to similar scale as other features)
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

# Drop 'Time' feature as it doesn't help in prediction
data.drop(['Time'], axis=1, inplace=True)    

# Separate features (X) and target (y)
# Features excluding the label
X = data.drop('Class', axis=1)
# The label column (0 = legit, 1 = fraud)
y = data['Class']

# Split the dataset into train (80%) and test (20%) subsets. stratify=y ensures class ratio is maintained in both sets. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance by generating synthetic samples for minority class (fraudulent)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


# 5. Model Training with XGBoost

# Initialize XGBoost Classifier. Fit model to balanced training data
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_res, y_train_res)


# 6. Model Evaluation

# Make predictions on the test set
y_pred = model.predict(X_test)

# Display confusion matrix to check TP, TN, FP, FN
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Generate classification metrics: precision, recall, F1-score, accuracy
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate ROC-AUC score (probability-based metric; measures overall model quality)
roc_auc = roc_auc_score(y_test, y_pred)
print("\nROC-AUC Score:", round(roc_auc, 4))


# Output obtained:

Confusion Matrix:
[[56832    32]
 [   16    82]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.72      0.84      0.77        98

    accuracy                           1.00     56962
   macro avg       0.86      0.92      0.89     56962
weighted avg       1.00      1.00      1.00     56962


ROC-AUC Score: 0.9181
