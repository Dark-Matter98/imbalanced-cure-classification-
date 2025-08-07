import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from model import PhysicsConstrainedADWB_RF
from utils import apply_safe_smote

process_constraints = {
    'Low': {
        'Power': (67, 98),
        'Speed': (40, 200),
        'Temperature': (62.2, 360.14),
        'optimal': {'Power': 75, 'Speed': 120, 'Temperature': 178.57}
    },
    'Moderate': {
        'Power': (67, 98),
        'Speed': (20, 180),
        'Temperature': (74.33, 360.14),
        'optimal': {'Power': 84, 'Speed': 100, 'Temperature': 231.67}
    },
    'High': {
        'Power': (79, 98),
        'Speed': (20, 140),
        'Temperature': (193.27, 360.14),
        'optimal': {'Power': 98, 'Speed': 60, 'Temperature': 360.14}
    }
}

domain_weights = {
    'Low': 1.0,
    'Moderate': 1.2,
    'High': 3.0
}

df = pd.read_csv("data/categorized_cure_data.csv")
X = df[['Power', 'Speed', 'Temperature']]
y = df['Cure_Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

print("\nTraining ADWB-RF...")
adwb_rf = PhysicsConstrainedADWB_RF(domain_weights, process_constraints)
adwb_rf.fit(X_train.values, y_train.values)

print("\nTraining baseline Random Forest...")
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

print("\nApplying SMOTE and training SMOTE-RF...")
X_smote, y_smote = apply_safe_smote(X_train, y_train)
smote_rf = RandomForestClassifier(n_estimators=200)
smote_rf.fit(X_smote, y_smote)

print("\nTraining Logistic Regression...")
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
])
lr_pipeline.fit(X_train, y_train)

print("\nTraining Support Vector Machine...")
svc_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(class_weight="balanced", random_state=42)),
])
svc_pipeline.fit(X_train, y_train)

print("\nClass distributions:")
print("Original:", Counter(y_train))
print("After SMOTE:", Counter(y_smote))

print("\nADWB-RF Performance:")
y_pred_adwb = adwb_rf.predict(X_test)
print(classification_report(y_test, y_pred_adwb))
print(confusion_matrix(y_test, y_pred_adwb))

print("\nStandard RF Performance:")
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

print("\nSMOTE-RF Performance:")
y_pred_smote = smote_rf.predict(X_test)
print(classification_report(y_test, y_pred_smote))
print(confusion_matrix(y_test, y_pred_smote))

print("\nLogistic Regression Performance:")
y_pred_lr = lr_pipeline.predict(X_test)
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))

print("\nSupport Vector Machine Performance:")
y_pred_svc = svc_pipeline.predict(X_test)
print(classification_report(y_test, y_pred_svc))
print(confusion_matrix(y_test, y_pred_svc))
