import os
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid_adwb = {
    'n_estimators': [100, 200],
    'k': [1.0, 1.5, 2.0]
}

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}

print("\nTraining ADWB-RF with GridSearchCV...")
adwb_rf = PhysicsConstrainedADWB_RF(domain_weights, process_constraints)
grid_adwb = GridSearchCV(adwb_rf, param_grid_adwb, cv=cv, scoring='f1_macro')
grid_adwb.fit(X_train.values, y_train.values)
print("Best ADWB-RF params:", grid_adwb.best_params_)
print("Best ADWB-RF CV score:", grid_adwb.best_score_)

print("\nTraining baseline Random Forest with GridSearchCV...")
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=cv, scoring='f1_macro')
grid_rf.fit(X_train, y_train)
print("Best RF params:", grid_rf.best_params_)
print("Best RF CV score:", grid_rf.best_score_)

print("\nApplying SMOTE and training SMOTE-RF with GridSearchCV...")
X_smote, y_smote = apply_safe_smote(X_train, y_train)
print("\nClass distributions:")
print("Original:", Counter(y_train))
print("After SMOTE:", Counter(y_smote))
smote_rf = RandomForestClassifier(random_state=42)
grid_smote = GridSearchCV(smote_rf, param_grid_rf, cv=cv, scoring='f1_macro')
grid_smote.fit(X_smote, y_smote)
print("Best SMOTE-RF params:", grid_smote.best_params_)
print("Best SMOTE-RF CV score:", grid_smote.best_score_)

print("\nADWB-RF Performance:")
y_pred_adwb = grid_adwb.best_estimator_.predict(X_test.values)
print(classification_report(y_test, y_pred_adwb))
print(confusion_matrix(y_test, y_pred_adwb))

print("\nStandard RF Performance:")
y_pred_rf = grid_rf.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

print("\nSMOTE-RF Performance:")
y_pred_smote = grid_smote.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred_smote))
print(confusion_matrix(y_test, y_pred_smote))

results = [
    ("ADWB-RF", grid_adwb.best_params_, grid_adwb.best_score_),
    ("RF", grid_rf.best_params_, grid_rf.best_score_),
    ("SMOTE-RF", grid_smote.best_params_, grid_smote.best_score_)
]

os.makedirs("results", exist_ok=True)
with open("results/best_params.txt", "w") as f:
    for name, params, score in results:
        f.write(f"{name} - Best Params: {params} | Best CV Score: {score:.4f}\n")

