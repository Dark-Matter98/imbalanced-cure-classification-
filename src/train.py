import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

metrics_summary = {
    'ADWB-RF': {'precision': [], 'recall': [], 'f1': [], 'accuracy': []},
    'RF': {'precision': [], 'recall': [], 'f1': [], 'accuracy': []},
    'SMOTE-RF': {'precision': [], 'recall': [], 'f1': [], 'accuracy': []}
}

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # ADWB-RF
    adwb_rf = PhysicsConstrainedADWB_RF(domain_weights, process_constraints)
    adwb_rf.fit(X_train.values, y_train.values)
    y_pred = adwb_rf.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    metrics_summary['ADWB-RF']['precision'].append(precision)
    metrics_summary['ADWB-RF']['recall'].append(recall)
    metrics_summary['ADWB-RF']['f1'].append(f1)
    metrics_summary['ADWB-RF']['accuracy'].append(accuracy)

    # Baseline RF
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_test, y_pred_rf)
    metrics_summary['RF']['precision'].append(precision)
    metrics_summary['RF']['recall'].append(recall)
    metrics_summary['RF']['f1'].append(f1)
    metrics_summary['RF']['accuracy'].append(accuracy)

    # SMOTE-RF
    X_smote, y_smote = apply_safe_smote(X_train, y_train)
    smote_rf = RandomForestClassifier(n_estimators=200, random_state=42)
    smote_rf.fit(X_smote, y_smote)
    y_pred_smote = smote_rf.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_smote, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_test, y_pred_smote)
    metrics_summary['SMOTE-RF']['precision'].append(precision)
    metrics_summary['SMOTE-RF']['recall'].append(recall)
    metrics_summary['SMOTE-RF']['f1'].append(f1)
    metrics_summary['SMOTE-RF']['accuracy'].append(accuracy)

for model_name, metrics in metrics_summary.items():
    print(f"\n{model_name} Performance:")
    for metric_name, values in metrics.items():
        print(f"{metric_name.capitalize()}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")
