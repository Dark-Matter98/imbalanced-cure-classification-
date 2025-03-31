from collections import Counter
from imblearn.over_sampling import SMOTE

def apply_safe_smote(X_train, y_train):
    class_counts = Counter(y_train)
    min_samples = min(class_counts.values())

    if min_samples < 6:
        k_neighbors = min_samples - 1
        print(f"\nUsing reduced k_neighbors={k_neighbors} for SMOTE due to small class size")
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    else:
        smote = SMOTE(random_state=42)

    return smote.fit_resample(X_train, y_train)