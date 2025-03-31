import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class PhysicsConstrainedADWB_RF:
    def __init__(self, domain_weights, process_constraints, n_estimators=200, k=1.5, random_state=42):
        self.domain_weights = domain_weights
        self.process_constraints = process_constraints
        self.k = k
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced_subsample',
            min_samples_leaf=1,
            random_state=random_state
        )
        self.le = LabelEncoder()
        self.feature_names = ['Power', 'Speed', 'Temperature']

    def _apply_constraints(self, X, cls):
        constraints = self.process_constraints[cls]
        mask = np.ones(X.shape[0], dtype=bool)
        tolerance = 1e-6
        mask &= (X[:, 0] >= constraints['Power'][0] - tolerance) & (X[:, 0] <= constraints['Power'][1] + tolerance)
        mask &= (X[:, 1] >= constraints['Speed'][0] - tolerance) & (X[:, 1] <= constraints['Speed'][1] + tolerance)
        mask &= (X[:, 2] >= constraints['Temperature'][0] - tolerance) & (X[:, 2] <= constraints['Temperature'][1] + tolerance)
        return X[mask]

    def _physics_guided_oversample(self, X, y, cls_label):
        np.random.seed(self.random_state)
        cls_encoded = self.le.transform([cls_label])[0]
        cls_idx = np.where(y == cls_encoded)[0]
        if len(cls_idx) == 0:
            return np.array([])

        X_cls = X[cls_idx]
        X_valid = self._apply_constraints(X_cls, cls_label)

        if len(X_valid) == 0:
            optimal_values = self.process_constraints[cls_label].get('optimal', {})
            optimal_samples = []
            n_samples = 5 if cls_label == 'High' else 1
            for _ in range(n_samples):
                sample = [
                    optimal_values['Power'] * np.random.uniform(0.98, 1.02),
                    optimal_values['Speed'] * np.random.uniform(0.98, 1.02),
                    optimal_values['Temperature'] * np.random.uniform(0.98, 1.02)
                ]
                optimal_samples.append(sample)
            return np.array(optimal_samples)

        base_samples = len(X_cls)
        class_weight = 3.0 if cls_label == 'High' else self.domain_weights[cls_label]
        required_samples = int(base_samples * class_weight * self.k)
        n_needed = max(0, required_samples - len(X_valid))

        if n_needed == 0:
            return X_valid

        synthetic = []
        optimal_values = self.process_constraints[cls_label].get('optimal', {})
        for _ in range(n_needed):
            base_sample = X_valid[np.random.choice(len(X_valid))]
            weights = np.random.uniform(0, 1, 3)
            synthetic_sample = [
                weights[0] * base_sample[0] + (1 - weights[0]) * optimal_values['Power'],
                weights[1] * base_sample[1] + (1 - weights[1]) * optimal_values['Speed'],
                weights[2] * base_sample[2] + (1 - weights[2]) * optimal_values['Temperature']
            ]
            synthetic.append(synthetic_sample)

        return np.vstack([X_valid] + synthetic)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        y = y.astype(str)
        self.le.fit(y)
        y_encoded = self.le.transform(y)
        self.classes_ = self.le.classes_

        resampled_X = []
        resampled_y = []

        for cls in self.classes_:
            X_cls = self._physics_guided_oversample(X, y_encoded, cls)
            if len(X_cls) > 0:
                resampled_X.append(X_cls)
                resampled_y.extend([self.le.transform([cls])[0]] * len(X_cls))

        if not resampled_X:
            raise ValueError("No valid samples generated for any class")

        X_resampled = np.vstack(resampled_X)
        y_resampled = np.array(resampled_y)
        self.clf.fit(X_resampled, y_resampled)
        return self

    def predict(self, X):
        return self.le.inverse_transform(self.clf.predict(X))
