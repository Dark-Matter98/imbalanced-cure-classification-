import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder

class PhysicsGAN:
    """Lightweight generative model respecting process constraints.

    This model approximates each class distribution with a multivariate
    normal and samples new points within the allowed process ranges.
    It serves as a simple physics-aware data augmentation technique.
    """

    def __init__(self, process_constraints, random_state=42):
        self.process_constraints = process_constraints
        self.random_state = random_state
        self.le = LabelEncoder()
        self.class_stats = {}

    def _apply_constraints(self, X, cls):
        constraints = self.process_constraints[cls]
        mask = np.ones(X.shape[0], dtype=bool)
        mask &= (X[:, 0] >= constraints['Power'][0]) & (X[:, 0] <= constraints['Power'][1])
        mask &= (X[:, 1] >= constraints['Speed'][0]) & (X[:, 1] <= constraints['Speed'][1])
        mask &= (X[:, 2] >= constraints['Temperature'][0]) & (X[:, 2] <= constraints['Temperature'][1])
        return X[mask]

    def fit(self, X, y):
        np.random.seed(self.random_state)
        y = y.astype(str)
        self.le.fit(y)
        for cls in self.le.classes_:
            cls_idx = np.where(y == cls)[0]
            X_cls = X[cls_idx]
            X_valid = self._apply_constraints(X_cls, cls)
            if len(X_valid) >= 2:
                mean = X_valid.mean(axis=0)
                cov = np.cov(X_valid, rowvar=False)
            else:
                opt = self.process_constraints[cls]['optimal']
                mean = np.array([opt['Power'], opt['Speed'], opt['Temperature']])
                cov = np.diag([1.0, 1.0, 1.0])
            self.class_stats[cls] = {'mean': mean, 'cov': cov}
        return self

    def _sample_class(self, cls, n_samples):
        stats = self.class_stats[cls]
        mean, cov = stats['mean'], stats['cov']
        samples = []
        attempts = 0
        while len(samples) < n_samples and attempts < n_samples * 10:
            sample = np.random.multivariate_normal(mean, cov)
            sample = self._apply_constraints(sample.reshape(1, -1), cls)
            if len(sample) > 0:
                samples.append(sample[0])
            attempts += 1
        if len(samples) < n_samples:
            opt = self.process_constraints[cls]['optimal']
            for _ in range(n_samples - len(samples)):
                samples.append([opt['Power'], opt['Speed'], opt['Temperature']])
        return np.array(samples)

    def augment(self, X, y):
        """Generate synthetic samples for minority classes.

        Returns the augmented dataset (X_aug, y_aug).
        """
        self.fit(X, y)
        y = y.astype(str)
        counter = Counter(y)
        max_count = max(counter.values())
        X_syn = []
        y_syn = []
        for cls, count in counter.items():
            n_needed = max_count - count
            if n_needed > 0:
                samples = self._sample_class(cls, n_needed)
                X_syn.append(samples)
                y_syn.extend([cls] * len(samples))
        if X_syn:
            X_aug = np.vstack([X] + X_syn)
            y_aug = np.array(list(y) + y_syn)
        else:
            X_aug, y_aug = X, np.array(y)
        return X_aug, y_aug
