import os
import sys
from collections import Counter

import numpy as np

# Ensure src is on the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import apply_safe_smote


def test_apply_safe_smote_zero_samples():
    X = np.empty((0, 2))
    y = np.array([])
    X_res, y_res = apply_safe_smote(X, y)
    assert X_res.shape == (0, 2)
    assert y_res.shape == (0,)


def test_apply_safe_smote_one_sample():
    X = np.array([[1, 2]])
    y = np.array([0])
    X_res, y_res = apply_safe_smote(X, y)
    assert np.array_equal(X_res, X)
    assert np.array_equal(y_res, y)


def test_apply_safe_smote_two_samples():
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 0],
            [2, 1],
        ]
    )
    y = np.array([0, 0, 0, 0, 1, 1])
    X_res, y_res = apply_safe_smote(X, y)
    counts = Counter(y_res)
    assert counts[0] == counts[1] == 4
    assert len(y_res) > len(y)

