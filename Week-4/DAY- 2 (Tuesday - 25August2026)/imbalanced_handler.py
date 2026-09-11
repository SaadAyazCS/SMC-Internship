import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from collections import Counter


class ImbalancedHandler:
    """
    Utilities to create and manage imbalanced datasets using SMOTE and class reweighting.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def create_imbalanced_dataset(self, n_samples=1200, weights=(0.90, 0.10)):
        """
        Creates a synthetic imbalanced binary classification dataset.
        Defaults to a 90:10 class imbalance ratio.
        """
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=6,
            n_redundant=2,
            weights=list(weights),
            random_state=self.random_state
        )
        feature_names = [f"feature_{i}" for i in range(10)]
        df_X = pd.DataFrame(X, columns=feature_names)
        s_y = pd.Series(y, name="target")
        return df_X, s_y

    def apply_smote(self, X, y):
        """
        Applies Synthetic Minority Over-sampling Technique (SMOTE) to balance classes.
        """
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        before_counts = dict(Counter(y))
        after_counts = dict(Counter(y_resampled))
        return X_resampled, y_resampled, before_counts, after_counts
