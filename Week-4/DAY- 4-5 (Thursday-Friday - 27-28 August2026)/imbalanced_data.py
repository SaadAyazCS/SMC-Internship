import pandas as pd
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE, ADASYN


class ImbalancedDataHandler:
    """
    Implements and compares strategies for handling class imbalance:
    - Baseline (unbalanced)
    - Algorithmic weighting (class_weight='balanced')
    - SMOTE oversampling
    - ADASYN adaptive oversampling
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def generate_imbalanced_data(self, n_samples=1200, weights=(0.92, 0.08)):
        """Generates a synthetic binary dataset with severe class imbalance."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=8,
            n_informative=5,
            n_redundant=1,
            weights=list(weights),
            random_state=self.random_state
        )
        cols = [f"feat_{i}" for i in range(8)]
        return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")

    def compare_strategies(self, X_train, X_test, y_train, y_test):
        """
        Trains RandomForestClassifier under 4 different imbalance mitigation strategies
        and returns comparative metrics for the minority class (Class 1).
        """
        strategies = {}

        # 1. Baseline (no intervention)
        clf_baseline = RandomForestClassifier(random_state=self.random_state)
        clf_baseline.fit(X_train, y_train)
        pred_base = clf_baseline.predict(X_test)
        strategies["1. Baseline (None)"] = pred_base

        # 2. Class Weight Balanced
        clf_weighted = RandomForestClassifier(class_weight="balanced", random_state=self.random_state)
        clf_weighted.fit(X_train, y_train)
        pred_weight = clf_weighted.predict(X_test)
        strategies["2. Class Weights"] = pred_weight

        # 3. SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        clf_smote = RandomForestClassifier(random_state=self.random_state)
        clf_smote.fit(X_smote, y_smote)
        pred_smote = clf_smote.predict(X_test)
        strategies["3. SMOTE Resampling"] = pred_smote

        # 4. ADASYN
        adasyn = ADASYN(random_state=self.random_state)
        X_ada, y_ada = adasyn.fit_resample(X_train, y_train)
        clf_ada = RandomForestClassifier(random_state=self.random_state)
        clf_ada.fit(X_ada, y_ada)
        pred_ada = clf_ada.predict(X_test)
        strategies["4. ADASYN Resampling"] = pred_ada

        comparison_records = []
        for name, preds in strategies.items():
            prec = precision_score(y_test, preds, pos_label=1, zero_division=0)
            rec = recall_score(y_test, preds, pos_label=1, zero_division=0)
            f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)

            comparison_records.append({
                "Strategy": name,
                "Minority Precision": round(prec, 4),
                "Minority Recall": round(rec, 4),
                "Minority F1-Score": round(f1, 4)
            })

        return pd.DataFrame(comparison_records)
