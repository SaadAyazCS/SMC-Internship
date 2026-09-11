import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from classification_models import ClassificationModels


class ClassificationComparison:
    """
    Evaluates and compares classification models using Stratified K-Fold Cross-Validation.
    """

    def __init__(self, n_splits=5, random_state=42, output_dir="results"):
        self.n_splits = n_splits
        self.random_state = random_state
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results_list = []

    def run_cross_validation(self, dataset_name, X, y, is_binary=True):
        """
        Runs Stratified K-Fold CV for all 8 classifiers and collects mean scores.
        """
        print(f"\n==================================================")
        print(f"  Cross-Validating Models on: {dataset_name}")
        print(f"==================================================")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision" if is_binary else "precision_weighted",
            "recall": "recall" if is_binary else "recall_weighted",
            "f1": "f1" if is_binary else "f1_weighted"
        }

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        models = ClassificationModels(random_state=self.random_state).models

        for name, model in models.items():
            cv_results = cross_validate(
                model,
                X_scaled,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                error_score="raise"
            )

            mean_acc = np.mean(cv_results["test_accuracy"])
            mean_prec = np.mean(cv_results[f"test_precision"])
            mean_rec = np.mean(cv_results[f"test_recall"])
            mean_f1 = np.mean(cv_results[f"test_f1"])

            self.results_list.append({
                "Dataset": dataset_name,
                "Model": name,
                "Accuracy": round(mean_acc, 4),
                "Precision": round(mean_prec, 4),
                "Recall": round(mean_rec, 4),
                "F1-Score": round(mean_f1, 4)
            })

    def get_summary_dataframe(self):
        """Returns comparison results DataFrame."""
        return pd.DataFrame(self.results_list)

    def print_summary(self):
        """Prints formatted tabular summary."""
        df = self.get_summary_dataframe()
        print("\n==================================================")
        print("     CLASSIFIER CROSS-VALIDATION SUMMARY TABLE    ")
        print("==================================================")
        print(df.to_string(index=False))
        return df

    def export_csv(self, filename="classification_comparison.csv"):
        """Exports comparison table to CSV."""
        df = self.get_summary_dataframe()
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"\nSaved cross-validation results to: {filepath}")
        return filepath
