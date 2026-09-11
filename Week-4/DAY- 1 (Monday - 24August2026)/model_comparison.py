import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from regression_models import RegressionModels


class ModelComparison:
    """
    Orchestrates training and comparative evaluation of multiple regression models
    across 3+ datasets.
    """

    def __init__(self, random_state=42, output_dir="results"):
        self.random_state = random_state
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results_list = []

    def compare_on_dataset(self, dataset_name, X, y, test_size=0.2):
        """
        Splits, scales features, fits all regression models, and calculates metrics.
        Returns the detailed evaluation dict (including predictions).
        """
        print(f"\n==================================================")
        print(f"  Evaluating Models on: {dataset_name}")
        print(f"==================================================")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        reg_models = RegressionModels(random_state=self.random_state)
        reg_models.train_all(X_train_scaled, y_train)
        eval_results = reg_models.evaluate_all(X_test_scaled, y_test)

        for model_name, metrics in eval_results.items():
            self.results_list.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "MAE": round(metrics["MAE"], 4),
                "MSE": round(metrics["MSE"], 4),
                "RMSE": round(metrics["RMSE"], 4),
                "R2": round(metrics["R2"], 4)
            })

        return {
            "eval_results": eval_results,
            "y_test": y_test,
            "X_train": X_train_scaled,
            "y_train": y_train
        }

    def get_summary_dataframe(self):
        """Returns the gathered comparison results as a pandas DataFrame."""
        return pd.DataFrame(self.results_list)

    def print_summary(self):
        """Prints formatted markdown/tabular summary of model performances."""
        df = self.get_summary_dataframe()
        print("\n==================================================")
        print("           MODEL COMPARISON SUMMARY TABLE         ")
        print("==================================================")
        print(df.to_string(index=False))
        return df

    def export_csv(self, filename="regression_comparison.csv"):
        """Saves comparison results to a CSV file."""
        df = self.get_summary_dataframe()
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"\nSaved comparison results to: {filepath}")
        return filepath
