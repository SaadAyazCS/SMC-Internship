import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class RegressionVisualizer:
    """
    Handles plotting for regression models:
    - Actual vs. Predicted values
    - Residual distributions
    - Comparative metric bar charts
    """

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def plot_predictions_vs_actual(self, y_actual, predictions_dict, dataset_name="Dataset"):
        """Plots Actual vs. Predicted scatter plots for each model."""
        n_models = len(predictions_dict)
        cols = 3
        rows = int(np.ceil(n_models / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = axes.flatten()

        for idx, (model_name, res) in enumerate(predictions_dict.items()):
            y_pred = res["predictions"]
            ax = axes[idx]
            ax.scatter(y_actual, y_pred, alpha=0.5, color="teal", edgecolors="k", s=30)
            
            # Perfect prediction line
            min_val = min(y_actual.min(), y_pred.min())
            max_val = max(y_actual.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal")
            
            ax.set_title(f"{model_name}\nR² = {res['R2']:.3f} | RMSE = {res['RMSE']:.3f}", fontsize=10)
            ax.set_xlabel("Actual Values", fontsize=9)
            ax.set_ylabel("Predicted Values", fontsize=9)
            ax.legend(fontsize=8)

        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(f"Predictions vs Actual - {dataset_name}", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{dataset_name.lower().replace(' ', '_')}_pred_vs_actual.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {save_path}")

    def plot_residuals(self, y_actual, predictions_dict, dataset_name="Dataset"):
        """Plots residuals distribution for each model."""
        n_models = len(predictions_dict)
        cols = 3
        rows = int(np.ceil(n_models / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = axes.flatten()

        for idx, (model_name, res) in enumerate(predictions_dict.items()):
            y_pred = res["predictions"]
            residuals = y_actual - y_pred
            ax = axes[idx]
            sns.histplot(residuals, kde=True, ax=ax, color="indigo", bins=20)
            ax.axvline(0, color="red", linestyle="--", lw=1.5)
            ax.set_title(f"Residuals: {model_name}", fontsize=10)
            ax.set_xlabel("Residual (Actual - Predicted)", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)

        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(f"Residual Distributions - {dataset_name}", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{dataset_name.lower().replace(' ', '_')}_residuals.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {save_path}")

    def plot_model_comparison(self, comparison_df, metric="R2"):
        """Plots a comparison bar chart of a specific metric across datasets and models."""
        plt.figure(figsize=(12, 6))
        chart = sns.barplot(
            data=comparison_df,
            x="Model",
            y=metric,
            hue="Dataset",
            palette="Set2"
        )
        plt.title(f"Model Comparison across Datasets ({metric})", fontsize=14, fontweight="bold")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"model_comparison_{metric.lower()}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {save_path}")
