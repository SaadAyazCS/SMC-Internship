import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc


class ClassificationVisualizer:
    """
    Visualizes classification results:
    - Confusion Matrix Heatmaps
    - ROC Curves
    - Cross-Validated Performance Benchmarks
    - Class Distribution changes before/after SMOTE
    """

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def plot_confusion_matrices(self, eval_results, dataset_name="Dataset"):
        """Plots confusion matrix subplots for each model."""
        n_models = len(eval_results)
        cols = 4
        rows = int(np.ceil(n_models / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten()

        for idx, (name, metrics) in enumerate(eval_results.items()):
            ax = axes[idx]
            cm = metrics["Confusion Matrix"]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_title(f"{name}\nAcc: {metrics['Accuracy']:.3f} | F1: {metrics['F1-Score']:.3f}", fontsize=9)
            ax.set_xlabel("Predicted Label", fontsize=8)
            ax.set_ylabel("True Label", fontsize=8)

        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(f"Confusion Matrices - {dataset_name}", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{dataset_name.lower().replace(' ', '_')}_confusion_matrices.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {save_path}")

    def plot_roc_curves(self, y_true, model_predictions, dataset_name="Dataset"):
        """Plots ROC curves for binary classification models."""
        plt.figure(figsize=(9, 7))

        for name, preds in model_predictions.items():
            y_proba = preds["y_proba"]
            if y_proba is not None:
                pos_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                fpr, tpr, _ = roc_curve(y_true, pos_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1.5, label="Random Guess")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=11)
        plt.ylabel("True Positive Rate", fontsize=11)
        plt.title(f"ROC Curves - {dataset_name}", fontsize=13, fontweight="bold")
        plt.legend(loc="lower right", fontsize=9)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, f"{dataset_name.lower().replace(' ', '_')}_roc_curves.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {save_path}")

    def plot_model_comparison(self, comparison_df):
        """Plots grouped bar charts comparing models across metrics."""
        metrics = ["Accuracy", "F1-Score"]
        if "ROC-AUC" in comparison_df.columns and comparison_df["ROC-AUC"].notnull().any():
            metrics.append("ROC-AUC")

        for metric in metrics:
            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=comparison_df,
                x="Model",
                y=metric,
                hue="Dataset",
                palette="viridis"
            )
            plt.title(f"Model Benchmark Comparison ({metric})", fontsize=13, fontweight="bold")
            plt.xticks(rotation=30, ha="right")
            plt.ylim(0, 1.05)
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, f"model_comparison_{metric.lower().replace('-', '_')}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  Saved plot: {save_path}")

    def plot_imbalance_distribution(self, before_counts, after_counts):
        """Plots bar chart of class counts before vs. after SMOTE."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        x_before = list(before_counts.keys())
        sns.barplot(x=x_before, y=list(before_counts.values()), hue=x_before, ax=axes[0], palette="Reds", legend=False)
        axes[0].set_title("Before SMOTE (Imbalanced)", fontsize=11)
        axes[0].set_xlabel("Class")
        axes[0].set_ylabel("Count")

        x_after = list(after_counts.keys())
        sns.barplot(x=x_after, y=list(after_counts.values()), hue=x_after, ax=axes[1], palette="Greens", legend=False)
        axes[1].set_title("After SMOTE (Balanced)", fontsize=11)
        axes[1].set_xlabel("Class")
        axes[1].set_ylabel("Count")

        plt.suptitle("SMOTE Resampling Class Distribution", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "smote_class_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {save_path}")
