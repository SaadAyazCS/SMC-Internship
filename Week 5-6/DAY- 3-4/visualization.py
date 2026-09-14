"""
Visualization module for Deep Learning Day 3-4.
Provides DLVisualizer class for:
- Training/validation loss and accuracy curves
- Confusion matrix heatmaps
- Model comparison bar charts
- Sample prediction grids
- CNN first-layer filter visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix


class DLVisualizer:
    """Visualization tools for CNN and MLP training experiments."""

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
        plt.style.use(style)

    def plot_training_curves(self, histories, filename="training_curves.png"):
        """
        Plots training and validation loss/accuracy curves for multiple models.
        histories: dict of {model_name: {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}}
        """
        n_models = len(histories)
        fig, axes = plt.subplots(n_models, 2, figsize=(14, 4 * n_models))
        if n_models == 1:
            axes = [axes]
        fig.suptitle("Training Curves: Loss & Accuracy", fontsize=15, fontweight="bold")

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, (name, hist) in enumerate(histories.items()):
            epochs = range(1, len(hist["train_loss"]) + 1)
            color  = colors[i % len(colors)]

            ax_loss = axes[i][0]
            ax_loss.plot(epochs, hist["train_loss"], label="Train Loss", color=color, linewidth=2)
            ax_loss.plot(epochs, hist["val_loss"],   label="Val Loss",   color=color, linewidth=2, linestyle="--")
            ax_loss.set_title(f"{name} -- Loss", fontweight="semibold")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)

            ax_acc = axes[i][1]
            ax_acc.plot(epochs, [a * 100 for a in hist["train_acc"]], label="Train Acc", color=color, linewidth=2)
            ax_acc.plot(epochs, [a * 100 for a in hist["val_acc"]],   label="Val Acc",   color=color, linewidth=2, linestyle="--")
            ax_acc.set_title(f"{name} -- Accuracy", fontweight="semibold")
            ax_acc.set_xlabel("Epoch")
            ax_acc.set_ylabel("Accuracy (%)")
            ax_acc.legend()
            ax_acc.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Saved Plot] {path}")

    def plot_confusion_matrix(self, y_true, y_pred, class_names, title="Confusion Matrix",
                               filename="confusion_matrix.png"):
        """Plots a normalized confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor="gray")
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Label", fontsize=11)
        plt.ylabel("True Label", fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Saved Plot] {path}")

    def plot_model_comparison(self, model_accs, filename="model_comparison.png"):
        """
        Bar chart comparing test accuracies of multiple models.
        model_accs: dict of {model_name: accuracy (float in [0,1])}
        """
        names  = list(model_accs.keys())
        accs   = [v * 100 for v in model_accs.values()]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][:len(names)]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(names, accs, color=colors, edgecolor="white", linewidth=1.2, width=0.5)
        for bar, acc in zip(bars, accs):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{acc:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        plt.title("Model Comparison: Test Accuracy", fontsize=14, fontweight="bold")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 110)
        plt.xticks(rotation=20, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Saved Plot] {path}")

    def plot_per_class_accuracy(self, per_class_acc, dataset_name="CIFAR-10",
                                 filename="per_class_accuracy.png"):
        """Horizontal bar chart of per-class accuracies."""
        names = list(per_class_acc.keys())
        accs  = [v * 100 for v in per_class_acc.values()]

        plt.figure(figsize=(8, 6))
        bars = plt.barh(names, accs, color="#2b5c8f", edgecolor="white")
        for bar, acc in zip(bars, accs):
            plt.text(acc + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{acc:.1f}%", va="center", fontsize=9)
        plt.title(f"{dataset_name} -- Per-class Test Accuracy", fontsize=14, fontweight="bold")
        plt.xlabel("Accuracy (%)")
        plt.xlim(0, 110)
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Saved Plot] {path}")

    def plot_transfer_vs_scratch(self, results, filename="transfer_learning_curves.png"):
        """Plots validation accuracy curves for transfer learning models."""
        plt.figure(figsize=(10, 5))
        colors = {"ResNet-18 (frozen backbone)": "#e6550d",
                  "MobileNetV2 (frozen backbone)": "#31a354"}

        for name, res in results.items():
            hist   = res["history"]
            epochs = range(1, len(hist["val_acc"]) + 1)
            color  = colors.get(name, "#756bb1")
            plt.plot(epochs, [a * 100 for a in hist["val_acc"]], label=name, linewidth=2.2, color=color)

        plt.title("Transfer Learning: Validation Accuracy Curves", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Saved Plot] {path}")

    def plot_sample_predictions(self, images, labels, preds, class_names,
                                 n=16, filename="sample_predictions.png"):
        """
        Plots a grid of sample images with true/predicted labels.
        Green title = correct, Red title = incorrect.
        """
        n = min(n, len(images))
        cols = 8
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.2))
        axes = axes.flatten()

        for i in range(n):
            ax = axes[i]
            img = images[i]
            # Handle CHW -> HWC for display
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.transpose(1, 2, 0)
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
                ax.imshow(img, cmap="gray")
            else:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                ax.imshow(img)
            correct = labels[i] == preds[i]
            color = "green" if correct else "red"
            ax.set_title(f"T:{class_names[labels[i]]}\nP:{class_names[preds[i]]}", fontsize=7, color=color)
            ax.axis("off")

        for i in range(n, len(axes)):
            axes[i].axis("off")

        fig.suptitle("Sample Predictions (Green=Correct, Red=Wrong)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Saved Plot] {path}")
