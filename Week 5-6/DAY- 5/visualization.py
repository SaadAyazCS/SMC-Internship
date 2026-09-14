"""
Visualization module for Deep Learning Day 5 (RNNs).
RNNVisualizer class for:
- Time series prediction plots (RNN vs LSTM vs GRU)
- Training convergence curves
- Generated text samples display
- Sentiment analysis confusion matrix
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class RNNVisualizer:
    """Visualization tools for RNN/LSTM/GRU experiments."""

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
        plt.style.use(style)

    def plot_time_series_comparison(self, sample_preds, sample_y, filename="ts_predictions.png"):
        """
        Plots predicted vs ground truth for each RNN architecture.
        sample_preds: dict {model_name: preds array (N, pred_steps)}
        sample_y: ground truth array (N, pred_steps)
        """
        n_models = len(sample_preds)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 3 * n_models), sharex=False)
        if n_models == 1:
            axes = [axes]
        fig.suptitle("Time Series Forecasting: Predicted vs Ground Truth", fontsize=14, fontweight="bold")

        colors = {"Vanilla RNN": "#7f7f7f", "LSTM": "#1f77b4", "GRU": "#2ca02c"}
        sample_idx = 0  # show first sample

        for ax, (name, preds) in zip(axes, sample_preds.items()):
            gt   = sample_y[sample_idx]       # (pred_steps,)
            pred = preds[sample_idx]           # (pred_steps,)
            steps = np.arange(len(gt))
            color = colors.get(name, "#d62728")

            ax.plot(steps, gt,   label="Ground Truth", color="black", linewidth=2.0, linestyle="--")
            ax.plot(steps, pred, label=f"{name} Prediction", color=color, linewidth=2.0)
            ax.fill_between(steps, gt, pred, alpha=0.15, color=color)
            ax.set_title(f"{name}", fontweight="semibold")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Saved Plot] {path}")

    def plot_rnn_loss_comparison(self, histories, filename="ts_loss_curves.png"):
        """
        Plots validation loss curves for multiple RNN architectures.
        histories: dict {model_name: {train_loss: [], val_loss: []}}
        """
        colors = {"Vanilla RNN": "#7f7f7f", "LSTM": "#1f77b4", "GRU": "#2ca02c"}
        plt.figure(figsize=(10, 5))
        for name, hist in histories.items():
            epochs = range(1, len(hist["val_loss"]) + 1)
            color  = colors.get(name, "#d62728")
            plt.plot(epochs, hist["val_loss"], label=f"{name} (val)", linewidth=2.2, color=color)
            plt.plot(epochs, hist["train_loss"], linewidth=1.2, color=color, linestyle="--", alpha=0.5)

        plt.title("Time Series RNN Comparison: Loss Curves", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Saved Plot] {path}")

    def plot_text_gen_loss(self, loss_history, filename="textgen_loss.png"):
        """Plots CharLSTM training loss curve."""
        epochs = range(1, len(loss_history) + 1)
        plt.figure(figsize=(9, 4))
        plt.plot(epochs, loss_history, color="#e6550d", linewidth=2.2, label="Train Loss")
        plt.fill_between(epochs, loss_history, alpha=0.15, color="#e6550d")
        plt.title("Character-Level LSTM: Training Loss Curve", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy Loss")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Saved Plot] {path}")

    def plot_sentiment_curves(self, history, filename="sentiment_curves.png"):
        """Plots sentiment BiLSTM training/validation loss and accuracy."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Sentiment Analysis (BiLSTM + Attention): Training Curves",
                     fontsize=14, fontweight="bold")
        epochs = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#3182bd", linewidth=2)
        axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#3182bd", linewidth=2, linestyle="--")
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, [a*100 for a in history["train_acc"]], label="Train Acc", color="#e6550d", linewidth=2)
        axes[1].plot(epochs, [a*100 for a in history["val_acc"]],   label="Val Acc",   color="#e6550d", linewidth=2, linestyle="--")
        axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Saved Plot] {path}")

    def plot_sentiment_confusion(self, y_true, y_pred, filename="sentiment_confusion.png"):
        """Plots confusion matrix for sentiment analysis."""
        cm      = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"],
                    linewidths=0.5)
        plt.title("Sentiment Analysis -- Confusion Matrix", fontsize=13, fontweight="bold")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Saved Plot] {path}")
