"""
Visualizations for Neural Network Fundamentals:
- Activation functions and their derivatives
- Loss landscapes (MSE vs Binary Cross-Entropy)
- Training loss convergence curves
- 2D Decision boundaries for non-linear datasets
- Multi-optimizer convergence benchmark
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from activations import ActivationFunctions
from loss_functions import LossFunctions


class NNVisualizer:
    """
    Visualization helper for neural networks, activations, losses, and training dynamics.
    """

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Apply standard clean styling
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams.update({
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 14
        })

    def plot_activation_functions(self, filename="activation_functions.png"):
        """
        Plots Sigmoid, Tanh, ReLU, Leaky ReLU, and their derivatives over [-5, 5].
        """
        x = np.linspace(-5, 5, 500)
        functions = [
            ("Sigmoid", ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative, "#2b5c8f"),
            ("Tanh", ActivationFunctions.tanh, ActivationFunctions.tanh_derivative, "#d95f02"),
            ("ReLU", ActivationFunctions.relu, ActivationFunctions.relu_derivative, "#2ca02c"),
            ("Leaky ReLU", lambda z: ActivationFunctions.leaky_relu(z, alpha=0.1),
             lambda z: ActivationFunctions.leaky_relu_derivative(z, alpha=0.1), "#e7298a")
        ]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle("Neural Network Activation Functions & Their Derivatives", fontsize=16, fontweight="bold", y=0.98)

        for i, (name, fn, deriv_fn, color) in enumerate(functions):
            # Top row: Activation Function
            ax_fn = axes[0, i]
            y_fn = fn(x)
            ax_fn.plot(x, y_fn, label=name, color=color, linewidth=2.5)
            ax_fn.set_title(f"{name} Activation", fontweight="semibold")
            ax_fn.set_xlabel("Input (z)")
            ax_fn.set_ylabel("Activation a = f(z)")
            ax_fn.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax_fn.axvline(0, color="gray", linestyle="--", alpha=0.5)
            ax_fn.legend(loc="upper left")
            ax_fn.grid(True, alpha=0.3)

            # Bottom row: Derivative
            ax_deriv = axes[1, i]
            y_deriv = deriv_fn(x)
            ax_deriv.plot(x, y_deriv, label=f"d({name})/dz", color=color, linestyle="--", linewidth=2.5)
            ax_deriv.set_title(f"{name} Derivative", fontweight="semibold")
            ax_deriv.set_xlabel("Input (z)")
            ax_deriv.set_ylabel("Derivative f'(z)")
            ax_deriv.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax_deriv.axvline(0, color="gray", linestyle="--", alpha=0.5)
            ax_deriv.legend(loc="upper left")
            ax_deriv.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Saved Plot] Activation functions plot saved to: {save_path}")

    def plot_loss_functions(self, filename="loss_functions.png"):
        """
        Plots MSE vs Binary Cross-Entropy curves across prediction probabilities [0.01, 0.99]
        for true labels y=1 and y=0.
        """
        p = np.linspace(0.01, 0.99, 300)

        # For y_true = 1
        mse_y1 = (p - 1.0) ** 2
        bce_y1 = -np.log(p)

        # For y_true = 0
        mse_y0 = p ** 2
        bce_y0 = -np.log(1.0 - p)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Loss Function Comparison: MSE vs Binary Cross-Entropy", fontsize=15, fontweight="bold")

        # Plot y_true = 1
        axes[0].plot(p, bce_y1, label="Binary Cross-Entropy", color="#d62728", linewidth=2.5)
        axes[0].plot(p, mse_y1, label="Mean Squared Error (MSE)", color="#1f77b4", linewidth=2.5, linestyle="--")
        axes[0].set_title("Loss Landscape when True Label y = 1", fontweight="semibold")
        axes[0].set_xlabel("Predicted Probability p = P(y=1)")
        axes[0].set_ylabel("Loss Value")
        axes[0].set_ylim(0, 5)
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Plot y_true = 0
        axes[1].plot(p, bce_y0, label="Binary Cross-Entropy", color="#d62728", linewidth=2.5)
        axes[1].plot(p, mse_y0, label="Mean Squared Error (MSE)", color="#1f77b4", linewidth=2.5, linestyle="--")
        axes[1].set_title("Loss Landscape when True Label y = 0", fontweight="semibold")
        axes[1].set_xlabel("Predicted Probability p = P(y=1)")
        axes[1].set_ylabel("Loss Value")
        axes[1].set_ylim(0, 5)
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Saved Plot] Loss functions comparison saved to: {save_path}")

    def plot_training_loss(self, loss_history, title="Training Loss Convergence", filename="training_loss.png"):
        """
        Plots the training loss curve across epochs.
        """
        epochs = np.arange(1, len(loss_history) + 1)

        plt.figure(figsize=(9, 5))
        plt.plot(epochs, loss_history, color="#2b5c8f", linewidth=2.2, label="Training Loss")
        plt.fill_between(epochs, loss_history, alpha=0.15, color="#2b5c8f")
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Saved Plot] Training loss curve saved to: {save_path}")

    def plot_decision_boundary(self, model, X, y, title="Learned Decision Boundary", filename="decision_boundary.png"):
        """
        Plots the 2D decision boundary generated by the neural network.
        """
        # Determine grid bounds
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))

        # Predict across grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        probas = model.predict_proba(grid_points)

        if probas.shape[1] == 1:
            Z = probas.reshape(xx.shape)
        else:
            Z = probas[:, 1].reshape(xx.shape)

        plt.figure(figsize=(9, 7))
        contour = plt.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.8)
        plt.colorbar(contour, label="P(y = 1)")
        plt.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2.0, linestyles="--")

        # Scatter observations
        scatter = plt.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap="coolwarm",
            edgecolors="k",
            s=45,
            linewidths=0.7,
            alpha=0.9
        )
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Feature 1 (X1)", fontsize=11)
        plt.ylabel("Feature 2 (X2)", fontsize=11)
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Saved Plot] Decision boundary plot saved to: {save_path}")

    def plot_optimizer_comparison(self, histories, filename="optimizer_comparison.png"):
        """
        Plots a comparative benchmark of different optimizers (SGD, Momentum, RMSprop, Adam).
        """
        colors = {"SGD": "#7f7f7f", "Momentum": "#ff7f0e", "RMSprop": "#2ca02c", "Adam": "#1f77b4"}

        plt.figure(figsize=(10, 6))
        for opt_name, history in histories.items():
            epochs = np.arange(1, len(history) + 1)
            color = colors.get(opt_name, None)
            plt.plot(epochs, history, label=opt_name, linewidth=2.2, color=color)

        plt.title("Optimizer Benchmark: Convergence Speed & Stability", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Training Loss", fontsize=11)
        plt.yscale("log")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(fontsize=11)

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Saved Plot] Optimizer comparison saved to: {save_path}")
