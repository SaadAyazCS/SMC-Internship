"""
WEEK 5-6: DEEP LEARNING & NEURAL NETWORKS
DAY 1-2: Neural Network Fundamentals

Demonstration Pipeline:
1. Activation Functions & Derivatives Visualization
2. Loss Functions & Optimization Landscapes
3. Binary Classification: Non-linear decision boundary on Make Moons dataset
4. Multi-class Classification: Hand-written digits recognition on Load Digits dataset
5. Optimizer Benchmark: SGD vs Momentum vs RMSprop vs Adam
"""

import os
import numpy as np
from sklearn.datasets import make_moons, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_network import NeuralNetwork
from visualization import NNVisualizer


def main():
    print("================================================================")
    print(" WEEK 5-6 - DAY 1-2: NEURAL NETWORK FUNDAMENTALS (NUMPY)")
    print("================================================================")

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    visualizer = NNVisualizer(output_dir=output_dir)

    # -----------------------------------------------------------------
    # Step 1: Visualize Activation Functions and Their Derivatives
    # -----------------------------------------------------------------
    print("\n[Step 1] Generating Activation Functions & Derivatives Curves...")
    visualizer.plot_activation_functions("activation_functions.png")

    # -----------------------------------------------------------------
    # Step 2: Visualize Loss Landscapes
    # -----------------------------------------------------------------
    print("\n[Step 2] Generating Loss Landscape Visualizations (MSE vs BCE)...")
    visualizer.plot_loss_functions("loss_functions.png")

    # -----------------------------------------------------------------
    # Step 3: Binary Classification on Make Moons (Non-Linear Boundary)
    # -----------------------------------------------------------------
    print("\n[Step 3] Training Neural Network on Non-Linear 'Make Moons' Dataset...")
    X_moons, y_moons = make_moons(n_samples=1200, noise=0.20, random_state=42)

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_moons, y_moons, test_size=0.2, random_state=42, stratify=y_moons
    )

    scaler_m = StandardScaler()
    X_train_m_scaled = scaler_m.fit_transform(X_train_m)
    X_test_m_scaled = scaler_m.transform(X_test_m)

    # Architecture: 2 -> 16 -> 8 -> 1
    nn_moons = NeuralNetwork(
        layer_sizes=[2, 16, 8, 1],
        activations=["relu", "relu", "sigmoid"],
        loss="bce",
        optimizer="adam",
        learning_rate=0.01,
        l2_lambda=0.001,
        dropout_rate=0.05,
        seed=42
    )

    print("  Architecture: Input(2) -> Dense(16, ReLU) -> Dense(8, ReLU) -> Dense(1, Sigmoid)")
    print("  Optimizer: Adam (lr=0.01) | Regularization: L2 (0.001) + Dropout (0.05)")
    nn_moons.fit(X_train_m_scaled, y_train_m, epochs=150, batch_size=32, verbose=True)

    # Evaluation
    train_eval_m = nn_moons.evaluate(X_train_m_scaled, y_train_m)
    test_eval_m = nn_moons.evaluate(X_test_m_scaled, y_test_m)
    print(f"\n  [Moons Results]")
    print(f"  Train Accuracy: {train_eval_m['accuracy'] * 100:.2f}% | Train Loss: {train_eval_m['loss']:.4f}")
    print(f"  Test Accuracy : {test_eval_m['accuracy'] * 100:.2f}% | Test Loss : {test_eval_m['loss']:.4f}")

    # Plot training loss & decision boundary
    visualizer.plot_training_loss(
        nn_moons.loss_history,
        title="Training Loss: Make Moons (Adam + L2 + Dropout)",
        filename="moons_training_loss.png"
    )

    # For decision boundary, wrap model to operate on unscaled space or pass scaled
    class ScaledModelWrapper:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler

        def predict_proba(self, X):
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)

    wrapped_model = ScaledModelWrapper(nn_moons, scaler_m)
    visualizer.plot_decision_boundary(
        wrapped_model,
        X_test_m,
        y_test_m,
        title="Learned Non-Linear Decision Boundary (Test Set)",
        filename="moons_decision_boundary.png"
    )

    # -----------------------------------------------------------------
    # Step 4: Multi-Class Classification on Scikit-Learn Digits
    # -----------------------------------------------------------------
    print("\n[Step 4] Training Neural Network on Multi-Class 'Load Digits' Dataset (10 classes)...")
    digits = load_digits()
    X_digits = digits.data
    y_digits = digits.target

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_digits, y_digits, test_size=0.2, random_state=42, stratify=y_digits
    )

    scaler_d = StandardScaler()
    X_train_d_scaled = scaler_d.fit_transform(X_train_d)
    X_test_d_scaled = scaler_d.transform(X_test_d)

    # Architecture: 64 -> 32 -> 16 -> 10
    nn_digits = NeuralNetwork(
        layer_sizes=[64, 32, 16, 10],
        activations=["relu", "relu", "softmax"],
        loss="cce",
        optimizer="adam",
        learning_rate=0.005,
        l2_lambda=0.0005,
        dropout_rate=0.1,
        seed=42
    )

    print("  Architecture: Input(64) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(10, Softmax)")
    print("  Optimizer: Adam (lr=0.005) | Regularization: L2 (0.0005) + Dropout (0.1)")
    nn_digits.fit(X_train_d_scaled, y_train_d, epochs=120, batch_size=32, verbose=True)

    train_eval_d = nn_digits.evaluate(X_train_d_scaled, y_train_d)
    test_eval_d = nn_digits.evaluate(X_test_d_scaled, y_test_d)
    print(f"\n  [Digits Results]")
    print(f"  Train Accuracy: {train_eval_d['accuracy'] * 100:.2f}% | Train Loss: {train_eval_d['loss']:.4f}")
    print(f"  Test Accuracy : {test_eval_d['accuracy'] * 100:.2f}% | Test Loss : {test_eval_d['loss']:.4f}")

    visualizer.plot_training_loss(
        nn_digits.loss_history,
        title="Training Loss: Digits Classification (Adam + Softmax)",
        filename="digits_training_loss.png"
    )

    # -----------------------------------------------------------------
    # Step 5: Optimizer Benchmark Comparison (SGD, Momentum, RMSprop, Adam)
    # -----------------------------------------------------------------
    print("\n[Step 5] Benchmarking Optimizers (SGD vs Momentum vs RMSprop vs Adam)...")
    optimizer_configs = [
        ("SGD", "sgd", 0.05),
        ("Momentum", "momentum", 0.03),
        ("RMSprop", "rmsprop", 0.005),
        ("Adam", "adam", 0.01)
    ]

    histories = {}
    for display_name, opt_name, lr in optimizer_configs:
        print(f"  Training with optimizer: {display_name} (lr={lr})...")
        net = NeuralNetwork(
            layer_sizes=[2, 16, 8, 1],
            activations=["relu", "relu", "sigmoid"],
            loss="bce",
            optimizer=opt_name,
            learning_rate=lr,
            l2_lambda=0.0,
            dropout_rate=0.0,
            seed=42
        )
        net.fit(X_train_m_scaled, y_train_m, epochs=100, batch_size=32, verbose=False)
        histories[display_name] = net.loss_history

    visualizer.plot_optimizer_comparison(histories, filename="optimizer_comparison.png")

    # -----------------------------------------------------------------
    # Final Summary Report
    # -----------------------------------------------------------------
    print("\n" + "=" * 64)
    print(" DAY 1-2 NEURAL NETWORK DEMONSTRATION SUMMARY")
    print("=" * 64)
    print(f"1. Make Moons (Binary Non-Linear Classification):")
    print(f"   - Architecture: [2, 16, 8, 1]")
    print(f"   - Test Accuracy: {test_eval_m['accuracy'] * 100:.2f}%")
    print(f"   - Test Loss:     {test_eval_m['loss']:.4f}")
    print(f"\n2. Hand-written Digits (Multi-Class 10-way Classification):")
    print(f"   - Architecture: [64, 32, 16, 10]")
    print(f"   - Test Accuracy: {test_eval_d['accuracy'] * 100:.2f}%")
    print(f"   - Test Loss:     {test_eval_d['loss']:.4f}")
    print(f"\n3. Visualizations Generated in '{output_dir}/':")
    print("   - activation_functions.png    (Sigmoid, Tanh, ReLU, Leaky ReLU & derivatives)")
    print("   - loss_functions.png          (MSE vs BCE landscapes)")
    print("   - moons_training_loss.png     (Binary convergence curve)")
    print("   - moons_decision_boundary.png (Non-linear decision contour)")
    print("   - digits_training_loss.png    (Multiclass convergence curve)")
    print("   - optimizer_comparison.png    (SGD vs Momentum vs RMSprop vs Adam)")
    print("================================================================")
    print(" DAY 1-2 COMPLETED SUCCESSFULLY!")
    print("================================================================")


if __name__ == "__main__":
    main()
