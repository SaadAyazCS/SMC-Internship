"""
Activation Functions and Derivatives for Neural Networks.
Pure NumPy implementation supporting:
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- Softmax
"""

import numpy as np


class ActivationFunctions:
    """
    Implements standard neural network activation functions and their derivatives.
    All functions support vectorized operations on NumPy arrays.
    """

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation: 1 / (1 + exp(-x)).
        Uses clipping to prevent overflow in exp(-x).
        """
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    @staticmethod
    def sigmoid_derivative(x, is_activated=False):
        """
        Derivative of sigmoid: s * (1 - s).
        If is_activated=True, x is assumed to already be sigmoid(z).
        """
        s = x if is_activated else ActivationFunctions.sigmoid(x)
        return s * (1.0 - s)

    @staticmethod
    def tanh(x):
        """Hyperbolic tangent activation: tanh(x)."""
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x, is_activated=False):
        """
        Derivative of tanh: 1 - tanh^2(x).
        If is_activated=True, x is assumed to already be tanh(z).
        """
        t = x if is_activated else np.tanh(x)
        return 1.0 - t ** 2

    @staticmethod
    def relu(x):
        """Rectified Linear Unit: max(0, x)."""
        return np.maximum(0.0, x)

    @staticmethod
    def relu_derivative(x, is_activated=False):
        """
        Derivative of ReLU: 1 if x > 0 else 0.
        If is_activated=True, uses x > 0.
        """
        return np.where(x > 0.0, 1.0, 0.0)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """Leaky ReLU: x if x > 0 else alpha * x."""
        return np.where(x > 0.0, x, alpha * x)

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01, is_activated=False):
        """Derivative of Leaky ReLU: 1 if x > 0 else alpha."""
        return np.where(x > 0.0, 1.0, alpha)

    @staticmethod
    def softmax(x):
        """
        Numerically stable Softmax activation along the last axis.
        Subtracts max value before exponentiation to prevent overflow.
        """
        # Subtract max for numerical stability along last axis
        exp_shifted = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)

    @staticmethod
    def softmax_derivative(a):
        """
        Jacobian matrix derivative of softmax for a single sample or batch.
        Note: When coupled with Categorical Cross-Entropy, the combined gradient
        simplifies to (y_pred - y_true), which is computed directly in the loss.
        """
        # For diagonal-dominant vectorized representation
        return a * (1.0 - a)

    @classmethod
    def get(cls, name):
        """
        Factory method returning (activation_fn, derivative_fn) tuple.
        """
        name_lower = name.lower()
        if name_lower == "sigmoid":
            return cls.sigmoid, cls.sigmoid_derivative
        elif name_lower == "tanh":
            return cls.tanh, cls.tanh_derivative
        elif name_lower == "relu":
            return cls.relu, cls.relu_derivative
        elif name_lower in ("leaky_relu", "leakyrelu"):
            return cls.leaky_relu, cls.leaky_relu_derivative
        elif name_lower == "softmax":
            return cls.softmax, cls.softmax_derivative
        elif name_lower in ("linear", "none", "identity"):
            return (lambda x: x), (lambda x, is_activated=False: np.ones_like(x))
        else:
            raise ValueError(f"Unsupported activation function: '{name}'")
