"""
Loss Functions and Optimization Algorithms for Neural Networks.
Pure NumPy implementations:
- Loss Functions: MSE, Binary Cross-Entropy, Categorical Cross-Entropy
- Optimizers: SGD, Momentum, RMSprop, Adam
"""

import numpy as np


class LossFunctions:
    """
    Implements standard neural network loss functions and their derivatives.
    """

    @staticmethod
    def mse(y_true, y_pred):
        """
        Mean Squared Error (MSE) loss:
        L = (1 / N) * sum((y_pred - y_true)^2)
        """
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def mse_derivative(y_true, y_pred):
        """
        Derivative of MSE with respect to y_pred:
        dL / dy_pred = (2 / N) * (y_pred - y_true)
        """
        n_samples = y_pred.shape[0] if y_pred.ndim > 0 else 1
        return (2.0 / n_samples) * (y_pred - y_true)

    @staticmethod
    def binary_cross_entropy(y_true, y_pred, eps=1e-15):
        """
        Binary Cross-Entropy (Log Loss):
        L = - (1 / N) * sum(y * log(p) + (1 - y) * log(1 - p))
        """
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
        loss = -(y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped))
        return np.mean(loss)

    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred, eps=1e-15):
        """
        Derivative of Binary Cross-Entropy with respect to y_pred:
        dL / dy_pred = (1 / N) * ((y_pred - y_true) / (y_pred * (1 - y_pred)))
        """
        n_samples = y_pred.shape[0] if y_pred.ndim > 0 else 1
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
        grad = (y_pred_clipped - y_true) / (y_pred_clipped * (1.0 - y_pred_clipped))
        return grad / n_samples

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred, eps=1e-15):
        """
        Categorical Cross-Entropy loss for multi-class classification:
        L = - (1 / N) * sum_i sum_k y_ik * log(p_ik)
        Supports both 1D integer class labels and 2D one-hot encoded targets.
        """
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
        n_samples = y_pred.shape[0]

        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            # Target as integer indices
            y_indices = y_true.flatten().astype(int)
            log_likelihood = -np.log(y_pred_clipped[np.arange(n_samples), y_indices])
            return np.mean(log_likelihood)
        else:
            # Target as one-hot encoded matrix
            return -np.sum(y_true * np.log(y_pred_clipped)) / n_samples

    @staticmethod
    def categorical_cross_entropy_derivative(y_true, y_pred, eps=1e-15):
        """
        Derivative of Categorical Cross-Entropy with respect to y_pred:
        dL / dy_pred = - (1 / N) * (y_true / y_pred)
        """
        n_samples = y_pred.shape[0]
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)

        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            y_one_hot = np.zeros_like(y_pred)
            y_one_hot[np.arange(n_samples), y_true.flatten().astype(int)] = 1.0
            return -(y_one_hot / y_pred_clipped) / n_samples
        else:
            return -(y_true / y_pred_clipped) / n_samples

    @classmethod
    def get(cls, name):
        """Factory method returning (loss_fn, grad_fn) tuple."""
        name_lower = name.lower()
        if name_lower in ("mse", "mean_squared_error"):
            return cls.mse, cls.mse_derivative
        elif name_lower in ("bce", "binary_cross_entropy", "log_loss"):
            return cls.binary_cross_entropy, cls.binary_cross_entropy_derivative
        elif name_lower in ("cce", "categorical_cross_entropy", "cross_entropy"):
            return cls.categorical_cross_entropy, cls.categorical_cross_entropy_derivative
        else:
            raise ValueError(f"Unsupported loss function: '{name}'")


# =====================================================================
# OPTIMIZERS
# =====================================================================

class BaseOptimizer:
    """Base class for all first-order optimizers."""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        raise NotImplementedError("Subclasses must implement update().")


class SGDOptimizer(BaseOptimizer):
    """
    Standard Stochastic Gradient Descent (SGD):
    W = W - lr * dW
    """
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def update(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad


class MomentumOptimizer(BaseOptimizer):
    """
    SGD with Momentum:
    v = beta * v + (1 - beta) * grad
    param = param - lr * v
    """
    def __init__(self, learning_rate=0.01, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.velocities = None

    def update(self, params, grads):
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.velocities[i] = self.beta * self.velocities[i] + (1.0 - self.beta) * grad
            param -= self.learning_rate * self.velocities[i]


class RMSpropOptimizer(BaseOptimizer):
    """
    RMSprop (Root Mean Square Propagation):
    s = beta * s + (1 - beta) * (grad ^ 2)
    param = param - (lr / (sqrt(s) + eps)) * grad
    """
    def __init__(self, learning_rate=0.001, beta=0.9, eps=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.eps = eps
        self.squared_avg = None

    def update(self, params, grads):
        if self.squared_avg is None:
            self.squared_avg = [np.zeros_like(p) for p in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.squared_avg[i] = self.beta * self.squared_avg[i] + (1.0 - self.beta) * (grad ** 2)
            param -= (self.learning_rate / (np.sqrt(self.squared_avg[i]) + self.eps)) * grad


class AdamOptimizer(BaseOptimizer):
    """
    Adam (Adaptive Moment Estimation):
    Combines momentum (1st moment) and RMSprop (2nd moment) with bias correction:
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ^ 2)
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    param = param - (lr / (sqrt(v_hat) + eps)) * m_hat
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update biased 1st and 2nd moment estimates
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)

            # Compute bias-corrected estimates
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            # Parameter update
            param -= (self.learning_rate / (np.sqrt(v_hat) + self.eps)) * m_hat


class Optimizers:
    """Factory helper to instantiate optimizer by name."""
    @staticmethod
    def get(name, learning_rate=0.01, **kwargs):
        name_lower = name.lower()
        if name_lower == "sgd":
            return SGDOptimizer(learning_rate=learning_rate)
        elif name_lower == "momentum":
            return MomentumOptimizer(learning_rate=learning_rate, **kwargs)
        elif name_lower == "rmsprop":
            return RMSpropOptimizer(learning_rate=learning_rate, **kwargs)
        elif name_lower == "adam":
            return AdamOptimizer(learning_rate=learning_rate, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer: '{name}'")
