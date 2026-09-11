"""
Neural Network Built from Scratch in Pure NumPy.
Features:
- Configurable layer architecture
- Xavier / He weight initialization
- Manual forward and backward propagation (chain rule)
- Multiple activation functions per layer
- L2 Regularization and Inverted Dropout
- Pluggable optimizers (SGD, Momentum, RMSprop, Adam)
- Mini-batch gradient descent training loop
"""

import numpy as np
from activations import ActivationFunctions
from loss_functions import LossFunctions, Optimizers


class NeuralNetwork:
    """
    Multilayer Perceptron (MLP) implemented from first principles in NumPy.
    Supports binary, multiclass classification, and regression.
    """

    def __init__(
        self,
        layer_sizes,
        activations=None,
        loss="bce",
        optimizer="adam",
        learning_rate=0.01,
        l2_lambda=0.0,
        dropout_rate=0.0,
        seed=42
    ):
        """
        Parameters:
        -----------
        layer_sizes : list of int
            List specifying number of neurons in each layer [n_in, h1, h2, ..., n_out].
        activations : list of str, or single str
            Activation function name for each layer transition.
            Defaults: 'relu' for hidden layers, 'sigmoid' (if n_out==1) or 'softmax' (if n_out>1).
        loss : str
            Loss function name ('bce', 'cce', 'mse').
        optimizer : str
            Optimizer algorithm ('sgd', 'momentum', 'rmsprop', 'adam').
        learning_rate : float
            Learning rate step size for optimization.
        l2_lambda : float
            L2 regularization strength (weight decay).
        dropout_rate : float
            Dropout probability for hidden layers (0.0 means no dropout).
        seed : int
            Random seed for reproducible weight initialization.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.loss_name = loss.lower()
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.rng = np.random.RandomState(seed)

        # Set default activations if not fully specified
        if activations is None:
            self.activation_names = []
            for i in range(self.num_layers - 1):
                self.activation_names.append("relu")
            # Output layer default
            if self.layer_sizes[-1] == 1:
                self.activation_names.append("sigmoid" if "bce" in self.loss_name else "linear")
            else:
                self.activation_names.append("softmax" if "cce" in self.loss_name else "linear")
        elif isinstance(activations, str):
            self.activation_names = [activations] * (self.num_layers - 1)
            self.activation_names.append("sigmoid" if self.layer_sizes[-1] == 1 else "softmax")
        else:
            self.activation_names = list(activations)

        # Load activation callables
        self.act_fns = []
        self.act_derivs = []
        for name in self.activation_names:
            fn, deriv = ActivationFunctions.get(name)
            self.act_fns.append(fn)
            self.act_derivs.append(deriv)

        # Loss function
        self.loss_fn, self.loss_grad_fn = LossFunctions.get(self.loss_name)

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self._initialize_parameters()

        # Build optimizer
        self.optimizer = Optimizers.get(optimizer, learning_rate=self.learning_rate)

        # Training history
        self.loss_history = []

    def _initialize_parameters(self):
        """
        Initializes weights using He or Xavier initialization depending on activation,
        and sets biases to zero vectors.
        """
        for l in range(self.num_layers):
            fan_in = self.layer_sizes[l]
            fan_out = self.layer_sizes[l + 1]
            act = self.activation_names[l].lower()

            if act in ("relu", "leaky_relu", "leakyrelu"):
                # He initialization: std = sqrt(2 / fan_in)
                std = np.sqrt(2.0 / fan_in)
                W = self.rng.randn(fan_in, fan_out) * std
            else:
                # Xavier / Glorot initialization: std = sqrt(2 / (fan_in + fan_out))
                std = np.sqrt(2.0 / (fan_in + fan_out))
                W = self.rng.randn(fan_in, fan_out) * std

            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

    def _get_param_lists(self):
        """Flatten weights and biases into a paired list for optimizer updates."""
        params = []
        for l in range(self.num_layers):
            params.append(self.weights[l])
            params.append(self.biases[l])
        return params

    def forward(self, X, training=True):
        """
        Forward propagation through all layers.
        Caches Z, A, and dropout masks for backpropagation.
        """
        self.A_cache = [X]
        self.Z_cache = []
        self.dropout_masks = []

        A = X
        for l in range(self.num_layers):
            Z = np.dot(A, self.weights[l]) + self.biases[l]
            self.Z_cache.append(Z)

            A = self.act_fns[l](Z)

            # Inverted dropout on hidden layers during training
            if training and self.dropout_rate > 0.0 and l < self.num_layers - 1:
                keep_prob = 1.0 - self.dropout_rate
                mask = (self.rng.rand(*A.shape) < keep_prob).astype(float) / keep_prob
                A = A * mask
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(None)

            self.A_cache.append(A)

        return A

    def backward(self, y_true):
        """
        Backward propagation (manual chain rule).
        Computes gradients for all weights and biases.
        """
        N = y_true.shape[0]
        grads = []
        dW_list = [None] * self.num_layers
        db_list = [None] * self.num_layers

        output_act = self.activation_names[-1].lower()
        A_out = self.A_cache[-1]

        # Calculate error gradient at output layer (dZ for final layer)
        if output_act == "sigmoid" and self.loss_name in ("bce", "binary_cross_entropy", "log_loss"):
            # Simplified stable gradient for Sigmoid + Binary Cross-Entropy
            y_target = y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true
            dZ = (A_out - y_target) / N
        elif output_act == "softmax" and self.loss_name in ("cce", "categorical_cross_entropy", "cross_entropy"):
            # Simplified stable gradient for Softmax + Categorical Cross-Entropy
            if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
                y_one_hot = np.zeros_like(A_out)
                y_one_hot[np.arange(N), y_true.flatten().astype(int)] = 1.0
            else:
                y_one_hot = y_true
            dZ = (A_out - y_one_hot) / N
        elif self.loss_name in ("mse", "mean_squared_error"):
            y_target = y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true
            dA = (2.0 / N) * (A_out - y_target)
            dZ = dA * self.act_derivs[-1](self.Z_cache[-1])
        else:
            # General fallback using explicit chain rule
            dA = self.loss_grad_fn(y_true, A_out)
            dZ = dA * self.act_derivs[-1](self.Z_cache[-1])

        # Backpropagation loop from output to input
        for l in reversed(range(self.num_layers)):
            A_prev = self.A_cache[l]
            # dW = A_prev^T . dZ + (l2_lambda / N) * W
            dW = np.dot(A_prev.T, dZ) + (self.l2_lambda / N) * self.weights[l]
            db = np.sum(dZ, axis=0, keepdims=True)

            dW_list[l] = dW
            db_list[l] = db

            if l > 0:
                dA_prev = np.dot(dZ, self.weights[l].T)

                # Backpropagate through dropout mask if applied
                if self.dropout_masks[l - 1] is not None:
                    dA_prev = dA_prev * self.dropout_masks[l - 1]

                # dZ_prev = dA_prev * g'(Z_prev)
                dZ = dA_prev * self.act_derivs[l - 1](self.Z_cache[l - 1])

        # Interleave dW and db into flattened list matching _get_param_lists
        for l in range(self.num_layers):
            grads.append(dW_list[l])
            grads.append(db_list[l])

        return grads

    def compute_total_loss(self, y_true, y_pred):
        """Computes data loss plus L2 regularization term."""
        base_loss = self.loss_fn(y_true, y_pred)
        if self.l2_lambda > 0.0:
            N = y_true.shape[0]
            l2_penalty = 0.5 * (self.l2_lambda / N) * sum(np.sum(W ** 2) for W in self.weights)
            return base_loss + l2_penalty
        return base_loss

    def fit(self, X, y, epochs=100, batch_size=32, verbose=True):
        """
        Trains the neural network using mini-batch gradient descent.
        """
        n_samples = X.shape[0]
        y_train = y.copy()

        # Reshape binary target if needed
        if self.layer_sizes[-1] == 1 and y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        self.loss_history = []
        batch_size = min(batch_size, n_samples)

        for epoch in range(1, epochs + 1):
            # Shuffle indices
            indices = self.rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_train[indices]

            # Mini-batch iteration
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass
                self.forward(X_batch, training=True)

                # Backward pass
                grads = self.backward(y_batch)

                # Update parameters via optimizer
                params = self._get_param_lists()
                self.optimizer.update(params, grads)

            # Compute epoch loss across full dataset without dropout
            y_pred_full = self.forward(X, training=False)
            epoch_loss = self.compute_total_loss(y_train, y_pred_full)
            self.loss_history.append(epoch_loss)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs):
                print(f"  Epoch [{epoch:4d}/{epochs:4d}] - Loss: {epoch_loss:.6f}")

        return self

    def predict_proba(self, X):
        """Returns predicted class probabilities (deterministic, no dropout)."""
        return self.forward(X, training=False)

    def predict(self, X):
        """Returns predicted class labels or continuous values."""
        probas = self.predict_proba(X)
        if self.layer_sizes[-1] == 1:
            if "bce" in self.loss_name or self.activation_names[-1].lower() == "sigmoid":
                return (probas >= 0.5).astype(int).flatten()
            return probas.flatten()
        else:
            return np.argmax(probas, axis=1)

    def evaluate(self, X, y):
        """
        Evaluates model performance on test/validation set.
        Returns dictionary with accuracy (classification) or MSE (regression) and loss.
        """
        y_pred = self.predict(X)
        probas = self.predict_proba(X)

        y_eval = y.reshape(-1, 1) if (self.layer_sizes[-1] == 1 and y.ndim == 1) else y
        loss = self.compute_total_loss(y_eval, probas)

        metrics = {"loss": float(loss)}
        if "mse" not in self.loss_name:
            y_true_labels = y.flatten().astype(int)
            accuracy = np.mean(y_pred == y_true_labels)
            metrics["accuracy"] = float(accuracy)

        return metrics
