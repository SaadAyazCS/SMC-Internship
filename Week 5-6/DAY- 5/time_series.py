"""
Time Series Prediction with LSTM, GRU, and Vanilla RNN.
Uses synthetic sine wave + noise dataset.
Compares RNN vs LSTM vs GRU on multi-step forecasting.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from rnn_basics import VanillaRNN, LSTMModel, GRUModel


# ===========================================================================
# Dataset
# ===========================================================================

def generate_sine_data(n_samples=2000, seq_len=50, pred_steps=10, noise_std=0.05, seed=42):
    """
    Generates a sine wave + noise and creates sliding window sequences.
    Returns X (n, seq_len, 1) and y (n, pred_steps) arrays.
    """
    np.random.seed(seed)
    t = np.linspace(0, 8 * np.pi, n_samples + seq_len + pred_steps)
    signal = np.sin(t) + 0.3 * np.sin(3 * t) + noise_std * np.random.randn(len(t))

    X, y = [], []
    for i in range(n_samples):
        X.append(signal[i: i + seq_len])
        y.append(signal[i + seq_len: i + seq_len + pred_steps])

    X = np.array(X, dtype=np.float32)[..., np.newaxis]  # (n, seq_len, 1)
    y = np.array(y, dtype=np.float32)                    # (n, pred_steps)
    return X, y, signal


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_ts_loaders(X, y, split=0.8, batch_size=64):
    n_train = int(len(X) * split)
    train_ds = TimeSeriesDataset(X[:n_train], y[:n_train])
    test_ds  = TimeSeriesDataset(X[n_train:], y[n_train:])
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False))


# ===========================================================================
# Training
# ===========================================================================

def train_ts_model(model, train_loader, test_loader, epochs, lr, device, name):
    """Trains a time series forecasting model and returns history."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    print(f"\n  Training {name}...")
    print(f"  {'Epoch':>5} | {'Train MSE':>10} | {'Val MSE':>10}")
    print("  " + "-" * 32)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(X_b)
            loss  = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        tr_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b)
                val_loss += criterion(preds, y_b).item()
        val_loss /= len(test_loader)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"  {epoch:>5} | {tr_loss:>10.6f} | {val_loss:>10.6f}")

    return history


def run_time_series(device, epochs=40):
    """
    Compares Vanilla RNN, LSTM, and GRU on sine wave forecasting.
    Returns histories, predictions, and ground truth.
    """
    print("\n  Generating sine wave dataset...")
    X, y, full_signal = generate_sine_data(n_samples=2000, seq_len=50, pred_steps=10)
    train_loader, test_loader = get_ts_loaders(X, y, batch_size=64)
    print(f"  Dataset -- Train: {int(len(X)*0.8):,} | Test: {int(len(X)*0.2):,} sequences")

    models_cfg = [
        ("Vanilla RNN", VanillaRNN(input_size=1, hidden_size=64, num_layers=2, output_size=10)),
        ("LSTM",        LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_size=10, dropout=0.1)),
        ("GRU",         GRUModel(input_size=1, hidden_size=64, num_layers=2, output_size=10, dropout=0.1)),
    ]

    histories  = {}
    test_losses = {}
    sample_preds = {}

    # Get a sample batch for visualization
    sample_X, sample_y = next(iter(test_loader))

    for name, model in models_cfg:
        model = model.to(device)
        hist = train_ts_model(model, train_loader, test_loader, epochs, lr=0.001, device=device, name=name)
        histories[name]   = hist
        test_losses[name] = hist["val_loss"][-1]

        # Sample predictions for plotting
        model.eval()
        with torch.no_grad():
            preds = model(sample_X.to(device)).cpu().numpy()
        sample_preds[name] = preds

        print(f"  [OK] {name} Final Test MSE: {hist['val_loss'][-1]:.6f}")

    print("\n  Model Comparison (Test MSE):")
    for name, mse in test_losses.items():
        print(f"    {name:<15}: {mse:.6f}")

    return histories, sample_preds, sample_y.numpy(), test_losses
