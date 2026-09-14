"""
Training utilities for MNIST handwritten digit classification.
Trains both an MLP and a CNN and returns training histories.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_builder import SequentialMLP, FunctionalMLP, BasicCNN, count_parameters


def get_mnist_loaders(batch_size=128, data_dir="./data"):
    """Downloads and returns MNIST train and test DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))   # MNIST mean/std
    ])
    try:
        train_ds = datasets.MNIST(data_dir, train=True,  download=False, transform=transform)
        test_ds  = datasets.MNIST(data_dir, train=False, download=False, transform=transform)
    except Exception:
        train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
        test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"  MNIST -- Train: {len(train_ds):,} | Test: {len(test_ds):,}")
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Runs one training epoch and returns avg loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += X.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluates model on loader and returns avg loss and accuracy."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += X.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def train_model(model, train_loader, test_loader, epochs, lr, device, model_name):
    """Full training loop with per-epoch reporting."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n  Training {model_name}  ({count_parameters(model):,} params)")
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("  " + "-" * 55)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"  {epoch:>5} | {tr_loss:>10.4f} | {tr_acc*100:>8.2f}% | "
                  f"{vl_loss:>8.4f} | {vl_acc*100:>6.2f}%")

    return history


def run_mnist_training(device, epochs_mlp=10, epochs_cnn=10, batch_size=128):
    """
    Trains MLP and CNN on MNIST. Returns histories and predictions for visualization.
    """
    print("\n  Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    results = {}

    # ------------------------------------------------------------------
    # Model 1: Sequential MLP
    # ------------------------------------------------------------------
    print("\n  [Model 1] Sequential MLP")
    mlp_seq = SequentialMLP(input_size=784, hidden_sizes=[256, 128], output_size=10, dropout_rate=0.3)
    mlp_seq = mlp_seq.to(device)

    # Flatten images for MLP
    class FlattenLoader:
        def __init__(self, loader):
            self.loader = loader
        def __iter__(self):
            for X, y in self.loader:
                yield X.view(X.size(0), -1), y
        def __len__(self):
            return len(self.loader)

    flat_train = FlattenLoader(train_loader)
    flat_test  = FlattenLoader(test_loader)

    hist_mlp_seq = train_model(mlp_seq, flat_train, flat_test, epochs_mlp, lr=0.001, device=device, model_name="Sequential MLP")
    _, test_acc_mlp_seq, preds_mlp, labels = evaluate(mlp_seq, flat_test, nn.CrossEntropyLoss(), device)
    results["Sequential MLP"] = {"history": hist_mlp_seq, "test_acc": test_acc_mlp_seq, "preds": preds_mlp, "labels": labels}
    print(f"\n  [OK] Sequential MLP Final Test Accuracy: {test_acc_mlp_seq * 100:.2f}%")

    # ------------------------------------------------------------------
    # Model 2: Functional MLP
    # ------------------------------------------------------------------
    print("\n  [Model 2] Functional API MLP")
    mlp_fn = FunctionalMLP(input_size=784, hidden1=256, hidden2=128, output_size=10, dropout_rate=0.3)
    mlp_fn = mlp_fn.to(device)
    hist_mlp_fn = train_model(mlp_fn, flat_train, flat_test, epochs_mlp, lr=0.001, device=device, model_name="Functional MLP")
    _, test_acc_mlp_fn, _, _ = evaluate(mlp_fn, flat_test, nn.CrossEntropyLoss(), device)
    results["Functional MLP"] = {"history": hist_mlp_fn, "test_acc": test_acc_mlp_fn}
    print(f"\n  [OK] Functional MLP Final Test Accuracy: {test_acc_mlp_fn * 100:.2f}%")

    # ------------------------------------------------------------------
    # Model 3: Basic CNN
    # ------------------------------------------------------------------
    print("\n  [Model 3] Basic CNN")
    cnn = BasicCNN(num_classes=10).to(device)
    hist_cnn = train_model(cnn, train_loader, test_loader, epochs_cnn, lr=0.001, device=device, model_name="Basic CNN")
    _, test_acc_cnn, preds_cnn, labels_cnn = evaluate(cnn, test_loader, nn.CrossEntropyLoss(), device)
    results["Basic CNN"] = {"history": hist_cnn, "test_acc": test_acc_cnn, "preds": preds_cnn, "labels": labels_cnn}
    print(f"\n  [OK] CNN Final Test Accuracy: {test_acc_cnn * 100:.2f}%")

    return results
