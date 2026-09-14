"""
Model Builder: Sequential and Functional API styles in PyTorch.
Covers:
- Sequential MLP (analogous to Keras Sequential)
- Functional MLP using nn.Module subclass (analogous to Keras Functional API)
- Basic CNN for image classification
- Advanced CNN with BatchNorm for CIFAR-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Sequential-style MLP (analogous to Keras Sequential)
# ===========================================================================

class SequentialMLP(nn.Module):
    """
    Multi-Layer Perceptron built with nn.Sequential.
    Equivalent to Keras: model = Sequential([Dense(...), ReLU(), Dense(...)])
    """

    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            ])
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.network(x)


# ===========================================================================
# Functional-style MLP (analogous to Keras Functional API / nn.Module subclass)
# ===========================================================================

class FunctionalMLP(nn.Module):
    """
    Multi-Layer Perceptron built as an nn.Module subclass with explicit forward().
    Equivalent to Keras Functional API -- allows skip connections, multiple heads, etc.
    """

    def __init__(self, input_size, hidden1, hidden2, output_size, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.bn1  = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2  = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        # Layer 1
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        # Layer 2
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        # Output (logits -- no softmax; handled by CrossEntropyLoss)
        return self.fc3(x)


# ===========================================================================
# Basic CNN for MNIST (1-channel 28x28 images)
# ===========================================================================

class BasicCNN(nn.Module):
    """
    Simple CNN: Conv -> Pool -> Conv -> Pool -> FC -> FC.
    Designed for grayscale images (e.g., MNIST 28x28).
    """

    def __init__(self, num_classes=10):
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (1,28,28) -> (32,28,28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # (32,28,28) -> (32,14,14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (32,14,14) -> (64,14,14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # (64,14,14) -> (64,7,7)
        )
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ===========================================================================
# Advanced CNN for CIFAR-10 (3-channel 32x32 images)
# ===========================================================================

class AdvancedCNN(nn.Module):
    """
    Deeper CNN with BatchNorm for CIFAR-10 RGB 32x32 images.
    Architecture inspired by VGG-style blocks.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),      # 32 -> 16
            nn.Dropout2d(0.1),
        )
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),      # 16 -> 8
            nn.Dropout2d(0.1),
        )
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),      # 8 -> 4
            nn.Dropout2d(0.1),
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# ===========================================================================
# Utilities
# ===========================================================================

def count_parameters(model):
    """Returns the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, name, input_shape):
    """Prints model architecture with parameter count."""
    total_params = count_parameters(model)
    print(f"\n  [{name}]")
    print(f"  Architecture:")
    for layer_name, module in model.named_children():
        print(f"    {layer_name}: {module.__class__.__name__}")
    print(f"  Total trainable parameters: {total_params:,}")

    # Quick forward pass to check shapes
    model.eval()
    dummy = torch.zeros(2, *input_shape)
    try:
        with torch.no_grad():
            out = model(dummy)
        print(f"  Input shape:  {tuple(dummy.shape)}")
        print(f"  Output shape: {tuple(out.shape)}")
    except Exception as e:
        print(f"  Forward pass failed: {e}")
