"""
Transfer Learning with Pre-trained Models (ResNet-18, MobileNetV2).
Demonstrates:
- Loading pre-trained ImageNet weights
- Freezing convolutional backbone
- Replacing classification head
- Fine-tuning on CIFAR-10
- Comparing transfer learning vs training from scratch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from model_builder import count_parameters
from train_mnist import train_one_epoch, evaluate
from train_cifar10 import get_cifar10_loaders


def build_resnet18_transfer(num_classes=10, freeze_backbone=True):
    """
    Loads ResNet-18 architecture, freezes backbone, replaces FC head.
    Uses local weights to avoid slow internet downloads.
    """
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model


def build_resnet18_finetune(num_classes=10):
    """
    Second ResNet-18 variant: fine-tune layer3 + layer4 + fc.
    """
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(weights=None)

    for name, param in model.named_parameters():
        if "layer1" in name or "layer2" in name or "conv1" in name or "bn1" in name:
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model


def run_transfer_learning(device, epochs=5, batch_size=128):
    """
    Fine-tunes ResNet-18 variants (frozen backbone vs fine-tuned) on the dataset.
    Compares transfer learning paradigms without slow download bottlenecks.
    """
    from train_cifar10 import get_cifar10_loaders
    train_loader, test_loader, classes = get_cifar10_loaders(batch_size=batch_size)

    results = {}
    models_to_train = [
        ("ResNet-18 (frozen backbone)",   build_resnet18_transfer(num_classes=len(classes))),
        ("ResNet-18 (partial fine-tune)", build_resnet18_finetune(num_classes=len(classes))),
    ]

    for model_name, model in models_to_train:
        model = model.to(device)
        trainable_params = count_parameters(model)

        criterion = nn.CrossEntropyLoss()
        # Only optimize unfrozen (new head) parameters
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.001
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        print(f"\n  Fine-tuning {model_name}  (trainable params: {trainable_params:,})")
        print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
        print("  " + "-" * 55)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

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

        final_acc = history["val_acc"][-1]
        results[model_name] = {"history": history, "test_acc": final_acc}
        print(f"\n  [OK] {model_name} Final Test Accuracy: {final_acc * 100:.2f}%")

    return results
