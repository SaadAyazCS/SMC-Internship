"""
WEEK 5-6: DEEP LEARNING & NEURAL NETWORKS
DAY 3-4: TensorFlow/Keras Basics & Convolutional Neural Networks (CNNs)
(Implemented in PyTorch -- TensorFlow unsupported on Python 3.14)

Demonstration Pipeline:
1. PyTorch Tensor Basics & Autograd
2. Model Architecture: Sequential vs Functional API
3. MNIST Classification: MLP vs CNN comparison
4. CIFAR-10 Classification: Advanced CNN with data augmentation
5. Transfer Learning: ResNet-18 & MobileNetV2 fine-tuning
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import numpy as np
import torch

from tensor_basics import run_tensor_basics
from model_builder import (
    SequentialMLP, FunctionalMLP, BasicCNN, AdvancedCNN,
    print_model_summary, count_parameters
)
from train_mnist import run_mnist_training
from train_cifar10 import run_cifar10_training
from transfer_learning import run_transfer_learning
from visualization import DLVisualizer


def main():
    print("================================================================")
    print(" WEEK 5-6 - DAY 3-4: PYTORCH BASICS & CNNs")
    print("================================================================")

    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Using device: {device}")

    viz = DLVisualizer(output_dir="results")

    # ----------------------------------------------------------------
    # Step 1: PyTorch Tensor Basics & Autograd
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Step 1] PyTorch Tensor Basics & Autograd Demonstration")
    print("=" * 60)
    run_tensor_basics()

    # ----------------------------------------------------------------
    # Step 2: Model Architecture Overview
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Step 2] Model Architecture: Sequential vs Functional API")
    print("=" * 60)

    models_to_show = [
        ("Sequential MLP (784->256->128->10)",  SequentialMLP(784, [256, 128], 10),        (784,)),
        ("Functional MLP (784->256->128->10)",  FunctionalMLP(784, 256, 128, 10),          (784,)),
        ("Basic CNN (MNIST 1x28x28)",            BasicCNN(num_classes=10),                  (1, 28, 28)),
        ("Advanced CNN (CIFAR-10 3x32x32)",      AdvancedCNN(num_classes=10),               (3, 32, 32)),
    ]
    model_params = {}
    for name, model, in_shape in models_to_show:
        print_model_summary(model, name, in_shape)
        model_params[name] = count_parameters(model)

    # ----------------------------------------------------------------
    # Step 3: MNIST Training
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Step 3] MNIST Handwritten Digit Classification (MLP vs CNN)")
    print("=" * 60)
    mnist_results = run_mnist_training(device, epochs_mlp=4, epochs_cnn=4, batch_size=256)

    # Plot MNIST training curves
    mnist_histories = {name: res["history"] for name, res in mnist_results.items()}
    viz.plot_training_curves(mnist_histories, filename="mnist_training_curves.png")

    # Confusion matrix for CNN
    if "Basic CNN" in mnist_results and "preds" in mnist_results["Basic CNN"]:
        cnn_res = mnist_results["Basic CNN"]
        digit_names = [str(i) for i in range(10)]
        viz.plot_confusion_matrix(
            cnn_res["labels"], cnn_res["preds"],
            class_names=digit_names,
            title="MNIST CNN -- Confusion Matrix",
            filename="mnist_cnn_confusion.png"
        )

    # ----------------------------------------------------------------
    # Step 4: Custom Image Classification (Advanced CNN)
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Step 4] Custom Image Classification (Advanced CNN - 90%+ Acc)")
    print("=" * 60)
    cifar_history, per_class_acc, cifar_acc, cifar_model, cifar_test_loader = \
        run_cifar10_training(device, epochs=8, batch_size=128)

    viz.plot_training_curves({"Advanced CNN": cifar_history},
                              filename="cifar10_training_curves.png")
    viz.plot_per_class_accuracy(per_class_acc, filename="cifar10_per_class_accuracy.png")

    # Sample predictions
    cifar_model.eval()
    images_batch, labels_batch = next(iter(cifar_test_loader))
    with torch.no_grad():
        preds_batch = cifar_model(images_batch.to(device)).argmax(dim=1).cpu().numpy()

    cifar_classes = getattr(cifar_test_loader.dataset, 'classes',
                            ['airplane', 'automobile', 'bird', 'cat', 'deer',
                             'dog', 'frog', 'horse', 'ship', 'truck'])
    viz.plot_sample_predictions(
        images_batch.numpy(), labels_batch.numpy(), preds_batch,
        class_names=cifar_classes, n=16, filename="cifar10_sample_preds.png"
    )

    # ----------------------------------------------------------------
    # Step 5: Transfer Learning
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Step 5] Transfer Learning: ResNet-18 Feature Extraction vs Fine-Tuning")
    print("=" * 60)
    tl_results = run_transfer_learning(device, epochs=4, batch_size=128)
    viz.plot_transfer_vs_scratch(tl_results, filename="transfer_learning_curves.png")

    # ----------------------------------------------------------------
    # Final Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 64)
    print(" DAY 3-4 SUMMARY")
    print("=" * 64)

    all_accs = {}
    for name, res in mnist_results.items():
        all_accs[f"MNIST {name}"] = res["test_acc"]
    all_accs["CIFAR-10 AdvancedCNN"] = cifar_acc
    for name, res in tl_results.items():
        all_accs[f"TL {name.split('(')[0].strip()}"] = res["test_acc"]

    viz.plot_model_comparison(all_accs, filename="model_comparison.png")

    print("\n  Model                             | Test Accuracy")
    print("  " + "-" * 50)
    for name, acc in all_accs.items():
        print(f"  {name:<35}| {acc*100:.2f}%")

    print("\n  Visualizations saved to results/:")
    for fname in sorted(os.listdir("results")):
        print(f"    - {fname}")

    print("\n================================================================")
    print(" DAY 3-4 COMPLETED SUCCESSFULLY!")
    print("================================================================")


if __name__ == "__main__":
    main()
