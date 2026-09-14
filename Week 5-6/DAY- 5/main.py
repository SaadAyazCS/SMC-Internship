"""
WEEK 5-6: DEEP LEARNING & NEURAL NETWORKS
DAY 5: Recurrent Neural Networks (RNNs, LSTM, GRU, Attention)

Demonstration Pipeline:
1. RNN/LSTM/GRU Architecture Comparison
2. Time Series Forecasting (Sine Wave) with RNN vs LSTM vs GRU
3. Character-Level Text Generation with LSTM
4. Sentiment Analysis with Bidirectional LSTM + Attention
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import torch

from rnn_basics import demonstrate_architectures
from time_series import run_time_series
from text_generation import run_text_generation
from sentiment_analysis import run_sentiment_analysis
from visualization import RNNVisualizer


def main():
    print("================================================================")
    print(" WEEK 5-6 - DAY 5: RECURRENT NEURAL NETWORKS (RNNs / LSTM)")
    print("================================================================")

    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Using device: {device}")

    viz = RNNVisualizer(output_dir="results")

    # ----------------------------------------------------------------
    # Step 1: Architecture Comparison
    # ----------------------------------------------------------------
    print("\n" + "=" * 64)
    print("[Step 1] RNN Architecture Overview (RNN vs LSTM vs GRU vs BiLSTM)")
    print("=" * 64)
    demonstrate_architectures()

    # ----------------------------------------------------------------
    # Step 2: Time Series Forecasting
    # ----------------------------------------------------------------
    print("\n" + "=" * 64)
    print("[Step 2] Time Series Forecasting: RNN vs LSTM vs GRU")
    print("=" * 64)
    ts_histories, sample_preds, sample_y, ts_test_losses = run_time_series(device, epochs=40)
    viz.plot_rnn_loss_comparison(ts_histories, filename="ts_loss_curves.png")
    viz.plot_time_series_comparison(sample_preds, sample_y, filename="ts_predictions.png")

    # ----------------------------------------------------------------
    # Step 3: Character-Level Text Generation
    # ----------------------------------------------------------------
    print("\n" + "=" * 64)
    print("[Step 3] Character-Level LSTM Text Generation")
    print("=" * 64)
    textgen_loss, generated_samples = run_text_generation(device, epochs=30)
    viz.plot_text_gen_loss(textgen_loss, filename="textgen_loss.png")

    # ----------------------------------------------------------------
    # Step 4: Sentiment Analysis
    # ----------------------------------------------------------------
    print("\n" + "=" * 64)
    print("[Step 4] Sentiment Analysis: BiLSTM + Bahdanau Attention")
    print("=" * 64)
    sentiment_history, preds_arr, labels_arr, sa_acc = run_sentiment_analysis(device, epochs=25)
    viz.plot_sentiment_curves(sentiment_history, filename="sentiment_curves.png")
    viz.plot_sentiment_confusion(labels_arr, preds_arr, filename="sentiment_confusion.png")

    # ----------------------------------------------------------------
    # Final Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 64)
    print(" DAY 5 SUMMARY")
    print("=" * 64)
    print("\n  1. Time Series Forecasting (Sine Wave, 10-step ahead):")
    for name, mse in ts_test_losses.items():
        print(f"     {name:<15}: MSE = {mse:.6f}")

    print("\n  2. Character-Level Text Generation:")
    print(f"     Final training loss: {textgen_loss[-1]:.4f}")
    if generated_samples:
        last_sample = list(generated_samples.values())[-1]
        print(f"     Sample: \"{last_sample[:80]}...\"")

    print(f"\n  3. Sentiment Analysis (BiLSTM + Attention):")
    print(f"     Test Accuracy: {sa_acc * 100:.2f}%")

    print("\n  Visualizations saved to results/:")
    for fname in sorted(os.listdir("results")):
        print(f"    - {fname}")

    print("\n================================================================")
    print(" DAY 5 COMPLETED SUCCESSFULLY!")
    print("================================================================")


if __name__ == "__main__":
    main()
