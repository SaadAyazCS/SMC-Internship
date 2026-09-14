"""
RNN, LSTM, and GRU Architecture Basics.
Covers:
- Vanilla RNN for sequence processing
- LSTM (Long Short-Term Memory) with gate breakdown
- GRU (Gated Recurrent Unit)
- Bidirectional RNN
- Simple additive (Bahdanau) attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Vanilla RNN
# ===========================================================================

class VanillaRNN(nn.Module):
    """
    Simple RNN built with torch.nn.RNN.
    For sequence classification or regression.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, h_n = self.rnn(x)
        # Take output at last timestep
        return self.fc(out[:, -1, :])


# ===========================================================================
# LSTM
# ===========================================================================

class LSTMModel(nn.Module):
    """
    LSTM-based sequence model.
    Uses torch.nn.LSTM; supports multilayer and bidirectional variants.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 bidirectional=False, dropout=0.2):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        # Concatenate last hidden state from both directions if bidirectional
        if self.bidirectional:
            final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            final_hidden = h_n[-1]
        return self.fc(self.dropout(final_hidden))


# ===========================================================================
# GRU
# ===========================================================================

class GRUModel(nn.Module):
    """GRU-based sequence model."""
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 bidirectional=False, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out, h_n = self.gru(x)
        return self.fc(self.dropout(h_n[-1]))


# ===========================================================================
# Bidirectional LSTM (convenience wrapper)
# ===========================================================================

class BiLSTMModel(LSTMModel):
    """Bidirectional LSTM -- convenience subclass."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__(input_size, hidden_size, num_layers, output_size,
                         bidirectional=True, dropout=dropout)


# ===========================================================================
# Simple Additive (Bahdanau) Attention
# ===========================================================================

class BahdanauAttention(nn.Module):
    """
    Additive / Bahdanau attention mechanism.
    Computes a context vector as a weighted sum of LSTM outputs.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out):
        """
        lstm_out: (batch, seq_len, hidden_size)
        Returns context: (batch, hidden_size), weights: (batch, seq_len)
        """
        scores  = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=1)              # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden)
        return context, weights


class AttentionLSTM(nn.Module):
    """LSTM with Bahdanau attention for sequence classification."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attention = BahdanauAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, return_attn=False):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        logits = self.fc(self.dropout(context))
        if return_attn:
            return logits, attn_weights
        return logits


# ===========================================================================
# Architecture Comparison Demo
# ===========================================================================

def demonstrate_architectures():
    """Runs a forward pass through all RNN variants and prints shape info."""
    batch, seq_len, input_size = 8, 20, 16
    hidden_size, num_layers, output_size = 32, 2, 2
    x = torch.randn(batch, seq_len, input_size)

    architectures = {
        "Vanilla RNN":         VanillaRNN(input_size, hidden_size, num_layers, output_size),
        "LSTM":                LSTMModel(input_size, hidden_size, num_layers, output_size),
        "GRU":                 GRUModel(input_size, hidden_size, num_layers, output_size),
        "Bidirectional LSTM":  BiLSTMModel(input_size, hidden_size, num_layers, output_size),
        "Attention LSTM":      AttentionLSTM(input_size, hidden_size, num_layers, output_size),
    }

    print("\n  RNN Architecture Forward-Pass Shapes:")
    print(f"  Input: batch={batch}, seq_len={seq_len}, input_size={input_size}")
    print("  " + "-" * 55)
    for name, model in architectures.items():
        with torch.no_grad():
            out = model(x)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {name:<25} | output: {tuple(out.shape)} | params: {total_params:,}")

    return architectures
