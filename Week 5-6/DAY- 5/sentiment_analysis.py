"""
Sentiment Analysis using Bidirectional LSTM with Attention.
Dataset: Synthetic movie review dataset (bundled -- no download needed).
Covers: text tokenization, vocabulary building, embedding layer,
        BiLSTM + attention, binary classification.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

from rnn_basics import AttentionLSTM, BahdanauAttention


# ===========================================================================
# Bundled synthetic sentiment dataset (400 reviews)
# ===========================================================================

POSITIVE_TEMPLATES = [
    "this movie was {adj} and i really enjoyed it",
    "absolutely {adj} film with a great story",
    "the acting was {adj} and the plot was engaging",
    "a {adj} masterpiece that kept me hooked",
    "loved every minute of this {adj} film",
    "the director did a {adj} job with this one",
    "brilliant and {adj} from start to finish",
    "highly recommend this {adj} movie to everyone",
    "the performances were {adj} and very convincing",
    "an {adj} experience that i will never forget",
]
NEG_TEMPLATES = [
    "this movie was {adj} and a complete waste of time",
    "absolutely {adj} film with a terrible story",
    "the acting was {adj} and the plot made no sense",
    "a {adj} mess that kept me bored",
    "hated every minute of this {adj} film",
    "the director did a {adj} job ruining this story",
    "dull and {adj} from start to finish",
    "do not recommend this {adj} movie to anyone",
    "the performances were {adj} and unconvincing",
    "a {adj} experience i wish i could forget",
]
POS_ADJ = ["wonderful", "amazing", "fantastic", "brilliant", "superb",
           "excellent", "outstanding", "incredible", "magnificent", "thrilling",
           "captivating", "remarkable", "stunning", "moving", "inspiring"]
NEG_ADJ = ["terrible", "awful", "dreadful", "boring", "horrible",
           "disappointing", "poor", "mediocre", "unbearable", "tedious",
           "dull", "bad", "pathetic", "atrocious", "wretched"]


def build_dataset(n_per_class=200, seed=42):
    """Generates synthetic positive/negative movie reviews."""
    rng = np.random.RandomState(seed)
    reviews, labels = [], []
    for _ in range(n_per_class):
        tmpl = POSITIVE_TEMPLATES[rng.randint(len(POSITIVE_TEMPLATES))]
        adj  = POS_ADJ[rng.randint(len(POS_ADJ))]
        reviews.append(tmpl.format(adj=adj))
        labels.append(1)
    for _ in range(n_per_class):
        tmpl = NEG_TEMPLATES[rng.randint(len(NEG_TEMPLATES))]
        adj  = NEG_ADJ[rng.randint(len(NEG_ADJ))]
        reviews.append(tmpl.format(adj=adj))
        labels.append(0)
    # Shuffle
    idx = rng.permutation(len(reviews))
    return [reviews[i] for i in idx], [labels[i] for i in idx]


# ===========================================================================
# Tokenization and vocabulary
# ===========================================================================

def tokenize(text):
    return re.sub(r"[^a-z ]", " ", text.lower()).split()


def build_vocab(texts, min_freq=1):
    counter = Counter(token for text in texts for token in tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_len=30):
    tokens = tokenize(text)[:max_len]
    ids    = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    # Pad or truncate to max_len
    ids   += [vocab["<pad>"]] * (max_len - len(ids))
    return ids[:max_len]


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=30):
        self.X = torch.tensor([encode(t, vocab, max_len) for t in texts], dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===========================================================================
# Sentiment Classifier
# ===========================================================================

class SentimentClassifier(nn.Module):
    """
    Bidirectional LSTM with Bahdanau Attention for binary sentiment classification.
    Architecture:
      Embedding -> BiLSTM -> Attention -> Dropout -> Linear
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.embed    = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm     = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                                batch_first=True, bidirectional=True,
                                dropout=dropout if num_layers > 1 else 0.0)
        self.attention = BahdanauAttention(hidden_size * 2)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size * 2, 2)

    def forward(self, x, return_attn=False):
        emb  = self.dropout(self.embed(x))
        out, _ = self.lstm(emb)          # (batch, seq, hidden*2)
        context, attn_weights = self.attention(out)
        logits = self.fc(self.dropout(context))
        if return_attn:
            return logits, attn_weights
        return logits


# ===========================================================================
# Training
# ===========================================================================

def run_sentiment_analysis(device, epochs=25):
    """Trains SentimentClassifier on synthetic IMDB-style reviews."""
    print("\n  Building sentiment dataset...")
    texts, labels = build_dataset(n_per_class=200)
    vocab = build_vocab(texts)

    n_train = int(len(texts) * 0.8)
    train_ds = ReviewDataset(texts[:n_train], labels[:n_train], vocab)
    test_ds  = ReviewDataset(texts[n_train:], labels[n_train:], vocab)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

    print(f"  Vocab size: {len(vocab)} | Train: {len(train_ds)} | Test: {len(test_ds)}")

    model = SentimentClassifier(
        vocab_size=len(vocab), embed_dim=64, hidden_size=64, num_layers=2, dropout=0.3
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  BiLSTM+Attention -- {total_params:,} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history  = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n  Training Sentiment Classifier ({epochs} epochs)...")
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("  " + "-" * 55)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * X_b.size(0)
            tr_correct += (logits.argmax(1) == y_b).sum().item()
            tr_total   += X_b.size(0)
        scheduler.step()

        # Validate
        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                loss   = criterion(logits, y_b)
                vl_loss += loss.item() * X_b.size(0)
                preds = logits.argmax(1)
                vl_correct += (preds == y_b).sum().item()
                vl_total   += X_b.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_b.cpu().numpy())

        tr_l = tr_loss / tr_total;  tr_a = tr_correct / tr_total
        vl_l = vl_loss / vl_total;  vl_a = vl_correct / vl_total

        history["train_loss"].append(tr_l); history["train_acc"].append(tr_a)
        history["val_loss"].append(vl_l);   history["val_acc"].append(vl_a)

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"  {epoch:>5} | {tr_l:>10.4f} | {tr_a*100:>8.2f}% | "
                  f"{vl_l:>8.4f} | {vl_a*100:>6.2f}%")

    # Final metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    preds_arr  = np.array(all_preds)
    labels_arr = np.array(all_labels)
    prec  = precision_score(labels_arr, preds_arr, zero_division=0)
    rec   = recall_score(labels_arr, preds_arr, zero_division=0)
    f1    = f1_score(labels_arr, preds_arr, zero_division=0)
    cm    = confusion_matrix(labels_arr, preds_arr)

    final_acc = history["val_acc"][-1]
    print(f"\n  Final Test Accuracy : {final_acc*100:.2f}%")
    print(f"  Precision           : {prec:.4f}")
    print(f"  Recall              : {rec:.4f}")
    print(f"  F1-Score            : {f1:.4f}")
    print(f"  Confusion Matrix:\n    {cm}")

    return history, preds_arr, labels_arr, final_acc
