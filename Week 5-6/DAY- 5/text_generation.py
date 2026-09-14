"""
Character-Level RNN Text Generator using LSTM.
Trains on a small Shakespeare corpus and generates text
by sampling from the model with temperature control.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Short Shakespeare excerpt (bundled to avoid downloads)
CORPUS = """
To be or not to be that is the question
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles
And by opposing end them to die to sleep
No more and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to tis a consummation
Devoutly to be wished to die to sleep
To sleep perchance to dream ay there is the rub
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil
Must give us pause there is the respect
That makes calamity of so long life
For who would bear the whips and scorns of time
The oppressors wrong the proud mans contumely
The pangs of despised love the laws delay
The insolence of office and the spurns
That patient merit of the unworthy takes
When he himself might his quietus make
With a bare bodkin who would fardels bear
To grunt and sweat under a weary life
But that the dread of something after death
The undiscovered country from whose bourn
No traveler returns puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of
Thus conscience does make cowards of us all
And thus the native hue of resolution
Is sicklied oer with the pale cast of thought
And enterprises of great pitch and moment
With this regard their currents turn awry
And lose the name of action soft you now
The fair ophelia nymph in thy orisons
Be all my sins remembered
""".strip()

# Repeat corpus to get sufficient training data
CORPUS = (CORPUS + "\n") * 8


class CharDataset(Dataset):
    """Sliding window character sequence dataset."""
    def __init__(self, text, seq_len=80):
        self.seq_len   = seq_len
        chars          = sorted(set(text))
        self.char2idx  = {c: i for i, c in enumerate(chars)}
        self.idx2char  = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.data       = torch.tensor([self.char2idx[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


class CharLSTM(nn.Module):
    """Character-level LSTM language model."""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.embed   = nn.Embedding(vocab_size, embed_size)
        self.lstm    = nn.LSTM(embed_size, hidden_size, num_layers=num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.dropout(self.embed(x))
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(self.dropout(out))
        return logits, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))


def generate_text(model, dataset, seed_text, length=200, temperature=0.8, device="cpu"):
    """
    Generates text by sampling from the model.
    temperature < 1 -> more deterministic, temperature > 1 -> more random.
    """
    model.eval()
    # Build seed indices
    indices = [dataset.char2idx.get(c, 0) for c in seed_text]
    inp     = torch.tensor([indices], dtype=torch.long).to(device)
    hidden  = model.init_hidden(1, device)

    generated = seed_text
    with torch.no_grad():
        # Warm up on seed
        for i in range(len(seed_text) - 1):
            _, hidden = model(inp[:, i:i+1], hidden)

        # Generate characters
        char_idx = indices[-1]
        for _ in range(length):
            x_in = torch.tensor([[char_idx]], dtype=torch.long).to(device)
            logits, hidden = model(x_in, hidden)
            # Apply temperature scaling
            probs = F.softmax(logits[0, 0] / temperature, dim=0).cpu().numpy()
            char_idx = np.random.choice(len(probs), p=probs)
            generated += dataset.idx2char[char_idx]

    return generated


def run_text_generation(device, epochs=30):
    """Trains CharLSTM and generates text samples at different checkpoints."""
    print("\n  Building character dataset...")
    dataset     = CharDataset(CORPUS, seq_len=80)
    loader      = DataLoader(dataset, batch_size=64, shuffle=True)
    vocab_size  = dataset.vocab_size
    print(f"  Corpus length: {len(CORPUS):,} chars | Vocab size: {vocab_size}")

    model = CharLSTM(vocab_size=vocab_size, embed_size=64,
                     hidden_size=256, num_layers=2, dropout=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  CharLSTM -- {total_params:,} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    loss_history = []
    generated_samples = {}

    seed = "to be or not"
    print(f"\n  Training CharLSTM ({epochs} epochs)...")
    print(f"  {'Epoch':>5} | {'Train Loss':>10}")
    print("  " + "-" * 22)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits, _ = model(X_b)
            # Reshape: (batch, seq, vocab) -> (batch*seq, vocab)
            loss = criterion(logits.reshape(-1, vocab_size), y_b.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"  {epoch:>5} | {avg_loss:>10.4f}")
            sample = generate_text(model, dataset, seed, length=120, temperature=0.7, device=device)
            generated_samples[f"Epoch {epoch}"] = sample

    # Show final generated samples
    print("\n  --- Generated Text Samples ---")
    for label, text in generated_samples.items():
        print(f"\n  [{label}]")
        print(f"  {text[:200]}")

    return loss_history, generated_samples
