"""
PyTorch Tensor Basics and Autograd Demonstration.
Covers:
- Tensor creation and operations
- Broadcasting and vectorized computation
- Automatic differentiation (autograd)
- GPU device awareness
"""

import numpy as np
import torch
import torch.nn.functional as F


class TensorBasics:
    """Demonstrates core PyTorch tensor operations."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tensor_creation(self):
        """Shows multiple ways to create tensors."""
        print("\n  --- Tensor Creation ---")

        # From Python list
        t_list = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        print(f"  From list:    shape={t_list.shape}, dtype={t_list.dtype}")

        # From NumPy array
        np_arr = np.random.randn(3, 4).astype(np.float32)
        t_numpy = torch.from_numpy(np_arr)
        print(f"  From NumPy:   shape={t_numpy.shape}, dtype={t_numpy.dtype}")

        # Special constructors
        t_zeros = torch.zeros(2, 3)
        t_ones  = torch.ones(2, 3)
        t_rand  = torch.randn(2, 3)
        t_eye   = torch.eye(3)
        print(f"  zeros(2,3):   {t_zeros.shape}")
        print(f"  ones(2,3):    {t_ones.shape}")
        print(f"  randn(2,3):   {t_rand.shape}")
        print(f"  eye(3):       {t_eye.shape}")

        return t_list, t_rand

    def tensor_operations(self, t1, t2):
        """Demonstrates arithmetic, matrix ops, and indexing."""
        print("\n  --- Tensor Operations ---")

        a = torch.randn(3, 4)
        b = torch.randn(3, 4)

        # Element-wise ops
        print(f"  a + b shape:        {(a + b).shape}")
        print(f"  a * b (elem-wise):  {(a * b).shape}")
        print(f"  a.pow(2) shape:     {a.pow(2).shape}")

        # Matrix multiplication
        m1 = torch.randn(3, 5)
        m2 = torch.randn(5, 4)
        print(f"  matmul(3x5, 5x4):  {torch.matmul(m1, m2).shape}")
        print(f"  @ operator:         {(m1 @ m2).shape}")

        # Reduction
        print(f"  sum:                {a.sum():.4f}")
        print(f"  mean:               {a.mean():.4f}")
        print(f"  max:                {a.max():.4f}")

        # Reshaping
        x = torch.arange(24, dtype=torch.float32)
        print(f"  arange(24) -> (2,3,4): {x.reshape(2, 3, 4).shape}")
        print(f"  flatten:               {x.reshape(2, 3, 4).flatten().shape}")

        # Indexing & slicing
        t = torch.randn(4, 6)
        print(f"  t[1, :]:            {t[1, :].shape}")
        print(f"  t[:, 2:5]:          {t[:, 2:5].shape}")
        print(f"  t[t > 0]:           {t[t > 0].shape}  (boolean mask)")

    def broadcasting_demo(self):
        """Shows NumPy-style broadcasting."""
        print("\n  --- Broadcasting ---")
        a = torch.ones(3, 1, 5)
        b = torch.ones(1, 4, 5)
        c = a + b
        print(f"  (3,1,5) + (1,4,5) = {c.shape}")

        # Normalization via broadcasting
        data = torch.randn(100, 10)
        mean = data.mean(dim=0, keepdim=True)     # (1, 10)
        std  = data.std(dim=0, keepdim=True)       # (1, 10)
        normalized = (data - mean) / (std + 1e-8)  # (100, 10) via broadcast
        print(f"  Normalized data:      {normalized.shape}, "
              f"mean~={normalized.mean():.4f}, std~={normalized.std():.4f}")

    def autograd_demo(self):
        """Demonstrates automatic differentiation with autograd."""
        print("\n  --- Autograd (Automatic Differentiation) ---")

        # Simple scalar gradient: f(x) = x^3 + 2x^2
        x = torch.tensor(3.0, requires_grad=True)
        f = x**3 + 2 * x**2
        f.backward()
        print(f"  f(x) = x^3 + 2x^2  at x=3")
        print(f"  f(3) = {f.item():.2f}")
        print(f"  f'(3) = 3x^2 + 4x = {x.grad.item():.2f}  (analytical: {3*9 + 4*3:.2f})")

        # Vector gradient: mean squared values
        w = torch.randn(4, requires_grad=True)
        loss = (w ** 2).mean()
        loss.backward()
        print(f"\n  w = {w.data.numpy().round(3)}")
        print(f"  loss = mean(w^2) = {loss.item():.4f}")
        print(f"  grad = 2w/n:    {w.grad.numpy().round(3)}")

        # Chain rule through a simple linear layer
        x_in = torch.randn(5, 3)
        W    = torch.randn(3, 2, requires_grad=True)
        b    = torch.zeros(2, requires_grad=True)
        out  = x_in @ W + b
        loss2 = out.sum()
        loss2.backward()
        print(f"\n  Linear layer: x(5x3) @ W(3x2) + b(2)")
        print(f"  W.grad shape: {W.grad.shape}")
        print(f"  b.grad shape: {b.grad.shape}")

    def device_demo(self):
        """Shows device (CPU/GPU) awareness."""
        print("\n  --- Device Management ---")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  Using device:   {self.device}")

        t = torch.randn(3, 3).to(self.device)
        print(f"  Tensor device:  {t.device}")
        print(f"  Tensor dtype:   {t.dtype}")


def run_tensor_basics():
    """Runs all tensor basics demonstrations."""
    demo = TensorBasics()
    t1, t2 = demo.tensor_creation()
    demo.tensor_operations(t1, t2)
    demo.broadcasting_demo()
    demo.autograd_demo()
    demo.device_demo()
