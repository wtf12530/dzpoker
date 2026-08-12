"""Training utilities and a small supervised trainer.

This script provides build_model (same API as the earlier stub) and
also a minimal CLI to train a supervised policy from a saved
training/data/selfplay.npz dataset (states, actions) and save a PyTorch
state_dict at training/models/policy.pt. It is intended for development
and CI smoke runs where a small quick training is acceptable.

Behavior:
- If torch is available, performs a few training steps and saves model.
- If no dataset is present, creates a tiny synthetic dataset so the
  CLI still produces a model for ONNX export testing.

The goal is to make the repo's training -> export flow runnable in CI
under reasonable (CPU) settings.
"""

import os
import argparse
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH = True
except Exception:
    TORCH = False


def build_model(input_dim: int, n_actions: int, hidden: int = 128):
    if TORCH:
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, n_actions, hidden):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, n_actions)
                )

            def forward(self, x):
                return self.net(x)

        return SimpleMLP(input_dim, n_actions, hidden)
    else:
        # Fallback minimal model from the previous stub
        class FallbackModel:
            def __init__(self, input_dim, n_actions, hidden):
                self.input_dim = input_dim
                self.n_actions = n_actions
                self._meta = {
                    'input_dim': input_dim,
                    'n_actions': n_actions,
                    'hidden': hidden,
                }

            def __call__(self, x):
                import numpy as _np
                x_arr = _np.asarray(x)
                batch = x_arr.shape[0] if x_arr.ndim > 1 else 1
                return _np.zeros((batch, self.n_actions), dtype=_np.float32)

            def state_dict(self):
                return {'_meta': self._meta}

            def load_state_dict(self, sd):
                return

        return FallbackModel(input_dim, n_actions, hidden)


def train_supervised(data_path: str, out_path: str, input_dim: int, n_actions: int,
                     hidden: int = 128, epochs: int = 5, batch_size: int = 128, lr: float = 1e-3):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not TORCH:
        print("Torch not available in environment; cannot perform real training.")
        # Create a dummy state dict for compatibility if possible
        model = build_model(input_dim, n_actions, hidden)
        try:
            sd = model.state_dict()
            # we can try to save using torch if available, but it's not
        except Exception:
            sd = {'_meta': {'input_dim': input_dim, 'n_actions': n_actions, 'hidden': hidden}}
        # Save as numpy file as a fallback
        np.savez_compressed(out_path + '.npz', state=sd)
        print(f"Saved fallback model metadata to {out_path}.npz")
        return

    # Load or synthesize dataset
    if os.path.exists(data_path):
        print(f"Loading dataset from {data_path}")
        data = np.load(data_path)
        states = data['states']
        actions = data['actions']
    else:
        print(f"Dataset {data_path} not found, creating synthetic data for smoke training")
        # create a tiny synthetic dataset
        N = 1024
        states = np.random.randn(N, input_dim).astype(np.float32)
        actions = np.random.randint(0, n_actions, size=(N,))

    # Convert to torch tensors
    X = torch.from_numpy(states).float()
    y = torch.from_numpy(actions).long()

    model = build_model(input_dim, n_actions, hidden)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for ep in range(1, max(1, epochs) + 1):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(dataset)
        print(f"Epoch {ep}/{epochs} avg_loss={avg:.4f}")

    # Save model state_dict
    torch.save(model.state_dict(), out_path)
    print(f"Saved model state_dict to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='training/data/selfplay.npz')
    parser.add_argument('--out', type=str, default='training/models/policy.pt')
    parser.add_argument('--input-dim', type=int, default=29)
    parser.add_argument('--n-actions', type=int, default=6)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    train_supervised(args.data, args.out, args.input_dim, args.n_actions,
                     hidden=args.hidden, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
