"""
Training utilities and a small supervised trainer.

This script provides build_model (same API as the earlier stub) and
also a minimal CLI to train a supervised policy from a saved
training/data/selfplay.npz dataset (states, actions) and save a PyTorch
state_dict at training/models/policy.pt. It is intended for development
and CI smoke runs where a small quick training is acceptable.
"""
import os
import argparse
import numpy as np
import csv

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


def _ensure_log_dir_and_header(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # If file not exists, write header
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'epochs', 'avg_loss', 'train_acc'])


def train_supervised(data_path: str, out_path: str, input_dim: int, n_actions: int,
                     hidden: int = 128, epochs: int = 5, batch_size: int = 128, lr: float = 1e-3):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log_path = os.path.join('training', 'train_log.csv')
    _ensure_log_dir_and_header(log_path)

    if not TORCH:
        print("Torch not available in environment; cannot perform real training.")
        model = build_model(input_dim, n_actions, hidden)
        try:
            sd = model.state_dict()
        except Exception:
            sd = {'_meta': {'input_dim': input_dim, 'n_actions': n_actions, 'hidden': hidden}}
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

        # compute training accuracy on full dataset (cheap for small datasets)
        model.eval()
        with torch.no_grad():
            logits_all = model(X)
            preds = logits_all.argmax(dim=1)
            train_acc = (preds == y).float().mean().item()
        model.train()

        print(f"Epoch {ep}/{epochs} avg_loss={avg:.4f} train_acc={train_acc:.4f}")

        # append to CSV log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, epochs, f"{avg:.6f}", f"{train_acc:.6f}"])

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
