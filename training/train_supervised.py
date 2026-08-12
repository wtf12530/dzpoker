"""
Training utilities and a small supervised trainer with validation logging and checkpointing.
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


def build_model(input_dim: int, n_actions: int, hidden: int = 128, dropout: float = 0.0):
    if TORCH:
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, n_actions, hidden, dropout):
                super().__init__()
                layers = [
                    nn.Linear(input_dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, n_actions)
                ]
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        return SimpleMLP(input_dim, n_actions, hidden, dropout)
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
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'epochs', 'avg_loss', 'train_acc', 'val_acc'])


def train_supervised(data_path: str, out_path: str, input_dim: int, n_actions: int,
                     hidden: int = 128, epochs: int = 5, batch_size: int = 128,
                     lr: float = 1e-3, weight_decay: float = 0.0, dropout: float = 0.0,
                     val_frac: float = 0.1, seed: int = 42, save_best: bool = True):
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

    # Deterministic split
    np.random.seed(seed)
    idx = np.random.permutation(len(states))
    n_val = int(len(states) * val_frac)
    if n_val > 0:
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
    else:
        val_idx = np.array([], dtype=int)
        train_idx = idx

    X_all = torch.from_numpy(states).float()
    y_all = torch.from_numpy(actions).long()

    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_val = X_all[val_idx] if n_val > 0 else None
    y_val = y_all[val_idx] if n_val > 0 else None

    model = build_model(input_dim, n_actions, hidden, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_acc = -1.0
    best_epoch = -1

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

        # compute training accuracy on training set
        model.eval()
        with torch.no_grad():
            logits_all = model(X_train)
            preds = logits_all.argmax(dim=1)
            train_acc = (preds == y_train).float().mean().item()

            # validation accuracy
            if n_val > 0:
                logits_val = model(X_val)
                preds_val = logits_val.argmax(dim=1)
                val_acc = (preds_val == y_val).float().mean().item()
            else:
                val_acc = float('nan')
        model.train()

        print(f"Epoch {ep}/{epochs} avg_loss={avg:.4f} train_acc={train_acc:.4f} val_acc={val_acc if not np.isnan(val_acc) else 'N/A'}")

        # append to CSV log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, epochs, f"{avg:.6f}", f"{train_acc:.6f}", f"{val_acc if not np.isnan(val_acc) else ''}"])

        # checkpoint best model by validation accuracy
        if save_best and (n_val > 0):
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = ep
                best_path = os.path.join(os.path.dirname(out_path), 'best_policy.pt')
                torch.save(model.state_dict(), best_path)
                print(f"Saved new best model (val_acc={best_val_acc:.6f}) to {best_path}")

    # Save final model state_dict (last epoch)
    torch.save(model.state_dict(), out_path)
    print(f"Saved final model state_dict to {out_path}")
    if save_best and best_epoch > 0:
        print(f"Best validation epoch: {best_epoch} val_acc={best_val_acc:.6f}")
