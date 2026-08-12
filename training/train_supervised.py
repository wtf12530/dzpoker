# training/train_supervised.py
"""
Train a simple supervised policy network on collected self-play data.
The network maps state feature vectors to action logits. This is a baseline step
that can be used to initialize or sanity-check downstream RL training.

Usage:
  python training/train_supervised.py --data training/data/selfplay.npz --epochs 10

Outputs:
  - saved PyTorch model at training/models/policy.pt

Notes:
- This is a minimal example using PyTorch. For NFSP/DeepCFR you would replace this with
  the corresponding algorithm implementations.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def build_model(input_dim, n_actions, hidden=128):
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_actions)
    )


def load_data(path):
    data = np.load(path)
    states = data['states']
    actions = data['actions']
    # filter invalid actions
    mask = actions >= 0
    states = states[mask]
    actions = actions[mask]
    return states, actions


def train(args):
    states, actions = load_data(args.data)
    input_dim = states.shape[1]
    # try to infer number of actions
    n_actions = int(actions.max() + 1) if actions.size else 2

    X = torch.from_numpy(states).float()
    y = torch.from_numpy(actions).long()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = build_model(input_dim, n_actions, hidden=args.hidden)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        avg_loss = total_loss / total if total>0 else 0.0
        acc = correct / total if total>0 else 0.0
        print(f"Epoch {epoch}/{args.epochs} loss={avg_loss:.4f} acc={acc:.4f}")
    torch.save(model.state_dict(), args.out)
    print(f"Saved model to {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='training/data/selfplay.npz')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--out', type=str, default='training/models/policy.pt')
    args = parser.parse_args()
    train(args)
