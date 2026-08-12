# training/README.md

This directory contains simple training pipeline scripts to bootstrap training for poker agents.

Files
- collect_selfplay.py: Run self-play in RLCard and collect transitions to a compressed .npz dataset.
- train_supervised.py: Train a simple PyTorch MLP policy on the collected dataset.

Quickstart
1. Install dependencies (from project root):
   pip install -r requirements.txt
2. Collect data (example using 1000 games):
   python training/collect_selfplay.py --game limit-holdem --num-games 1000 --out training/data/selfplay.npz
3. Train supervised policy:
   python training/train_supervised.py --data training/data/selfplay.npz --epochs 10

Notes and next steps
- These scripts are intentionally minimal and designed to be a starting point. For NFSP/DeepCFR
  you should implement the corresponding algorithm training loop and replay buffers.
- You can use the collected dataset to pretrain policy/value networks or to debug state encodings.
