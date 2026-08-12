# training/export_onnx.py
"""
training/export_onnx.py

Loads a PyTorch model saved by train_supervised.py and exports it to ONNX.
Usage:
  python -m training.export_onnx --model training/models/policy.pt --out training/models/policy.onnx --input-dim 29 --n-actions 6
Or (less recommended):
  python training/export_onnx.py --model training/models/policy.pt --out training/models/policy.onnx --input-dim 29 --n-actions 6
"""

# Ensure repo root is on sys.path when running as a script so "import training..." works.
# This must run before any imports that depend on the package layout.
import os
import sys
if __package__ is None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import argparse
import torch
import numpy as np
from training.train_supervised import build_model

def export(model_path, out_path, input_dim=None, n_actions=None):
    # load state dict
    state = torch.load(model_path, map_location='cpu')
    if input_dim is None or n_actions is None:
        raise ValueError('Please provide input_dim and n_actions to export to ONNX')
    model = build_model(input_dim, n_actions)
    # If state is a numpy-saved fallback (.npz), handle that gracefully
    if isinstance(state, dict) and '_meta' in state and not hasattr(state, 'keys'):
        # fallback meta-only state (from environments without torch)
        # Try to continue with randomly initialized model (best-effort)
        print("Loaded fallback metadata state; continuing with fresh model weights.")
    else:
        model.load_state_dict(state)
    model.eval()
    dummy = torch.zeros(1, input_dim)
    torch.onnx.export(model, dummy, out_path, input_names=['input'], output_names=['logits'],
                      dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}}, opset_version=11)
    print(f"Exported ONNX model to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--input-dim', type=int, required=True)
    parser.add_argument('--n-actions', type=int, required=True)
    args = parser.parse_args()
    export(args.model, args.out, args.input_dim, args.n_actions)
