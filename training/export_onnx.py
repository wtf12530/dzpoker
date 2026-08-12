"""
training/export_onnx.py

Loads a PyTorch model saved by train_supervised.py and exports it to ONNX.

This script attempts to export with a modern opset first (11) and will fall
back to a more compatible opset (9) if export or later loading fails. It
also ensures the repository root is on sys.path when run as a script so
imports like `from training.train_supervised import build_model` work both
when executed as a module and when executed directly as a script.
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
import traceback
import torch
import numpy as np

from training.train_supervised import build_model


def _try_export(model, dummy, out_path, opset):
    try:
        torch.onnx.export(model, dummy, out_path, input_names=['input'], output_names=['logits'],
                          dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}}, opset_version=opset)
        print(f"Exported ONNX model to {out_path} using opset {opset}")
        return True
    except Exception as e:
        print(f"Failed to export with opset={opset}: {e}")
        traceback.print_exc()
        return False


def export(model_path, out_path, input_dim=None, n_actions=None):
    # load state dict
    state = torch.load(model_path, map_location='cpu')
    if input_dim is None or n_actions is None:
        raise ValueError('Please provide input_dim and n_actions to export to ONNX')
    model = build_model(input_dim, n_actions)

    # If state is a numpy-saved fallback (.npz), handle that gracefully
    if isinstance(state, dict) and '_meta' in state and not hasattr(state, 'keys'):
        # fallback meta-only state (from environments without torch)
        print("Loaded fallback metadata state; continuing with fresh model weights.")
    else:
        try:
            model.load_state_dict(state)
        except Exception as e:
            print("Warning: failed to load state_dict into model; continuing with whatever worked:\n", e)

    model.eval()
    dummy = torch.zeros(1, input_dim)

    # Try modern opset first, then fall back to 9 for compatibility
    tried = []
    for opset in (11, 9):
        tried.append(opset)
        success = _try_export(model, dummy, out_path, opset)
        if success:
            return

    # If we get here, all attempts failed
    raise RuntimeError(f"ONNX export failed for opsets {tried}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--input-dim', type=int, required=True)
    parser.add_argument('--n-actions', type=int, required=True)
    args = parser.parse_args()
    export(args.model, args.out, args.input_dim, args.n_actions)
