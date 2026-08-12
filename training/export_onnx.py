"""
training/export_onnx.py

Loads a PyTorch model saved by train_supervised.py and exports it to ONNX.
Usage:
  python training/export_onnx.py --model training/models/policy.pt --out training/models/policy.onnx

The script requires the input_dim and n_actions to be provided or inferred from a sample dataset.
"""
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
