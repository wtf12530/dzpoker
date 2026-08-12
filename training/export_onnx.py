"""
training/export_onnx.py

Exports a PyTorch model to ONNX and validates the exported file can be loaded.
If validation fails for a higher opset, the script will retry with a lower opset.
"""
# Ensure repo root is on sys.path when running as a script.
import os
import sys
if __package__ is None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import argparse
import traceback
import torch

from training.train_supervised import build_model

def _export_model(model, dummy, out_path, opset):
    try:
        torch.onnx.export(
            model,
            dummy,
            out_path,
            input_names=['input'],
            output_names=['logits'],
            dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
            opset_version=opset,
        )
        print(f"Exported ONNX model to {out_path} using opset {opset}")
        return True
    except Exception as e:
        print(f"Failed to export with opset={opset}: {e}")
        traceback.print_exc()
        return False

def _validate_onnx(out_path):
    # Prefer onnxruntime validation if available (stronger check); fall back to onnx.checker
    try:
        import onnxruntime as ort
        try:
            _ = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
            print("onnxruntime successfully loaded the exported model")
            return True
        except Exception as e:
            print(f"onnxruntime failed to load the model: {e}")
            return False
    except Exception:
        # onnxruntime not installed: try basic onnx load/check
        try:
            import onnx
            model = onnx.load(out_path)
            onnx.checker.check_model(model)
            print("onnx.load and onnx.checker succeeded")
            return True
        except Exception as e:
            print(f"ONNX sanity check failed: {e}")
            traceback.print_exc()
            return False

def export(model_path, out_path, input_dim=None, n_actions=None):
    state = torch.load(model_path, map_location='cpu')
    if input_dim is None or n_actions is None:
        raise ValueError('Please provide input_dim and n_actions to export to ONNX')
    model = build_model(input_dim, n_actions)

    # Try to load state dict if possible
    if isinstance(state, dict):
        try:
            model.load_state_dict(state)
        except Exception as e:
            print("Warning: failed to load state_dict into model; continuing with current weights:", e)

    model.eval()
    dummy = torch.zeros(1, input_dim)

    # Try opsets in order: prefer newer, but validate and fallback if needed
    for opset in (11, 9):
        ok = _export_model(model, dummy, out_path, opset)
        if not ok:
            continue
        # Validate that the exported file can be loaded
        if _validate_onnx(out_path):
            print(f"Export validated with opset={opset}")
            return
        else:
            print(f"Validation failed for opset={opset}, will try next lower opset (if any)")
    raise RuntimeError("ONNX export failed or no valid ONNX file could be produced")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--input-dim', type=int, required=True)
    p.add_argument('--n-actions', type=int, required=True)
    args = p.parse_args()
    export(args.model, args.out, args.input_dim, args.n_actions)
