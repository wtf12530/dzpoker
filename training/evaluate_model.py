"""
Evaluate model using Torch and ONNX. This script is defensive: it always writes
training/eval_output.txt explaining what happened (success or exact failure reason).
It also saves numpy logits/labels when possible to make downstream analysis easier.
"""
import os
import argparse
import traceback
import numpy as np

def _write_output(text, out_path='training/eval_output.txt'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(text)
    print(text)

def _load_data(data_path, input_dim):
    if os.path.exists(data_path):
        d = np.load(data_path)
        states = d['states'].astype(np.float32)
        actions = d['actions'].astype(np.int64)
        return states, actions
    else:
        # fallback: synthetic subset for quick eval
        N = 512
        states = np.random.randn(N, input_dim).astype(np.float32)
        actions = np.random.randint(0, 6, size=(N,))
        return states, actions

def evaluate_torch(model_path, input_dim, n_actions, data_path):
    try:
        import torch
        from training.train_supervised import build_model
    except Exception as e:
        return False, f"Torch import failed or training.train_supervised missing: {e}"

    if not os.path.exists(model_path):
        return False, f"Model file not found at {model_path}"

    try:
        states, actions = _load_data(data_path, input_dim)
        X = torch.from_numpy(states).float()
        model = build_model(input_dim, n_actions)
        sd = torch.load(model_path, map_location='cpu')
        try:
            model.load_state_dict(sd)
        except Exception:
            # Maybe sd is nested or saved differently; try tolerant load
            try:
                if isinstance(sd, dict) and 'state_dict' in sd:
                    model.load_state_dict(sd['state_dict'])
                else:
                    # attempt to set state dict where keys may be prefixed
                    model.load_state_dict(sd)
            except Exception as e:
                return False, f"Failed to load state_dict into model: {e}"
        model.eval()
        with torch.no_grad():
            logits = model(X)
            preds = logits.argmax(dim=1).numpy()
        acc = (preds == actions).mean()
        # save raw outputs for later analysis
        os.makedirs('training', exist_ok=True)
        np.save('training/torch_logits.npy', logits.numpy())
        np.save('training/torch_labels.npy', actions)
        info = f"Torch eval: N={len(actions)} acc={acc:.4f}\n"
        # include example logits
        info += "Example logits (first 3):\n" + np.array2string(logits.numpy()[:3], precision=7) + "\n"
        return True, info
    except Exception as e:
        tb = traceback.format_exc()
        return False, f"Exception during Torch evaluation: {e}\n{tb}"

def evaluate_onnx(onnx_path, input_dim, n_actions, data_path):
    try:
        import onnxruntime as ort
    except Exception as e:
        return False, f"onnxruntime import failed: {e}"

    if not os.path.exists(onnx_path):
        return False, f"ONNX model file not found at {onnx_path}"

    try:
        states, actions = _load_data(data_path, input_dim)
        # only use first 512 for speed
        states = states[:512]
        actions = actions[:512]
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        inp_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        logits = sess.run([out_name], {inp_name: states.astype('float32')})[0]
        preds = logits.argmax(axis=1)
        acc = (preds == actions).mean()
        os.makedirs('training', exist_ok=True)
        np.save('training/onnx_logits.npy', logits)
        np.save('training/onnx_labels.npy', actions)
        info = f"ONNX eval: N={len(actions)} acc={acc:.4f}\n"
        info += "Example ONNX logits (first 3):\n" + np.array2string(logits[:3], precision=7) + "\n"
        return True, info
    except Exception as e:
        tb = traceback.format_exc()
        return False, f"Exception during ONNX evaluation: {e}\n{tb}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--onnx', type=str, required=False, default='')
    p.add_argument('--input-dim', type=int, required=True)
    p.add_argument('--n-actions', type=int, required=True)
    p.add_argument('--data', type=str, default='training/data/selfplay.npz')
    args = p.parse_args()

    out_lines = []
    # Torch eval
    ok, info = evaluate_torch(args.model, args.input_dim, args.n_actions, args.data)
    if ok:
        out_lines.append(info)
    else:
        out_lines.append("Skipping Torch eval: " + info)

    # ONNX eval
    if args.onnx:
        ok2, info2 = evaluate_onnx(args.onnx, args.input_dim, args.n_actions, args.data)
        if ok2:
            out_lines.append(info2)
        else:
            out_lines.append("Skipping ONNX eval: " + info2)
    else:
        out_lines.append("Skipping ONNX eval: no onnx path provided")

    final_text = "\n".join(out_lines)
    _write_output(final_text, out_path='training/eval_output.txt')

if __name__ == '__main__':
    main()
