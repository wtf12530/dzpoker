# training/evaluate_model.py
import os
import argparse
import numpy as np

def has_torch():
    try:
        import torch
        return True
    except Exception:
        return False

def has_onnxruntime():
    try:
        import onnxruntime as ort
        return True
    except Exception:
        return False

def load_dataset(path, input_dim, n_actions):
    if path and os.path.exists(path):
        data = np.load(path)
        X = data['states'].astype(np.float32)
        y = data['actions'].astype(np.int64)
        return X, y
    # synthesize small dataset
    N = 1024
    X = np.random.randn(N, input_dim).astype(np.float32)
    y = np.random.randint(0, n_actions, size=(N,), dtype=np.int64)
    return X, y

def eval_torch(model_path, input_dim, n_actions, X, y):
    import torch
    from training.train_supervised import build_model
    model = build_model(input_dim, n_actions)
    try:
        state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state)
    except Exception:
        print("Warning: failed to load state_dict; evaluating with random init")
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X)
        logits = model(xt).cpu().numpy()
    preds = logits.argmax(axis=1)
    acc = (preds == y).mean()
    print(f"Torch eval: N={len(y)} acc={acc:.4f}")
    # print some example logits
    print("Example logits (first 3):")
    print(logits[:3])
    return acc

def eval_onnx(onnx_path, input_dim, n_actions, X, y):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    logits = sess.run(None, {input_name: X[:512]})[0]  # limit to 512 for speed
    preds = logits.argmax(axis=1)
    acc = (preds == y[:len(preds)]).mean()
    print(f"ONNX eval: N={len(preds)} acc={acc:.4f}")
    print("Example ONNX logits (first 3):")
    print(logits[:3])
    return acc

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='training/models/policy.pt')
    p.add_argument('--onnx', default='training/models/policy.onnx')
    p.add_argument('--data', default='training/data/selfplay.npz')
    p.add_argument('--input-dim', type=int, default=29)
    p.add_argument('--n-actions', type=int, default=6)
    args = p.parse_args()

    X, y = load_dataset(args.data, args.input_dim, args.n_actions)
    print(f"Loaded dataset X={X.shape} y={y.shape}")

    if has_torch() and os.path.exists(args.model):
        try:
            _ = eval_torch(args.model, args.input_dim, args.n_actions, X, y)
        except Exception as e:
            print("Torch evaluation failed:", e)
    else:
        print("Skipping Torch eval (either torch missing or model file not found).")

    if has_onnxruntime() and os.path.exists(args.onnx):
        try:
            _ = eval_onnx(args.onnx, args.input_dim, args.n_actions, X, y)
        except Exception as e:
            print("ONNX evaluation failed:", e)
    else:
        print("Skipping ONNX eval (onnxruntime missing or onnx file not found).")

if __name__ == '__main__':
    main()
