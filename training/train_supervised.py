"""Minimal training.train_supervised stub used by CI and other scripts.

This module provides a `build_model(input_dim, n_actions, hidden=128)` factory
that returns a model object compatible with downstream code that expects a
PyTorch `nn.Module` (has `.state_dict()` and `.load_state_dict()` and is callable).

Behavior:
- If PyTorch is available, we return a simple torch.nn.Sequential MLP.
- If PyTorch is not available in the environment (common in lightweight CI runs),
  we return a tiny fallback object that provides the minimal API so imports and
  smoke tests succeed.

This file is intentionally small and conservative — replace with a full training
model when you iterate on training.
"""

from typing import Any

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


def build_model(input_dim: int, n_actions: int, hidden: int = 128) -> Any:
    """Return a model object with a torch-like API.

    Args:
        input_dim: dimensionality of input features
        n_actions: number of output actions / logits
        hidden: hidden layer size (only used when PyTorch is present)

    Returns:
        If PyTorch is installed: an instance of torch.nn.Module (MLP)
        Otherwise: a lightweight fallback object with state_dict/load_state_dict
    """
    if TORCH_AVAILABLE and torch is not None and nn is not None:
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

    # Fallback minimal model when torch is not present.
    class FallbackModel:
        def __init__(self, input_dim, n_actions, hidden):
            self.input_dim = input_dim
            self.n_actions = n_actions
            # minimal pseudo-parameters for compatibility
            self._meta = {
                'input_dim': input_dim,
                'n_actions': n_actions,
                'hidden': hidden,
            }

        def __call__(self, x):
            # Accept anything and return zeros-like logits
            try:
                import numpy as _np
                x_arr = _np.asarray(x)
                batch = x_arr.shape[0] if x_arr.ndim > 1 else 1
                return _np.zeros((batch, self.n_actions), dtype=_np.float32)
            except Exception:
                # best-effort fallback
                return [[0.0] * self.n_actions]

        def state_dict(self):
            return {'_meta': self._meta}

        def load_state_dict(self, sd):
            # accept and ignore; keep compatibility with torch.load/state dict usage
            if isinstance(sd, dict) and '_meta' in sd:
                # optionally validate shape keys
                return
            return

    return FallbackModel(input_dim, n_actions, hidden)


__all__ = ["build_model"]
