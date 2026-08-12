"""Collect self-play trajectories using RLCard environment and simple agents.
Now defaults to using the rule-based agent as opponents if available.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

try:
    import rlcard
    from rlcard.agents.random_agent import RandomAgent
except Exception as e:
    rlcard = None

# Import rule agent if present
try:
    from training.rule_agent import RuleAgent
except Exception:
    RuleAgent = None

# Shared encoding helpers
try:
    from training.encoding import card_to_int  # type: ignore
except Exception:
    def card_to_int(card: str) -> int:
        try:
            return int(card)
        except Exception:
            return -1

def default_state_to_feature(state):
    feats = []
    hand = state.get('hand', []) or state.get('raw_obs', {}).get('hand', [])
    for i in range(2):
        if i < len(hand):
            try:
                feats.append(int(card_to_int(hand[i])))
            except Exception:
                feats.append(-1)
        else:
            feats.append(-1)
    community = state.get('public_cards', []) or state.get('raw_obs', {}).get('public_cards', []) or []
    for i in range(5):
        if i < len(community):
            try:
                feats.append(int(card_to_int(community[i])))
            except Exception:
                feats.append(-1)
        else:
            feats.append(-1)
    for k in ['pot', 'to_call', 'min_raise', 'current_player']:
        v = state.get(k, None)
        if v is None:
            v = state.get('raw_obs', {}).get(k, 0)
        try:
            feats.append(float(v))
        except Exception:
            feats.append(0.0)
    stacks = state.get('stacks', []) or state.get('raw_obs', {}).get('stacks', []) or []
    in_hand = state.get('in_hand', []) or state.get('raw_obs', {}).get('in_hand', []) or []
    for i in range(9):
        feats.append(float(stacks[i]) if i < len(stacks) else 0.0)
        feats.append(1.0 if (i < len(in_hand) and in_hand[i]) else 0.0)
    return np.array(feats, dtype=np.float32)

# -- rest of file unchanged --
# (keep the other functions and __main__ from your existing file)
