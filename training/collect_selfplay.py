# training/collect_selfplay.py
"""
Collect self-play trajectories using RLCard environment and simple agents.
Saves experiences to a numpy .npz file containing arrays: states, actions, rewards.

Usage:
  python training/collect_selfplay.py --game no-limit-holdem --num-games 1000 --out data.npz

Notes:
- Requires `rlcard` to be installed. If the requested game is not available, try 'limit-holdem' or 'leduc-holdem'.
- This is a data collection tool for the training pipeline. It collects observations and actions
  from self-play of random agents or rule-based agents (configurable).
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


def default_state_to_feature(state):
    """Convert RLCard state (dict) to a flat numeric feature vector.
    This function implements a conservative encoder that extracts commonly available fields.
    It may need adjustment for different game implementations.
    """
    feats = []
    # hero cards
    hand = state.get('hand', []) or state.get('raw_obs', {}).get('hand', [])
    # Represent cards as two integers (0-51) or -1 if unknown
    for i in range(2):
        if i < len(hand):
            c = hand[i]
            try:
                # rlcard uses integers for cards in some envs
                feats.append(int(c))
            except Exception:
                feats.append(-1)
        else:
            feats.append(-1)
    # community cards
    community = state.get('public_cards', []) or state.get('raw_obs', {}).get('public_cards', []) or []
    for i in range(5):
        if i < len(community):
            try:
                feats.append(int(community[i]))
            except Exception:
                feats.append(-1)
        else:
            feats.append(-1)
    # simple numeric fields
    for k in ['pot', 'to_call', 'min_raise', 'current_player']:
        v = state.get(k, None)
        if v is None:
            v = state.get('raw_obs', {}).get(k, 0)
        try:
            feats.append(float(v))
        except Exception:
            feats.append(0.0)
    # stacks/in_hand: try to flatten first 9 players
    stacks = state.get('stacks', []) or state.get('raw_obs', {}).get('stacks', []) or []
    in_hand = state.get('in_hand', []) or state.get('raw_obs', {}).get('in_hand', []) or []
    for i in range(9):
        feats.append(float(stacks[i]) if i < len(stacks) else 0.0)
        feats.append(1.0 if (i < len(in_hand) and in_hand[i]) else 0.0)
    return np.array(feats, dtype=np.float32)


def collect_games(game, num_games, out_file):
    if rlcard is None:
        raise RuntimeError('rlcard not installed or failed to import. Install rlcard to use this script.')

    try:
        env = rlcard.make(game)
    except Exception as e:
        raise RuntimeError(f"Failed to create RLCard env for game '{game}': {e}")

    num_players = env.player_num
    # Use random agents as placeholders; swap in stronger agents later
    agents = [RandomAgent(num_actions=env.action_num) for _ in range(num_players)]
    for i, a in enumerate(agents):
        env.set_agent(i, a)

    states = []
    actions = []
    rewards = []

    for g in range(num_games):
        trajectories, payoffs = env.run(is_training=False)
        # trajectories is a list of lists (per player) of (state, action, reward, next_state)
        for pid in range(len(trajectories)):
            for step in trajectories[pid]:
                s = step[0]
                a = step[1]
                states.append(default_state_to_feature(s))
                actions.append(int(a) if a is not None else -1)
                rewards.append(float(step[2] if len(step) > 2 else 0.0))

    states = np.stack(states) if len(states) else np.zeros((0, 2+5 + 4 + 9*2), dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)
    rewards = np.array(rewards, dtype=np.float32)
    np.savez_compressed(out_file, states=states, actions=actions, rewards=rewards)
    print(f"Saved {len(actions)} transitions to {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='no-limit-holdem', help='RLCard game id (e.g., no-limit-holdem, limit-holdem, leduc-holdem)')
    parser.add_argument('--num-games', type=int, default=1000)
    parser.add_argument('--out', type=str, default='training/data/selfplay.npz')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    collect_games(args.game, args.num_games, args.out)
