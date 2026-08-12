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

def default_state_to_feature(state):
    feats = []
    hand = state.get('hand', []) or state.get('raw_obs', {}).get('hand', [])
    for i in range(2):
        if i < len(hand):
            c = hand[i]
            try:
                feats.append(int(c))
            except Exception:
                feats.append(-1)
        else:
            feats.append(-1)
    community = state.get('public_cards', []) or state.get('raw_obs', {}).get('public_cards', []) or []
    for i in range(5):
        if i < len(community):
            try:
                feats.append(int(community[i]))
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


class RuleAgentWrapper:
    """Adapter to make training.rule_agent.RuleAgent compatible with RLCard agent API.
    RLCard expects agents to implement step(state) returning an action index.
    """
    def __init__(self, num_actions):
        self.num_actions = num_actions
        if RuleAgent is not None:
            self.inner = RuleAgent(num_actions)
        else:
            self.inner = None

    def step(self, state):
        if self.inner is None:
            # fallback to random
            return np.random.randint(0, self.num_actions)
        try:
            return int(self.inner.choose_action(state))
        except Exception:
            return np.random.randint(0, self.num_actions)

    def eval_step(self, state):
        # RLCard sometimes calls eval_step
        return self.step(state), {}

def collect_games(game, num_games, out_file, use_rule_agent=False):
    if rlcard is None:
        raise RuntimeError('rlcard not installed or failed to import. Install rlcard to use this script.')

    try:
        env = rlcard.make(game)
    except Exception as e:
        raise RuntimeError(f"Failed to create RLCard env for game '{game}': {e}")

    num_players = env.player_num
    # Default: use rule-based agents when requested, else random
    agents = []
    for i in range(num_players):
        if use_rule_agent:
            agents.append(RuleAgentWrapper(env.action_num))
        else:
            agents.append(RandomAgent(num_actions=env.action_num))
    for i, a in enumerate(agents):
        env.set_agent(i, a)

    states = []
    actions = []
    rewards = []

    for g in range(num_games):
        trajectories, payoffs = env.run(is_training=False)
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
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savez_compressed(out_file, states=states, actions=actions, rewards=rewards)
    print(f"Saved {len(actions)} transitions to {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='limit-holdem', help='RLCard game id (e.g., limit-holdem, leduc-holdem)')
    parser.add_argument('--num-games', type=int, default=1000)
    parser.add_argument('--out', type=str, default='training/data/selfplay.npz')
    parser.add_argument('--use-rule-agent', action='store_true', help='Use rule-based agents instead of random')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    collect_games(args.game, args.num_games, args.out, use_rule_agent=args.use_rule_agent)
