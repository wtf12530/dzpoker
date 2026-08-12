# NFSP training skeleton
"""
training/nfsp_train.py

Skeleton implementation of NFSP-style training using RLCard environment.
This script is a starting point and not a full production NFSP implementation.
It alternates between training a DQN RL policy and updating an average-policy network
by supervised learning on a reservoir replay buffer.

Usage:
  python training/nfsp_train.py --game limit-holdem --num-episodes 10000

Notes:
- Requires rlcard and torch installed (see training/requirements.txt)
- The implementation here is intentionally minimal to demonstrate structure. Replace
  components (DQN agent, replay buffers, epsilon schedule) with more robust versions
  for large-scale training.
"""

import argparse
import os
import random
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import rlcard
    from rlcard.agents.random_agent import RandomAgent
except Exception:
    rlcard = None

from training.train_supervised import build_model, default=None  # reuse model builder


class ReservoirBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.n_seen = 0

    def add(self, sample):
        self.n_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            # reservoir sampling
            r = random.randrange(self.n_seen)
            if r < self.capacity:
                self.buffer[r] = sample

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,s2,d = map(np.array, zip(*batch))
        return s,a,r,s2,d


def build_q_network(input_dim, n_actions, hidden=128):
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_actions)
    )


def default_state_to_feature(state):
    # Delegate to collect_selfplay's encoder to keep consistency
    from training.collect_selfplay import default_state_to_feature as enc
    return enc(state)


def train(args):
    if rlcard is None:
        raise RuntimeError('rlcard not installed')

    env = rlcard.make(args.game)
    num_players = env.player_num
    num_actions = env.action_num

    # Simple agents: use random for all except training agent (player 0)
    agents = [RandomAgent(num_actions) for _ in range(num_players)]
    env.set_agents(agents)

    input_dim = 2 + 5 + 4 + 9*2  # must match encoder in collect_selfplay.py

    q_net = build_q_network(input_dim, num_actions, hidden=args.hidden)
    target_q = build_q_network(input_dim, num_actions, hidden=args.hidden)
    target_q.load_state_dict(q_net.state_dict())
    avg_net = build_model(input_dim, num_actions, hidden=args.hidden)

    q_opt = optim.Adam(q_net.parameters(), lr=args.lr)
    avg_opt = optim.Adam(avg_net.parameters(), lr=args.lr)

    replay = ReplayBuffer(capacity=args.replay_capacity)
    reservoir = ReservoirBuffer(capacity=args.reservoir_capacity)

    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = args.epsilon_decay

    global_step = 0

    for ep in range(1, args.num_episodes + 1):
        trajectories, payoffs = env.run(is_training=True)
        # collect transitions into replay and reservoir
        for pid in range(len(trajectories)):
            for step in trajectories[pid]:
                s = default_state_to_feature(step[0])
                a = int(step[1]) if step[1] is not None else 0
                r = float(step[2]) if len(step) > 2 else 0.0
                # next state not always present in RLCard traces; we skip bootstrapping simplicity
                replay.add(s, a, r, s, False)
                reservoir.add((s, a))

        # train Q network from replay
        if len(replay.buffer) >= args.batch_size:
            s,a,r,s2,d = replay.sample(args.batch_size)
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).long()
            r = torch.from_numpy(r).float()
            s2 = torch.from_numpy(s2).float()

            q_vals = q_net(s)
            q_a = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = target_q(s2).max(1)[0]
                target = r + args.gamma * q_next
            loss = nn.functional.mse_loss(q_a, target)
            q_opt.zero_grad()
            loss.backward()
            q_opt.step()

        # periodically update target
        if ep % args.target_update == 0:
            target_q.load_state_dict(q_net.state_dict())

        # periodically train average policy on reservoir samples
        if ep % args.avg_update == 0 and len(reservoir.buffer) >= args.batch_size:
            batch = reservoir.sample(args.batch_size)
            states = np.stack([b[0] for b in batch])
            actions = np.array([b[1] for b in batch])
            states_t = torch.from_numpy(states).float()
            actions_t = torch.from_numpy(actions).long()
            logits = avg_net(states_t)
            loss = nn.CrossEntropyLoss()(logits, actions_t)
            avg_opt.zero_grad()
            loss.backward()
            avg_opt.step()

        if ep % args.save_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(q_net.state_dict(), os.path.join(args.out_dir, f"q_net_ep{ep}.pt"))
            torch.save(avg_net.state_dict(), os.path.join(args.out_dir, f"avg_net_ep{ep}.pt"))
            print(f"Saved models at episode {ep}")

        if ep % args.log_every == 0:
            print(f"Episode {ep}/{args.num_episodes}")

    # final save
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(q_net.state_dict(), os.path.join(args.out_dir, "q_net_final.pt"))
    torch.save(avg_net.state_dict(), os.path.join(args.out_dir, "avg_net_final.pt"))
    print("Training complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='limit-holdem')
    parser.add_argument('--num-episodes', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--replay-capacity', type=int, default=100000)
    parser.add_argument('--reservoir-capacity', type=int, default=200000)
    parser.add_argument('--epsilon-decay', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target-update', type=int, default=100)
    parser.add_argument('--avg-update', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--out-dir', type=str, default='training/models')
    args = parser.parse_args()
    train(args)
