"""
A minimal rule-based RLCard agent. Implements choose_action(state) returning action index.
The rules are intentionally simple and are meant to be a stronger baseline than RandomAgent.
"""
from treys import Evaluator, Card
import random

class RuleAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.evaluator = Evaluator()

    def _hand_strength_est(self, hand, community):
        # crude estimator: pair or high card -> stronger
        try:
            if not hand:
                return 0.0
            ranks = [Card.get_rank_int(Card.new(c)) for c in hand]
            if ranks[0] == ranks[1]:
                return 0.9
            high = max(ranks)
            if high >= 11:  # J or higher
                return 0.75
        except Exception:
            pass
        return 0.3

    def choose_action(self, state):
        # state: RLCard state dict; try to read 'hand' and 'public_cards'
        hand = state.get('hand', []) or state.get('raw_obs', {}).get('hand', [])
        community = state.get('public_cards', []) or state.get('raw_obs', {}).get('public_cards', []) or []
        strength = self._hand_strength_est(hand, community)
        # basic mapping to actions: prefer raise if strong, call if medium, fold if weak
        # Map to action indices naively: 0=fold,1=call,2=raise,... depends on env
        if strength > 0.8:
            # try to raise if available
            return min(2, self.num_actions-1)
        elif strength > 0.5:
            return 1 if self.num_actions > 1 else 0
        else:
            return 0
