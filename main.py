from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import random
from treys import Deck, Evaluator, Card

app = FastAPI(title="Poker Suggestion API - Demo")

# ---------- Pydantic models ----------
class LastAction(BaseModel):
    player: int
    action: str
    amount: Optional[int] = None

class Discretization(BaseModel):
    type: str = Field("pot-fraction", example="pot-fraction")
    bins: int = Field(12, ge=2, le=32)

class SuggestRequest(BaseModel):
    game_type: str
    num_players: int
    hero_index: int
    hero_cards: Optional[List[str]] = []
    community_cards: Optional[List[str]] = []
    stacks: List[int]
    in_hand: List[bool]
    contributed: Optional[List[int]] = []
    pot: int
    big_blind: int
    small_blind: Optional[int] = 0
    to_call: int
    min_raise: int
    betting_round: str
    last_actions: Optional[List[LastAction]] = []
    discretization: Optional[Discretization] = Discretization()

class SuggestResponse(BaseModel):
    legal_actions: List[str]
    recommended_action: str
    action_probabilities: Dict[str, float]
    raise_amount: Optional[int]
    raise_bin_index: Optional[int]
    estimated_winrate: float
    estimated_EV: float
    confidence: float
    explain: Optional[str] = ""

# ---------- Helper functions ----------
evaluator = Evaluator()

DEFAULT_FRACTIONS = [0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0,8.0,12.0]

def card_str_to_treys(card: str) -> int:
    return Card.new(card)

def estimate_winrate_montecarlo(hero_cards, community_cards, num_players, in_hand, stacks, sims=100):
    wins = 0
    ties = 0
    losses = 0
    deck = Deck()
    known = []
    for c in hero_cards + community_cards:
        if c:
            try:
                known.append(card_str_to_treys(c))
            except Exception:
                pass
    for k in known:
        try:
            deck.cards.remove(k)
        except ValueError:
            pass

    active_indices = [i for i,alive in enumerate(in_hand) if alive]
    num_opponents = max(0, len(active_indices)-1)
    hero_cards_t = [card_str_to_treys(c) for c in hero_cards] if hero_cards else []
    community_t = [card_str_to_treys(c) for c in community_cards] if community_cards else []

    for _ in range(sims):
        d = Deck()
        for c in hero_cards_t + community_t:
            if c in d.cards:
                d.cards.remove(c)
        sampled_opps = []
        needed = num_opponents * 2
        if needed > 0:
            sampled = d.draw(needed)
            for i in range(num_opponents):
                sampled_opps.append([sampled[2*i], sampled[2*i+1]])
        rem = 5 - len(community_t)
        community_draw = d.draw(rem) if rem>0 else []
        community_full = community_t + community_draw
        if not hero_cards_t:
            losses += 1
            continue
        hero_score = evaluator.evaluate(community_full, hero_cards_t)
        opp_best = []
        for h in sampled_opps:
            s = evaluator.evaluate(community_full, h)
            opp_best.append(s)
        better = sum(1 for s in opp_best if s < hero_score)
        equal = sum(1 for s in opp_best if s == hero_score)
        if better == 0 and equal == 0:
            wins += 1
        elif better == 0 and equal > 0:
            ties += 1
        else:
            losses += 1

    total = wins + ties + losses
    winrate = (wins + 0.5 * ties) / total if total>0 else 0.0
    return winrate

def map_bins_to_amounts(pot, to_call, min_raise, effective_stack, bins):
    fractions = DEFAULT_FRACTIONS.copy()
    if bins <= len(fractions):
        fracs = fractions[:bins]
    else:
        extra = bins - len(fractions)
        new = list(np.logspace(np.log10(12.0), np.log10(12.0*(1.5**extra)), extra))
        fracs = fractions + new
    amounts = []
    max_allowed = effective_stack
    for f in fracs:
        desired = int(round(f * (pot + to_call)))
        desired = max(desired, min_raise)
        desired = min(desired, max_allowed)
        amounts.append(desired)
    if amounts[-1] != max_allowed:
        amounts[-1] = max_allowed
    uniq = sorted(list(dict.fromkeys(amounts)))
    return uniq

def softmax(scores):
    ex = np.exp(np.array(scores) - np.max(scores))
    probs = ex / ex.sum()
    return probs.tolist()

# ---------- API endpoints ----------
@app.post("/v1/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest):
    if req.num_players < 2 or req.num_players > 9:
        raise HTTPException(status_code=400, detail="num_players must be between 2 and 9")
    if len(req.stacks) != req.num_players or len(req.in_hand) != req.num_players:
        raise HTTPException(status_code=400, detail="stacks and in_hand must be length num_players")
    hero_cards = req.hero_cards or []
    community = req.community_cards or []
    sims = 100
    try:
        winrate = estimate_winrate_montecarlo(hero_cards, community, req.num_players, req.in_hand, req.stacks, sims=sims)
    except Exception as e:
        winrate = 0.33

    legal = []
    if req.to_call == 0:
        legal = ["check","bet","fold"]
    else:
        legal = ["fold","call","raise","allin"]

    pot_after_call = req.pot + req.to_call
    est_EV = winrate * pot_after_call - req.to_call

    effective_stack = req.stacks[req.hero_index] if 0 <= req.hero_index < len(req.stacks) else max(req.stacks)
    bins = req.discretization.bins if req.discretization else 12
    amounts = map_bins_to_amounts(req.pot, req.to_call, req.min_raise, effective_stack, bins)
    scores = {"fold":0.0,"call":0.0,"raise":0.0,"allin":0.0,"check":0.0,"bet":0.0}
    if req.to_call > 0:
        pot_odds = req.to_call / (req.pot + req.to_call)
        if winrate > pot_odds + 0.10:
            scores["raise"] += (winrate - pot_odds)
            scores["call"] += 0.2
        elif winrate > pot_odds - 0.03:
            scores["call"] += 1.0
            scores["fold"] += 0.1
        else:
            scores["fold"] += 1.0
    else:
        if winrate > 0.70:
            scores["bet"] += 2.0
            scores["raise"] += 1.0
        elif winrate > 0.45:
            scores["bet"] += 0.8
            scores["check"] += 0.5
        else:
            scores["check"] += 1.0
    if winrate > 0.92:
        scores["allin"] += 3.0
    legal_scores = [scores[a] if a in scores else 0.0 for a in legal]
    probs = softmax(legal_scores)
    action_probabilities = {a: float(p) for a,p in zip(legal, probs)}
    best_idx = int(np.argmax(probs))
    recommended = legal[best_idx]
    raise_amount = None
    raise_bin = None
    if recommended in ("raise","bet"):
        idx = int(min(len(amounts)-1, max(0, int(round(winrate * (len(amounts)-1))))))
        raise_amount = int(amounts[idx])
        raise_bin = idx
    top = float(np.max(probs))
    second = float(np.partition(probs, -2)[-2]) if len(probs)>1 else 0.0
    confidence = max(0.0, min(1.0, top - second + abs(winrate-0.5)))
    explain = f"Demo policy: winrate≈{winrate:.2f}, est_EV≈{est_EV:.1f}. Bins={len(amounts)}."
    return SuggestResponse(
        legal_actions=legal,
        recommended_action=recommended,
        action_probabilities=action_probabilities,
        raise_amount=raise_amount,
        raise_bin_index=raise_bin,
        estimated_winrate=float(winrate),
        estimated_EV=float(est_EV),
        confidence=float(confidence),
        explain=explain
    )

@app.get("/v1/model_info")
def model_info():
    return {
        "model_name": "demo_montecarlo_policy",
        "version": "0.1.0",
        "discretization": {"type":"pot-fraction","bins":12},
        "supported_game_types": ["no-limit","pot-limit","limit"]
    }

@app.get("/v1/health")
def health():
    return {"status":"ok"}
