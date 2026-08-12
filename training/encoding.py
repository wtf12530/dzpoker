"""
Shared encoding utilities for card strings.

Provides:
- HAS_TREYS: whether treys is importable
- card_str_to_treys(card) -> int (treys encoding or -1)
- parse_rank_from_card(card) -> int (2..14 or -1)
- card_to_int(card) -> int (prefer treys encoding; fall back to rank or -1)
"""
try:
    from treys import Card  # type: ignore
    HAS_TREYS = True
except Exception:
    Card = None
    HAS_TREYS = False

def card_str_to_treys(card: str) -> int:
    """Return treys Card.new(card) int, or -1 if treys not available or parse fails."""
    if not HAS_TREYS or Card is None:
        return -1
    try:
        return Card.new(card)
    except Exception:
        return -1

def parse_rank_from_card(card: str) -> int:
    """Parse card rank from strings like 'As','Kd','10h' -> 2..14, return -1 on failure."""
    if not card or not isinstance(card, str):
        return -1
    s = card.strip()
    # rank part is everything except the last char (suit). '10s' -> '10'
    if len(s) >= 2 and s[-1].isalpha():
        rank_part = s[:-1].upper()
    else:
        rank_part = s.upper()
    rank_map = {
        'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '10': 10,
        '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2
    }
    return rank_map.get(rank_part, -1)

def card_to_int(card: str) -> int:
    """
    Preferred integer encoding for a card:
      - If treys available, return Card.new(card)
      - Otherwise return rank (2..14)
      - If both fail return -1
    """
    v = card_str_to_treys(card)
    if v != -1:
        return v
    r = parse_rank_from_card(card)
    return r if r != -1 else -1
