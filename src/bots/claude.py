import random

# DISLCAIMER:
# This bot was created by ANTHROPIC'S CLAUDE OPUS 4.6 so that I could test my bot (my own work) against a strategic agent
# instead of random and flat bots.
# PROMPT:
# (after supplying it the assignment sheet and lecture notes)
"""
can you quickly code a strategic bot. Doesnt have to be good, just something strategic and dynamic to play against
"""

class Bot(object):
    def __init__(self):
        self.name = "claude_bot"

    def get_bid(
        self,
        current_round,
        bots,
        winner_pays,
        artists_and_values,
        round_limit,
        starting_budget,
        painting_order,
        target_collection,
        my_bot_details,
        current_painting,
        winner_ids,
        amounts_paid,
    ):
        budget = my_bot_details["budget"]
        paintings = my_bot_details["paintings"]
        remaining = round_limit - current_round

        if budget <= 0 or remaining <= 0:
            return 0

        # How many of this painter do I have?
        current_count = paintings.get(current_painting, 0)

        # Gap set: painters I have zero of
        artists = ["Picasso", "Van Gogh", "Rembrandt", "Da Vinci"]
        gaps = [a for a in artists if paintings.get(a, 0) == 0]

        # Track average opponent spending from recent rounds
        recent = amounts_paid[-20:] if len(amounts_paid) >= 20 else amounts_paid
        avg_paid = sum(recent) / max(1, len(recent)) if recent else 10

        # Base bid: pace budget across remaining rounds
        base = budget / max(1, remaining)

        # Adjust based on need
        if current_painting in gaps:
            # Desperate: bid aggressively to fill gap
            # More desperate if fewer rounds remain with this painter
            future_chances = painting_order[current_round:].count(current_painting)
            urgency = max(1, 10 - future_chances)  # higher urgency = fewer chances left
            bid = base * (2 + urgency * 0.5)
            bid = max(bid, avg_paid * 1.3)  # beat recent competition
        elif current_count <= 2:
            # Still building: moderate bid
            bid = base * 1.5
            bid = max(bid, avg_paid * 1.1)
        else:
            # Already have plenty: bid low or skip
            if len(gaps) > 0:
                # Save money for gaps
                bid = base * 0.3
            else:
                bid = base * 0.8

        # Never exceed budget
        bid = min(int(bid), budget)
        # Always bid at least 1 if in gap set
        if current_painting in gaps:
            bid = max(1, bid)

        return max(0, bid)