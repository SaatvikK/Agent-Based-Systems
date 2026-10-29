import random
import math

# DISLCAIMER:
# This bot was created by ANTHROPIC'S CLAUDE OPUS 4.6 so that I could test my bot (my own work) against a strategic agent
# instead of random and flat bots.
# PROMPT:
# (after supplying it the assignment sheet and lecture notes)
"""
Can you develop another adversary bot that i can test it against? it should be strategic, and using the theory provided in the lecture notes.
"""
class Bot(object):
    """
    A bot grounded in game-theoretic principles:
    
    1. Expected Utility Maximisation (Risk & Decisions lectures):
       Each bid is evaluated as a lottery: P(win|bid) * marginal_value - bid_cost.
       The bot picks the bid that maximises expected utility.
    
    2. Marginal Value via Product Scoring:
       The value of winning a painting is the marginal increase in the product score.
       score(b) = prod(paintings). So marginal value of painter x is:
       prod(paintings + e_x) - prod(paintings)
       This captures diminishing returns and the catastrophic cost of zeroes.
    
    3. Best Response to Estimated Opponent Strategy (Nash/GT lectures):
       The bot maintains a running estimate of opponent bid distributions
       and computes a best response: the bid that maximises EU given beliefs.
    
    4. Mixed Strategy Element:
       Adds noise to bids to avoid being predictable (exploitable).
    """

    def __init__(self):
        self.name = "eu_bot"
        self.opponent_bids = []  # estimated from winning bids
        self.my_id = None

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
        artists = ["Picasso", "Van Gogh", "Rembrandt", "Da Vinci"]

        if budget <= 0 or remaining <= 0:
            return 0

        if self.my_id is None:
            self.my_id = my_bot_details["bot_unique_id"]

        # --- 1. Compute marginal value of current painting ---
        counts = [paintings.get(a, 0) for a in artists]
        current_idx = artists.index(current_painting)

        current_product = math.prod(counts) if all(c > 0 for c in counts) else 0

        new_counts = counts.copy()
        new_counts[current_idx] += 1
        new_product = math.prod(new_counts) if all(c > 0 for c in new_counts) else 0

        marginal_value = new_product - current_product

        # Bonus: if this painter is in the gap set, assign high urgency
        gap_painters = [a for a in artists if paintings.get(a, 0) == 0]
        is_gap = current_painting in gap_painters

        # How many future chances to get this painter?
        future_chances = painting_order[current_round:].count(current_painting)

        # Urgency multiplier: fewer chances = more urgent
        if is_gap and future_chances > 0:
            urgency = min(5.0, 10.0 / future_chances)
        elif is_gap:
            urgency = 10.0  # last chance
        else:
            urgency = 1.0

        # --- 2. Estimate opponent bid distribution ---
        # Use recent winning bids as proxy for max opponent bid
        recent_window = min(30, len(amounts_paid))
        if recent_window > 0:
            recent = amounts_paid[-recent_window:]
            mean_winning = sum(recent) / len(recent)
            # Rough std estimate
            variance = sum((x - mean_winning) ** 2 for x in recent) / max(1, len(recent))
            std_winning = max(1, variance ** 0.5)
        else:
            mean_winning = starting_budget / round_limit
            std_winning = mean_winning * 0.5

        # --- 3. Expected utility maximisation ---
        # For each candidate bid b, compute:
        #   EU(b) = P(win | b) * (marginal_value * urgency) - b
        # where P(win | b) is estimated from opponent distribution
        
        n_opponents = len(bots) - 1
        best_eu = float('-inf')
        best_bid = 0
        
        # Pacing constraint: don't spend more than fair share
        max_bid = min(budget, int(budget / max(1, remaining) * 3))

        # Candidate bids
        candidates = set([0, 1, 5])
        candidates.add(max(1, int(mean_winning * 0.8)))
        candidates.add(max(1, int(mean_winning)))
        candidates.add(max(1, int(mean_winning * 1.1)))
        candidates.add(max(1, int(mean_winning * 1.3)))
        candidates.add(max(1, int(mean_winning * 0.5)))
        candidates.add(max_bid)
        candidates = sorted([c for c in candidates if 0 <= c <= max_bid])

        for b in candidates:
            if b == 0:
                p_win = 0
            elif std_winning > 0:
                # P(win) = P(b > max opponent bid)
                # Approximate: P(b > X) where X ~ N(mean, std) for each opponent
                # P(all opponents below b) = P(one below b)^n
                z = max(-50, min(50, (b - mean_winning) / std_winning))
                # Simple sigmoid approximation of normal CDF
                p_one = 1.0 / (1.0 + math.exp(-1.7 * z))
                p_win = p_one ** n_opponents
            else:
                p_win = 1.0 if b > mean_winning else 0.5

            eu = p_win * (marginal_value * urgency) - b * (1 - p_win * 0.5)

            if eu > best_eu:
                best_eu = eu
                best_bid = b

        # --- 4. Mixed strategy: add small noise to avoid predictability ---
        noise = random.randint(-2, 2)
        best_bid = max(0, min(budget, best_bid + noise))

        # --- 5. Gap set safety: always bid at least 1 on gap painters ---
        if is_gap and best_bid == 0:
            best_bid = max(1, min(budget, int(mean_winning * 1.1)))

        return int(best_bid)