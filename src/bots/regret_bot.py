import math
import random

# ---------------------------------------------------------------------------
# ADVERSARY BOT: Minimax Regret Agent
# ---------------------------------------------------------------------------
# Theoretical grounding
# ---------------------
# All three reference bots optimise *expected* quantities (Bayesian posterior
# mean, Q-value, EU). This bot instead minimises *regret* — a radically
# different decision criterion that requires no probabilistic model of
# opponents at all, only ordinal reasoning about outcomes.
#
# At round r, define two regret quantities for any candidate bid b:
#
#   R_lose(b) = regret of losing the item = marginal_score_gain(painter)
#               i.e. the geometric value we forego by not winning
#
#   R_over(b) = regret of overbidding = b - reference_price
#               i.e. budget wasted above a "fair" reference price
#
# The minimax regret bid b* solves:
#
#   b* = argmin_b  max( R_lose(b) * I[we lose], R_over(b) * I[we win] )
#
# Because we don't know who wins at bid time, we treat both branches and
# find the bid where R_lose == R_over — the *indifference point*. This is
# the bid that equalises worst-case regret across the two failure modes.
#
# Concretely:
#   b* = (marginal_value + reference_price) / 2
#
# The reference price is a market-learned anchor: the running empirical
# *median* of winning bids (not mean — median is more robust to auction
# spikes and requires no distributional assumptions).
#
# Additional mechanisms:
# - Gap closure urgency: if a painter is in the gap set (count = 0), the
#   marginal_value is treated as "infinity capped at budget" since a zero
#   in the product score is catastrophic. This forces aggressive regret
#   balancing toward the upper end.
# - Budget pacing via a *spending rate* constraint: at most
#   budget / remaining_rounds * PACE_FACTOR per round, preventing ruin.
# - Opponent count correction: with n opponents, the effective regret of
#   losing scales with n (more rivals → higher opportunity cost of passivity),
#   so marginal_value is multiplied by log(n+1) as a soft correction.
# ---------------------------------------------------------------------------

PACE_FACTOR   = 2.5   # how many times the "fair share" we're willing to spend
ARTISTS       = ["Picasso", "Van Gogh", "Rembrandt", "Da Vinci"]


def _product_score(counts: list[int]) -> int:
    result = 1
    for c in counts:
        result *= c
    return result


def _marginal_gain(counts: list[int], idx: int) -> int:
    """
    Geometric marginal value of one more painting of artist `idx`.
    score(counts + e_idx) - score(counts).
    If any count is 0 and idx is *not* that zero, the marginal is still 0.
    If idx *is* a zero, the marginal is the product of all other counts
    (filling in the catastrophic zero).
    """
    new_counts = counts[:]
    new_counts[idx] += 1
    return _product_score(new_counts) - _product_score(counts)


class _MedianTracker:
    """
    Online approximate median via two heaps (max-heap of lower half,
    min-heap of upper half). No imports beyond stdlib needed.
    """
    def __init__(self):
        self._lower = []  # max-heap (store negatives)
        self._upper = []  # min-heap

    def push(self, val: float):
        import heapq
        heapq.heappush(self._lower, -val)
        # rebalance: lower's max <= upper's min
        if self._lower and self._upper and (-self._lower[0]) > self._upper[0]:
            heapq.heappush(self._upper, -heapq.heappop(self._lower))
        # size balance
        if len(self._lower) > len(self._upper) + 1:
            heapq.heappush(self._upper, -heapq.heappop(self._lower))
        elif len(self._upper) > len(self._lower):
            heapq.heappush(self._lower, -heapq.heappop(self._upper))

    def median(self) -> float:
        if not self._lower:
            return 10.0  # cold start
        if len(self._lower) == len(self._upper):
            return (-self._lower[0] + self._upper[0]) / 2.0
        return float(-self._lower[0])


class Bot(object):
    """
    Minimax Regret Bidder.

    Decision rule (per round r):
        b* = clip( (V_marginal + P_ref) / 2,  lo=lo_bid, hi=pace_cap )

    where:
        V_marginal  = geometric marginal value of this painter (with gap bonus)
        P_ref       = empirical median of historical winning bids
        lo_bid      = 1 if painter in gap set, else 0
        pace_cap    = budget / remaining * PACE_FACTOR
    """

    def __init__(self):
        self.name = "regret_bot"
        self._median_tracker = _MedianTracker()
        self._rounds_seen    = 0   # how many bids we've observed

    # No deepcopy issue since we hold no circular references.

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
    ) -> int:

        budget    = my_bot_details["budget"]
        paintings = my_bot_details["paintings"]
        remaining = max(1, round_limit - current_round)
        n_opp     = max(1, len(bots) - 1)

        if budget <= 0:
            return 0

        # ------------------------------------------------------------------
        # 1. Ingest the latest winning bid into the median tracker
        # ------------------------------------------------------------------
        if len(amounts_paid) > self._rounds_seen:
            for paid in amounts_paid[self._rounds_seen:]:
                self._median_tracker.push(paid)
            self._rounds_seen = len(amounts_paid)

        P_ref = self._median_tracker.median()

        # ------------------------------------------------------------------
        # 2. Compute geometric marginal value of current painting
        # ------------------------------------------------------------------
        counts     = [paintings.get(a, 0) for a in ARTISTS]
        paint_idx  = ARTISTS.index(current_painting)
        gap_set    = [a for a in ARTISTS if paintings.get(a, 0) == 0]
        is_gap     = current_painting in gap_set

        if is_gap:
            # Filling a zero is infinitely valuable in a product score;
            # cap at remaining budget as practical upper bound.
            # The marginal value is the product of all non-zero counts,
            # which is what we'd "unlock" — clamp to budget.
            other_counts = [c for i, c in enumerate(counts) if i != paint_idx]
            raw_marginal = math.prod(other_counts) if all(c > 0 for c in other_counts) else 1
            V_marginal   = min(budget, max(raw_marginal, 30))
        else:
            V_marginal = _marginal_gain(counts, paint_idx)
            # Apply opponent-count correction: log(n+1) soft scaling
            V_marginal = V_marginal * math.log(n_opp + 1)

        # ------------------------------------------------------------------
        # 3. Minimax regret indifference point
        #
        #    R_lose(b) ≈ V_marginal (constant in b, incurred if we lose)
        #    R_over(b) ≈ b - P_ref  (grows with b, incurred if we overbid)
        #
        #    Equating: V_marginal = b* - P_ref  =>  b* = V_marginal + P_ref
        #    But b* is the *indifference* bid; we want to bid just enough
        #    to be on the right side of the balance, so:
        #
        #        b* = (V_marginal + P_ref) / 2
        #
        #    Intuition: below this, regret-of-losing dominates;
        #               above this, regret-of-overbidding dominates.
        # ------------------------------------------------------------------
        b_star = (V_marginal + P_ref) / 2.0

        # ------------------------------------------------------------------
        # 4. Constraints
        # ------------------------------------------------------------------
        pace_cap = (budget / remaining) * PACE_FACTOR
        lo_bid   = 1 if is_gap else 0

        bid = max(lo_bid, min(int(math.ceil(b_star)), int(pace_cap), budget))

        # ------------------------------------------------------------------
        # 5. Tiny dither to break ties (avoids deterministic exploitation)
        # ------------------------------------------------------------------
        dither = random.randint(-1, 1)
        bid    = max(lo_bid, min(bid + dither, budget))

        return bid
