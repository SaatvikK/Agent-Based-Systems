import numpy as np
from scipy import stats
import math

class Bot(object):
    """
    SigmaPoisoner — adversarial bot targeting adaptive-sigma Bayesian inference.

    ═══════════════════════════════════════════════════════════════════════════
    ATTACK THEORY
    ═══════════════════════════════════════════════════════════════════════════

    Target bot recap:
        sigma_{r+1}  =  std( last window winning bids )
        B*  =  min B' s.t.  P(B' > expected_winning_bid | N(mu, sigma)) >= alpha

    More precisely, the target's predWinningBid() computes:
        E[mu] = sum_theta theta * P(theta | history)
        B* = Phi^{-1}( confidence^{1/n}, loc=E[mu], scale=sigma )

    Three exploitable properties:

    (1) SIGMA INFLATION
        updateSigma() sets sigma = std( last `window` winning bids ).
        If the adversary injects alternating HIGH/LOW bids into the winning
        bids history, the rolling std is maximised:
            sigma_inject ~ (h - l) / 2
        This inflates B* since:
            B* = Phi^{-1}( confidence^{1/n}, loc=E[mu], scale=sigma )
        A large sigma => large B* => target overbids => budget depletion.

    (2) POSTERIOR MU INFLATION
        The target's Bayesian posterior tracks E[mu] = expected mean opponent bid.
        If the adversary wins rounds with HIGH bids, the likelihood update:
            P(theta | i_won) ∝ CDF(m_i, theta, sigma)^n * P(theta)
        shifts the posterior toward higher theta values, raising E[mu] and
        therefore B* further.

    (3) SHADOW MODELLING + SNIPING
        Since the target's model is a function of the public (amounts_paid,
        winner_ids) history, the adversary can replicate it locally.
        In the exploitation phase, the adversary bids just above the shadow
        estimate of the target's B*, cheaply sniping desired paintings while
        the target is locked into overbidding.

    ═══════════════════════════════════════════════════════════════════════════
    PHASE STRUCTURE
    ═══════════════════════════════════════════════════════════════════════════
    Phase 1 [rounds 0 .. POISON_END]:   Alternating high/low wins to inflate sigma
    Phase 2 [rounds POISON_END .. 200]: Shadow-model target + snipe paintings

    BUDGET: Poison phase allocated POISON_BUDGET_FRAC of starting_budget.
    ═══════════════════════════════════════════════════════════════════════════
    """

    POISON_END         = 40
    POISON_BUDGET_FRAC = 0.38
    HIGH_MIN_FRAC      = 0.025
    HIGH_MAX_FRAC      = 0.055
    LOW_BID            = 1
    SNIPE_DELTA_MIN    = 2
    SNIPE_DELTA_MAX    = 6
    TARGET_CONFIDENCE  = 0.6    # target bot's confidence hyperparameter
    SHADOW_WINDOW      = 20     # target bot's window hyperparameter
    EXPLOIT_RESERVE    = 40     # never spend below this in exploitation
    ANCHOR_SPIKE_ROUND = 5
    ANCHOR_SPIKE_FRAC  = 0.07

    def __init__(self):
        self.name = "sigma_poisoner"

        self._poison_budget_cap = int(1001 * self.POISON_BUDGET_FRAC)
        self._poison_spent      = 0
        self._poison_toggle     = True   # True => HIGH bid, False => LOW bid

        # Shadow model of target's Bayesian state
        self._shadow_sigma  = 10.0
        self._shadow_mu     = 15.0      # rolling mean of winning bids

    # ─────────────────────────────────────────────────────────────────────────
    # Main interface — matches the auctioneer's exact call signature
    # ─────────────────────────────────────────────────────────────────────────

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
        r       = current_round
        s       = my_bot_details
        budget  = s["budget"]
        n_opp   = len(bots) - 1

        # Reset poison budget cap at start of each auction
        if r == 0:
            self._poison_budget_cap = int(starting_budget * self.POISON_BUDGET_FRAC)
            self._poison_spent      = 0
            self._poison_toggle     = True
            self._shadow_sigma      = 10.0
            self._shadow_mu         = 15.0
            return self.LOW_BID

        # Update shadow model from newly public information
        if len(amounts_paid) > 0:
            self._update_shadow_model(amounts_paid)

        if budget <= 0:
            return 0

        if r < self.POISON_END:
            bid = self._poisoning_bid(r, budget)
        else:
            bid = self._exploitation_bid(
                current_painting, s, painting_order, r, budget, n_opp, amounts_paid
            )

        return int(min(max(0, bid), budget))

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Sigma Poisoning
    # ─────────────────────────────────────────────────────────────────────────

    def _poisoning_bid(self, r, budget):
        """
        Alternate HIGH/LOW bids to inject maximum variance into the target's
        rolling sigma window.

        For a window of n bids alternating between h and l:
            sigma_window ~ (h - l) / 2

        One anchor spike at ANCHOR_SPIKE_ROUND also inflates the posterior
        mu of the target's Bayesian model by pushing a high value through
        its likelihood update when i_won=False (the target observes a high
        winning bid it didn't make, which shifts its theta posterior upward).
        """
        if r == self.ANCHOR_SPIKE_ROUND:
            spike = int(budget * self.ANCHOR_SPIKE_FRAC)
            self._poison_spent += spike
            return max(1, spike)

        if self._poison_spent >= self._poison_budget_cap:
            return self.LOW_BID

        if self._poison_toggle:
            allowance = self._poison_budget_cap - self._poison_spent
            high = int(budget * np.random.uniform(self.HIGH_MIN_FRAC, self.HIGH_MAX_FRAC))
            high = min(high, allowance)
            bid  = max(1, high)
            self._poison_spent += bid
        else:
            bid = self.LOW_BID

        self._poison_toggle = not self._poison_toggle
        return bid

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Shadow Modelling + Sniping
    # ─────────────────────────────────────────────────────────────────────────

    def _update_shadow_model(self, amounts_paid):
        """
        Replicate the target's updateSigma() on the public amounts_paid list.
        The target uses a rolling window of `window` bids (default 20):
            sigma = std( amounts_paid[-window:] )
        We maintain the same quantity as our shadow estimate.
        """
        window = amounts_paid[-self.SHADOW_WINDOW:]
        if len(window) >= 2:
            self._shadow_sigma = float(np.std(window, ddof=1))
            self._shadow_mu    = float(np.mean(window))

    def _estimate_target_bid(self, n_opp):
        """
        Replicate the target's predWinningBid():

            targetQuant = confidence^{1/n}
            B* = Phi^{-1}( targetQuant, loc=E[mu], scale=sigma )

        We approximate E[mu] ~ shadow_mu (the rolling mean of winning bids,
        which is a reasonable proxy for the posterior mean theta under a
        near-uniform prior when history is short).

        This gives us a cheap but accurate shadow bid estimate without
        needing to replicate the full Bayesian posterior.
        """
        if n_opp < 1:
            n_opp = 1
        target_quant = self.TARGET_CONFIDENCE ** (1.0 / n_opp)
        est = stats.norm.ppf(target_quant, loc=self._shadow_mu, scale=self._shadow_sigma)
        return max(1, math.ceil(est))

    def _exploitation_bid(self, current_painting, s, painting_order, r, budget, n_opp, amounts_paid):
        """
        Snipe paintings we need at just above the shadow-estimated target bid.
        Bid LOW on paintings we don't urgently need, letting the target waste
        budget on rounds we deliberately inflated.

        Priority: product scoring means a zero count on any artist collapses
        the score to 0. We always prioritise the rarest artist in our collection.
        """
        paintings = s.get("paintings", {})
        artists   = ["Picasso", "Van Gogh", "Rembrandt", "Da Vinci"]
        counts    = {a: paintings.get(a, 0) for a in artists}
        min_count = min(counts.values())
        need_this = counts.get(current_painting, 0) <= min_count

        if not need_this:
            return self.LOW_BID

        target_est = self._estimate_target_bid(n_opp)
        delta      = np.random.randint(self.SNIPE_DELTA_MIN, self.SNIPE_DELTA_MAX + 1)
        snipe      = target_est + delta

        # Budget pacing: don't over-commit even on desired paintings
        remaining_rounds    = max(1, 200 - r)
        usable              = max(0, budget - self.EXPLOIT_RESERVE)
        per_round_allowance = usable / remaining_rounds
        snipe = min(snipe, int(per_round_allowance * 4))

        return max(1, snipe)