from collections import Counter, defaultdict


class Bot(object):
    def __init__(self):
        self.name = "die flatbot die"

        # Learned price statistics
        self.artist_avg_price = defaultdict(lambda: 10.0)
        self.global_avg_price = 10.0

        # Artist names used by the auctioneer
        self.all_artists = ["Picasso", "Van Gogh", "Rembrandt", "Da Vinci"]

    # -----------------------------
    # Basic helpers
    # -----------------------------
    def _my_counts(self, my_bot_details):
        paintings = my_bot_details.get("paintings", {}) or {}
        counts = {a: 0 for a in self.all_artists}
        for a, c in paintings.items():
            counts[a] = int(c)
        return counts

    def _remaining_supply(self, painting_order, current_round):
        remaining = Counter(painting_order[current_round:])
        return {a: int(remaining.get(a, 0)) for a in self.all_artists}

    def _safe_int(self, x, default=0):
        try:
            return int(x)
        except Exception:
            return default

    # -----------------------------
    # Price learning
    # -----------------------------
    def _update_price_model(self, painting_order, amounts_paid):
        if not amounts_paid:
            return

        n = min(len(painting_order), len(amounts_paid))
        sums = defaultdict(float)
        counts = defaultdict(int)
        total_sum = 0.0

        for r in range(n):
            artist = painting_order[r]
            paid = float(amounts_paid[r])
            sums[artist] += paid
            counts[artist] += 1
            total_sum += paid

        self.global_avg_price = total_sum / max(1, n)

        for artist in counts:
            observed_avg = sums[artist] / counts[artist]
            # Smoothed update so one strange bid does not dominate
            self.artist_avg_price[artist] = 0.65 * self.artist_avg_price[artist] + 0.35 * observed_avg

    def _price_anchor(self, artist):
        return max(1.0, 0.6 * self.artist_avg_price[artist] + 0.4 * self.global_avg_price)

    # -----------------------------
    # Scoring logic
    # -----------------------------
    def _virtual_product(self, counts):
        """
        True score is product of counts across all 4 artists.
        Early in the game that gives many zeros, so for decision-making
        we use a smoothed proxy: product(count + 1).
        This strongly rewards balance and first copies.
        """
        prod = 1
        for artist in self.all_artists:
            prod *= (counts.get(artist, 0) + 1)
        return prod

    def _marginal_gain(self, counts, artist):
        before = self._virtual_product(counts)
        new_counts = dict(counts)
        new_counts[artist] = new_counts.get(artist, 0) + 1
        after = self._virtual_product(new_counts)
        return after - before

    def _need_weight(self, counts, artist):
        """
        Strongly favour artists we currently have fewer of.
        """
        artist_count = counts.get(artist, 0)
        min_count = min(counts[a] for a in self.all_artists)

        weight = 1.0

        # Missing or low-count artists matter most
        if artist_count == 0:
            weight *= 2.3
        elif artist_count == 1:
            weight *= 1.6
        elif artist_count == 2:
            weight *= 1.2

        # Extra bonus if this artist is currently one of our lowest counts
        if artist_count == min_count:
            weight *= 1.5
        elif artist_count == min_count + 1:
            weight *= 1.15

        return weight

    def _scarcity_weight(self, counts, remaining_supply, artist, current_round, round_limit):
        """
        Bid more when an artist is running out, especially if we still do not have many of it.
        """
        owned = counts.get(artist, 0)
        remain = max(0, remaining_supply.get(artist, 0))
        rounds_left = max(1, round_limit - current_round)

        weight = 1.0

        # Relative scarcity in the future order
        freq = remain / float(rounds_left)
        if freq < 0.15:
            weight *= 1.55
        elif freq < 0.22:
            weight *= 1.30
        elif freq < 0.30:
            weight *= 1.12

        # Very low remaining stock is dangerous if we still need this artist
        if remain <= 3 and owned <= 2:
            weight *= 1.55
        elif remain <= 5 and owned <= 1:
            weight *= 1.25

        return weight

    def _coverage_urgency(self, counts, remaining_supply):
        """
        If we are missing an artist and very few remain, increase urgency a lot.
        """
        urgency = 1.0
        for artist in self.all_artists:
            owned = counts.get(artist, 0)
            remain = remaining_supply.get(artist, 0)
            if owned == 0:
                if remain <= 2:
                    urgency = max(urgency, 2.4)
                elif remain <= 4:
                    urgency = max(urgency, 1.8)
            elif owned == 1 and remain <= 3:
                urgency = max(urgency, 1.35)
        return urgency

    # -----------------------------
    # Opponent modelling
    # -----------------------------
    def _opponent_pressure(self, bots, current_painting):
        """
    Estimate how dangerous it is to let someone else take this painting.
    Works whether `bots` is a list or a dict-like structure.
        """
        max_pressure = 1.0

    # Handle both list and dict formats safely
        if isinstance(bots, dict):
            bot_iterable = bots.values()
        elif isinstance(bots, list):
            bot_iterable = bots
        else:
            return max_pressure

        for bot_info in bot_iterable:
            if not isinstance(bot_info, dict):
                continue

            paintings = bot_info.get("paintings", {}) or {}
            opp_counts = {a: 0 for a in self.all_artists}

            # paintings may itself be dict-like
            if isinstance(paintings, dict):
                for artist, c in paintings.items():
                    opp_counts[artist] = self._safe_int(c)

            opp_budget = self._safe_int(bot_info.get("budget", 0))
            if opp_budget <= 0:
                continue

            gain = self._marginal_gain(opp_counts, current_painting)
            total_virtual = self._virtual_product(opp_counts)

            pressure = 1.0 + (gain / max(1.0, total_virtual))

            if opp_counts[current_painting] == min(opp_counts[a] for a in self.all_artists):
                pressure *= 1.2

            if opp_budget >= 20:
                pressure *= 1.1

            if pressure > max_pressure:
                max_pressure = pressure

        return max_pressure

    # -----------------------------
    # Budget pacing
    # -----------------------------
    def _future_useful_opportunities(self, counts, remaining_supply):
        """
        Estimate how many future paintings are still genuinely useful for balancing us.
        """
        min_count = min(counts[a] for a in self.all_artists)
        useful = 0
        for artist in self.all_artists:
            # We especially care about artists at the bottom or near the bottom
            if counts[artist] <= min_count + 1:
                useful += remaining_supply.get(artist, 0)
            else:
                useful += int(0.35 * remaining_supply.get(artist, 0))
        return max(1, useful)

    def _budget_cap_for_round(self, my_budget, counts, remaining_supply, current_round, round_limit, artist):
        """
        Prevent overspending early while allowing aggressive late-game bids.
        """
        rounds_left = max(1, round_limit - current_round)
        useful_opps = self._future_useful_opportunities(counts, remaining_supply)

        base_cap = my_budget / float(useful_opps)

        # Late game: spend more assertively
        progress = current_round / float(max(1, round_limit))
        if progress > 0.85:
            base_cap *= 2.0
        elif progress > 0.65:
            base_cap *= 1.5
        elif progress > 0.45:
            base_cap *= 1.2

        # If artist is currently one of our weakest, allow extra spend
        if counts[artist] == min(counts[a] for a in self.all_artists):
            base_cap *= 1.35

        # If very few of this artist remain and we still need coverage/balance, spend more
        if remaining_supply.get(artist, 0) <= 3:
            base_cap *= 1.35

        return max(1, int(base_cap))

    # -----------------------------
    # Main bidding policy
    # -----------------------------
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
        my_budget = self._safe_int(my_bot_details.get("budget", 0))
        if my_budget <= 0:
            return 0

        # Learn from previous winning prices
        self._update_price_model(painting_order, amounts_paid)

        counts = self._my_counts(my_bot_details)
        remaining_supply = self._remaining_supply(painting_order, current_round)

        # If no current painting string or unknown artist, play safely
        if current_painting not in self.all_artists:
            return 0

        # Direct value of taking this painting
        gain = self._marginal_gain(counts, current_painting)
        need_weight = self._need_weight(counts, current_painting)
        scarcity_weight = self._scarcity_weight(
            counts, remaining_supply, current_painting, current_round, round_limit
        )
        coverage_urgency = self._coverage_urgency(counts, remaining_supply)
        opponent_pressure = self._opponent_pressure(bots, current_painting)

        # Price expectation from history
        price_anchor = self._price_anchor(current_painting)

        # Convert strategic usefulness into a bid multiplier
        strategic_multiplier = need_weight * scarcity_weight

        # Coverage urgency mainly matters if this artist is still low for us
        if counts[current_painting] <= 1:
            strategic_multiplier *= coverage_urgency

        # Opponent blocking should matter, but not dominate everything
        strategic_multiplier *= min(1.35, opponent_pressure)

        # Slight late-game aggression
        progress = current_round / float(max(1, round_limit))
        if progress > 0.80:
            strategic_multiplier *= 1.25
        elif progress > 0.60:
            strategic_multiplier *= 1.10

        # Main target bid
        target_bid = price_anchor * strategic_multiplier

        # Extra push if this artist is missing entirely for us
        if counts[current_painting] == 0:
            target_bid += 4

        # Budget pacing cap
        cap = self._budget_cap_for_round(
            my_budget, counts, remaining_supply, current_round, round_limit, current_painting
        )

        # We should not bid huge amounts for already-rich artists unless late game forces it
        min_count = min(counts[a] for a in self.all_artists)
        if counts[current_painting] >= min_count + 3 and progress < 0.75:
            cap = min(cap, max(3, int(0.5 * cap)))

        bid = int(target_bid)

        # Respect cap and budget
        bid = min(bid, cap, my_budget)

        # If the bid is too small to matter but the painting is genuinely important, ensure a floor
        if gain > 0 and counts[current_painting] <= min_count + 1:
            floor_bid = min(my_budget, max(3, int(0.55 * price_anchor)))
            bid = max(bid, floor_bid)

        # Avoid wasting money on very redundant paintings early
        if counts[current_painting] >= min_count + 4 and progress < 0.70:
            bid = min(bid, 2)

        # Flat bot bids 10 every round, so useful paintings should often clear that level
        if bid > 0 and counts[current_painting] <= min_count + 1:
            bid = max(bid, 11 if my_budget >= 11 else my_budget)

        return int(max(0, min(my_budget, bid)))














