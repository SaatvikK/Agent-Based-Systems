import random
import math

class Bot(object):
    def __init__(self):
        self.name = "Pressure_Bot_9000"
        self.target = [3, 2, 1]
        self.full_set = ["Da Vinci", "Picasso", "Rembrandt", "Van Gogh"]

    def get_bid(self, current_round, bots, winner_pays, artists_and_values, 
                round_limit, starting_budget, painting_order, target_collection, 
                my_bot_details, current_painting, winner_ids, amounts_paid):
        
        me = my_bot_details
        my_budget = me["budget"]
        my_paintings = me["paintings"]
        
        # 1. Calculate Game Progress
        remaining_rounds = round_limit - current_round
        
        # 2. Identify our "Need" level for the current painting
        # We want to fill the 3, then the 2, then the 1.
        current_count = my_paintings[current_painting]
        
        # Determine priority: 0 (don't need) to 3 (MUST HAVE)
        if current_count < 1:
            priority = 3 # High priority to get the first of any type
        elif current_count < 2:
            priority = 2 # Medium for the second
        elif current_count < 3:
            priority = 1 # Lower for the third
        else:
            priority = 0.5 # We already have 3, only buy if dirt cheap
            
        # 3. Analyze the Competition (Specifically looking for YOU/5714960)
        competitor_needs_this = False
        for bot in bots:
            if "5714960" in bot["bot_unique_id"]:
                # If you have 2 of this painting, you are about to win. We must block.
                if bot["paintings"][current_painting] >= 2:
                    competitor_needs_this = True

        # 4. Base Bid Calculation
        # Average price per painting if we want to buy ~10-15 paintings to guarantee a set
        avg_base_bid = my_budget / max(1, (15 - sum(my_paintings.values())))
        
        # Apply multipliers
        bid = avg_base_bid * priority
        
        # 5. Strategic Adjustments
        if competitor_needs_this:
            bid *= 1.5 # Strategic overbid to block the player
            
        # Late game desperation
        if remaining_rounds < 20 and priority > 1:
            bid = my_budget * 0.4 # Throw the kitchen sink to finish the collection
            
        # 6. Random "Noise" (To mess with your Bayesian inference)
        if random.random() < 0.05:
            bid += random.randint(20, 50)

        # 7. Safety Checks
        bid = math.ceil(bid)
        if bid > my_budget:
            bid = my_budget
            
        # Never bid 0 on something we need unless broke
        if priority >= 2 and bid < 15 and my_budget > 15:
            bid = 15
            
        return int(bid)