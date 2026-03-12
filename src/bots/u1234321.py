import random
import numpy as np
import collections as coll
from dataclasses import *
import scipy 

N0 = int  # ℕ₀ = {0, 1, 2, ...}
Z = int   # ℤ
Psp = str # Painter space P
Bsp = dict # Bot space B

@dataclass
class KnowledgeBase:
  sigma_r : list # list of winning bids
  sigma_p : list # list of paintings
  sigma_w : list # list of winners

  def paintings(self, b: Bsp, p: Psp) -> N0:
    return b["paintings"].get(p, 0)

  def budget(self, b: Bsp) -> Z:
    return b["budget"]

  def sale(self, r: N0) -> Psp:
    return self.sigma_p[r]
  
  def bid(self, s: Bsp, r: N0, rT: N0, bayes) -> N0:
    # apply rule 3: pacing constraint
    pacingb = self.budget(s) / (5 * max(1, rT - r))

    # apply rule 2 first as it matters regardless of rule 1:
    upperb = self.budget(s) * 1/(self.paintings(s, self.sale(r)) + 1)

    # out of rules 2 and 3 (both are upper bounds) only take the tighter one
    #upperb = min(pacingb, upperb)

    # apply rule 1:
    paint0 = [ x for x in ["Picasso", "Van Gogh", "Rembrandt", "Da Vinci"] if self.paintings(s, x) == 0 ]
    bid = self.computeBid(s, bayes)
    if self.sale(r) in paint0:
      return min(upperb, max(1, bid)) # ensure 0 < (rule 1) bid <= upper bound (rule 2).
    
    return min(upperb, bid) # ensure bid <= upper bound (rule 2)

  def computeBid(self, s: Bsp, bayes):
    if len(self.sigma_r) > 0:
      m_i = self.sigma_r[-1]
      i_won = self.sigma_w[-1] == s["bot_unique_id"]
      bayes.update(m_i, i_won)
    return min(bayes.predWinningBid(confidence=0.5), self.budget(s))

class BayesianModel:
  def __init__(self, nOpponents, budget=1001):
    self.nOpponents = nOpponents
    self.budget = budget
    
    # θ = μ (mean opponent bid)
    # Prior: uniform over possible mean bid values
    self.thetaVals = np.linspace(1, budget, 200)
    self.posterior = np.ones(len(self.thetaVals)) / len(self.thetaVals)
    
    # Fixed assumed variance
    self.sigma = 50

  def likelihood(self, m_i, theta, i_won): # m_i is a single winning bid for round i
    """P(m_i | θ) for a single round"""
    if i_won:
      # All opponents bid below m_i
      return scipy.stats.norm.cdf(m_i, loc=theta, scale=self.sigma) ** self.nOpponents
    else:
      # Winning bid m_i was max of opponent draws
      n = self.nOpponents
      return (
        n
        * scipy.stats.norm.pdf(m_i, loc=theta, scale=self.sigma)
        * scipy.stats.norm.cdf(m_i, loc=theta, scale=self.sigma) ** (n - 1)
      )

  def update(self, m_i, i_won):
    """P(θ | σ_{1:r}) ∝ P(m_i | θ) · P(θ | σ_{1:r-1})"""
    likelihoods = np.array([
        self.likelihood(m_i, theta, i_won) for theta in self.thetaVals
    ])
    self.posterior *= likelihoods

    # scipy.stats.normalise: divide by P(σ_r | σ_{1:r-1})
    total = self.posterior.sum()
    if total > 0:
        self.posterior /= total
    else:
        # Reset if numerical underflow
        self.posterior = np.ones(len(self.thetaVals)) / len(self.thetaVals)

  def predWinningBid(self, confidence=0.8):
    """Estimate bid needed to win with probability >= confidence"""
    # Expected μ under posterior
    eMu = np.dot(self.thetaVals, self.posterior)
    
    # Bid that beats max of n opponents with probability >= confidence
    # F(bid)^n >= confidence => F(bid) >= confidence^(1/n)
    targetQuant = confidence ** (1 / self.nOpponents)
    bid = scipy.stats.norm.ppf(targetQuant, loc=eMu, scale=self.sigma)
    
    return max(1, int(bid))

class Bot(object):
  def __init__(self):
    self.name = (
        #random.randint(1000000, 10000000)
        "5714960"  # Put your id number her. String or integer will both work
    )
    # Add your own variables here, if you want to.

    self.KB = KnowledgeBase([], [], [])
    self.bayes = BayesianModel(0, 0)

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
    """Strategy for collection type games.

    Parameters:
    current_round(int): 			The current round of the auction game
    bots(dict): 					A dictionary holding the details of all of the bots in the auction
                                                                    For each bot, you are given these details:
                                                                    bot_name(str):		The bot's name
                                                                    bot_unique_id(str):	A unique id for this bot
                                                                    paintings(dict):	A dict of the paintings won so far by this bot
                                                                    budget(int):		How much budget this bot has left
                                                                    score(int):			Current value of paintings (for value game)
    winner_pays(int):				Rank of bid that winner plays. 1 is 1st price auction. 2 is 2nd price auction.
    artists_and_values(dict):		A dictionary of the artist names and the painting value to the score (for value games)
    round_limit(int):				Total number of rounds in the game - will always be 200
    starting_budget(int):			How much budget each bot started with - will always be 1001
    painting_order(list str):		A list of the full painting order
    target_collection(list int):	A list of the type of collection required to win, for collection games - will always be [3,2,1]
                                                                    [5] means that you need 5 of any one type of painting
                                                                    [4,2] means you need 4 of one type of painting and 2 of another
                                                                    [3,2,1] means you need 3 of one type of painting, 2 of another, and 1 of another
    my_bot_details(dict):			Your bot details. Same as in the bots dict, but just your bot.
                                                                    Includes your current paintings, current score and current budget
    current_painting(str):			The artist of the current painting that is being bid on
    winner_ids(list str):			A list of the ids of the winners of each round so far
    amounts_paid(list int):			List of amounts paid for paintings in the rounds played so far

    Returns:
    int:Your bid. Return your bid for this round.
    """

    # WRITE YOUR STRATEGY HERE FOR COLLECTION TYPE GAMES - FIRST TO COMPLETE A FULL COLLECTION
    self.KB.sigma_r = amounts_paid
    self.KB.sigma_p = painting_order
    self.KB.sigma_w = winner_ids

    s = my_bot_details
    r, rT = current_round, round_limit

    self.bayes.budget = s["budget"]
    self.bayes.nOpponents = len(bots) - 1
    
    #print(s["paintings"])

    return self.KB.bid(s, r, rT, self.bayes)
