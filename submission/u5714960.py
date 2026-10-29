###################### IMPORTS ######################
import random
import collections as coll
from dataclasses import *
import scipy 
import os
import math

#### Non-Standard Modules
# pip install scipy numpy
import numpy as np
import scipy 
#####################################################

N0 = int  # ℕ₀ = {0, 1, 2, ...}
Z = int   # ℤ
Psp = str # Painter space P
Bsp = dict # Bot space B

###################### TRACE OF CODE ######################
"""
1. Start on L184 - The auctioneer passes all relevant information to the Bot class, in the `Bot.get_bid(...)` method.
2. The knowledge base (KB) then updates all of its information including each `seq_x` list, and the current auction number.
3. The bayesian model is also updated here with the new budget and number of opponents.
4. `Bot.get_bid(...) then calls KB.bid(...)
5. KB.bid() computes the values of the upper bounds (the general R2 upper bound and the specific pacing bound of R3), and finds the tighter of the two bounds.
6. Then, the KB.bid() calls KB.computeBid() which is a method-dispatcher that handles the bayesian inference & bidding computation.
7. The bayesian bid is then returned to KB.bid() which then applies, in order, rule 1 and then rules 2/3 (depending on which is tighter).
8. The final, bounded bid is then returned to the auctioneer.
"""
###########################################################

# Dataclass and strict typing were used for large portions of the knowledge base as its largely propositional logic and a data-holder.
@dataclass
class KnowledgeBase:
  seq_r : list # list of winning bids
  seq_p : list # list of paintings
  seq_w : list # list of winners
  rT      : N0
  totalb  : N0

  # PROBLEM: When running the lots of auctions method in arena.py, the auctioneer, for each auction, performs a deep copy of the bot.
  # This deep copy cascades to every class that the bot class uses, including KnowledgeBase, BayesianModel, and QLearner, and then to
  # Every object and variable *those* classes use and call.
  # This is unproblematic for *one* auction, however it becomes a big issue for many auctions, resulting in a recusion depth exceeded error.
  # As such, we overright copy.deepcopy() using __deepcopy__ for each class we wrote, which simply returns the object itself.
  def __deepcopy__(self, memo):
    return self
  
  def paintings(self, b: Bsp, p: Psp) -> N0:
    return b["paintings"].get(p, 0)

  def budget(self, b: Bsp) -> Z:
    return b["budget"]
  
  def sale(self, r: N0) -> Psp:
    return self.seq_p[r]
  
  def bid(self, s: Bsp, r: N0, bayes) -> N0:
    # apply rule 3: pacing constraint

    # Compute the number of times this painter will be up for auctioning in this auction's leftover rounds.
    Np = len([ p for p in self.seq_p[r:] if p == self.sale(r) ])

    # Pace our bidding based on how many more times we'll see this painter again AND how many paintings of this painter we already own.
    pacingb = (self.budget(s) / (max(1, Np) * max(1, self.paintings(s, self.sale(r)))))*3

    # apply rule 2 first as it matters regardless of rule 1:
    upperborg = self.budget(s) * 1/(self.paintings(s, self.sale(r)) + 1)
    
    # out of rules 2 and 3 (both are upper bounds) only take the tighter one
    upperb = math.ceil(min(pacingb, upperborg))
    
    # apply rule 1:
    paint0 = [ x for x in ["Picasso", "Van Gogh", "Rembrandt", "Da Vinci"] if self.paintings(s, x) == 0 ]

    bid = self.computeBid(bayes, s)

    finalbid = bid

    # If the round is less than 10. During testing, it was seen that many bots bid exceptionally high bids in the first handful of rounds. This
    # bot reacts to other bots, meaning in the subsequent few rounds it will also bid highly in response.
    # Thus, we just bid 0 initially in order to not waste budget until the other bots stabilise into behaviour that is actually learnable.
    if r < 10:
      finalbid = 0
    
    # If the previous winning bid was very low, match it rather than overbidding and overpaying (specific flatbot10 edge case)
    elif len(self.seq_r) > 0 and self.seq_r[-1] <= 11:
      finalbid = min(upperb, 11)
    
    # Apply rule 1
    elif self.sale(r) in paint0:
      # Apply rule 2/3
      finalbid = min(upperb, max(11, bid))
    else:
      # Apply rule 2/3
      finalbid = min(upperb, bid)

    return finalbid

  def computeBid(self, bayes, s):
    # Apply the adaptive-sigma algorithm each round
    # Don't compute the std on the first round as there's no window of previous winners, std requires at least two data points.
    # Instead use the default initial value in BayesianModel.__init__().
    if len(self.seq_r) > 1: bayes.updateSigma(self.seq_r, 20) # 20 is the window size

    # Only update the bayes model if there's already been a round
    if len(self.seq_r) > 0:
      winneri = self.seq_r[-1]
      botWon = self.seq_w[-1] == s["bot_unique_id"]
      bayes.update(winneri, botWon)

    bayesbid = min(bayes.predWinningBid(confidence=0.6), self.budget(s))
    return bayesbid


class BayesianModel:
  def __init__(self, nOpponents, budget=1001):
    self.nOpponents = nOpponents
    self.budget = budget
    
    # theta = mu (mean opponent bid)
    # Prior: uniform over possible mean bid values
    self.thetaVals = np.linspace(1, budget, 200)
    self.posterior = np.ones(len(self.thetaVals)) / len(self.thetaVals)
    
    # Fixed assumed variance
    self.sigma = 10
    self.lastSigma = self.sigma

  # See L25 - 29 for reasoning.
  def __deepcopy__(self, memo):
    return self

  def likelihood(self, winneri, theta, botWon): # winneri is a single winning bid for round i, and the `botWon` flag is a boolean to signify if this bot won or not.
    # P(winneri | theta) for a single round

    if botWon:
      # All opponents bid below winneri
      return scipy.stats.norm.cdf(winneri, loc=theta, scale=self.sigma) ** self.nOpponents
    else:
      # Winning bid winneri was max of opponent draws
      n = self.nOpponents
      return (n*scipy.stats.norm.pdf(winneri, loc=theta, scale=self.sigma)*scipy.stats.norm.cdf(winneri, loc=theta, scale=self.sigma)**(n - 1))

  def update(self, winneri, botWon):
    """P(theta | w_{1:r}) proportional to P(w_r | theta) * P(theta | w_{1:r-1})"""
    likelihoods = np.array([ self.likelihood(winneri, theta, botWon) for theta in self.thetaVals ])
    self.posterior *= likelihoods

    # scipy.stats.normalise: divide by P(w_r | w_{1:r-1})
    total = self.posterior.sum()
    if total > 0:
        self.posterior /= total
    else:
        # Reset if numerical underflow
        self.posterior = np.ones(len(self.thetaVals)) / len(self.thetaVals)

  def predWinningBid(self, confidence=0.8):
    # Estimate bid needed to win with probability >= confidence
    # Expected μ under posterior
    eMu = np.dot(self.thetaVals, self.posterior)
    
    # Bid that beats max of n opponents with probability >= confidence
    # F(bid)^n >= confidence => F(bid) >= confidence^(1/n)
    targetQuant = confidence ** (1 / self.nOpponents)
    bid = scipy.stats.norm.ppf(targetQuant, loc=eMu, scale=self.sigma)
    return max(1, int(bid))
    
  def updateSigma(self, winningBids, window = 50):

    # If the winow is greater than the number of rounds so far, just set the window to the number of rounds
    if window > len(winningBids):
      window = len(winningBids)
    
    # Get all winning bids in the window
    wins = winningBids[len(winningBids)-window:]
    self.lastSigma = self.sigma
    self.sigma = np.std(wins) # And then find the std of those winning bids
    self.sigma = max(1, self.sigma)
    return self.sigma


class Bot(object):
  def __init__(self):
    self.name = "5714960"
    # Add your own variables here, if you want to.
    self.KB = KnowledgeBase([], [], [], 0, 0)
    self.bayes = BayesianModel(0, 1001)

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
    self.KB.seq_r = amounts_paid
    self.KB.seq_p = painting_order
    self.KB.seq_w = winner_ids
    s = my_bot_details
    r, rT = current_round, round_limit

    self.bayes.budget = s["budget"]
    if len(bots) - 1 == 0: return 0 # no opponents results in bayesian model crashing
    self.bayes.nOpponents = len(bots) - 1
    
    # Reset at the start of each auction
    if r == 0: 
      self.KB.totalb = s["budget"]
      self.KB.rT = rT
      self.bayes = BayesianModel(len(bots) - 1, s["budget"])

      return 11 # On the first round we're not updating the bayesian model, so just return 11 (in case flat bot 10 is in the auction).

    bid = self.KB.bid(s, r, self.bayes)
    return bid
