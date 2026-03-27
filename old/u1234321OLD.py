import random
import numpy as np
import collections as coll
from dataclasses import *
import scipy 
from .old import qlearningold as ql
import os
import math

N0 = int  # ℕ₀ = {0, 1, 2, ...}
Z = int   # ℤ
Psp = str # Painter space P
Bsp = dict # Bot space B

# gamma is whether the final bid places more importance on Q learning or bayesian inference
# confident is how confident the bayesian bid should be
# learning rate and discount factors are for Q Learning
# Do gamma closer to 1 for testing (as the q table has been optimised at that point), else do closer to 0.
hyperparams = {"gamma": 1, "confidence": 0.6, "discfactor": 0.95, "lr": 0.4, "epsilon": lambda episode: max(0.05, 0.5 - (episode / 5000) * 0.45)}

@dataclass
class KnowledgeBase:
  sigma_r : list # list of winning bids
  sigma_p : list # list of paintings
  sigma_w : list # list of winners
  rT      : N0
  totalb  : N0
  currentAuction : N0
  qler    : ql.QLearner

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
  
  def gapset(self, b: Bsp) -> frozenset:
    return frozenset([ x for x in ["Picasso", "Van Gogh", "Rembrandt", "Da Vinci"] if self.paintings(b, x) == 0 ])

  def sale(self, r: N0) -> Psp:
    return self.sigma_p[r]
  
  def bid(self, s: Bsp, r: N0, bayes, training = True) -> N0:
    # apply rule 3: pacing constraint

    # Compute the number of times this painter will be up for auctioning in this auction's leftover rounds.
    Np = len([ p for p in self.sigma_p[r:] if p == self.sale(r) ])

    # Pace our bidding based on how many more times we'll see this painter again AND how many paintings of this painter we already own.
    pacingb = (self.budget(s) / (max(1, Np) * max(1, self.paintings(s, self.sale(r)))))*8

    # apply rule 2 first as it matters regardless of rule 1:
    upperborg = self.budget(s) * 1/(self.paintings(s, self.sale(r)) + 1)
    
    # out of rules 2 and 3 (both are upper bounds) only take the tighter one
    upperb = np.inf if training == True else math.ceil(min(pacingb, upperborg))
    
    # apply rule 1:
    paint0 = [ x for x in ["Picasso", "Van Gogh", "Rembrandt", "Da Vinci"] if self.paintings(s, x) == 0 ]
    bid = self.computeBid(bayes, s, training = training)
    
    # if r == 1 or r == 199 or r == 100:
    #   print(" ")
    #   print(f"======= DIAGNOSTIC: ROUND {r} =======")
    #   #print(f"upper bound = {upperborg}")
    #   #print(f"pacing bound = {pacingb}")
    #   print(f"budget = {self.budget(s)}")
    #   print(f"computed bid = {bid}, bidded bid = {math.ceil(min(upperb, bid))}")
    #   print(f"WINNER OF LAST ROUND = {self.sigma_w[-1]}")
    #   print(" ")
    #   print(f"=====================================")
    #   print(" ")
    #print(f"{r}, {bid}")
    if self.sale(r) in paint0:
      return min(upperb, max(11, bid)) # ensure 0 < (rule 1) bid <= upper bound (rule 2).
  
    return min(upperb, bid) # ensure bid <= upper bound (rule 2)

  def computeBid(self, bayes, s, training = True):
    if len(self.sigma_r) > 0:
      m_i = self.sigma_r[-1]
      i_won = self.sigma_w[-1] == s["bot_unique_id"]
      bayes.update(m_i, i_won)
    bayesbid = min(bayes.predWinningBid(confidence=hyperparams["confidence"]), self.budget(s))

    return self.qlearn(bayesbid, s, training = training)
  
  def qlearn(self, bayesbid, s, training = True):
    if len(self.sigma_w) == 0:
      return self.qler.selectAction(bayesbid, self.budget(s), gamma = hyperparams["gamma"], epsilon = hyperparams["epsilon"], currAuction = self.currentAuction)
    
    painter, spent = None, 0
    # If this bot won last round then determine which painter it won and how much it spent
    if self.sigma_w[-1].startswith("5714960"):
      painter = self.sigma_p[len(self.sigma_w) - 1]
      spent = self.sigma_r[len(self.sigma_w) - 1]
    
    budgetbin = self.qler.budgetbinner(self.budget(s), self.totalb)
    if training: 
      self.qler.transition(painter, budgetbin)
      self.qler.bellmans(spent)
    return self.qler.selectAction(bayesbid, self.budget(s), gamma = hyperparams["gamma"], epsilon = hyperparams["epsilon"], currAuction = self.currentAuction)



class BayesianModel:
  def __init__(self, nOpponents, budget=1001):
    self.nOpponents = nOpponents
    self.budget = budget
    
    # θ = μ (mean opponent bid)
    # Prior: uniform over possible mean bid values
    self.thetaVals = np.linspace(1, budget, 200)
    self.posterior = np.ones(len(self.thetaVals)) / len(self.thetaVals)
    
    # Fixed assumed variance
    self.sigma = 10

  # PROBLEM: When running the lots of auctions method in arena.py, the auctioneer, for each auction, performs a deep copy of the bot.
  # This deep copy cascades to every class that the bot class uses, including KnowledgeBase, BayesianModel, and QLearner, and then to
  # Every object and variable *those* classes use and call.
  # This is unproblematic for *one* auction, however it becomes a big issue for many auctions, resulting in a recusion depth exceeded error.
  # As such, we overright copy.deepcopy() using __deepcopy__ for each class we wrote, which simply returns the object itself.
  def __deepcopy__(self, memo):
    return self

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
    self.TRAININGQLEARN = True
    self.KB = KnowledgeBase([], [], [], 0, 0, 0, ql.QLearner(lr = hyperparams["lr"], discfactor = hyperparams["discfactor"], epochs = 1000))

    if not self.TRAININGQLEARN: self.KB.qler.loadQtable("data/qtable.json")

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
    self.KB.sigma_r = amounts_paid
    self.KB.sigma_p = painting_order
    self.KB.sigma_w = winner_ids
    self.rT = round_limit
    s = my_bot_details
    r, rT = current_round, round_limit
    print(s["paintings"])
    self.bayes.budget = s["budget"]
    self.bayes.nOpponents = len(bots) - 1
    
    # Reset at the start of each auction
    if r == 0: 
      self.KB.currentAuction += 1
      self.KB.totalb = s["budget"]
      self.KB.qler.lr = hyperparams["lr"]
      self.KB.qler.discfactor = hyperparams["discfactor"]
      self.bayes = BayesianModel(len(bots) - 1, s["budget"])

      return 11

    bid = self.KB.bid(s, r, self.bayes, training=self.TRAININGQLEARN)
    return bid
