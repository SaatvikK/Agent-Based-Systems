import numpy as np
import json
import random as rand
import math
import os

class QLearner:
  def __init__(self, lr = 0.8, discfactor = 0.95, epochs = 1000):
    self.lr, self.discfactor, self.epochs = lr, discfactor, epochs
   ### self.actions = [0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0] # THIS IS FOR QLEARNING-BAYESIAN HYBRID BIDDING!
    #self.actions = [-1.25, -1, -0.75, 0, 0.75, 1.0, 1.25] # THIS IS FOR QLEARNING-BAYESIAN SCALING!
    self.actions = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.75]
    self.nActions = len(self.actions)
    
    # 0 = gap = the bot owns 0 of one or more painters, resulting in a 0 score at the end
    # 2 = 2 or less paintings
    # 5 = 5 "" "" ""
    # 10 = 10 "" "" ""
    # 25 = 25 "" "" ""
    # 40 = 40 "" "" ""
    # 41 = more than 40 paintings
    self.paintingsBins = [0, 2, 5, 10, 25, 40, 41]
    self.state = [0, "high"] # (collection level, budget bin)
    self.lastState = [0, "high"]
    self.paintings = [0, 0, 0, 0]
    self.lastPaintings = self.paintings
    self.Qvalue = 0
    self.Qtable = {}  # key: (p1, p2, p3, p4, budget_bin), value: np.zeros(nActions)
    self.lastBid = 0
    self.bid = 0
    self.currAuction = 0
     # {0,0.02,0.05,0.1,0.15,0.2,0.3,0.5,0.8,1.0} <-- ratios of budget

  # PROBLEM: When running the lots of auctions method in arena.py, the auctioneer, for each auction, performs a deep copy of the bot.
  # This deep copy cascades to every class that the bot class uses, including KnowledgeBase, BayesianModel, and QLearner, and then to
  # Every object and variable *those* classes use and call.
  # This is unproblematic for *one* auction, however it becomes a big issue for many auctions, resulting in a recusion depth exceeded error.
  # As such, we overright copy.deepcopy() using __deepcopy__ for each class we wrote, which simply returns the object itself.
  def __deepcopy__(self, memo):
    return self

  @staticmethod
  def budgetbinner(budget, total, lbr = 0.25, ubr = 0.75): 
    # lbr = lower bound ratio, ubr = upper bound ratio.
    # Because the bins using the bounds of lb = lbr * total, ub = ubr * total

    if budget >= total*ubr: return "high"
    elif total*lbr < budget < total*ubr: return "med"
    elif 0 < budget <= total*lbr: return "low"
    elif budget <= 0: return "empty"
  
  @staticmethod
  def paintingsBinner(paintings, bins): 
    if 0 in list(paintings.values()): return 0
    if QLearner.rawRewardFunc(paintings) > bins[-2]: return bins[-1]

    for bin in bins:
      if QLearner.rawRewardFunc(paintings) <= bin:
        return bin

  @staticmethod
  def rawRewardFunc(paintings):
    vals = paintings.values()
    score = 1
    for val in vals:
      score *= val

    return score           
  
  @staticmethod
  def scoreFunc(current, previous, spent):
    # Use log-sum instead of product to avoid zero problem
    # log(product) = sum(log), but handle zeros
    
    def value(s):
        return sum(math.log(c + 1) for c in s)
    
    delta = value(current) - value(previous)
    if spent > 0:
      return (delta / spent) * 100 # reward efficiency

    elif spent == 0 and any(c == 0 for c in current): # a penalty for *not* bidding when it owns zero paintings from this painter -- which is deterimental.
      return -0.1
    return 0

  @staticmethod
  def bin(budget, bid, bidAmounts):
    idx = 0
    if budget > 0:
      fraction = bid / budget
    else:
      fraction = 0

    # Snap to nearest bin
    idx = np.argmin([abs(fraction - b) for b in bidAmounts])
    
    return idx

  def selectAction(self, bayesianBid, budget, epsilon = 0.1, currAuction = 1):
    if callable(epsilon):
      epsilon = epsilon(currAuction)

    if np.random.random() < epsilon:
      idx = rand.randint(0, len(self.actions) - 1)
      self.lastBid = idx
      return min(self.actions[idx] * bayesianBid, budget)

    idxQ = np.argmax(self.getQ(self.state))
    self.lastBid = idxQ
    #print(math.ceil(min(self.actions[idxQ] * bayesianBid, budget)))
    return math.ceil(min(self.actions[idxQ] * bayesianBid, budget))

  def getQ(self, state):
    state = tuple(state)
    if state not in self.Qtable:
      self.Qtable[state] = np.ones(self.nActions)
    return self.Qtable[state]

  def transition(self, paintings, paintingBin, budgetbin):

    self.lastState = self.state.copy()
    self.lastPaintings = self.paintings.copy()
    self.paintings = paintings
    self.state = [paintingBin, budgetbin]

    return self.state
  
  def bellmans(self, spent):
    reward = QLearner.scoreFunc(self.paintings, self.lastPaintings, spent)
    futureQ = np.max(self.getQ(self.state))
    oldQ = self.getQ(self.lastState)[self.lastBid]
    
    Qnew = (1 - self.lr) * oldQ + self.lr * (reward + self.discfactor * futureQ)
    #print(oldQ, futureQ, Qnew)
    self.getQ(self.lastState)[self.lastBid] = Qnew

  def saveQtable(self, path="qtable.json"):
    # Convert tuple keys to strings since JSON only allows string keys
    serialised = {str(k): v.tolist() for k, v in self.Qtable.items()}
    with open(path, "w+") as f:
      json.dump(serialised, f)

  def loadQtable(self, path="qtable.json"):
    if not os.path.exists(path):
      with open(path, "w+") as f:
        json.dump({}, f)
        self.Qtable = {}
        return
      
    with open(path, "r") as f:
      serialised = json.load(f)
    self.Qtable = {eval(k): np.array(v) for k, v in serialised.items()}