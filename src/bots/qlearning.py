import numpy as np
import matplotlib.pyplot as plt
import json
import random as rand

class QLearner:
  def __init__(self, lr = 0.8, discfactor = 0.95, epochs = 1000):
    self.lr, self.discfactor, self.epochs = lr, discfactor, epochs
    self.nActions = len([0,0.02,0.05,0.1,0.15,0.2,0.3,0.5,0.8,1.0])
    self.state = [0, 0, 0, 0, "high"] # (P, G, R, V, budget bin)
    self.lastState = [0, 0, 0, 0, "high"]
    self.Qvalue = 0
    self.Qtable = {}  # key: (p1, p2, p3, p4, budget_bin), value: np.zeros(nActions)
    self.lastBid = 0
    self.bid = 0
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
  def scoreFunc(current, previous):
    score = lambda s: s[0] * s[1] * s[2] * s[3]
    return score(current) - score(previous)

  @staticmethod
  def bin(budget, bid):
    bidAmounts = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]
    idx = 0
    if budget > 0:
      fraction = bid / budget
    else:
      fraction = 0

      # Snap to nearest bin
      idx = np.argmin([abs(fraction - b) for b in bidAmounts])
    
    return idx
  
  def selectAction(self, bayesianBid, budget,  alpha = 0.7):
    bidAmounts = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]
    
    # Convert raw bid to fraction of budget
    # Under the assumption that maybe the bayesian bid will always be the same, we introduce randomness via a threshold value.
    # If the random value exceeds the threshold, we select a bid bin at random and use that.
    # This ensures the Q Learner gains sufficient experience across all actions.

    # Snap to nearest bin
    idxBayes = QLearner.bin(budget, bayesianBid)
    
    # then get the bid from the qtable:
    idxQ = np.argmax(self.getQ(self.state))

    # average the two bidding candidates, modified by gamma: 0 indicates a bayesian bid only, 1 indicates a Q bid
    Qbid, Bbid = bidAmounts[idxQ]*budget, bidAmounts[idxBayes]*budget
    bidd = alpha*Qbid + (1-alpha)*Bbid
    idxFinal = rand.choice([idxBayes, idxQ])
    bidd = bidAmounts[idxFinal] * budget

    # Snap averaged value back into bin
    self.lastBid = QLearner.bin(budget, bidd)
    return bidd

  def getQ(self, state):
    state = tuple(state)
    if state not in self.Qtable:
      self.Qtable[state] = np.zeros(self.nActions)
    return self.Qtable[state]

  def transition(self, painterWon, budgetbin):
    # for many iterations, the state space with uncapped number of paintings gets incredibly large.
    # If the number of paintings owned exceeds the cap, the state[i] will be set to the cap.
    cap = 5 

    self.lastState = self.state.copy()
    match(painterWon):
      case "Picasso": self.state[0] = min(self.state[0] + 1, cap)
      case "Van Gogh": self.state[1] = min(self.state[1] + 1, cap)
      case "Rembrandt": self.state[2] = min(self.state[2] + 1, cap)
      case "Da Vinci": self.state[3] = min(self.state[3] + 1, cap)
      case _: self.state = self.state
    
    self.state[4] = budgetbin
    return self.state
  
  def bellmans(self):
    reward = QLearner.scoreFunc(self.state, self.lastState)
    futureQ = np.max(self.getQ(self.state))
    oldQ = self.getQ(self.lastState)[self.lastBid]
    
    Qnew = (1 - self.lr) * oldQ + self.lr * (reward + self.discfactor * futureQ)
    
    self.getQ(self.lastState)[self.lastBid] = Qnew

  def saveQtable(self, path="qtable.json"):
    # Convert tuple keys to strings since JSON only allows string keys
    serialised = {str(k): v.tolist() for k, v in self.Qtable.items()}
    with open(path, "w+") as f:
      json.dump(serialised, f)

  def loadQtable(self, path="qtable.json"):
    with open(path, "r") as f:
      serialised = json.load(f)
    self.Qtable = {eval(k): np.array(v) for k, v in serialised.items()}