import numpy as np
import matplotlib.pyplot as plt

class QLearner:
  def __init__(self, nStates, nActions, goalS, lr = 0.8, discfactor = 0.95, epochs = 1000):
    self.nStates, self.nActions, self.goalS, self.lr, self.discfactor, self.epochs = nStates, nActions, goalS, lr, discfactor, epochs
    
    self.state = [0, 0, 0, 0, "high"] # (P, G, R, V, budget bin)
    self.lastState = [0, 0, 0, 0, "high"]
    self.Qvalue = 0
    self.Qtable = {}  # key: (p1, p2, p3, p4, budget_bin), value: np.zeros(nActions)
    self.lastBid = 0
    self.bid = 0
     # {0,0.02,0.05,0.1,0.15,0.2,0.3,0.5,0.8,1.0} <-- ratios of budget


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

  def selectAction(self, bayesianBid, budget, threshold):
    bidAmounts = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]
    
    # Convert raw bid to fraction of budget
    # Under the assumption that maybe the bayesian bid will always be the same, we introduce randomness via a threshold value.
    # If the random value exceeds the threshold, we select a bid bin at random and use that.
    # This ensures the Q Learner gains sufficient experience across all actions.

    if np.random.random() < threshold:
      idx = np.random.randint(len(bidAmounts))
    else:
      if budget > 0:
        fraction = bayesianBid / budget
      else:
        fraction = 0

      # Snap to nearest bin
      idx = np.argmin([abs(fraction - b) for b in bidAmounts])
    
    self.lastBid = idx
    return int(bidAmounts[idx] * budget)

  def getQ(self, state):
    state = tuple(state)
    if state not in self.Qtable:
      self.Qtable[state] = np.zeros(self.nActions)
    return self.Qtable[state]

  def transition(self, painterWon, budgetbin):
    self.lastState = self.state.copy()
    match(painterWon):
      case "Picasso": self.state[0] += 1
      case "Van Gogh": self.state[1] += 1
      case "Rembrandt": self.state[2] += 1
      case "Da Vinci": self.state[3] += 1
      case _: self.state = self.state
    
    self.state[4] = budgetbin
    return self.state
  
  def bestBid(self):
    qvals = self.getQ(self.state)
    best = np.argmax(qvals)
    bidamounts = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]
    return bidamounts[best], qvals[best]

  def bellmans(self):
    reward = QLearner.scoreFunc(self.state, self.lastState)
    futureQ = np.max(self.getQ(self.state))
    oldQ = self.getQ(self.lastState)[self.lastBid]
    
    Qnew = (1 - self.lr) * oldQ + self.lr * (reward + self.discfactor * futureQ)
    
    self.getQ(self.lastState)[self.lastBid] = Qnew