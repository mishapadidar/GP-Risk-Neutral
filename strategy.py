# optimization strategy
#
#

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


class RandomStrategy():
# choose a random point

  def __init__(self,lb,ub):
    self.lb = lb;
    self.ub = ub;

  def generate_evals(self,surrogate):
    return np.random.uniform(self.lb,self.ub)



class EIStrategy():
# expected improvement

  def __init__(self,lb,ub):
    self.lb = lb;
    self.ub = ub;
    self.num_multistart = 10;

  # equation (35) Jones 2001
  # xx: evaluation point as a 1D-array
  # This function cannot evaluate a vector of points
  # args: [surrogate]
  def objective(self,xx,args):
    # unpack surrogate from optimizer
    surrogate = args[0]
    # optimizer inputs point in wrong shape
    xx = xx.reshape((len(xx),surrogate.dim))

    # predict surrogate 
    y,s = surrogate.predict(xx,std = True)
    y   = float(y)
    s   = float(s)
    # compute min(fX)
    fmin = min(surrogate.fX)
    # define u
    u = (fmin-y)/s
    # compute Phi(u): standard normal cdf at u
    Phi = norm.cdf(u)
    # compute phi(u): standard normal pdf at u
    phi = norm.pdf(u)
    # objective
    EI  = s*(u*Phi + phi)

    return EI

  # negative expected improvement
  def negobjective(self,xx,args):
    return -self.objective(xx,args)

  # optimize the expected improvement objective
  def generate_evals(self,surrogate):
    # candidate points
    candidates = np.zeros(self.num_multistart);
    vals       = 0*candidates;
    # use multistart
    for i in range(self.num_multistart):
      # minimize expected improvement objective
      x0      = np.random.uniform(self.lb,self.ub)
      x0 = x0.reshape((len(x0),surrogate.dim))
      args    = [surrogate]
      bounds  = list(zip(self.lb,self.ub))
      # MAXIMIZE expected improvement
      sol = minimize(self.negobjective, x0, args =args, method='SLSQP',bounds =bounds)
      candidates[i] = sol.x
      vals[i]       = sol.fun
    iopt    = np.argmin(vals);
    next_pt = candidates[iopt]
    return next_pt




class POIStrategy():

  def __init__(self,lb,ub):
    self.lb = lb;
    self.ub = ub;
    self.num_multistart = 10;
    self.alpha = 0.001 # Percent improvement desired

  # equation (35) Jones 2001
  # xx: evaluation point as a 1D-array
  # This function cannot evaluate a vector of points
  # args: [surrogate]
  def objective(self,xx,args):
    # unpack surrogate from optimizer
    surrogate = args[0]
    # optimizer inputs point in wrong shape
    xx = xx.reshape((len(xx),surrogate.dim))

    # predict surrogate 
    y,s = surrogate.predict(xx,std = True)
    y   = float(y)
    s   = float(s)
    # compute min(fX)
    fmin = min(surrogate.fX)
    # improvement goal
    fgoal = (1.0-self.alpha)*fmin
    # define u
    u = (fgoal-y)/s
    # compute Phi(u): standard normal cdf at u
    POI = norm.cdf(u)
    return POI

  # negative probability of improvement
  def negobjective(self,xx,args):
    return -self.objective(xx,args)

  # maximize the probability of improvement objective
  # using multistart
  def generate_evals(self,surrogate):
    # candidate points
    candidates = np.zeros(self.num_multistart);
    vals       = 0*candidates;
    # use multistart
    for i in range(self.num_multistart):
      # set up optimizer
      x0      = np.random.uniform(self.lb,self.ub)
      x0      = x0.reshape((len(x0),surrogate.dim))
      args    = [surrogate]
      bounds  = list(zip(self.lb,self.ub))
      # MAXIMIZE probability of improvement
      sol = minimize(self.negobjective, x0, args =args, method='SLSQP',bounds =bounds)
      candidates[i] = sol.x
      vals[i]       = sol.fun
    iopt    = np.argmin(vals);
    next_pt = np.array([candidates[iopt]])
    return next_pt





class SRBFStrategy():
  """
  Global Metric SRBF strategy from

  Rommel G. Regis, Christine A. Shoemaker, (2007) A Stochastic 
  Radial Basis Function Method for the Global Optimization of
  Expensive Functions. INFORMS Journal on Computing 19(4):497-509

  cycle through weights to determine local vs global search.

  Return the next evaluation as a 2D array.
  """

  def __init__(self,lb,ub,num_candidates=10):
    self.lb = lb;
    self.ub = ub;
    self.num_candidates = num_candidates
    # for weights
    self.cycle_length = 5
    self.weights      = np.linspace(0,1,self.cycle_length)
    self.weight_index = 0 # initialize at 0


  def generate_evals(self,surrogate):
    # generate candidates
    dim  = surrogate.dim
    C    = np.random.uniform(self.lb,self.ub, (self.num_candidates, dim))
    # estimate function value
    fC   = surrogate.predict(C)
    df   = max(fC)-min(fC)
    # evaluate minimum distance from previous points
    D    = np.array([min(np.linalg.norm(c-surrogate.X,axis=1)) for c in C]).flatten()
    # largest minus smallest distance
    dD   = max(D)-min(D)
    # compute score for response surface criterion
    if df == 0.0:
      VR = np.ones(len(C))
    else:
      VR = (fC - min(fC))/df
    # compute score for distance criterion
    if dD == 0.0:
      VD = np.ones(len(C))
    else:
      VD = (max(D)-D)/dD
    # compute weighted score
    w     = self.weights[self.weight_index]
    score = w*VR + (1-w)*VD
    # choose minimizer
    iopt = np.argmin(score)
    xopt = C[iopt]
    # update weight for next time
    self.wi = (self.weight_index + 1)%self.cycle_length

    return xopt
