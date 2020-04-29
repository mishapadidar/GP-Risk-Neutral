# optimization strategy
#
#

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


# choose a random point
class RandomStrategy():

  def __init__(self,lb,ub):
    self.lb = lb;
    self.ub = ub;

  def generate_evals(self,surrogate):
    return np.random.uniform(self.lb,self.ub)

# expected improvement
class EIStrategy():

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
    next_pt = np.array([candidates[iopt]])
    return next_pt



# probability of improvement
# default 1 percent improvement
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

  # maximize probaility of improvement by randomly sampling it
  # def generate_evals(self,surrogate):
  #   # candidate points
  #   candidates = np.random.uniform(self.lb,self.ub,(20,surrogate.dim))
  #   vals       = np.zeros(20);
  #   for i in range(self.num_multistart):
  #     args = [surrogate]
  #     # evaluate the points
  #     vals[i] = self.POI_objective(candidates[i],args)
  #   iopt    = np.argmax(vals);
  #   next_pt = np.array([candidates[iopt]])
  #   return next_pt


    
