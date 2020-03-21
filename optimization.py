#
# risk-neutral optimization with gaussian processes
#
#

import numpy as np
from surrogate import GPRiskNeutral


"""
A risk-neutral optimization class. 



ex:

from experimental_design import SymmetricLatin Hypercube as SLHC
from strategy import randomSample
f          = lambda x: np.linalg.norm(x)
dim        = 2
Sigma      = np.eye(dim)
max_evals  = 50
lb         = -10*np.ones(dim)
ub         = 10*np.ones(dim)
num_pts    = 2*dim + 1
exp_design = SLHC(dim, num_pts)
strategy   = randomSample(lb,ub)


problem    = RiskNeutralOptimization(f, dim, Sigma, max_evals, exp_design, strategy, lb, ub)
xopt,fopt  = problem.optimize()
"""
class RiskNeutralOptimization():


  """
  f: function handle
  dim: int, problem dimension
  Sigma: 2D - covariance matrix for perturbations u
          size (dim,dim)
  max_evals: int, maximum function evaluations
  exp_des: experimental design object
           ex: SymmetricLatinHypercube(dim,num_pts)
  strategy: strategy object
  			ex: randomSample(lb, ub)
  lb: np.array(), lower-bounds
  ub: np.array(), upper-bounds
  """
  def __init__(self, f, dim, Sigma, max_evals, exp_design, strategy, lb, ub):
    self.f         = f
    self.dim       = dim
    self.Sigma     = Sigma
    self.max_evals = max_evals
    self.exp_des   = exp_design
    self.strategy  = strategy
    self.surrogate = GPRiskNeutral(self.Sigma)  # use the risk neutral GP surrogate
    self.lb        = lb
    self.ub        = ub
    self.X         = None; # data points X
    self.fX        = None; # function evals
    self.fX_rn     = None; # risk-neutral function evals


  # solve the optimization problem
  def optimize(self):

    # initial sampling
    X  = self.exp_des.generate_points(self.lb, self.ub);
    # evaluate f at X; ensure it returns number not array
    fX = np.array([float(self.f(x)) for x in X])
    evals = len(fX)
    # fit a gp with gaussian kernel
    self.surrogate.fit(X,fX)

    # start optimizing
    for i in range(self.max_evals-evals):
    	# choose next point with strategy
      xi = self.strategy.generate_evals(self.surrogate)
      # evaluate point; ensure it returns number not array
      fi = float(self.f(xi))
      # update surrogate
      self.surrogate.update(xi,fi)

    # extract points from surrogate
    X  = self.surrogate.X
    fX = self.surrogate.fX

    # predict the Risk-Neutral surrogate at all evaluated points
    fX_rn = self.surrogate.predict(X)

    # return best Risk-Neutral eval so far
    iopt = np.argmin(fX_rn)
    fopt = fX_rn[iopt]
    xopt = X[iopt]

    # save the evaluation history
    self.X     = X
    self.fX    = fX
    self.fX_rn = fX_rn

    return xopt,fopt




