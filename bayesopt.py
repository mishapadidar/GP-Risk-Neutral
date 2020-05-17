import numpy as np


"""

Bayesian Optimization


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


problem    = BayesianOptimization(f, dim, Sigma, max_evals, exp_design, strategy, surrogate, lb, ub)
xopt,fopt  = problem.minimize()
"""
class BayesianOptimization():


  """
  f: R^n -> R, function handle
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
  def __init__(self, f, dim, max_evals, exp_design, strategy, surrogate, lb, ub):
    self.f         = f
    self.dim       = dim
    self.max_evals = max_evals
    self.exp_des   = exp_design
    self.strategy  = strategy
    self.surrogate = surrogate  
    self.lb        = lb
    self.ub        = ub
    self.X         = None; # data points X
    self.fX        = None; # function evals

  # solve the optimization problem
  def minimize(self):

    # initial sampling
    X  = self.exp_des.generate_points(self.lb, self.ub);
    # evaluate f at X; ensure it returns number not array
    fX = np.array([float(self.f(x)) for x in X])
    evals = len(fX)
    # fit a gp 
    self.surrogate.fit(X,fX)

    # start optimizing
    for i in range(self.max_evals-evals):
      # choose next point with strategy
      xi = self.strategy.generate_evals(self.surrogate)
      # evaluate point; ensure it returns number not array
      fi = float(self.f(xi))
      # update surrogate
      self.surrogate.update(np.atleast_2d(xi),fi)

    # save the evaluation history
    self.X  = self.surrogate.X
    self.fX = self.surrogate.fX

    # return surrogate's best evaluation (not function's)
    f_surr = self.surrogate.predict(self.X)
    iopt = np.argmin(f_surr)
    fopt = f_surr[iopt]
    xopt = self.X[iopt]

    return xopt,fopt




