"""
Run Bayesian Optimization on Risk-Neutral GP

Plot the true function and Risk-Neutral GP
"""

import numpy as np
from experimental_design import SymmetricLatinHypercube as SLHC
from bayesopt import BayesianOptimization
from surrogate import GPRiskNeutral
from strategy import RandomStrategy, EIStrategy
import matplotlib.pyplot as plt

f = lambda x: -np.exp(-100*(x-0.8)**2)+np.exp(-2*(x-1)**2)+np.exp(-2*(x+1.5)**2)
dim        = 1
Sigma      = 0.01*np.eye(dim)
max_evals  = 50
lb         = -1.5*np.ones(dim)
ub         = 1.5*np.ones(dim)
num_pts    = 12*dim + 1
exp_design = SLHC(dim, num_pts)
#strategy   = randomSample(lb,ub)
strategy   = EIStrategy(lb,ub)
surrogate  = GPRiskNeutral(Sigma)

# initialize the problem
problem    = BayesianOptimization(f, dim, max_evals, exp_design, strategy, surrogate, lb, ub)
# solve it
xopt,fopt  = problem.minimize()

#plot it
Xtest = np.linspace(lb,ub,100).reshape((100,1))
ftest = problem.surrogate.predict(Xtest)
ftrue = np.array([f(x) for x in Xtest])
plt.plot(Xtest.flatten(),ftrue,linewidth=3,color='b',label='True Function')
plt.scatter(problem.X.flatten(),problem.fX,color='k',label='Data')
plt.plot(Xtest.flatten(),ftest, linewidth=3,color='orange', label='Risk-Neutral')
plt.legend()
plt.title('Risk Neutral Bayesian Optimization')
plt.show()
