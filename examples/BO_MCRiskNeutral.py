"""
Perform Bayesian Optimization with the Risk Neutral
surrogate computed via Monte Carlo on a GP for the 
function.
"""
import sys
sys.path.insert(1, '../')

import numpy as np
from bayesopt import BayesianOptimization

from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF
from surrogate import MCRiskNeutral

from strategy import SRBFStrategy
from experimental_design import SymmetricLatinHypercube as SLHC
import matplotlib.pyplot as plt


#=============================================================
# Run Bayesian Optimization
#=============================================================


# basic info
f = lambda x: -np.exp(-100*(x-0.8)**2)+np.exp(-2*(x-1)**2)+np.exp(-2*(x+1.5)**2)
dim        = 1
max_evals  = 80
Sigma      = 0.01*np.eye(dim)
lb         = -1.5*np.ones(dim)
ub         = 1.5*np.ones(dim)

# experimental design
num_pts    = 15*dim + 1 # initial evaluations
exp_design = SLHC(dim, num_pts)

# strategy
strategy   = SRBFStrategy(lb,ub)

# uncertainty information 
mu         = 0.0
sigma      = np.sqrt(0.01)
p = lambda num_pts: np.random.normal(mu, sigma, (num_pts,dim))

# Monte Carlo parameters
num_points_MC = 5000

# surrogate
kernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(1, (0.1, 100)) + \
    WhiteKernel(1e-3, (1e-6, 1e-2))
surrogate = MCRiskNeutral(kernel, p, num_points_MC)

# initialize the problem
problem    = BayesianOptimization(f,dim, max_evals, exp_design, strategy, surrogate,lb, ub)
# solve it
xopt,fopt  = problem.minimize()


#=============================================================
# Plot
#=============================================================

# test points
Ntest       = 300;
Xtest = np.linspace(lb,ub,Ntest).reshape((Ntest,dim))

# plot final Mean-Variance surrogate
plt.plot(Xtest.flatten(),surrogate.predict(Xtest),linewidth=2, color='r',label='MC Risk-Neutral')

# plot true function
plt.plot(Xtest.flatten(),f(Xtest.flatten()),linewidth=2, color='b',label='function')

# plot GP
plt.plot(Xtest.flatten(),surrogate.GP.predict(Xtest),linewidth=2, color='orange',label='GP')

# plot solution to Mean Variance Bayesian Optimization
plt.scatter(xopt,fopt,color='red',s = 150,marker=(5,1),label='Best Evaluation')

# plot figure
plt.title('Monte Carlo Risk-Neutral Surrogate')
plt.legend()
plt.show()

