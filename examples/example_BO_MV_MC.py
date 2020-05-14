"""
Example file for computing the risk-neutral with the Monte-Carlo approximation as another checkpoint

Strategy used is SRBF_MC_MV
"""
import sys
sys.path.insert(1, '../')

import numpy as np
from bayesopt import BayesianOptimization

from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF
from riskkernel import Normal_SEKernel
from surrogate import GaussianProcessRegressor

from MonteCarlo_strategies import MonteCarlo_MV
from strategy import SRBFStrategy, SRBFStrategy_MC_MV
from experimental_design import SymmetricLatinHypercube as SLHC
import matplotlib.pyplot as plt


#=============================================================
# Run Bayesian Optimization
#=============================================================


f = lambda x: -np.exp(-100*(x-0.8)**2)+np.exp(-2*(x-1)**2)+np.exp(-2*(x+1.5)**2)
#f = lambda x: np.exp(-(x-0.5)**2)*np.sin(30*x)
dim        = 1
max_evals  = 60
Sigma      = 0.01*np.eye(dim)
lb         = -1.5*np.ones(dim)
ub         = 1.5*np.ones(dim)
num_pts    = 15*dim + 1 # initial evaluations
exp_design = SLHC(dim, num_pts)
strategy   = SRBFStrategy_MC_MV(lb,ub)
kernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(1, (0.1, 100)) + \
    WhiteKernel(1e-3, (1e-6, 1e-2))
surrogate = GaussianProcessRegressor(kernel =kernel)

# initialize the problem
problem    = BayesianOptimization(f,dim, max_evals, exp_design, strategy, surrogate,lb, ub)
# solve it
xopt,fopt  = problem.minimize()


#=============================================================
# Plot
#=============================================================
Ntest       = 100;
num_pts_MC  = 100;
eta         = 1;
#Compute MC for plot
Xtest = np.linspace(lb,ub,Ntest).reshape((Ntest,dim))

# Monte-Carlo on surrogate
mv = MonteCarlo_MV(surrogate,Xtest,num_pts_MC, eta)

# plot true function
ftrue = np.array([f(x) for x in Xtest])
plt.plot(Xtest.flatten(),ftrue,linewidth=2, color='b',label='function')

# plot mean-variance function
plt.plot(Xtest.flatten(),mv,linewidth=2, color='r',label='MC mean-variance')

# plot GP
m = surrogate.predict(Xtest.reshape((Ntest,dim)), False)
m = m.flatten()
plt.plot(Xtest.flatten(),m,linewidth=2, color='orange',label='GP')

# plot figure
plt.title('Monte-Carlo for BO mean-variance')
plt.legend()
plt.show()

