
"""
Test EI strategy in 2 dimensions
"""
import sys
sys.path.insert(1, '../')

import numpy as np
from bayesopt import BayesianOptimization

from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF
from riskkernel import Normal_SEKernel
from surrogate import GaussianProcessRegressor

from strategy import RandomStrategy, EIStrategy, POIStrategy, SRBFStrategy
from experimental_design import SymmetricLatinHypercube as SLHC
import matplotlib.pyplot as plt


#=============================================================
# Run Bayesian Optimization
#=============================================================


# function f: R^n -> R
f = lambda x: np.sqrt((x[0]-0.382)**2 + x[1]**2)
g = lambda x,y: np.sqrt((x-0.382)**2 + y**2)
#f = lambda x: -np.exp(-100*(x-0.8)**2)+np.exp(-2*(x-1)**2)+np.exp(-2*(x+1.5)**2)

# basic info
dim        = 2
lb         = -np.ones(dim)
ub         = np.ones(dim)
max_evals  = 50

# experimental design
num_pts    = 12*dim + 1 # initial evaluations
exp_design = SLHC(dim, num_pts)

# strategy
strategy    = POIStrategy(lb,ub)
kernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(1, (0.01, 100)) + \
    WhiteKernel(1e-3, (1e-6, 1e-2))
surrogate = GaussianProcessRegressor(kernel =kernel)

# initialize the problem
problem    = BayesianOptimization(f,dim, max_evals, exp_design, strategy, surrogate,lb, ub)
# solve it
xopt,fopt  = problem.minimize()


#=============================================================
# Plot
#=============================================================

Ntest = 300
x = np.linspace(lb[0],ub[0],Ntest)
y = np.linspace(lb[1],ub[1],Ntest)
X,Y = np.meshgrid(x,y)


# plot function
plt.contour(X,Y,g(X,Y))
#plt.plot(np.linspace(-1,1,Ntest),f(np.linspace(-1,1,Ntest)))

# plot optimum
plt.scatter(xopt[0],xopt[1],color='red',s = 150,marker=(5,1), label='Optimum')

plt.title('Bayesian Optimization')
plt.legend()
plt.show()
