
"""
sequentially plot expected improvement

Plot the expected improvement, Risk-Neutral Surrogate
and function.

The variable II between [1,max_evals] allows you to look
at what the GP and expected improvement looked like
after the first II points were evaluated.

You can also change the strategy in the Bayesian Optimization
Section. There are currently three options: randomStrategy,
EIStrategy, POIStrategy (random sampling, expected improvement
probability of improvement)
"""
import numpy as np
from bayesopt import BayesianOptimization

from riskkernel import Normal_SEKernel
from surrogate import GaussianProcessRiskNeutral

from strategy import RandomStrategy, EIStrategy, POIStrategy
from experimental_design import SymmetricLatinHypercube as SLHC
import matplotlib.pyplot as plt


# number of points used in plots
# use II < max_evals
II = 55

#=============================================================
# Run Bayesian Optimization
#=============================================================


f = lambda x: -np.exp(-100*(x-0.8)**2)+np.exp(-2*(x-1)**2)+np.exp(-2*(x+1.5)**2)
#f = lambda x: np.exp(-(x-0.5)**2)*np.sin(30*x)
dim        = 1
max_evals  = 60
Sigma      = 0.001*np.eye(dim)
lb         = -1.5*np.ones(dim)
ub         = 1.5*np.ones(dim)
num_pts    = 10*dim + 1 # initial evaluations
exp_design = SLHC(dim, num_pts)
#strategy   = POIStrategy(lb,ub)
strategy   = RandomStrategy(lb,ub)
#strategy    = EIStrategy(lb,ub)
kernel     = Normal_SEKernel(Sigma)
surrogate  = GaussianProcessRiskNeutral(kernel)

# initialize the problem
problem    = BayesianOptimization(f,dim, max_evals, exp_design, strategy, surrogate,lb, ub)
# solve it
xopt,fopt  = problem.minimize()


#=============================================================
# Plot
#=============================================================

# get the function evaluations
X  = problem.X
fX = problem.fX
Ntest = 300
Xtest = np.linspace(lb,ub,Ntest).reshape((Ntest,dim))

# fit the surrogate to the first II points
surrogate.fit(X[:II],fX[:II])

# predict the risk neutral and standard error
ftest, std = surrogate.predict(Xtest, std=True)


#compute aquisition function
# args    = [surrogate]
# acquisition = []
# for x in Xtest:
#   acquisition.append(strategy.objective(x,args))

# # plot acquisition function
# plt.plot(Xtest.flatten(),acquisition,color='red',label='acquisition function')

# plot Next Evaluation
plt.scatter(X[II+1],fX[II+1],color='red',s = 150,marker=(5,1), label='Next Evaluation')

# plot gp
plt.plot(Xtest.flatten(),surrogate.GP.predict(Xtest),color='green',label='GP')

# plot surrogate
plt.plot(Xtest.flatten(),ftest,linewidth=3, color='orange',label='GPRN')
# plot the 95% confidence interval
plt.fill_between(Xtest.flatten(),ftest-1.96*std,ftest+1.96*std,alpha=0.3)

# plot true function
ftrue = np.array([f(x) for x in Xtest])
plt.plot(Xtest.flatten(),ftrue,linewidth=3, color='b',label='function')

# plot data
plt.scatter(X[:II].flatten(),fX[:II],color='k', label='data')
plt.title('Bayesian Optimization Under Uncertainty')
plt.legend()
plt.show()
