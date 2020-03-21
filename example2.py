
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
from experimental_design import SymmetricLatinHypercube as SLHC
from optimization import RiskNeutralOptimization
from surrogate import GPRiskNeutral
from strategy import randomStrategy, EIStrategy, POIStrategy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel


# number of points used in plots
II = 45

#=============================================================
# Run Bayesian Optimization
#=============================================================


f = lambda x: -np.exp(-100*(x-0.8)**2)+np.exp(-2*(x-1)**2)+np.exp(-2*(x+1.5)**2)
#f = lambda x: (x-1)**4 + 2*(x-1)**3 + 0.01*(x-2)**2 - 1.5*x
dim        = 1
Sigma      = 0.01*np.eye(dim)
max_evals  = 50
lb         = -1.5*np.ones(dim)
ub         = 2.5*np.ones(dim)
num_pts    = 4*dim + 1
exp_design = SLHC(dim, num_pts)
#strategy   = randomSample(lb,ub)
strategy   = POIStrategy(lb,ub)

# initialize the problem
problem    = RiskNeutralOptimization(f, dim, Sigma, max_evals, exp_design, strategy, lb, ub)
# solve it
xopt,fopt  = problem.optimize()


#=============================================================
# Plot
#=============================================================


# get the function evaluations
X  = problem.X
fX = problem.fX
fX_rn = problem.fX_rn
Ntest = 300
Xtest = np.linspace(lb,ub,Ntest).reshape((Ntest,dim))

# fit the surrogate to the first II points
GP = GPRiskNeutral(Sigma)
GP.fit(X[:II],fX[:II])

# predict the risk neutral and standard error
ftest, std = GP.predict(Xtest, std=True)

# compute expected improvement
strategy   = EIStrategy(lb,ub)
args       = [GP]
ei = []
for x in Xtest:
  ei.append(strategy.EI_objective(x,args))

# compute probability improvement
strategy   = POIStrategy(lb,ub)
args       = [GP]
poi = []
for x in Xtest:
  poi.append(strategy.POI_objective(x,args))


# plot expected improvement
#plt.plot(Xtest.flatten(),ei,color='r',label='expected improvement')

# plot probability of improvement
plt.plot(Xtest.flatten(),poi,color='red',label='probabilty of improvement')

# plot Next Evaluation
plt.scatter(X[II+1],fX[II+1],color='red',s = 150,marker=(5,1), label='Next Evaluation')

# plot risk-neutral GP
plt.plot(Xtest.flatten(),ftest,linewidth=3, color='orange',label='Risk-Neutral')
# plot the 95% confidence interval
plt.fill_between(Xtest.flatten(),ftest-1.96*std,ftest+1.96*std,alpha=0.3)

# plot true function
ftrue = np.array([f(x) for x in Xtest])
plt.plot(Xtest.flatten(),ftrue,linewidth=3, color='b',label='function')

# plot data
plt.scatter(problem.X[:II].flatten(),problem.fX[:II],color='k', label='data')
plt.title('Bayesian Optimization: Risk-Neutral Probability of Improvement')
plt.legend()
plt.show()
