
"""
Plot the CVaR function
"""
import sys
sys.path.insert(1, '../')

import numpy as np
from bayesopt import BayesianOptimization

from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF
from surrogate import CVaR

from strategy import SRBFStrategy
from experimental_design import SymmetricLatinHypercube as SLHC
import matplotlib.pyplot as plt


#=============================================================
# Set up the problem
#=============================================================


f = lambda x: -np.exp(-100*(x-0.8)**2)+np.exp(-2*(x-1)**2)+np.exp(-2*(x+1.5)**2)

# basic info
dim        = 1
N = 300
a = -1.5
b = 1.5
X = np.linspace(a,b,N)
fX = f(X)

# probability of U
mu         = 0.0
sigma      = np.sqrt(0.01)
p = lambda num_pts: np.random.normal(mu, sigma, (num_pts,dim))
num_points_MC = 5000
beta = 0.95

# surrogate
kernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(1, (0.01, 100)) + \
    WhiteKernel(1e-3, (1e-6, 1e-2))
surrogate = CVaR(kernel, p, beta=beta,num_points_MC=num_points_MC)

# fit the surrogate 
surrogate.fit(X.reshape((N,dim)),fX)



#=============================================================
# Predict the CVaR
#=============================================================

# test points
Ntest = 100
Xtest = np.linspace(a,b,Ntest)

# predict the surrogate
ftest = surrogate.predict(Xtest.reshape((Ntest,dim)), std=False)
plt.plot(Xtest,ftest,linewidth=3,color='orange',label='CVaR')

# plot GP
plt.plot(Xtest,surrogate.GP.predict(Xtest.reshape((Ntest,1))),linewidth=2, color='green',label='GP')

# plot true function
plt.plot(Xtest,f(Xtest),linewidth=3, color='b',label='function')

# plot data
#plt.scatter(X[:II].flatten(),fX[:II],color='k', label='data')
plt.title('CVaR')
plt.legend()
plt.show()
