"""
This example yields a Monte-Carlo calculation with f being the surrogate and should serve as a first step of understanding of how to include the Monte-Carlo in the full Bayesian optimization
"""
import sys
sys.path.insert(1, '../')
import numpy as np
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF
from surrogate import GaussianProcessRegressor
import matplotlib.pyplot as plt
from MonteCarlo1D_fun import MonteCarlo1D_surrogate

#=============================================================
# Setting
#=============================================================
f = lambda x: -np.exp(-100*(x-0.8)**2)+np.exp(-2*(x-1)**2)+np.exp(-2*(x+1.5)**2)
N           = 35
dim         = 1
lb          = -1.5*np.ones(dim)
ub          = 1.5*np.ones(dim)
num_pts_MC  = 100
X           = np.linspace(lb,ub,N).reshape((N,dim))
fX          = f(X).flatten();

kernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(1, (0.1, 100)) + \
    WhiteKernel(1e-3, (1e-6, 1e-2))
surrogate = GaussianProcessRegressor(kernel =kernel,n_restarts_optimizer=3)
surrogate.fit(X,fX)

#=============================================================
# Compute Monte-Carlo evaluations
#=============================================================
eta   = 1;
Ntest = 100;
Xtest = np.linspace(lb,ub,Ntest).reshape((Ntest,dim))

# Monte-Carlo on surrogate
ev,va = MonteCarlo1D_surrogate(surrogate,Xtest,Ntest,num_pts_MC)

#=============================================================
# Plot
#=============================================================
# plot true function
ftrue = np.array([f(x) for x in Xtest])
plt.plot(Xtest.flatten(),ftrue,linewidth=2, color='b',label='function')

# plot risk-neutral function
plt.plot(Xtest.flatten(),ev,linewidth=2, color='r',label='risk-neutral')

# plot risk-averse function
plt.plot(Xtest.flatten(),ev+eta*va,linewidth=2, color='orange',label='risk-averse')

# plot GP
m = surrogate.predict(Xtest.reshape((Ntest,dim)), False)
m = m.flatten()
plt.plot(Xtest.flatten(),m,linewidth=2, color='k',label='GP')

# plot figure
plt.title('Monte-Carlo on surrogate')
plt.legend()
plt.show()
