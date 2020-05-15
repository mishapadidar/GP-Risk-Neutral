"""
Example file for Monte-Carlo 1D

Compute the Risk-Neutral and Mean-Variance surrogates by 
Monte Carlo integration on the function f
"""
import sys
sys.path.insert(1, '../')

import numpy as np
import matplotlib.pyplot as plt
from MonteCarlo1D_fun import MonteCarlo1D

#=============================================================
# Settings
#=============================================================
f = lambda x: -np.exp(-100*(x-0.8)**2)+np.exp(-2*(x-1)**2)+np.exp(-2*(x+1.5)**2)
#f = lambda x: np.exp(-(x-0.5)**2)*np.sin(30*x)
dim             = 1
lb              = -1.5*np.ones(dim)
ub              = 1.5*np.ones(dim)
num_pts_MC      = 100


#=============================================================
# Compute Monte-Carlo evaluations
#=============================================================
eta   = 1;
Ntest = 100;
Xtest = np.linspace(lb,ub,Ntest).reshape((Ntest,dim))

ev,va = MonteCarlo1D(f,Xtest,Ntest,num_pts_MC)


#=============================================================
# Plot
#=============================================================
# plot true function
ftrue = np.array([f(x) for x in Xtest])
plt.plot(Xtest.flatten(),ftrue,linewidth=3, color='b',label='function')

# plot risk-neutral function
plt.plot(Xtest.flatten(),ev,linewidth=3, color='r',label='risk-neutral')

# plot risk-averse function
plt.plot(Xtest.flatten(),ev+eta*va,linewidth=3, color='orange',label='risk-averse')

# plot figure
plt.title('Measures of Risk Calculated via Monte Carlo on function')
plt.legend()
plt.show()
