import numpy as np;
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF
from surrogate import GaussianProcessRegressor
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel

"""
Verify the fit of a GP. 
Plot the Likelihood function.
"""
import sys
sys.path.insert(1, '../')

dim =1
N = 100
a = -1
b = 1
X = np.random.uniform(a,b,(N,dim))
f = lambda x: np.exp(-(x-0.5)**2)*np.sin(30*x)
#f = lambda x: -np.exp(-100*(x-0.8)**2)+np.exp(-2*(x-1)**2)+np.exp(-2*(x+1.5)**2)
fX = f(X).flatten();

kernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(1, (0.1, 100)) + \
    WhiteKernel(1e-3, (1e-6, 1e-2))
surrogate = GaussianProcessRegressor(kernel =kernel,n_restarts_optimizer=3)

#================================
# Plot a GP


surrogate.fit(X,fX)
M = 400
xx = np.linspace(a,b,M)
m,v = surrogate.predict(xx.reshape((M,dim)), True)
m = m.flatten()
v = v.flatten()
plt.plot(xx,m,label='GP')
plt.fill_between(xx,m,m+v,color='b',alpha=0.5)
plt.fill_between(xx,m,m-v,color='b',alpha=0.5)
plt.scatter(X.flatten(),fX,color='k',label='data')
plt.legend()
plt.show()


