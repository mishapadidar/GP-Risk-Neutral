import numpy as np;
#from gp import GaussianProcessRegressor
from surrogate import GaussianProcessRegressor
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel

"""
Verify the fit of a GP. 
Plot the Likelihood function.
"""

dim =1
N = 10
a = -1
b = 1
X = np.random.uniform(a,b,(N,dim))
f = lambda x: np.exp(-(x-0.5)**2)*np.sin(30*x)
f = lambda x: -np.exp(-100*(x-0.8)**2)+np.exp(-2*(x-1)**2)+np.exp(-2*(x+1.5)**2)
fX = f(X).flatten();

GP = GaussianProcessRegressor()

#================================
# Plot a GP


GP.fit(X,fX)
M = 400
xx = np.linspace(a,b,M)
m,v = GP.predict(xx.reshape((M,dim)), True)
print(GP.kernel.hyperparams)
m = m.flatten()
v = v.flatten()
plt.plot(xx,m,label='GP')
plt.fill_between(xx,m,m+v,color='b',alpha=0.5)
plt.fill_between(xx,m,m-v,color='b',alpha=0.5)
plt.scatter(X.flatten(),fX,color='k',label='data')
plt.legend()
plt.show()


#====================================
# Plot the Likelihood function
#

GP.X = X;
GP.fX = fX;
theta = np.linspace(1e-2,10.0,500)

L1 = []
L2 = []
D = []
for h in theta:
 K = rbf_kernel(X,X,1.0/h**2) + (1e-2)**2*np.eye(N)
 L = np.linalg.cholesky(K)
 _, comp = np.linalg.slogdet(L)
 fit = -np.dot(fX,np.linalg.solve(L.T,np.linalg.solve(L,fX)))
 L1.append(fit)
 L2.append(2*comp)

plt.plot(theta,L1,label='data fit')
plt.plot(theta,L2,label='complexity')
plt.plot(theta,np.array(L1)-np.array(L2),label='marginal likelihood')
plt.title('Log Marginal Likelihood')
plt.legend()
plt.xscale('log')
plt.show()
