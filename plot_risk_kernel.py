import numpy as np;
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF
from surrogate import GaussianProcessRegressor
from surrogate import GaussianProcessRiskNeutral
from riskkernel import Normal_SEKernel
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel

"""
Plot the once mollified risk kernel analytically and with monte carlo
"""

# once mollified kernel
def psi_MonteCarlo(kernel,r,mu,sigma,num_pts_MC):
  N = len(r);
  z = np.array([[0]])
  U = np.random.normal(mu, sigma, num_pts_MC)
  ev = np.zeros(N)
  for i in range(N):
    fxmU = np.array([ kernel(z,np.array([[ r[i]-U[j] ]])) for j in range(num_pts_MC) ])
    ev[i] = np.mean(fxmU)
  return ev

#================================
# basic info
dim =1
N = 100
b = 5
r = np.linspace(0,b,N) # radius

# kernel 
kernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(1, (0.01, 100)) + \
    WhiteKernel(1e-3, (1e-6, 1e-2))

#================================
# plot the kernel
Z = np.array([[0.0]])
R = np.reshape(r,(N,1))
plt.plot(r,kernel(Z,R)[0],label='kernel')

#================================
# plot risk kernel psi with monte carlo
mu         = 0
var        = 0.1
sigma      = np.sqrt(var)
num_pts_MC = 1000
ev = psi_MonteCarlo(kernel,r,mu,sigma,num_pts_MC)
plt.plot(r,ev,color='g',label='psi Monte Carlo')

#================================
# plot the risk kernel
riskkernel = Normal_SEKernel(var*np.eye(dim))
psi_risk   = riskkernel.mollifiedx1(Z,R)[0]
plt.plot(r,psi_risk,'--',color='g',label='psi risk')

plt.title('Analytic versus Monte Carlo Risk Kernel')
plt.legend()
plt.show()


