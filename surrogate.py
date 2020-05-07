# risk-neutral GP example
# gaussian kernel

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from scipy.optimize import minimize


class GaussianProcessRegressor():
  """Wrapper on sklearn GaussianProcessRegressor
  """
  
  def __init__(self,kernel,n_restarts_optimizer=0):
    self.dim    = 0;    # input data dimension
    self.X      = None; # data points
    self.fX     = None; # function evals
    #self.kernel = kernel; # kernel function
    #self.n_restarts_optimizer = n_restarts_optimizer
    self.GP     = GPR(kernel=kernel,n_restarts_optimizer=n_restarts_optimizer)


  # "fit" a GP to the data
  def fit(self, X,fX):
    # update data
    self.X  = X;
    self.fX = fX;
    self.dim = X.shape[1]

    # fit sklearn GP
    self.GP.fit(X,fX)


  def predict(self, xx, std = False):
    # ensure GP is trained
    #self.fit(self.X,self.fX);

    return self.GP.predict(xx, return_std = std)



  def update(self, xx,yy):
    """  update gp with new points
    """
    self.X = np.vstack((self.X,xx))
    self.fX = np.concatenate((self.fX,[yy]))
    self.fit(self.X,self.fX)




class GaussianProcessRiskNeutral():
  
  def __init__(self,RiskKernel):
    self.dim   = 0;  # input data dimension
    self.X     = None; # data points
    self.fX    = None; # function evals
    self.RiskKernel = RiskKernel;
    self.GP    = GPR(kernel=RiskKernel.GPkernel)
    

  # "fit" a GP to the data
  def fit(self, X,fX):
    # update data
    self.X  = X;
    self.fX = fX;
    self.dim = X.shape[1]
    # fit the GP
    self.GP.fit(X,fX)
    # update the kernel with the tuned hyperparams
    self.RiskKernel.updatekernel(self.GP.kernel_)


  """
  Predict by:
  1. Fit a GP to the data (X,fX)
  2. Compute the single and double convolutional kernel 
      matrices. 
  3. Predict by computing the posterior GP fhat|fX, the 
      risk-neutral GP conditioned on the GP on f. 

  xx: 2d np.array; prediction points
  std: Bool; return standard error if True
  """

  def predict(self, xx, std = False):
    # ensure GP is trained
    #self.fit(self.X,self.fX);

    # once mollified kernel matrix
    Psi    = self.RiskKernel.mollifiedx1(self.X, xx)
    # twice mollified kernel matrix
    Psihat = self.RiskKernel.mollifiedx2(xx)
    
    # compute the predictive mean and covariance
    m = Psi.T @ np.linalg.solve(self.GP.L_.T,np.linalg.solve(self.GP.L_,self.fX));
    K = Psihat - Psi.T @ np.linalg.solve(self.GP.L_.T,np.linalg.solve(self.GP.L_, Psi))

    if std is False:
      # return the mean
      return m
    else:
      # return mean and standard error
      v = np.sqrt(np.diag(K))
      return m, v

  # update gp with new points
  def update(self, xx,yy):
    self.X  = np.vstack((self.X,xx))
    self.fX = np.concatenate((self.fX,[yy]))
    self.fit(self.X, self.fX)
