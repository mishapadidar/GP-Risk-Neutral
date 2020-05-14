# risk-neutral GP example
# gaussian kernel

import numpy as np
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from scipy.optimize import minimize
import cvxpy as cp


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




class CVaR():
  """CVaR/VaR surrogate
  p: function handle for distribution
  p takes in an integer number of points and returns a 2D array
  of vectors
  beta: confidence level for CVaR/VaR
  num_points_MC: number of points used in Monte Carlo
  """
  
  def __init__(self,kernel,p,beta = 0.95, num_points_MC = 500):
    self.dim    = 0;    # input data dimension
    self.X      = None; # data points
    self.fX     = None; # function evals
    self.GP     = GPR(kernel=kernel) # gaussian process
    self.p      = p     # pdf for U
    self.beta   = beta  # confidence level for CVaR
    self.num_points_MC = num_points_MC # number of points for monte carlo


  # "fit" a GP to the data
  def fit(self, X,fX):
    # update data
    self.X  = X;
    self.fX = fX;
    self.dim = X.shape[1]

    # fit sklearn GP
    self.GP.fit(X,fX)


  def predict(self, xx, std = False):
    """predict Ghat(x) = min_alpha G_beta(x,alpha)
    xx: 2D array of points
    std: Bool
    """
    if std == True:
      print('')
      print("ERROR: CVaR has no variance")
      quit()

    # storage
    N    = np.shape(xx)[0]
    Ghat = np.zeros(N) 

    # for each x in xx calculate Ghat(x)
    for i in range(N):
      # f(x-U) with Monte Carlo
      U    = self.p(self.num_points_MC)
      A    = self.GP.predict(xx[i]-U)

      # solve the optimization problem
      alpha = cp.Variable()
      # average the positive values
      cost = alpha + cp.sum(cp.pos(A-alpha))/(self.num_points_MC*(1.0-self.beta))
      prob = cp.Problem(cp.Minimize(cost))
      prob.solve()
      Ghat[i] = prob.value

    return Ghat



  def update(self, xx,yy):
    """  update gp with new points
    """
    self.X = np.vstack((self.X,xx))
    self.fX = np.concatenate((self.fX,[yy]))
    self.fit(self.X,self.fX)
