# risk-neutral GP example
# gaussian kernel

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process import GaussianProcessRegressor
"""
The sklearn GPRegressor uses the kernel
k(x,y) = a*np.exp(-gamma*||x-y||^2)

a is the 'k1__constant_value' extracted below and 
gamma is the 'k2__length_scale'
"""

class GPRiskNeutral():
  
  def __init__(self,Sigma):
    self.Sigma = Sigma   # gaussian perturbation covariance
    self.dim   = len(Sigma[0])  # input data dimension
    self.X     = None; # data points
    self.fX    = None; # function evals
    self.gp    = None; # sklearn gp
    self.gamma1 = 1.0; # hyperparam: coefficient
    self.gamma2 = 1.0; # hyperparam: length scale
    self.L      = None; # cholesky factorization of rbf kernel matrix
    

  # def fit(self,X,fX):
  #   # fit a sklearn GP to the data
  #   gp       = GaussianProcessRegressor().fit(X,fX)
  #   self.X   = X;
  #   self.fX  = fX;
  #   self.gp  = gp;

  #   # save the trained hyperparameters 
  #   params        = self.gp.kernel_.get_params()
  #   self.gamma1   = params['k1__constant_value'] # rbf coefficient
  #   self.gamma2   = params['k2__length_scale']  # rbf length scale

  # "fit" a GP to the data
  def fit(self, X,fX):
    # update data
    self.X  = X;
    self.fX = fX;

    # manually optimize hyperparams
    self.gamma1 = 1.0;
    self.gamma2 = len(self.fX)/self.dim**2 # proxy for choosing hyperparameter
    # nugget
    nugget = 1e-8*np.eye(len(self.fX))
    # kernel matrix
    Phi_XX = self.gamma1*rbf_kernel(self.X,self.X,self.gamma2) + nugget
    # cholesky factorization
    self.L = np.linalg.cholesky(Phi_XX)


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
    self.fit(self.X,self.fX);

    # extract cholesky 
    #L       = self.gp.L_;
    
    # compute the convolution kernels
    A          = 1/(2*self.gamma2)*np.eye(self.dim);
    ASigma     = A + self.Sigma
    A2Sigma    = A + 2*self.Sigma
    detA       = np.linalg.det(A);        # det(A)
    detASigma  = np.linalg.det(ASigma);   # det(A + Sigma)
    detA2Sigma = np.linalg.det(A2Sigma);  # det(A + 2*Sigma)
    invASigma  =np.linalg.inv(ASigma);    # inv(A + Sigma)
    invA2Sigma =np.linalg.inv(A2Sigma);   # inv(A + 2Sigma)
    # convolution gaussian kernel function
    conv1 = lambda i,j: np.exp(-0.5*(self.X[i]-xx[j])@invASigma@(self.X[i]-xx[j]));
    conv2 = lambda i,j: np.exp(-0.5*(xx[i]-xx[j])@invA2Sigma@(xx[i]-xx[j]));
    # vectorize the function
    g1 = np.vectorize(conv1)
    g2 = np.vectorize(conv2)
    # create the kernel matrix
    Psi_Xx    = np.fromfunction(g1,(len(self.X),len(xx)),dtype=int);
    Psihat_xx = np.fromfunction(g2,(len(xx),len(xx)),dtype=int);
    # multiply by constant
    Psi_Xx     = self.gamma1*Psi_Xx*np.sqrt(detA/detASigma) # single convolutional kernel
    Psihat_xx  = self.gamma1*Psihat_xx*np.sqrt(detA/detA2Sigma) # double convolutional kernel

    # compute the predictive mean and covariance
    m = Psi_Xx.T @ np.linalg.solve(self.L.T,np.linalg.solve(self.L,self.fX));
    K = Psihat_xx - Psi_Xx.T @ np.linalg.solve(self.L.T,np.linalg.solve(self.L, Psi_Xx))

    if std is False:
      # return the mean
      return m
    else:
      # return mean and standard error
      return m, np.sqrt(np.diag(K))

  # update gp with new points
  def update(self, xx,yy):
    self.X = np.vstack((self.X,xx))
    self.fX = np.concatenate((self.fX,[yy]))
    self.fit(self.X,self.fX)
