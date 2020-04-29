# risk-neutral GP example
# gaussian kernel

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from kernel import RBFKernel
from scipy.optimize import minimize


class GaussianProcessRegressor():
  
  def __init__(self):
    self.dim    = 0;    # input data dimension
    self.X      = None; # data points
    self.fX     = None; # function evals
    self.N      = 0     # number of training points
    self.L      = None; # cholesky factorization of rbf kernel matrix
    self.num_multistart = 3; # for optimizing hyperparameters
    self.kernel = RBFKernel(); # kernel function


  # "fit" a GP to the data
  def fit(self, X,fX):
    # update data
    self.X  = X;
    self.fX = fX;
    self.dim = X.shape[1]
    self.N  = len(fX)

    # optimize hyperparameters
    #self.kernel.hyperparams[0] = 1.0/len(self.fX)
    self.train()

    # kernel matrix
    K = self.kernel.eval(X)

    # cholesky factorization
    self.L = np.linalg.cholesky(K)


  def predict(self, xx, std = False):
    # ensure GP is trained
    #self.fit(self.X,self.fX);

    # compute kernel at new points
    K_Xx = self.kernel.eval(self.X,xx)
    K_xx = self.kernel.eval(xx)

    # compute the predictive mean and covariance
    m = K_Xx.T @ np.linalg.solve(self.L.T,np.linalg.solve(self.L,self.fX));
    K = K_xx - K_Xx.T @ np.linalg.solve(self.L.T,np.linalg.solve(self.L, K_Xx))

    if std is False:
      # return the mean
      return m
    else:
      # return mean and standard error
      return m, np.sqrt(np.diag(K))



  def update(self, xx,yy):
    """  update gp with new points
    """
    self.X = np.vstack((self.X,xx))
    self.fX = np.concatenate((self.fX,[yy]))
    self.fit(self.X,self.fX)


  def likelihood(self, hyperparams):
    """ log marginal likelihood function;
    hyperparams: vector of hyperparametes for kernel
    """

    # kernel matrix   
    K = self.kernel.eval(self.X, None, hyperparams)
    # cholesky factorization
    L = np.linalg.cholesky(K)
    _, logdet  = np.linalg.slogdet(L) # more stable than det
    complexity = 2*logdet
    fit = self.fX @ np.linalg.solve(L.T,np.linalg.solve(L,self.fX));
    #fit = self.fX @ np.linalg.inv(K) @ self.fX
    return -0.5*(complexity+fit) 


  def gradlikelihood(self,hyperparams):
    """gradient of likelihood wrt hyperparams 
    """
        # kernel matrix   
    K = self.kernel.eval(self.X, None, hyperparams)
    # cholesky factorization
    L = np.linalg.cholesky(K)

    # kernel derivative
    Dk = self.kernel.deriv(self.X, None,hyperparams)

    # gradient vector storage
    grad = np.zeros(self.kernel.num_hyperparams)

    # derivative for each hyper
    for i in range(self.kernel.num_hyperparams):
      # kernel deriv for hyper
      Dk_i = Dk[i]
      # M = K^{-1}*dK/dh
      M = np.linalg.solve(L.T,np.linalg.solve(L,Dk_i));
      # trace term
      Tr = 0.5*np.trace(M)
      # quadratic term
      Q  = 0.5*self.fX @ Dk_i @ M @ self.fX
      grad[i] = Tr + Q
      
    return grad
  

  def neglikelihood(self,hyperparams):
    # NEGATIVE log marginal likelihood function;
    return -self.likelihood(hyperparams)


  def gradneglikelihood(self,hyperparams):
    # gradient of negative likelihood
    return -self.gradlikelihood(hyperparams)



  def train(self):
    """ optimize hyperparameters via 
    maximizing log marginal likelihood
    """
    #print("Tuning Hyperparams")

    # current hyperparams
    best = self.kernel.hyperparams
    val  = self.likelihood(best)

    # hyperparam bounds
    bounds  = self.kernel.bounds    

    # use multistart
    II = 0
    while II < self.num_multistart:
      # initial guess
      if II == 0:
        x0 = best
      else:
        #x0 = np.random.beta(1,6,self.kernel.num_hyperparams) # support on [0,1]
        #x0 = bounds[:,0] + (bounds[:,1]-bounds[:,0])*x0
        x0      = np.random.uniform(bounds[:,0],bounds[:,1],self.kernel.num_hyperparams)
      
      # MAXIMIZE likelihood
      # TNC == Newton-Conjugate Gradients
      sol = minimize(self.neglikelihood, x0, method='TNC',jac = self.gradneglikelihood, bounds =bounds) 
      #print(sol.x, sol.fun)  
      # replace best hypers
      if sol.fun < val:
        best = sol.x
        val  = sol.fun;
      II +=1
      #print(sol)
      #print(sol.x)
      #print(sol.fun)
      #print('')

    # save optimized hypers
    self.kernel.hyperparams = best;
    #print('')
    print(best)
    #print('done')






class GPRiskNeutral():
  
  def __init__(self,Sigma):
    self.Sigma = Sigma   # gaussian perturbation covariance
    self.dim   = len(Sigma[0])  # input data dimension
    self.X     = None; # data points
    self.fX    = None; # function evals
    self.GP    = GaussianProcessRegressor(); # gaussian process
    

  # "fit" a GP to the data
  def fit(self, X,fX):
    # update data
    self.X  = X;
    self.fX = fX;
    self.GP.fit(X,fX)

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
    
    # compute the convolution kernels
    A          = (0.5*self.GP.kernel.hyperparams[0]**2)*np.eye(self.dim);
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
    Psihat_xx = np.fromfunction(g2,(len(xx),len(xx)),dtype=int) 
    # multiply by constant
    Psi_Xx     = Psi_Xx*np.sqrt(detA/detASigma) # single convolutional kernel
    Psihat_xx  = Psihat_xx*np.sqrt(detA/detA2Sigma) + self.GP.kernel.hyperparams[1]**2*np.eye(len(xx)); # double convolutional kernel

    # compute the predictive mean and covariance
    m = Psi_Xx.T @ np.linalg.solve(self.GP.L.T,np.linalg.solve(self.GP.L,self.fX));
    K = Psihat_xx - Psi_Xx.T @ np.linalg.solve(self.GP.L.T,np.linalg.solve(self.GP.L, Psi_Xx))

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
    self.GP.update(xx,yy)




