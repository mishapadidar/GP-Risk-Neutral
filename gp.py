import numpy as np
from kernel import RBFKernel
from scipy.optimize import minimize



"""
Gaussian Process class
"""
class GaussianProcessRegressor():
  
  def __init__(self):
    self.dim    = 0;    # input data dimension
    self.X      = None; # data points
    self.fX     = None; # function evals
    self.N      = 0     # number of training points
    self.L      = None; # cholesky factorization of rbf kernel matrix
    self.num_multistart = 7; # for optimizing hyperparameters
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
