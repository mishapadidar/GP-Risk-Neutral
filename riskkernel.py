import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF
import scipy.spatial.distance
from sklearn.metrics.pairwise import euclidean_distances

# kernels class
class Kernel(ABC):
    """Base class for a radial kernel.
    :ivar order: Order of the conditionally positive definite kernel
    """
    def __init__(self):  # pragma: no cover
        self.hyperparams = []

    @abstractmethod
    def updatekernel(self, kernel):  # pragma: no cover
        """Evaluate the radial kernel.
        :param dists: Array of size n x n with pairwise distances
        :type dists: numpy.ndarray
        :return: Array of size n x n with kernel values
        :rtype: numpy.ndarray
        """
        pass

    @abstractmethod
    def get_hyperparameters(self):
      """ Return the hyperparameters as a list"""
      pass

    @abstractmethod
    def mollifiedx1(self, X,Y):  # pragma: no cover
        """Once mollified kernel
        :param dists: Array of size n x n with pairwise distances
        :type dists: numpy.ndarray
        :return: Array of size n x n with kernel derivatives
        :rtype: numpy.ndarray
        """
        pass

    @abstractmethod
    def mollifiedx2(self, X):  # pragma: no cover
        """Twice mollified kernel
        :param dists: Array of size n x n with pairwise distances
        :type dists: numpy.ndarray
        :return: Array of size n x n with kernel derivatives
        :rtype: numpy.ndarray
        """
        pass



class Normal_SEKernel(Kernel):
    """ kernels for Normal pdf convolved with Squared Exponential kernel.

    :math:`\\varphi(r) = \\exp(\\-theta_2^2||x-x'||^2) + \\theta_3^2\\delta_{ii'}` which is
    positive definite.
    """
    def __init__(self, Sigma):
      super().__init__()
      # squared exponential kernel for GP
      self.GPkernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(1, (1e-3, 100)) + \
                        WhiteKernel(1e-3, (1e-6, 1e-1))
      self.Sigma = Sigma
      self.dim   = Sigma.shape[1]


    def updatekernel(self,kernel):
      """ update the GPkernel with a new kernel
      """
      self.GPkernel = kernel


    def get_hyperparameters(self):
      """ get hyperparameters from GPkernel
      """
      theta0 = self.GPkernel.get_params()['k1__k1__constant_value']
      theta1 = self.GPkernel.get_params()['k1__k2__length_scale']
      theta2 = self.GPkernel.get_params()['k2__noise_level']
      return np.array([theta0,theta1,theta2])


    def mollifiedx1(self, X,Y): 
      """ Once mollified kernel
      """
      # get optimized hyperparameters
      theta = self.get_hyperparameters()

      # A + 2Sigma
      A     = (theta[1]**2)*np.eye(self.dim);
      B     = A + self.Sigma
      Binv  = np.linalg.inv(B);    # inv(A + Sigma)
      # constant
      c    = theta[0]*np.sqrt(np.linalg.det(A)/np.linalg.det(B))
      # kernel
      kernel = lambda i,j: c*np.exp(-0.5*(X[i]-Y[j])@Binv@(X[i]-Y[j]));
      # kernel
      g    = np.vectorize(kernel)
      # make kernel matrix
      K    = np.fromfunction(g,(len(X),len(Y)),dtype=int);
      return K


    def mollifiedx2(self, X):
      """ Twice mollified kernel
      """
      # get optimized hyperparameters
      theta = self.get_hyperparameters()

      # A + 2Sigma
      A    = (theta[1]**2)*np.eye(self.dim);
      B    = A + 2*self.Sigma
      Binv = np.linalg.inv(B);   # inv(A + 2Sigma)
      # constant
      c    = theta[0]*np.sqrt(np.linalg.det(A)/np.linalg.det(B))
      # kernel
      kernel = lambda i,j: c*np.exp(-0.5*(X[i]-X[j])@Binv@(X[i]-X[j]));
      g    = np.vectorize(kernel)
      # make kernel matrix
      K    = np.fromfunction(g,(len(X),len(X)),dtype=int) + (theta[2])*np.eye(len(X))
      return K


