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
    def mollifiedx1(self, dists):  # pragma: no cover
        """Once mollified kernel
        :param dists: Array of size n x n with pairwise distances
        :type dists: numpy.ndarray
        :return: Array of size n x n with kernel derivatives
        :rtype: numpy.ndarray
        """
        pass

    @abstractmethod
    def mollifiedx2(self, dists):  # pragma: no cover
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
      self.GPkernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(1, (0.1, 100)) + \
                        WhiteKernel(1e-3, (1e-6, 1e-2))
      self.Sigma = Sigma


    def updatekernel(self,kernel):
      """ update the GPkernel with a new kernel
      """
      self.GPkernel = kernel


    def get_hyperparameters(self):
      """ get hyperparameters from GPkernel
      """

      theta1 = self.GPkernel.get_params()['k1__k1__constant_value']
      theta2 = self.GPkernel.get_params()['k1__k2__length_scale']
      theta3 = self.GPkernel.get_params()['k2__noise_level']
      return np.array([theta1,theta2,theta3])


    def mollifiedx1(self, X,Y):
      """ Once mollified kernel
      """
      theta = self.get_hyperparameters()
      dim = X.shape[1]
      # compute the convolution kernels
      A          = (0.5*theta[1]**2)*np.eye(dim);
      ASigma     = A + self.Sigma
      detA       = np.linalg.det(A);        # det(A)
      detASigma  = np.linalg.det(ASigma);   # det(A + Sigma)
      invASigma  = np.linalg.inv(ASigma);    # inv(A + Sigma)
      # convolution gaussian kernel function
      conv = lambda i,j: np.exp(-0.5*(X[i]-Y[j])@invASigma@(X[i]-Y[j]));
      # vectorize the function
      g    = np.vectorize(conv)
      # create the kernel matrix
      K    = np.fromfunction(g,(len(X),len(Y)),dtype=int);
      K    = theta[0]*K*np.sqrt(detA/detASigma) # single convolutional kernel
      return K


    def mollifiedx2(self, X):
      """ Twice mollified kernel
      """
      theta = self.get_hyperparameters()
      dim = X.shape[1]


      A          = (0.5*theta[1]**2)*np.eye(dim);
      detA       = np.linalg.det(A)
      A2Sigma    = A + 2*self.Sigma
      detA2Sigma = np.linalg.det(A2Sigma);  # det(A + 2*Sigma)
      invA2Sigma = np.linalg.inv(A2Sigma);   # inv(A + 2Sigma)
      conv = lambda i,j: np.exp(-0.5*(X[i]-X[j])@invA2Sigma@(X[i]-X[j]));
      g    = np.vectorize(conv)
      K    = np.fromfunction(g,(len(X),len(X)),dtype=int) 
      K    = K*np.sqrt(detA/detA2Sigma) + theta[2]**2*np.eye(len(X)); 
      return K


