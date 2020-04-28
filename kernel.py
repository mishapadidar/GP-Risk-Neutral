import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import rbf_kernel
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
    def eval(self, dists):  # pragma: no cover
        """Evaluate the radial kernel.
        :param dists: Array of size n x n with pairwise distances
        :type dists: numpy.ndarray
        :return: Array of size n x n with kernel values
        :rtype: numpy.ndarray
        """
        pass

    @abstractmethod
    def deriv(self, dists):  # pragma: no cover
        """Evaluate derivatives of radial kernel wrt hyperparams.
        :param dists: Array of size n x n with pairwise distances
        :type dists: numpy.ndarray
        :return: Array of size n x n with kernel derivatives
        :rtype: numpy.ndarray
        """
        pass



class RBFKernel(Kernel):
    """Thin-plate spline RBF kernel.
    This is a basic class Squared Exponential RBF kernel:
    :math:`\\varphi(r) = \\exp(\\-theta_2^2||x-x'||^2) + \\theta_3^2\\delta_{ii'}` which is
    positive definite.
    """
    def __init__(self):
        super().__init__()
        self.hyperparams = np.array([1e-1,1e-1])
        self.bounds      = np.array([[1e-6,10],[1e-3,1.0]])
        self.num_hyperparams  = len(self.hyperparams)


    def eval(self, X,Y=None,hyperparams =None):
        """Evaluate the RBF kernel.
        :param X,Y: 2D-arrays of points (as rows)
        :type X,Y: numpy.array
        :param hyperparams: array of hyperparams
        :type hyperparams: numpy.array
        :returns: a matrix where element :math:`(i,j)` is
            :math:`\\exp(-\\theta_2||X_i-Y_j||^2) + \\theta_3^2\\delta_{ij}`
        :rtype: numpy.array
        """

        # set the hypers
        if hyperparams is None:
          hyperparams = self.hyperparams

        # add the nugget 
        if Y is None:
          N      = X.shape[0]        
          nugget = (hyperparams[1]**2)*np.eye(N)
        else:
          # When X != Y
          nugget = 0.0

        # kernel matrix
        K = rbf_kernel(X,Y,1.0/hyperparams[0]**2) + nugget
        return K


    def eval_conv1(self, X,Y=None,hyperparams =None):
        """Evaluate the Mollified RBF kernel.
        :param X,Y: 2D-arrays of points (as rows)
        :type X,Y: numpy.array
        :param hyperparams: array of hyperparams
        :type hyperparams: numpy.array
        :returns: a matrix where element :math:`(i,j)` is
            :math:`\\exp(\\-theta_2||X_i-Y_j||^2) + \\theta_3^2\\delta_{ij}`
        :rtype: numpy.array
        """
        return None

    def eval_conv2(self, X,Y=None,hyperparams =None):
        """Evaluate the RBF kernel.
        :param X,Y: 2D-arrays of points (as rows)
        :type X,Y: numpy.array
        :param hyperparams: array of hyperparams
        :type hyperparams: numpy.array
        :returns: a matrix where element :math:`(i,j)` is
            :math:`\\exp(\\-theta_2||X_i-Y_j||^2) + \\theta_3^2\\delta_{ij}`
        :rtype: numpy.array
        """
        
        return None


    def deriv(self, X,Y=None,hyperparams =None):
        """Derivative of kernel w.r.t hyperparameters
        :param X,Y: 2D-arrays of points (as rows)
        :type X,Y: numpy.array
        :param hyperparams: array of hyperparams
        :type hyperparams: numpy.array
        :returns: a matrix where element :math:`(i,j)` is
            :math:`\\exp(-\\theta_2||X_i-Y_j||^2) + \\theta_3^2\\delta_{ij}`
        :rtype: numpy.array
        """

        # set the hypers
        if hyperparams is None:
          hyperparams = self.hyperparams


        # derivative wrt lengthscale
        dkdx = rbf_kernel(X,Y,1.0/hyperparams[0]**2) 
        dxdl = 2*euclidean_distances(X,Y)*(1.0/hyperparams[0]**3)
        dkdl = np.multiply(dkdx,dxdl)

        # derivative wrt nugget variable
        if Y is None:
          N      = X.shape[0]        
          dkdn = 2*hyperparams[1]*np.eye(N)
        else:
          # When X != Y
          dkdn = np.zeros((N,N))

        jac = np.array([dkdl,dkdn])
        return jac