# plot the risk-neutral and rbf kernel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel

"""
build the risk-neutral from a set of data points.
Then plot the rbf kernel and risk-neutral kernel
"""

#=================================================
# Build the risk-neutral kernel

def risk_kernel(X,Sigma, gamma1, gamma2):
    # manually optimize hyperparams
    #gamma1 = 1.0;
    #gamma2 = num_data_pts/dim**2 # proxy for choosing hyperparameter
    #gamma2 = 10
    # nugget
    nugget = 1e-8*np.eye(num_data_pts)
    # kernel matrix
    Phi_XX = gamma1*rbf_kernel(X,X,gamma2) + nugget
    # cholesky factorization
    L = np.linalg.cholesky(Phi_XX)


    # compute the convolution kernels
    A          = 1/(2*gamma2)*np.eye(dim);
    ASigma     = A + Sigma
    A2Sigma    = A + 2*Sigma
    detA       = np.linalg.det(A);        # det(A)
    detASigma  = np.linalg.det(ASigma);   # det(A + Sigma)
    detA2Sigma = np.linalg.det(A2Sigma);  # det(A + 2*Sigma)
    invASigma  = np.linalg.inv(ASigma);    # inv(A + Sigma)
    invA2Sigma = np.linalg.inv(A2Sigma);   # inv(A + 2Sigma)
    # convolution gaussian kernel function
    conv1 = lambda i,j: np.exp(-0.5*(X[i]-xx[j])@invASigma@(X[i]-xx[j]));
    conv2 = lambda i,j: np.exp(-0.5*(xx[i]-xx[j])@invA2Sigma@(xx[i]-xx[j]));
    # vectorize the function
    g1 = np.vectorize(conv1)
    g2 = np.vectorize(conv2)
    # create the kernel matrix
    Psi_Xx    = np.fromfunction(g1,(len(X),len(xx)),dtype=int);
    Psihat_xx = np.fromfunction(g2,(len(xx),len(xx)),dtype=int);
    # multiply by constant
    Psi_Xx     = gamma1*Psi_Xx*np.sqrt(detA/detASigma) # single convolutional kernel
    Psihat_xx  = gamma1*Psihat_xx*np.sqrt(detA/detA2Sigma) # double convolutional kernel

    # build the risk neutral kernel matrix
    K_rn = Psihat_xx - Psi_Xx.T @ np.linalg.solve(L.T,np.linalg.solve(L, Psi_Xx))# build the rbf kernel matrix

    # risk kernel matrix
    return K_rn



#=================================================
# initialize the data

# number of data points
num_data_pts = 10;
# interval 
a   = -2;
b   = 2
# dimension
dim = 1
# generate data
X = np.linspace(a,b,num_data_pts)
X = np.reshape(X,(num_data_pts,dim))
# points for plotting
num_plot_pts = 200
xx = np.linspace(a,b,num_plot_pts)
xx = np.reshape(xx,(num_plot_pts,dim))


#=================================================
# Fix Sigma, plot rbf and risk kernel for various gamma2
#

# perturbation variance
var = 0.01
Sigma = var*np.eye(dim)

# hypers
gamma1 = 1;
gamma2_list = [1.0,10.,100.]

# colors for plot
colors = ['r','b','orange','green','yellow']

for i in range(len(gamma2_list)):
    gamma2 = gamma2_list[i]
    K_rn = risk_kernel(X,Sigma, gamma1, gamma2)
    # just plot one row of the kernel matrices
    plt.plot(xx,K_rn[int(num_plot_pts/2),:],color=colors[i],label='risk: $\gamma_2/\sigma^2 = $%.0f'%(gamma2/var))
    plt.plot(xx,gamma1*rbf_kernel(xx,[[0]],gamma2),'--',color=colors[i],label='rbf: $\gamma_2/\sigma^2 = $%.0f'%(gamma2/var))

plt.title('RBF and Risk Kernels $\sigma^2 = $%.3f, N = %d'%(var,num_data_pts))
plt.legend()
plt.show()


#=================================================
# Fix gamma2, plot rbf and risk-kernel for varying Sigma
#


# hypers
gamma1      = 1.0;
gamma2      = 10;

# perturbation variance
var_list    = gamma2*np.array([0.0001,0.001,0.01,0.1])

# plot colors
colors = ['r','b','orange','green','yellow']

for i in range(len(var_list)):
    var   = var_list[i]
    Sigma = var*np.eye(dim)
    K_rn = risk_kernel(X,Sigma, gamma1, gamma2)
    # just plot one row of the kernel matrices
    plt.plot(xx,K_rn[int(num_plot_pts/2),:],color=colors[i],label='risk: $\gamma_2/\sigma^2 = $%.0f'%(gamma2/var))

#plt.scatter(X.flatten(),gamma1*rbf_kernel(X,[[0]],gamma2), color='k',label='data')
plt.plot(xx,gamma1*rbf_kernel(xx,[[0]],gamma2),'--',color='k',label='rbf kernel')
plt.title('Effect of Perturbation Variance Risk Kernel; Fixed hypers $\gamma_2 =$ %.2f, N = %d'%(gamma2,num_data_pts))
plt.legend()
plt.show()

