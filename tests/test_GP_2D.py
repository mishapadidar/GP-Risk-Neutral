
"""
fit the GP on 2D data
"""
import sys
sys.path.insert(1, '../')

import numpy as np
from experimental_design import SymmetricLatinHypercube as SLHC
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF
from surrogate import GaussianProcessRegressor

import matplotlib.pyplot as plt




# function f: R^n -> R
f = lambda x: x[0]**2 *(4-2.1*x[0]**2 + x[0]**4/3) + x[0]*x[1] + (-4+4*x[1]**2)*x[1]**2
# just for plotting...easier to evaluate with meshgrid
g = lambda x,y: x**2 *(4-2.1*x**2 + x**4/3) + x*y + (-4+4*y**2)*y**2

# basic info
dim        = 2
lb         = np.array([-3.0,-2.0]) 
ub         = np.array([3.0,2.0])

# experimental design
num_pts    = 500 # initial evaluations
exp_design = SLHC(dim, num_pts)
X = exp_design.generate_points(lb,ub)
fX = [float(f(x)) for x in X]

kernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(1, (0.01, 100)) + \
    WhiteKernel(1e-3, (1e-6, 1e-2))
surrogate = GaussianProcessRegressor(kernel =kernel)
surrogate.fit(X,fX)



#=============================================================
# Plot
#=============================================================

# plot the function
Ntest = 500
x = np.linspace(lb[0],ub[0],Ntest)
y = np.linspace(lb[1],ub[1],Ntest)
X,Y = np.meshgrid(x,y)
Z   = g(X,Y)
plt.contour(X,Y,Z,range(1,5000,30))

# plot the GP 
X_grid = np.c_[ np.ravel(X), np.ravel(Y) ]
zz = surrogate.predict(X_grid)
zz = zz.reshape(X.shape)
#plt.contour(X,Y,zz-Z)


plt.colorbar()
plt.title('Difference Between GP and 6 Hump Camel')
plt.legend()
plt.show()
