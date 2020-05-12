"""
Function to evaluate Monte-Carlo in 1D
"""
import numpy as np

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def MonteCarlo1D(f,Xtest,Ntest,num_pts_MC):
    #draw num_pts_MC realizations of RV (here gaussian)
    mu      = 0;
    sigma   = 0.1;
    U       = np.random.normal(mu, sigma, num_pts_MC)
    ev      = np.zeros(Ntest);
    va      = np.zeros(Ntest);
    help_ev = np.zeros(num_pts_MC);
    help_va = np.zeros(num_pts_MC);
    for i in range(0,Ntest):
        for j in range(0,num_pts_MC):
            help_ev[j] = f(Xtest[i]-U[j]);
        #compute expected value
        ev[i] = 1/num_pts_MC*np.sum(help_ev);
        for j in range(0,num_pts_MC):
            help_va[j] = (help_ev[j]-ev[i])**2
        # compute variance
        va[i] = 1/num_pts_MC*sum(help_va);
    
    return ev,va
    
def MonteCarlo1D_surrogate(surrogate,Xtest,Ntest,num_pts_MC):
    #draw num_pts_MC realizations of RV (here gaussian)
    mu      = 0;
    sigma   = 0.1;
    U       = np.random.normal(mu, sigma, num_pts_MC)
    ev      = np.zeros(Ntest);
    va      = np.zeros(Ntest);
    help_ev = np.zeros(num_pts_MC);
    help_va = np.zeros(num_pts_MC);
    X_help  = np.zeros(num_pts_MC);
    for i in range(0,Ntest):
        X_help = np.ones(num_pts_MC)*Xtest[i]-U;
        help_ev = surrogate.predict(X_help.reshape((num_pts_MC,1)))
        #compute expected value
        ev[i] = 1/num_pts_MC*np.sum(help_ev);
        for j in range(0,num_pts_MC):
            help_va[j] = (help_ev[j]-ev[i])**2
        # compute variance
        va[i] = 1/num_pts_MC*sum(help_va);
    
    return ev,va
