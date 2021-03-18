# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 15:27:01 2020

@author: rebec
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime
startTime = datetime.now()

np.random.seed(250)


###   ###   ###   ###   ###   ###

# SET UP THE MODEL

# Set the dimension of your problem, with N as the number of observations and 
# d as the number of parameters

n = 400
d = 4

# Generate a sequence of random numbers from a Normal distribution to then be
# able to fill in the matrix X

x = np.random.randn(n,d-1)

X = np.ones((n,d))
X[:,1:] = x

# Make a guess on the true value of theta

# true_theta = np.array([-0.5,3.3]).reshape((d,1))

true_theta = np.array([0.5,0.2,0.3,-0.2]).reshape((d,1))

# By making use of the probit link, compute the probabilities p (that is, the mean)
# of your Bernoulli Multivariate model

p = stats.norm.cdf(X.dot(true_theta))

# Then again, simulate the outcomes by making use of the probabilities we have.
# We used here a random draw from a binomial with parameter n = 1, to get a
# random draw from a Bernoulli distribution

y = np.random.binomial(1, p, size=None)

# Count the number of ones and zeros

N1 = int(sum(y))
N0 = n- N1

###   ###   ###   ###   ###   ###

# INITIALIZING THE PARAMETERS 

theta_0 = np.zeros((d,1))
Q_0 = np.eye(d)*10

theta = theta_0
z = np.zeros((n,1))

N_sim = 10000
burn_in = 5000

theta_chain = np.zeros((N_sim,d))

###   ###   ###   ###   ###   ###

# GIBBS SAMPLING ALGORITHM

prec_0 = np.linalg.inv(Q_0)
V = np.linalg.inv(prec_0 + X.T.dot(X))

for t in range(1,N_sim):
    
    # Update the mean of Z
    
    mu_z = X.dot(theta)
    
    # Draw latent variable Z from its full conditional
    
    z[y == 0] = stats.truncnorm.rvs(a = -np.inf, b = -mu_z[y == 0], loc = mu_z[y == 0], scale=1)
    
    z[y == 1] = stats.truncnorm.rvs(a = -mu_z[y == 1], b = np.inf, loc=mu_z[y == 1], scale=1)
    
    # Compute posterior mean of theta
        
    M = V.dot(prec_0.dot(theta_0) + X.T.dot(z))
    
    # Draw variable theta from its full conditional
    
    theta = np.random.multivariate_normal(M.ravel(),V).reshape((d,1))
    
    theta_chain[t,:] = theta.T
    
# Get the posterior mean of theta

post_theta = theta_chain[burn_in:,].mean(axis=0)


###   ###   ###   ###   ###   ###

# CONVERGENCE CHECK
    
# Trace Plots


fig, ax = plt.subplots(2, d, figsize=(24, 12))
for i in range(2):
    if i == 0:
        for j in range(d):
            ax[i][j].hist(theta_chain[burn_in:,j], bins = 40)
            ax[i][j].axvline(x=post_theta[j], ymin=0, ymax=400, label='Final value of Theta', color = "tomato")
            ax[i][j].axvline(x=true_theta[j], ymin=0, ymax=400, label='True value of Theta',color = "plum" )
            ax[i][j].set_title("Distribution of parameter "+str(j)+" post burn-in")
    elif i == 1:
       for j in range(d):
            ax[i][j].plot(theta_chain[burn_in:burn_in+1000,j], zorder = 5, linewidth = 0.5)
            ax[i][j].hlines(y = post_theta[j],xmin=0, xmax=1000, label='Final value of Theta', zorder  = 10, color = "tomato")
            ax[i][j].hlines(y=true_theta[j], xmin=0, xmax=1000, label='True value of Theta',  zorder  = 10,color = "plum")
            ax[i][j].set_title("Convergence of parameter "+str(j))


# Timing the Algorithm
time = datetime.now() - startTime

print(time)
print(true_theta)
print(post_theta)

