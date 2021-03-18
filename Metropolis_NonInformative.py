# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:07:45 2021

@author: rebec
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.discrete.discrete_model import Probit
plt.style.use('seaborn')

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

true_theta = np.array([0.5,0.2,0.3,-0.2]).T

# By making use of the probit link, compute the probabilities p (that is, the mean)
# of your Bernoulli Multivariate model

p = stats.norm.cdf(X.dot(true_theta))

# Then again, simulate the outcomes by making use of the probabilities we have.
# We used here a random draw from a binomial with parameter n = 1, to get a
# random draw from a Bernoulli distribution

y = np.random.binomial(1, p, size=None)

# Initialize the theta

theta = np.zeros(d).T

###   ###   ###   ###   ###

# DEFINITION OF THE VARIANCE OF THE PROPOSAL

# Proposal (Multivariate Normal centered at the current update with 
# variance matrix given by the inverse of Fisher Information)

V = np.linalg.inv(-Probit(y,X).hessian(theta))

###   ###   ###   ###

# DEFINITION OF THE TARGET FUNCTION


def target(theta):
    return np.exp(Probit(y,X).loglike(theta))


###   ###   ###   ###

# RUN THE MODEL

N_sim = 10000
burn_in = 5000
naccept = 0
theta_chain = np.zeros((N_sim,d))

# Iterations

for i in range(N_sim):
    V = np.linalg.inv(-Probit(y,X).hessian(theta))
    theta_p = np.random.multivariate_normal(theta,V)
    rho = min(1, target(theta_p)/target(theta))
    u = np.random.uniform()
    if u < rho:
        naccept += 1
        theta = theta_p
    theta_chain[i,:] = theta.reshape((1,d))
    
print("The acceptance rate is",naccept/N_sim)

###   ###   ###

# CONVERGENCE TRACE PLOTS

post_theta = theta_chain[burn_in:,].mean(axis=0)

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
            ax[i][j].plot(theta_chain[:,j], zorder = 5, linewidth = 0.5)
            ax[i][j].hlines(y = post_theta[j],xmin=0, xmax=N_sim, label='Final value of Theta', zorder  = 10, color = "tomato")
            ax[i][j].hlines(y=true_theta[j], xmin=0, xmax=N_sim, label='True value of Theta',  zorder  = 10,color = "plum")
            ax[i][j].set_title("Convergence of parameter "+str(j))

# Timing the Algorithm

print(datetime.now() - startTime)
print(true_theta)
print(post_theta)