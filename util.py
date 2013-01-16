from __future__ import division
import numpy as np
na = np.newaxis
from matplotlib import pyplot as plt

def solve_diagonal_plus_lowrank(diag_of_A,B,C,b):
    '''
    like np.linalg.solve(np.diag(diag_of_A)+B.dot(C),b) but better!
    b can be a matrix
    see p.673 of Convex Optimization by Boyd and Vandenberghe
    '''
    one_dim = b.ndim == 1
    if one_dim:
        b = np.reshape(b,(-1,1))
    z = b/diag_of_A[:,na]
    E = C.dot(B/diag_of_A[:,na])
    E.flat[::E.shape[0]+1] += 1
    w = np.linalg.solve(E,C.dot(z))
    z -= B.dot(w)/diag_of_A[:,na]
    return z if not one_dim else z.ravel()

def unscented_transform(mu,Sigma,alpha,kappa,beta=2):
    n = mu.shape[0]
    lmbda = alpha**2*(n+kappa)-n

    points = np.empty((2*n+1,n))
    mean_weights = np.empty(2*n+1)
    cov_weights = np.empty(2*n+1)

    points[0] = mu
    mean_weights[0] = lmbda/(n+lmbda)
    cov_weights[0] = lmbda/(n+lmbda)+(1-alpha**2+beta)

    chol = np.linalg.cholesky(Sigma)
    points[1:] = np.sqrt(n+lmbda)*np.hstack((chol,-chol)).T + mu
    cov_weights[1:] = mean_weights[1:] = 1./(2*(n+lmbda))

    return points, mean_weights, cov_weights

def inverse_unscented_transform(points,mean_weights,cov_weights):
    mu = mean_weights.dot(points)
    shifted_points = points - mu
    Sigma = (shifted_points.T * cov_weights).dot(shifted_points)
    return mu, Sigma

def plot_gaussian_2D(mu, lmbda, color='b', centermarker=True):
    '''
    Plots mean and cov ellipsoid into current axes. Must be 2D. lmbda is a covariance matrix.
    '''
    assert len(mu) == 2

    t = np.hstack([np.arange(0,2*np.pi,0.01),0])
    circle = np.vstack([np.sin(t),np.cos(t)])
    ellipse = np.dot(np.linalg.cholesky(lmbda),circle)

    if centermarker:
        plt.plot([mu[0]],[mu[1]],marker='D',color=color,markersize=4)
    plt.plot(ellipse[0,:] + mu[0], ellipse[1,:] + mu[1],linestyle='-',linewidth=2,color=color)

