from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import Kalman
from Gaussians import Gaussian

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


theta = 0.1*np.pi
A = 1.02*np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
B = 0.1*np.eye(2)
C = np.eye(2)
D = 0.75*np.eye(2)

x = np.zeros((100,2))
x[0] = (1,1)
for i in range(1,100):
    x[i] = A.dot(x[i-1]) + B.dot(np.random.randn(2))

y = (C.dot(x.T) + D.dot(np.random.normal(size=x.shape).T)).T

initial_distn = Gaussian(np.zeros(2),np.eye(2))

smoothed_distns = Kalman.smooth(A,B,C,D,initial_distn,y)
mus = np.array([d.mu for d in smoothed_distns])

plt.plot(x[:,0],x[:,1],'kx-.')
plt.plot(y[:,0],y[:,1],'bx-')
plt.plot(mus[:,0],mus[:,1],'r.-')
for d in smoothed_distns:
    plot_gaussian_2D(d.mu,d.Sigma,color='r',centermarker=False)

plt.show()
