from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import kalman
from gaussians import Gaussian
from util import plot_gaussian_2D

### generate some noisy spiral data

theta = 0.1*np.pi
A = 1.02*np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
B = 0.25*np.eye(2)
C = np.eye(2)
D = np.eye(2)

x = np.zeros((100,2))
x[0] = (0,2)
for i in range(1,100):
    x[i] = A.dot(x[i-1]) + B.dot(np.random.randn(2))

y = (C.dot(x.T) + D.dot(np.random.normal(size=x.shape).T)).T

### run Kalman filters and smoothers

initial_distn = Gaussian(np.zeros(2),np.eye(2))

filtered_distns = kalman.filter_generic(A,B,C,D,initial_distn,y)
filtered_mus = np.array([d.mu for d in filtered_distns])
smoothed_distns = kalman.smooth_rts_optimized(A,B,C,D,initial_distn,y)
smoothed_mus = np.array([d.mu for d in smoothed_distns])

### plot things

plt.plot(x[:,0],x[:,1],'kx-.',label='true state')
plt.plot(y[:,0],y[:,1],'bx-',label='observations')
plt.plot(filtered_mus[:,0],filtered_mus[:,1],'g.-',label='filtered')
plt.plot(smoothed_mus[:,0],smoothed_mus[:,1],'r.-',label='smoothed')
for df,ds in zip(filtered_distns,smoothed_distns):
    plot_gaussian_2D(df.mu,df.Sigma,color='g',centermarker=False)
    plot_gaussian_2D(ds.mu,ds.Sigma,color='r',centermarker=False)
plt.legend()

plt.show()
