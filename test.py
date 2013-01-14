from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import Kalman
from Gaussians import Gaussian

theta = 0.1*np.pi
A = 1.01*np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
B = 0.1*np.eye(2)
C = np.eye(2)
D = 0.5*np.eye(2)

x = np.zeros((100,2))
x[0] = (1,1)
for i in range(1,100):
    x[i] = A.dot(x[i-1]) + B.dot(np.random.randn(2))

y = (C.dot(x.T) + D.dot(np.random.normal(size=x.shape).T)).T

initial_distn = Gaussian(np.zeros(2),np.eye(2))

smoothed_distns = Kalman.smooth(A,B,C,D,initial_distn,y)
mus = np.array([d.mu for d in smoothed_distns])

plt.plot(y[:,0],y[:,1],'bx-')
plt.plot(mus[:,0],mus[:,1],'r.-')
plt.show()
