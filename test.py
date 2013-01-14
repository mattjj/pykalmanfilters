from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import Kalman
from Gaussians import Gaussian

x = np.sin(np.linspace(0,8*np.pi,100))
y = np.cos(np.linspace(0,8*np.pi,100))
data = (np.vstack((x,y))*np.arange(100.)).T

A = np.eye(2)
B = 2*np.eye(2)
C = np.eye(2)
D = np.eye(2)

initial_distn = Gaussian(np.zeros(2),np.eye(2))

smoothed_distns = Kalman.smooth(A,B,C,D,initial_distn,data)
mus = np.array([d.mu for d in smoothed_distns])

plt.plot(data[:,0],data[:,1],'bx-')
plt.plot(mus[:,0],mus[:,1],'r.-')
plt.show()
