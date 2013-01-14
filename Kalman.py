from __future__ import division
import numpy as np

from Gaussians import Gaussian

class Filter(object):
    '''
    implemented in a real-time style
    '''

    def __init__(self,A,B,C,D,initial_distn):
        A,B,C,D = map(np.atleast_2d,(A,B,C,D))
        self.A = A
        self.BBT = B.dot(B.T)
        self.C = C
        self.DDT = D.dot(D.T)

        self.n = A.shape[0]

        self.distns = [initial_distn]

    def step(self,data):
        data = np.atleast_1d(data)

        # predict next state
        new_distn = self.distns[-1].linear_transform(self.A) + Gaussian(mu=np.zeros(self.n),Sigma=self.BBT)
        # condition on new observation
        new_distn *= Gaussian(mu=data,Sigma=self.DDT).ilinear_substitution(self.C)

        self.distns.append(new_distn)
        return new_distn


class FasterFilter(Filter):
    '''
    like Filter but uses special methods to avoid inverses and does updates in-place
    '''
    def __init__(self,*args,**kwargs):
        super(FasterFilter,self).__init__(*args,**kwargs)
        self.distn = self.distns[0]
        del self.distns

    def step(self,data):
        data = np.atleast_1d(data)
        # predict
        self.distn.ilinear_transform(self.A)
        self.distn += Gaussian(mu=np.zeros(data.shape[0]),Sigma=self.BBT)
        # condition
        self.distn.condition_on(data,self.DDT,self.C.dot(self.distn.Sigma))

        return self.distn


def smooth(self,A,B,C,D,initial_distn,data):
    # construct distributions
    # forward pass
    # backwards pass
    # return smoothed
    pass


def faster_smooth():
    pass
