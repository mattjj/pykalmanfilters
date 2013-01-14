from __future__ import division
import numpy as np

from Gaussians import Gaussian

class ForwardFilter(object):
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
        new_distn *= Gaussian(mu=data,Sigma=self.DDT).inplace_linear_substitution(self.C)

        self.distns.append(new_distn)
        return new_distn

def smooth(A,B,C,D,initial_distn,data):
    # two-filter version (not RTS version)
    n = A.shape[0]
    BBT = B.dot(B.T)
    DDT = D.dot(D.T)

    # get the obs factors
    obs = [Gaussian(mu=d,Sigma=DDT).inplace_linear_substitution(C) for d in data]

    # forward pass
    forward_distns = [initial_distn]
    for o in obs:
        forward_distns.append(o*(forward_distns[-1].linear_transform(A) + Gaussian(mu=np.zeros(n),Sigma=BBT)))
    forward_distns = forward_distns[1:]

    # get final (unconditional) distn, p(x_{T}) if data goes (y_0,...,y_{T-1})
    final_distn = initial_distn
    for i in range(data.shape[0]):
        final_distn.inplace_linear_transform(A)
        final_distn += Gaussian(mu=np.zeros(n),Sigma=BBT)

    # backwards pass
    backward_distns = [final_distn]
    for o in obs[::-1]:
        backward_distns.append(o*(backward_distns[-1].linear_substitution(A) + Gaussian(mu=np.zeros(n),Sigma=BBT)))
    backward_distns = backward_distns[:0:-1]

    # return smoothed distributions
    return [d1*d2 for d1,d2 in zip(forward_distns,backward_distns)]


class FasterForwardFilter(ForwardFilter):
    '''
    like Filter but uses special methods to avoid inverses and does updates in-place
    '''
    def __init__(self,*args,**kwargs):
        super(FasterForwardFilter,self).__init__(*args,**kwargs)
        self.distn = self.distns[0]
        del self.distns

    def step(self,data):
        data = np.atleast_1d(data)
        # predict
        self.distn.inplace_linear_transform(self.A)
        self.distn += Gaussian(mu=np.zeros(data.shape[0]),Sigma=self.BBT)
        # condition
        self.distn.condition_on(data,self.DDT,self.C.dot(self.distn.Sigma))

        return self.distn

