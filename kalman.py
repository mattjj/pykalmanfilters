from __future__ import division
import numpy as np

from gaussians import Gaussian

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

        self.distns = []
        self.next_prediction = initial_distn

    def step(self,data):
        data = np.atleast_1d(data)

        # fold in observation
        new_distn = self.next_prediction * Gaussian(mu=data,Sigma=self.DDT).inplace_linear_substitution(self.C)

        # make prediction
        self.next_prediction = new_distn.linear_transform(self.A) + Gaussian(mu=np.zeros(self.n),Sigma=self.BBT)

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
    next_prediction = initial_distn
    forward_distns = []
    for o in obs:
        forward_distns.append(next_prediction * o)
        next_prediction = forward_distns[-1].linear_transform(A) + Gaussian(mu=np.zeros(n),Sigma=BBT)

    # get final (unconditional) distn, p(x_{T-1}) if data goes (y_0,...,y_{T-1})
    final_distn = initial_distn
    for i in range(data.shape[0]-1):
        final_distn.inplace_linear_transform(A)
        final_distn += Gaussian(mu=np.zeros(n),Sigma=BBT)

    # backwards pass
    prev_prediction = final_distn
    backward_distns = []
    for o in obs[::-1]:
        backward_distns.append(prev_prediction * o)
        prev_prediction = backward_distns[-1].linear_substitution(A) + Gaussian(mu=np.zeros(n),Sigma=BBT)
    backward_distns = backward_distns[::-1]

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
        self.distn.condition_on(self.C.dot(self.distn.Sigma),self.C.dot(self.distn.Sigma).dot(self.C.T),
                                            Gaussian(mu=data,Sigma=self.DDT))

        return self.distn

