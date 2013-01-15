from __future__ import division
import numpy as np

from gaussians import Gaussian

def filter_step(A,B,C,D,prev_distn,data):
    # predict
    new_distn = prev_distn.linear_transform(A) + Gaussian(mu=np.zeros(A.shape[1]),Sigma=B.dot(B.T))
    # fold in observation
    new_distn *= Gaussian(mu=data,Sigma=D.dot(D.T)).inplace_linear_substitution(C)
    return new_distn

def smooth_twofilter(A,B,C,D,initial_distn,data):
    'two-filter version (not RTS version) written for generic ops, not speed'
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

# TODO smooth_rts_simple

def filter_step_efficient():
    raise NotImplementedError

def smooth_rts_efficient():
    raise NotImplementedError
