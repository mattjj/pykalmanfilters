from __future__ import division
import numpy as np

from gaussians import Gaussian, OptimizedGaussian

'''
Demo file of simple Kalman filter and smoother implementations

There are versions using generic Gaussian operations and also versions using the
somewhat more efficient solver-based methods provided by OptimizedGaussian in
gaussians.py.

Additionally, there are two smoothing algorithm options: the two-filter version
and the slightly less clear Rauch-Tung-Striebel version.
'''

###############
#  Filtering  #
###############

def filter_generic(A,B,C,D,initial_distn,data):
    'written to use generic operations for transparency'
    n = A.shape[0]
    BBT = B.dot(B.T)
    DDT = D.dot(D.T)

    next_prediction = initial_distn
    forward_distns = []
    for d in data:
        forward_distns.append(next_prediction * Gaussian(mu=d,Sigma=DDT).inplace_linear_substitution(C))
        next_prediction = forward_distns[-1].linear_transform(A) + Gaussian(mu=np.zeros(n),Sigma=BBT)

    return forward_distns

def filter_optimized(A,B,C,D,initial_distn,data):
    'written to stay in distribution form using solvers'
    initial_distn = OptimizedGaussian(initial_distn.mu,initial_distn.Sigma)

    n = A.shape[0]
    BBT = B.dot(B.T)
    DDT = D.dot(D.T)

    next_prediction = initial_distn
    forward_distns = []
    for d in data:
        forward_distns.append(next_prediction.inplace_condition_on(C,OptimizedGaussian(d,DDT)))
        next_prediction = forward_distns[-1].linear_transform(A) + OptimizedGaussian(np.zeros(n),BBT)

    return forward_distns

###############
#  Smoothing  #
###############

def smooth_generic_twofilter(A,B,C,D,initial_distn,data):
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

def smooth_rts_optimized(A,B,C,D,initial_distn,data):
    'RTS smoother written to stay in distribution form'
    initial_distn = OptimizedGaussian(initial_distn.mu,initial_distn.Sigma)

    n = A.shape[0]
    BBT = B.dot(B.T)
    DDT = D.dot(D.T)

    # forward pass
    prediction = initial_distn
    distns = []
    for d in data:
        prediction.inplace_condition_on(C,OptimizedGaussian(d,DDT))
        distns.append(prediction)
        prediction = prediction.linear_transform(A) + OptimizedGaussian(np.zeros(n),BBT)

    # backward pass
    for d1,d2 in zip(distns[::-1][1:],distns[:0:-1]):
        _rts_backward_step_optimized(A,BBT,d1,d2)

    return distns

# TODO not sure about the right abstraction for this operation yet... it should
# probably be a combination of conditioning operations, but I haven't massaged
# the algebra into the right form
def _rts_backward_step_optimized(A,BBT,dprev,dnext):
    P_tp1_t = A.dot(dprev.Sigma).dot(A.T) + BBT
    dprev._mu += dprev.Sigma.dot(A.T).dot(np.linalg.solve(P_tp1_t,dnext.mu - A.dot(dprev.mu)))
    temp = np.linalg.solve(P_tp1_t,dnext.Sigma)
    temp.flat[::temp.shape[0]+1] -= 1
    dprev._Sigma += dprev.Sigma.dot(A.T).dot(np.linalg.solve(P_tp1_t, temp.dot(A).dot(dprev.Sigma)))

