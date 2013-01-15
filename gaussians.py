from __future__ import division
import numpy as np

from util import solve_diagonal_plus_lowrank

class Gaussian(object):
    '''
    + means return the marginal sum (convolve the pdfs)
    * means condition one on the other (multiply the pdfs)
    '''

    def __init__(self,mu=None,Sigma=None,h=None,J=None):
        assert (mu is not None and Sigma is not None) ^ (h is not None and J is not None)
        self._mu = mu
        self._Sigma = Sigma
        self._h = h
        self._J = J

        self._is_diagonal = Sigma.ndim == 1 if Sigma is not None else J.ndim == 1

    @property
    def mu(self):
        if self._mu is None:
            self._mu = np.linalg.solve(self._J,self._h) if not self._is_diagonal else self._h / self._J
        return self._mu

    @property
    def Sigma(self):
        if self._Sigma is None:
            self._Sigma = np.linalg.inv(self._J) if not self._is_diagonal else 1./self._J
            self._mu = self._Sigma.dot(self._h) if not self._is_diagonal else self._h*self._Sigma
        return self._Sigma if not self._is_diagonal else np.diag(self._Sigma)

    @property
    def h(self):
        if self._h is None:
            self._h = np.linalg.solve(self._Sigma,self._mu) if not self._is_diagonal else self._mu / self._Sigma
        return self._h

    @property
    def J(self):
        if self._J is None:
            self._J = np.linalg.inv(self._Sigma) if not self._is_diagonal else 1./self._Sigma
            self._h = self._J.dot(self._mu) if not self._is_diagonal else self._mu*self._J
        return self._J if not self._is_diagonal else np.diag(self._J)

    def __iadd__(self,other):
        self.Sigma, self.mu # make sure we have these
        self._mu += other.mu
        if self._is_diagonal == other._is_diagonal:
            self._Sigma += other.Sigma
        elif self._is_diagonal:
            self._Sigma = np.diag(self.Sigma) + other.Sigma
            self._is_diagonal = False
        else:
            self._Sigma[np.diag_indices_from(self._Sigma)] += other.Sigma
        self._J = self._h = None # invalidate
        return self

    def inplace_linear_transform(self,A):
        if self._is_diagonal:
            self._Sigma = np.diag(self.Sigma)
            self._is_diagonal = False
        self._Sigma = A.dot(self.Sigma).dot(A.T)
        self._mu = A.dot(self.mu)
        self._J = self._h = None # invalidate
        return self

    def __imul__(self,other):
        self.J, self.mu # make sure we have these
        self._h += other.h
        if self._is_diagonal == other._is_diagonal:
            self._J += other.J
        elif self._is_diagonal:
            self._J = np.diag(self.J) + other.J
            self._is_diagonal = False
        else:
            self._J[np.diag_indices_from(self._J)] += other.J
        self._Sigma = self._mu = None # invalidate
        return self

    def inplace_linear_substitution(self,A):
        if self._is_diagonal:
            self._J = np.diag(self.J)
            self._is_diagonal = False
        self._h = A.T.dot(self.h)
        self._J = A.T.dot(self.J).dot(A)
        self._Sigma = self._mu = None # invalidate
        return self

    # these are written to take covariance/information blocks so that they can
    # be used with approximations to nonlinear observations (e.g. using the
    # unscented transform in the unscented Kalman filter)

    def inplace_condition_on(self,Sigma_xy,Sigma_yy,my_prediction,other):
        'maintains distribution form, computes updates using Schur complements'
        self.Sigma, self.mu # make sure we are in distribution form
        self._mu += Sigma_xy.dot(np.linalg.solve(Sigma_yy+other.Sigma,other.mu - my_prediction))
        self._Sigma -= Sigma_xy.dot(np.linalg.solve(Sigma_yy+other.Sigma,Sigma_xy.T))
        self._J = self._h = None # invalidate

    def inplace_condition_on_diag_plus_lowrank(self,Sigma_xy,Sigma_yy_factor,my_prediction,other):
        'maintains distribution form, faster and better Schur complement solving!'
        assert other._is_diagonal
        self.Sigma, self.mu # make sure we are in distribution form
        self._mu += Sigma_xy.dot(solve_diagonal_plus_lowrank(other._Sigma,Sigma_yy_factor,
                                    Sigma_yy_factor.T,other.mu-my_prediction))
        self._Sigma -= Sigma_xy.dot(solve_diagonal_plus_lowrank(other._Sigma,Sigma_yy_factor,
                                        Sigma_yy_factor.T,Sigma_xy.T))
        self._J = self._h = None # invalidate

    def inplace_marginalize_against(self,Sigma_xy,Sigma_yy,my_prediction,other):
        '''
        maintains distribution form, useful in backwards RTS smoothing pass
        Sigma_xy should be something like Sigma_{t|t}.dot(A.T)
        Sigma_yy should be P_{t+1|t}
        other.Sigma should be P{t+1|T}
        '''
        self.Sigma, self.mu # make sure we are in distirbution form
        self._mu += Sigma_xy.dot(np.linalg.solve(Sigma_yy,other.mu - my_prediction))
        # TODO a little wasteful
        # drops T's because of symmetric matrices
        self._Sigma += Sigma_xy.dot(np.linalg.solve(Sigma_yy,
            np.linalg.solve(Sigma_yy,other.Sigma - Sigma_yy).dot(Sigma_xy.T)))
        self._J = self._h = None # invalidate

    ### boilerplate to provide non-destructive functions in terms of in-place versions

    def __add__(self,other):
        return self.__class__(mu=self.mu.copy(),Sigma=self.Sigma.copy()).__iadd__(other)

    def __mul__(self,other):
        return self.__class__(h=self.h.copy(),J=self.J.copy()).__imul__(other)

    def linear_transform(self,*args,**kwargs):
        return self.__class__(mu=self.mu,Sigma=self.Sigma).inplace_linear_transform(*args,**kwargs)

    def linear_substitution(self,*args,**kwargs):
        return self.__class__(h=self.h,J=self.J).inplace_linear_substitution(*args,**kwargs)

    def condition_on(self,*args,**kwargs):
        return self.__class__(mu=self.mu.copy(),Sigma=self.Sigma.copy()).inplace_condition_on(*args,**kwargs)

    def condition_on_diag_plus_lowrank(self,*args,**kwargs):
        return self.__class__(mu=self.mu.copy(),Sigma=self.Sigma.copy()).inplace_condition_on_diag_plus_lowrank(*args,**kwargs)

    def marginalize_agaist(self,*args,**kwargs):
        return self.__class__(mu=self.mu.copy(),Sigma=self.Sigma.copy()).inplace_marginalize_against(*args,**kwargs)

