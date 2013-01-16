from __future__ import division
import numpy as np

from util import solve_diagonal_plus_lowrank

# TODO split this into two classes, wrap them into one

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

    ### boilerplate to provide non-destructive functions in terms of in-place versions

    def __add__(self,other):
        return self.__class__(mu=self.mu.copy(),Sigma=self.Sigma.copy()).__iadd__(other)

    def __mul__(self,other):
        return self.__class__(h=self.h.copy(),J=self.J.copy()).__imul__(other)

    def linear_transform(self,*args,**kwargs):
        return self.__class__(mu=self.mu,Sigma=self.Sigma).inplace_linear_transform(*args,**kwargs)

    def linear_substitution(self,*args,**kwargs):
        return self.__class__(h=self.h,J=self.J).inplace_linear_substitution(*args,**kwargs)


class OptimizedGaussian(Gaussian):
    'adds functions that maintain distribution form and avoid inverses by using solvers'

    def inplace_condition_on(self,C,obs_distn):
        'same as self*obs_distn.linear_substitution(C)'
        return self.inplace_condition_on_cov(
                self.Sigma.dot(C.T),C.dot(self.Sigma).dot(C.T),C.dot(self.mu),obs_distn)

    def inplace_condition_on_cov(self,Sigma_xy,Sigma_yy,my_prediction,obs_distn):
        'cov version useful for linearized approximations; compare to inplace_condition_on for the idea'
        self.Sigma, self.mu # make sure we are in distribution form
        self._mu += Sigma_xy.dot(np.linalg.solve(Sigma_yy+obs_distn.Sigma,obs_distn.mu - my_prediction))
        self._Sigma -= Sigma_xy.dot(np.linalg.solve(Sigma_yy+obs_distn.Sigma,Sigma_xy.T))
        self._J = self._h = None # invalidate
        return self

    def inplace_condition_on_diag_plus_lowrank(self,C,obs_distn):
        return self.inplace_condition_on_diag_plus_lowrank_cov(
                self.Sigma.dot(C.T),C.dot(self.Sigma),C.T,C.dot(self.mu),obs_distn)

    def inplace_condition_on_diag_plus_lowrank_cov(self,Sigma_xy,
            Sigma_yy_factor1,Sigma_yy_factor2,my_prediction,obs_distn):
        assert obs_distn._is_diagonal
        self.Sigma, self.mu # make sure we are in distribution form
        self._mu += Sigma_xy.dot(solve_diagonal_plus_lowrank(obs_distn._Sigma,Sigma_yy_factor1,
                                    Sigma_yy_factor2,obs_distn.mu-my_prediction))
        self._Sigma -= Sigma_xy.dot(solve_diagonal_plus_lowrank(obs_distn._Sigma,Sigma_yy_factor1,
                                        Sigma_yy_factor2,Sigma_xy.T))
        self._J = self._h = None # invalidate
        return self

    ### boilerplate

    def condition_on(self,*args,**kwargs):
        return self.__class__(mu=self.mu.copy(),Sigma=self.Sigma.copy()).inplace_condition_on(*args,**kwargs)

    def condition_on_cov(self,*args,**kwargs):
        return self.__class__(mu=self.mu.copy(),Sigma=self.Sigma.copy()).inplace_condition_on_cov(*args,**kwargs)

    def condition_on_diag_plus_lowrank(self,*args,**kwargs):
        return self.__class__(mu=self.mu.copy(),Sigma=self.Sigma.copy()).inplace_condition_on_diag_plus_lowrank(*args,**kwargs)

    def condition_on_diag_plus_lowrank_cov(self,*args,**kwargs):
        return self.__class__(mu=self.mu.copy(),Sigma=self.Sigma.copy()).inplace_condition_on_diag_plus_lowrank_cov(*args,**kwargs)

    ### tests

    @classmethod
    def _test_condition_on(cls):
        randn = np.random.randn
        A = randn(3,3); A = A.dot(A.T);
        B = randn(3,3); B = B.dot(B.T);
        C = randn(3,3)

        a = cls(np.zeros(3),A)
        b = cls(np.ones(3),B)

        res1 = a*b.linear_substitution(C)
        res2 = a.condition_on(C,b)

        assert np.allclose(res1.mu,res2.mu)
        assert np.allclose(res1.Sigma,res2.Sigma)

    @classmethod
    def _test_condition_on_diag_plus_lowrank(cls):
        randn = np.random.randn
        A = randn(3,3); A = A.dot(A.T)
        Bdiag = randn(10); Bdiag = Bdiag*Bdiag;
        C = randn(10,3)

        a = cls(np.zeros(3),A)
        bdiag = cls(np.ones(10),Bdiag)
        b = cls(np.ones(10),np.diag(Bdiag))

        res1 = a.condition_on(C,b)
        res2 = a.condition_on_diag_plus_lowrank(C,bdiag)

        assert np.allclose(res1.mu,res2.mu)
        assert np.allclose(res1.Sigma,res2.Sigma)

