from __future__ import division
import numpy as np

class Gaussian(object):
    '''
    + means return the marginal sum (convolve the pdfs)
    * means condition one on the other (multiply the pdfs)

    the extra methods condition_on() and marginalize_against() can be written in
    terms of + and *, but under the hood condition_on() preserves the
    distribution form and marginalize_against() preserves the information form
    using calls to solvers which may be more efficient and more numerically
    stable

    if the object's state is low-rank in one domain and an operation that
    requires an inverse is requested, a LinAlgError will probably pop out
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

    def condition_on(self,crosscov,other):
        raise NotImplementedError

    def marginalize_against(self,crosscov,other):
        raise NotImplementedError

    ### boilerplate

    def __add__(self,other):
        return self.__class__(mu=self.mu.copy(),Sigma=self.Sigma.copy()).__iadd__(other)

    def __mul__(self,other):
        return self.__class__(h=self.h.copy(),J=self.J.copy()).__imul__(other)

    def linear_transform(self,A):
        return self.__class__(mu=self.mu,Sigma=self.Sigma).inplace_linear_transform(A)

    def linear_substitution(self,A):
        return self.__class__(h=self.h,J=self.J).inplace_linear_substitution(A)

