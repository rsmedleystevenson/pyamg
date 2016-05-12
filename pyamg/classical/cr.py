"""Compatible Relaxation"""
from __future__ import print_function

__docformat__ = "restructuredtext en"

import numpy as np
import scipy as sp
from scipy.linalg import norm
from scipy.sparse import isspmatrix, spdiags, isspmatrix_csr

from ..relaxation.relaxation import gauss_seidel, gauss_seidel_indexed

__all__ = ['CR', 'binormalize']


def _CRsweep(A, Findex, Cindex, nu, thetacr, method):

    n = A.shape[0]    # problem size
    numax = nu
    z = np.zeros((n,))
    e = np.ones((n,))
    e[Cindex] = 0.0
    enorm = norm(e)
    rhok = 1

    for it in range(1, numax+1):

        if method == 'habituated':
            gauss_seidel(A, e, z, iterations=1)
            e[Cindex] = 0.0
        elif method == 'concurrent':
            gauss_seidel_indexed(A, e, z, indices=Findex, iterations=1)
        else:
            raise NotImplementedError('method not recognized: need \
                                       habituated or concurrent')

        enorm_old = enorm
        enorm = norm(e)
        rhok_old = rhok
        rhok = enorm / enorm_old

        # criteria 1
        if (abs(rhok - rhok_old) / rhok < 0.1) and (it >= nu):
            return rhok, e

        # criteria 2
        if rhok < 0.1 * thetacr:
            return rhok, e


def CR(A, method='habituated', nu=3, thetacr=0.7, thetacs=[0.3, 0.5],
            maxiter=20):
    """Use Compatible Relaxation to compute a C/F splitting

    Parameters
    ----------
    S : csr_matrix
        sparse matrix (n x n) usually matrix A of Ax=b
    method : {'habituated','concurrent'}
        Method used during relaxation:
            - concurrent: GS relaxation on F-points, leaving e_c = 0
            - habituated: full relaxation, setting e_c = 0
    nu : 

    thetacr :

    thetacs : 
    
    maxiter : int
        maximum number of outer iterations (lambda)

    Returns
    -------
    splitting : array
        C/F list of 1's (coarse pt) and 0's (fine pt) (n x 1)

    References
    ----------



    Examples 
    --------
    >>> from pyamg.gallery import poisson
    >>> from cr import CR
    >>> A = poisson((20,20),format='csr')
    >>> splitting = CR(A)
    """
    n = A.shape[0]    # problem size

    thetacs = list(thetacs)
    thetacs.reverse()

    if not isspmatrix_csr(A):
        raise TypeError('expecting csr sparse matrix A')

    if A.dtype == complex:
        raise NotImplementedError('complex A not implemented')

    # 3.1a
    splitting = np.zeros((n,), dtype='intc')
    gamma = np.zeros((n,))

    # 3.1b
    Cindex = np.where(splitting == 1)[0]
    Findex = np.where(splitting == 0)[0]
    rho, e = _CRsweep(A, Findex, Cindex, nu, thetacr, method=method)

    # 3.1c
    for it in range(0, maxiter):

        print(it)
        # 3.1d (assuming constant initial e in _CRsweep)
        # should already be zero at C pts (Cindex)
        gamma[Findex] = np.abs(e[Findex]) / np.abs(e[Findex]).max()

        # 3.1e
        Uindex = np.where(gamma > thetacs[0])[0]
        if len(thetacs) > 1:
            thetacs.pop()

        # 3.1f
        # first find the weights: omega_i = |N_i\C| + gamma_i
        omega = -np.inf * np.ones((n,))
        for i in Uindex:
            J = A.indices[np.arange(A.indptr[i], A.indptr[i+1])]
            J = np.where(splitting[J] == 0)[0]
            omega[i] = len(J) + gamma[i]

        # independent set
        Usize = len(Uindex)
        while Usize > 0:
            # step 1
            i = omega.argmax()
            splitting[i] = 1
            gamma[i] = 0.0
            # step 2
            J = A.indices[np.arange(A.indptr[i], A.indptr[i+1])]
            J = np.intersect1d(J, Uindex, assume_unique=True)
            omega[i] = -np.inf
            omega[J] = -np.inf

            # step 3
            for j in J:
                K = A.indices[np.arange(A.indptr[j], A.indptr[j+1])]
                K = np.intersect1d(K, Uindex, assume_unique=True)
                omega[K] = omega[K] + 1.0

            Usize -= 1

        Cindex = np.where(splitting == 1)[0]
        Findex = np.where(splitting == 0)[0]
        rho, e = _CRsweep(A, Findex, Cindex, nu, thetacr, method=method)

        print(rho)
        if rho < thetacr:
            break

    return splitting


def binormalize(A, tol=1e-5, maxiter=10):
    """Binormalize matrix A.  Attempt to create unit l_1 norm rows.

    Parameters
    ----------
    A : csr_matrix
        sparse matrix (n x n)
    tol : float
        tolerance
    x : array
        guess at the diagonal
    maxiter : int
        maximum number of iterations to try

    Returns
    -------
    C : csr_matrix
        diagonally scaled A, C=DAD

    Notes
    -----
        - Goal: Scale A so that l_1 norm of the rows are equal to 1:
        - B = DAD
        - want row sum of B = 1
        - easily done with tol=0 if B=DA, but this is not symmetric
        - algorithm is O(N log (1.0/tol))

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical import binormalize
    >>> A = poisson((10,),format='csr')
    >>> C = binormalize(A)

    References
    ----------
    .. [1] Livne, Golub, "Scaling by Binormalization"
       Tech Report SCCM-03-12, SCCM, Stanford, 2003
       http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.3.1679

    """
    if not isspmatrix(A):
        raise TypeError('expecting sparse matrix A')

    if A.dtype == complex:
        raise NotImplementedError('complex A not implemented')

    n = A.shape[0]
    it = 0
    x = np.ones((n, 1)).ravel()

    # 1.
    B = A.multiply(A).tocsc()  # power(A,2) inconsistent in numpy, scipy.sparse
    d = B.diagonal().ravel()

    # 2.
    beta = B * x
    betabar = (1.0/n) * np.dot(x, beta)
    stdev = rowsum_stdev(x, beta)

    # 3
    while stdev > tol and it < maxiter:
        for i in range(0, n):
            # solve equation x_i, keeping x_j's fixed
            # see equation (12)
            c2 = (n-1)*d[i]
            c1 = (n-2)*(beta[i] - d[i]*x[i])
            c0 = -d[i]*x[i]*x[i] + 2*beta[i]*x[i] - n*betabar
            if (-c0 < 1e-14):
                print('warning: A nearly un-binormalizable...')
                return A
            else:
                # see equation (12)
                xnew = (2*c0)/(-c1 - np.sqrt(c1*c1 - 4*c0*c2))
            dx = xnew - x[i]

            # here we assume input matrix is symmetric since we grab a row of B
            # instead of a column
            ii = B.indptr[i]
            iii = B.indptr[i+1]
            dot_Bcol = np.dot(x[B.indices[ii:iii]], B.data[ii:iii])

            betabar = betabar + (1.0/n)*dx*(dot_Bcol + beta[i] + d[i]*dx)
            beta[B.indices[ii:iii]] += dx*B.data[ii:iii]

            x[i] = xnew

        stdev = rowsum_stdev(x, beta)
        it += 1

    # rescale for unit 2-norm
    d = np.sqrt(x)
    D = spdiags(d.ravel(), [0], n, n)
    C = D * A * D
    C = C.tocsr()
    beta = C.multiply(C).sum(axis=1)
    scale = np.sqrt((1.0/n) * np.sum(beta))
    return (1/scale)*C


def rowsum_stdev(x, beta):
    """Compute row sum standard deviation

    Compute for approximation x, the std dev of the row sums
    s(x) = ( 1/n \sum_k  (x_k beta_k - betabar)^2 )^(1/2)
    with betabar = 1/n dot(beta,x)

    Parameters
    ----------
    x : array
    beta : array

    Returns
    -------
    s(x)/betabar : float

    Notes
    -----
    equation (7) in Livne/Golub

    """
    n = x.size
    betabar = (1.0/n) * np.dot(x, beta)
    stdev = np.sqrt((1.0/n) *
                    np.sum(np.power(np.multiply(x, beta) - betabar, 2)))
    return stdev/betabar
