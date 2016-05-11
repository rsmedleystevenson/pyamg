"""Tentative prolongator"""

__docformat__ = "restructuredtext en"

import pdb
import sys
import warnings

import numpy as np
from scipy.sparse import isspmatrix_csr, isspmatrix_bsr, bsr_matrix, csr_matrix, identity, vstack, dia_matrix
from scipy import linalg, array
from pyamg import amg_core

from copy import deepcopy

__all__ = ['fit_candidates', 'ben_ideal_interpolation']


def fit_candidates(AggOp, B, tol=1e-10):
    """Fit near-nullspace candidates to form the tentative prolongator

    Parameters
    ----------
    AggOp : csr_matrix
        Describes the sparsity pattern of the tentative prolongator.
        Has dimension (#blocks, #aggregates)
    B : array
        The near-nullspace candidates stored in column-wise fashion.
        Has dimension (#blocks * blocksize, #candidates)
    tol : scalar
        Threshold for eliminating local basis functions.
        If after orthogonalization a local basis function Q[:, j] is small,
        i.e. ||Q[:, j]|| < tol, then Q[:, j] is set to zero.

    Returns
    -------
    (Q, R) : (bsr_matrix, array)
        The tentative prolongator Q is a sparse block matrix with dimensions
        (#blocks * blocksize, #aggregates * #candidates) formed by dense blocks
        of size (blocksize, #candidates).  The coarse level candidates are
        stored in R which has dimensions (#aggregates * #candidates,
        #candidates).

    See Also
    --------
    amg_core.fit_candidates

    Notes
    -----
        Assuming that each row of AggOp contains exactly one non-zero entry,
        i.e. all unknowns belong to an aggregate, then Q and R satisfy the
        relationship B = Q*R.  In other words, the near-nullspace candidates
        are represented exactly by the tentative prolongator.

        If AggOp contains rows with no non-zero entries, then the range of the
        tentative prolongator will not include those degrees of freedom. This
        situation is illustrated in the examples below.

    References
    ----------
    .. [1] Vanek, P. and Mandel, J. and Brezina, M.,
       "Algebraic Multigrid by Smoothed Aggregation for
       Second and Fourth Order Elliptic Problems",
       Computing, vol. 56, no. 3, pp. 179--196, 1996.
       http://citeseer.ist.psu.edu/vanek96algebraic.html


    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.aggregation.tentative import fit_candidates
    >>> # four nodes divided into two aggregates
    ... AggOp = csr_matrix( [[1, 0],
    ...                      [1, 0],
    ...                      [0, 1],
    ...                      [0, 1]] )
    >>> # B contains one candidate, the constant vector
    ... B = [[1],
    ...      [1],
    ...      [1],
    ...      [1]]
    >>> Q, R = fit_candidates(AggOp, B)
    >>> Q.todense()
    matrix([[ 0.70710678,  0.        ],
            [ 0.70710678,  0.        ],
            [ 0.        ,  0.70710678],
            [ 0.        ,  0.70710678]])
    >>> R
    array([[ 1.41421356],
           [ 1.41421356]])
    >>> # Two candidates, the constant vector and a linear function
    ... B = [[1, 0],
    ...      [1, 1],
    ...      [1, 2],
    ...      [1, 3]]
    >>> Q, R = fit_candidates(AggOp, B)
    >>> Q.todense()
    matrix([[ 0.70710678, -0.70710678,  0.        ,  0.        ],
            [ 0.70710678,  0.70710678,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.70710678, -0.70710678],
            [ 0.        ,  0.        ,  0.70710678,  0.70710678]])
    >>> R
    array([[ 1.41421356,  0.70710678],
           [ 0.        ,  0.70710678],
           [ 1.41421356,  3.53553391],
           [ 0.        ,  0.70710678]])
    >>> # aggregation excludes the third node
    ... AggOp = csr_matrix( [[1, 0],
    ...                      [1, 0],
    ...                      [0, 0],
    ...                      [0, 1]] )
    >>> B = [[1],
    ...      [1],
    ...      [1],
    ...      [1]]
    >>> Q, R = fit_candidates(AggOp, B)
    >>> Q.todense()
    matrix([[ 0.70710678,  0.        ],
            [ 0.70710678,  0.        ],
            [ 0.        ,  0.        ],
            [ 0.        ,  1.        ]])
    >>> R
    array([[ 1.41421356],
           [ 1.        ]])

    """
    if not isspmatrix_csr(AggOp):
        raise TypeError('expected csr_matrix for argument AggOp')

    B = np.asarray(B)
    if B.dtype not in ['float32', 'float64', 'complex64', 'complex128']:
        B = np.asarray(B, dtype='float64')

    if len(B.shape) != 2:
        raise ValueError('expected 2d array for argument B')

    if B.shape[0] % AggOp.shape[0] != 0:
        raise ValueError('dimensions of AggOp %s and B %s are \
                          incompatible' % (AggOp.shape, B.shape))

    N_fine, N_coarse = AggOp.shape

    K1 = int(B.shape[0] / N_fine)  # dof per supernode (e.g. 3 for 3d vectors)
    K2 = B.shape[1]                # candidates

    # the first two dimensions of R and Qx are collapsed later
    R = np.empty((N_coarse, K2, K2), dtype=B.dtype)  # coarse candidates
    Qx = np.empty((AggOp.nnz, K1, K2), dtype=B.dtype)  # BSR data array

    AggOp_csc = AggOp.tocsc()

    fn = amg_core.fit_candidates
    fn(N_fine, N_coarse, K1, K2,
       AggOp_csc.indptr, AggOp_csc.indices, Qx.ravel(),
       B.ravel(), R.ravel(), tol)

    # TODO replace with BSC matrix here
    Q = bsr_matrix((Qx.swapaxes(1, 2).copy(), AggOp_csc.indices,
                    AggOp_csc.indptr), shape=(K2*N_coarse, K1*N_fine))
    Q = Q.T.tobsr()
    R = R.reshape(-1, K2)

    return Q, R


# # -------------------------------- CHECK ON / FIX -------------------------------- #
#     # NEED TO IMPLEMENT THIS FOR SUPER NODES AS WELL...
#       This means that Cpts will refer to blocks. In this case, the minmization is over 2x2 or 3x3, etc. blocks... 
#     # K1 = B.shape[0] / num_pts  # dof per supernode (e.g. 3 for 3d vectors)
#     # K2 = B.shape[1]           # candidates
def ben_ideal_interpolation(A, AggOp, Cnodes, B=None, SOC=None, weighting=10.0, tol=1e-10):

    blocksize = None

    if not isspmatrix_csr(AggOp):
        raise TypeError('expected csr_matrix for argument AggOp')

    if B is not None:
        B = np.asarray(B)
        if B.dtype not in ['float32', 'float64', 'complex64', 'complex128']:
            B = np.asarray(B, dtype='float64')

        if len(B.shape) != 2:
            raise ValueError('expected 2d array for argument B')

        if B.shape[0] % AggOp.shape[0] != 0:
            raise ValueError('dimensions of AggOp %s and B %s are \
                              incompatible' % (AggOp.shape, B.shape))

    # BSR matrix - get blocksize and convert to csr matrix to extract submatrices.
    if isspmatrix_bsr(A):
        blocksize = A.blocksize
        A = A.tocsr()
        num_pts = A.shape[0]
        Cpts = [blocksize[0]*i+j for i in Cnodes for j in range(0,blocksize[0])]
        Fpts = [i for i in range(0,num_pts) if i not in Cnodes]
        num_Fpts = len(Fpts)
        num_Cpts = len(Cpts)
        num_bad_guys = B.shape[1]
        if isspmatrix_bsr(B):
            B = B.tocsr()

        if blocksize[0] != blocksize[1]:
            warnings.warn('A has rectangular block size.\n New ideal interpolation is not set up to accomodate this.')

    # CSR matrix
    else:
        num_pts = AggOp.shape[0]
        Cpts = deepcopy(Cnodes)
        Fpts = [i for i in range(0,num_pts) if i not in Cpts]
        num_Fpts = len(Fpts)
        num_Cpts = len(Cpts)
        num_bad_guys = B.shape[1]


    # ----------------------------------------------------------------------------- #
    # Shouldn't even compute F points in Python as above. If I can avoid
    # constructing submatrices at all, just pass A and C-points into the C-code...
    #   - Is it better to form Acc a-priori for w = \hat{w}*Acc? \hat{B}_c = AccB_c?
    #   - Can I form LQ minimization operators w/o Afc explicit? 
    #   - How am I computing sparsity pattern? Preferably in C I suppose? 
    # ----------------------------------------------------------------------------- #


    # Unconstrained new ideal interpolation if no bad guys are provided
    if B is None:
        warnings.warn("No bad guys provided - using unconstrained minimization.")
        fn = amg_core.unconstrained_new_ideal
        fn( Y.indptr,
            Y.indices,
            Y.data,
            lqTopOp.indptr,
            lqTopOp.indices,
            lqTopOp.data,
            rhsTop.indptr,
            rhsTop.indices,
            rhsTop.data,
            num_Fpts,
            num_Cpts )
    # Constrained new ideal interpolation if bad guys are provided
    else:
        lqBottomOp = weighting*(B[Cpts,:].T*Acc)
        rhsBottom = weighting*B[Fpts,:].T
        fn = amg_core.new_ideal_interpolation
        fn( Y.indptr,
            Y.indices,
            Y.data,
            lqTopOp.indptr,
            lqTopOp.indices,
            lqTopOp.data,
            lqBottomOp.ravel(order='F'),
            rhsTop.indptr,
            rhsTop.indices,
            rhsTop.data,
            rhsBottom.ravel(order='F'),
            num_Fpts,
            num_Cpts,
            num_bad_guys )

    # Form P


    return P


