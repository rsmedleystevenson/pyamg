"""Tentative prolongator"""

__docformat__ = "restructuredtext en"

import numpy as np
from scipy.sparse import isspmatrix_csr, bsr_matrix, csr_matrix, vstack, identity
from pyamg.util.utils import filter_matrix_rows, truncate_rows
from pyamg import amg_core
import pdb

__all__ = ['fit_candidates','ben_ideal_interpolation']


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
        raise ValueError('dimensions of AggOp %s and B %s are '
                         'incompatible' % (AggOp.shape, B.shape))

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


def ben_ideal_interpolation(A, B, SOC, Cpts, AggOp=None, deg=1, prefilter={}):
    """ Ben ideal interpolation - form P = [W; I] via minimizing 

            ||I - WAcc^{-1}Acf||_F,  s.t.  PBc = B            
    
    Parameters
    ----------
    A : csr_matrix
        Operator on current level of hierarchy, dimension nxn.
    AggOp : csr_matrix
        Describes the sparsity pattern of the tentative prolongator.
        Has dimension (n, #aggregates)
    Cpts : array-like
        List of designated C-points, assumed to be passed in sorted.
    B : array
        The near-nullspace candidates stored in column-wise fashion.
        Has dimension (n, #candidates)
    SOC : csr_matrix

    d : int : Default 1
        Degree of expanding the sparsity pattern for P via multiplying
        with the SOC matrix.
    prefilter : {dictionary} : Default {}
        Filters elements by row in sparsity pattern for P to reduce operator and
        setup complexity. If None or empty dictionary, no dropping in P is done.
        If postfilter has key 'k', then the largest 'k' entries  are kept in each
        row.  If postfilter has key 'theta', all entries such that
            P[i,j] < kwargs['theta']*max(abs(P[i,:]))
        are dropped.  If postfilter['k'] and postfiler['theta'] are present, then
        they are used in conjunction, with the union of their patterns used.

    Returns
    -------
    P : csr_matrix
        Interpolation operator
    Bc : array-like
        Coarse-grid bad guys as B restricted to the coarse grid. 

    """

    if ('theta' in prefilter) and (prefilter['theta'] == 0):
        prefilter.pop('theta', None)

    B = np.asarray(B)
    if B.dtype not in ['float32', 'float64', 'complex64', 'complex128']:
        B = np.asarray(B, dtype='float64')

    if len(B.shape) != 2:
        raise ValueError('expected 2d array for argument B')

    # What if AggOp is none
    # if B.shape[0] % AggOp.shape[0] != 0:
    #     raise ValueError('dimensions of AggOp %s and B %s are \
    #                       incompatible' % (AggOp.shape, B.shape))

    if not isspmatrix_csr(A):
        try: 
            A = A.tocsr()
            warnings.warn("Warning, implicit conversion of A to csr.")
        except:
            raise TypeError("Incompatible matrix type A.")
    
    Cpts = np.array(Cpts)

    # Sort indices of A
    A.sort_indices()
    n = A.shape[0]
    num_Cpts = len(Cpts)
    num_bad_guys = B.shape[1]

    # Form initial sparsity pattern
    if AggOp == None:
        rowptr = np.zeros((n+1,),dtype='intc')
        rowptr[Cpts+1] = 1
        np.cumsum(rowptr, out=rowptr)
        S = csr_matrix((np.ones((num_Cpts,), dtype='intc'),
                        np.arange(0,num_Cpts),
                        rowptr),
                       dtype='float64')
    else:
        S = csr_matrix(AggOp, dtype='float64')

    # pdb.set_trace()

    # Expand sparsity pattern by multiplying with SOC 
    for i in range(0,deg):
        S = SOC * S

    # Make sure all rows have at least one nonzero. If not, 
    # 
    #   TODO :  Then what? 
    #
    missing = np.where( np.ediff1d(S.indptr) == 0)[0]
    # for row in missing:


    # Filter sparsity pattern
    if 'theta' in prefilter and 'k' in prefilter:
        temp = filter_matrix_rows(S, prefilter['theta'])
        S = truncate_rows(S, prefilter['k'])
        # Union two sparsity patterns
        S += temp
    elif 'k' in prefilter:
        S = truncate_rows(S, prefilter['k'])
    elif 'theta' in prefilter:
        S = filter_matrix_rows(S, prefilter['theta'])
    elif len(prefilter) > 0:
        raise ValueError("Unrecognized prefilter option")

    S.eliminate_zeros()
    S.sort_indices()

    # Form empty array for row pointer of P
    P_rowptr = np.empty((n+1,), dtype='intc')

    # Ben ideal interpolation
    fn = amg_core.ben_ideal_interpolation
    output = fn(A.indptr,
                A.indices,
                A.data,
                S.indptr,
                S.indices,
                P_rowptr,
                B.ravel(),
                Cpts,
                n,
                num_bad_guys )

    # P = csr_matrix((P_data, P_colinds, P_rowptr), shape=[n,num_Cpts])
    P = csr_matrix((np.array(output[1]),
                    np.array(output[0]),
                    P_rowptr),
                   shape=[n,num_Cpts])

    # Form coarse-grid bad guys as B restricted to coarse grid
    Bc = B[Cpts,:]

    # pdb.set_trace()

    return P, Bc



def test_ideal_interpolation(A, B, SOC, Cpts, AggOp=None, d=1, prefilter={}):
    """ 
    Test python implementation

    """

    if ('theta' in prefilter) and (prefilter['theta'] == 0):
        prefilter.pop('theta', None)

    B = np.asarray(B)
    if B.dtype not in ['float32', 'float64', 'complex64', 'complex128']:
        B = np.asarray(B, dtype='float64')

    if len(B.shape) != 2:
        raise ValueError('expected 2d array for argument B')

    if not isspmatrix_csr(A):
        try: 
            A = A.tocsr()
            warnings.warn("Warning, implicit conversion of A to csr.")
        except:
            raise TypeError("Incompatible matrix type A.")
    
    Cpts = np.array(Cpts)
    splitting = np.zeros((A.shape[0],))
    splitting[Cpts] = 1
    Fpts = np.where(splitting == 0)[0]

    # Sort indices of A
    A.sort_indices()
    n = A.shape[0]
    num_Cpts = len(Cpts)
    num_Fpts = len(Fpts)
    num_bad_guys = B.shape[1]

    # Form initial sparsity pattern
    if AggOp == None:
        rowptr = np.zeros((n+1,),dtype='intc')
        rowptr[Cpts+1] = 1
        np.cumsum(rowptr, out=rowptr)
        S = csr_matrix((np.ones((num_Cpts,), dtype='intc'),
                        np.arange(0,num_Cpts),
                        rowptr),
                       dtype='float64')
    else:
        S = csr_matrix(AggOp, dtype='float64')

    # Expand sparsity pattern by multiplying with SOC 
    for i in range(0,d):
        S = SOC * S

    S = csr_matrix(S[Fpts,:])
    S.eliminate_zeros()
    S.sort_indices()

    # pdb.set_trace()

    # Submatrices 
    Acc = A[Cpts,:][:,Cpts]
    Acf = -A[Cpts,:][:,Fpts]
    Bf = B[Fpts,:]
    Bc = B[Cpts,:]
    Bc_hat = Acc * Bc

    # Form empty array for row pointer of P
    P_rowptr = np.empty((n+1,), dtype='intc')

    # Ben ideal interpolation
    # solve minimization for every row
    for row in range(0,num_Fpts):

        # get indices for rows / columns of A needed for the sparsity pattern of this row
        row_ind = S.indices[ S.indptr[row]:S.indptr[row+1] ]
        col_ind = np.concatenate([Acf[i,:].indices for i in row_ind])
        col_ind = np.array(sorted(list(set(col_ind))) )

        if len(row_ind) > 0:
            # compute right hand side vectors 
            rhs = np.zeros((1,len(col_ind)))
            l_ind = np.where(col_ind == row)[0]
            if len(l_ind) > 0:
                rhs[0,l_ind[0]] = 1.0
            else:
                print "row ",row," not in sparsity"
                # pdb.set_trace()

            # form least squares operator
            operator = Acf[row_ind,:][:,col_ind].todense()

            # form constraint
            constraint = Bc_hat[row_ind,:]
            constraint_rhs = Bf[row,:]

        # No rows in sparsity
        else: 
            pdb.set_trace()

        # Solve constraint exactly
        qc, rc = np.linalg.qr(constraint, mode='complete')
        con_sol = np.linalg.solve(rc[0:num_bad_guys,0:num_bad_guys].T, constraint_rhs).T

        # Move to rhs
        new_op = np.dot(operator.T, qc)
        rhs -= np.dot(new_op[:,0:num_bad_guys], con_sol)

        # solve reduced least squares
        if new_op.shape[1] > num_bad_guys:  
            new_op = new_op[:,num_bad_guys:]
            q, r = np.linalg.qr(new_op)
            rhs = np.dot(q.T, rhs.T)
            lq_sol = np.array(np.linalg.solve(r, rhs))
        else:
            lq_sol = np.array(())

        # form final solution
        S.data[ S.indptr[row]:S.indptr[row+1] ] = np.dot(qc, np.concatenate([con_sol.ravel(),lq_sol.ravel()]) )


    P = vstack([ S*Acc, identity(num_Cpts)], 'csr')
 
    # Arrange rows of P in proper order
    permute = identity(A.shape[0],format='csr')
    permute.indices = np.concatenate((Fpts,Cpts))
    permute = permute.T;
    P = csr_matrix(permute*P)
    P.sort_indices()

    # pdb.set_trace()

    return P, Bc

