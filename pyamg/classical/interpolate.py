"""Classical AMG Interpolation methods"""


__docformat__ = "restructuredtext en"

from warnings import warn
import numpy as np
from scipy.sparse import csr_matrix, bsr_matrix, isspmatrix_csr, \
        isspmatrix_bsr, SparseEfficiencyWarning
from pyamg import amg_core
from pyamg.relaxation.relaxation import boundary_relaxation
from pyamg.strength import classical_strength_of_connection

__all__ = ['direct_interpolation', 'standard_interpolation',
           'trivial_interpolation', 'injection_interpolation',
           'approximate_ideal_restriction']


# TODO : Figure out why other classical interpolate routines want zero diagonal
#        in SOC matrix, and if SOC matrix actually has zero diagonal???


def direct_interpolation(A, C, splitting, cost=[0]):
    """Create prolongator using direct interpolation

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format
    C : {csr_matrix}
        Strength-of-Connection matrix
        Must have zero diagonal
    splitting : array
        C/F splitting stored in an array of length N

    Returns
    -------
    P : {csr_matrix}
        Prolongator using direct interpolation

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical import direct_interpolation
    >>> import numpy as np
    >>> A = poisson((5,),format='csr')
    >>> splitting = np.array([1,0,1,0,1], dtype='intc')
    >>> P = direct_interpolation(A, A, splitting)
    >>> print P.todense()
    [[ 1.   0.   0. ]
     [ 0.5  0.5  0. ]
     [ 0.   1.   0. ]
     [ 0.   0.5  0.5]
     [ 0.   0.   1. ]]

    """
    if not isspmatrix_csr(A):
        raise TypeError('expected csr_matrix for A')

    if not isspmatrix_csr(C):
        raise TypeError('expected csr_matrix for C')

    # Interpolation weights are computed based on entries in A, but subject to
    # the sparsity pattern of C.  So, copy the entries of A into the
    # sparsity pattern of C.
    C = C.copy()
    C.data[:] = 1.0
    C = C.multiply(A)

    Pp = np.empty_like(A.indptr)

    amg_core.rs_direct_interpolation_pass1(A.shape[0],
                                           C.indptr, C.indices, splitting, Pp)

    nnz = Pp[-1]
    Pj = np.empty(nnz, dtype=Pp.dtype)
    Px = np.empty(nnz, dtype=A.dtype)

    amg_core.rs_direct_interpolation_pass2(A.shape[0],
                                           A.indptr, A.indices, A.data,
                                           C.indptr, C.indices, C.data,
                                           splitting,
                                           Pp, Pj, Px)

    return csr_matrix((Px, Pj, Pp))


def standard_interpolation(A, C, splitting, cost=[0]):
    """Create prolongator using standard interpolation

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format
    C : {csr_matrix}
        Strength-of-Connection matrix
        Must have zero diagonal
    splitting : array
        C/F splitting stored in an array of length N

    Returns
    -------
    P : {csr_matrix}
        Prolongator using standard interpolation

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical import standard_interpolation
    >>> import numpy as np
    >>> A = poisson((5,),format='csr')
    >>> splitting = np.array([1,0,1,0,1], dtype='intc')
    >>> P = standard_interpolation(A, A, splitting)
    >>> print P.todense()
    [[ 1.   0.   0. ]
     [ 0.5  0.5  0. ]
     [ 0.   1.   0. ]
     [ 0.   0.5  0.5]
     [ 0.   0.   1. ]]

    """
    if not isspmatrix_csr(A):
        raise TypeError('expected csr_matrix for A')

    if not isspmatrix_csr(C):
        raise TypeError('expected csr_matrix for C')

    # Interpolation weights are computed based on entries in A, but subject to
    # the sparsity pattern of C.  So, copy the entries of A into the
    # sparsity pattern of C.
    C = C.copy()
    C.data[:] = 1.0
    C = C.multiply(A)

    Pp = np.empty_like(A.indptr)
    amg_core.rs_standard_interpolation_pass1(A.shape[0], C.indptr,
    										 C.indices, splitting, Pp)

    nnz = Pp[-1]
    Pj = np.empty(nnz, dtype=Pp.dtype)
    Px = np.empty(nnz, dtype=A.dtype)

    amg_core.rs_standard_interpolation_pass2(A.shape[0],
                                             A.indptr, A.indices, A.data,
                                             C.indptr, C.indices, C.data,
                                             splitting,
                                             Pp, Pj, Px)
    return  csr_matrix((Px, Pj, Pp))


def trivial_interpolation(A, splitting, cost=[0]):
    """ Create trivial classical interpolation operator, that is
    C-points are interpolated by injection and F-points are not
    interpolated.

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format or BSR format
    splitting : array
        C/F splitting stored in an array of length N

    Returns
    -------
    NxNc interpolation operator, P
    """
    if isspmatrix_bsr(A):
        blocksize = A.blocksize[0]
        n = A.shape[0] / blocksize
    elif isspmatrix_csr(A):
        n = A.shape[0]
        blocksize = 1
    else:
        try:
            A = A.tocsr()
            warn("Implicit conversion of A to csr", sparse.SparseEfficiencyWarning)
            n = A.shape[0]
            blocksize = 1
        except:
            raise TypeError("Invalid matrix type, must be CSR or BSR.")

    P_rowptr = np.append(np.array([0],dtype='int32'), np.cumsum(splitting,dtype='int32') )
    nc = P_rowptr[-1]
    P_colinds = np.arange(start=0, stop=nc, step=1, dtype='int32')

    if blocksize == 1:
        return csr_matrix((np.ones((nc,), dtype=A.dtype), P_colinds, P_rowptr), shape=[n,nc])
    else:
        P_data = np.array(nc*[np.identity(blocksize, dtype=A.dtype)], dtype=A.dtype)
        return bsr_matrix((P_data, P_colinds, P_rowptr), blocksize=[blocksize,blocksize],
                          shape=[n*blocksize,nc*blocksize])


def injection_interpolation(A, C, splitting, cost=[0]):
    """ Create full injection interpolation operator, that is
    C-points are interpolated by injection and F-points are
    interpolated by injection from their strongest-connected
    C-point neighbor.

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format
    C : {csr_matrix}
        Strength-of-Connection matrix (does not need zero diagonal)
    splitting : array
        C/F splitting stored in an array of length N

    Returns
    -------
    NxNc interpolation operator, P
    """
    if isspmatrix_bsr(A):
        blocksize = A.blocksize[0]
        n = A.shape[0] / blocksize
    elif isspmatrix_csr(A):
        n = A.shape[0]
        blocksize = 1
    else:
        try:
            A = A.tocsr()
            warn("Implicit conversion of A to csr", sparse.SparseEfficiencyWarning)
            n = A.shape[0]
            blocksize = 1
        except:
            raise TypeError("Invalid matrix type, must be CSR or BSR.")

    nc = np.sum(splitting)
    P_rowptr = np.arange(start=0, stop=(n+1), step=1, dtype='int32')
    P_colinds = np.empty((n,),dtype='int32')
    amg_core.injection_interpolation(P_rowptr, P_colinds, C.indptr,
                                     C.indices, C.data, splitting)
    if blocksize == 1:
        P_data = np.ones((n,), dtype=A.dtype)
        return csr_matrix((P_data,P_colinds,P_rowptr), shape=[n,nc])
    else:
        P_data = np.array(n*[np.identity(blocksize, dtype=A.dtype)], dtype=A.dtype)
        return bsr_matrix((P_data,P_colinds,P_rowptr), blocksize=[blocksize,blocksize],
                          shape=[blocksize*n,blocksize*nc])


def approximate_ideal_restriction(A, splitting, theta=0.1, max_row=None, degree=1, cost=[0]):
    """ Compute approximate ideal restriction by setting RA = 0, within the
    sparsity pattern of R. Sparsity pattern of R for the ith row (i.e. ith
    C-point) is the set of all strongly connected F-points, or the max_row
    *most* strongly connected F-points.

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR or BSR format
    theta : float, default 0.1
        Solve local system for each row of R for all values
            |A_ij| >= 0.1 * max_{i!=k} |A_ik|
    degree : int, default 1
        Expand sparsity pattern for R by considering strongly connected
        neighbors within 'degree' of a given node 
    splitting : array
        C/F splitting stored in an array of length N
    max_row : int
        Maximum size of sparsity pattern for any row in R

    Returns
    -------
    Approximate ideal restriction, R, in same sparse format as A.

    """

    # Get SOC matrix containing neighborhood to be included in local solve
    if isspmatrix_bsr(A):
        C = classical_strength_of_connection(A=A, theta=theta, block='amalgamate', norm='abs')
        blocksize = A.blocksize[0]
    elif isspmatrix_csr(A):
        blocksize = 1
        C = classical_strength_of_connection(A=A, theta=theta, block=None, norm='abs')
    else:
        try:
            A = A.tocsr()
            warn("Implicit conversion of A to csr", sparse.SparseEfficiencyWarning)
            C = classical_strength_of_connection(A=A, theta=theta, block=None, norm='abs')
            blocksize = 1
        except:
            raise TypeError("Invalid matrix type, must be CSR or BSR.")

    Cpts = np.array(np.where(splitting == 1)[0], dtype='int32')
    nc = Cpts.shape[0]
    n = C.shape[0]

    # Expand sparsity pattern for R
    if degree == 1:
        pass
    elif degree == 2:
        C = csr_matrix(C*C)
    elif degree == 3:
        C = csr_matrix(C*C*C)
    elif degree == 4:
        C = csr_matrix(C*C)
        C = csr_matrix(C*C)
    else:
        raise ValueError("Only sparsity degree 1-4 supported.")

    # Form row pointer for R
    R_rowptr = np.empty(nc+1, dtype='int32')
    if max_row is None:
        amg_core.approx_ideal_restriction_pass1(R_rowptr, C.indptr, C.indices,
                                                C.data, Cpts, splitting)
    else:
        amg_core.approx_ideal_restriction_pass1(R_rowptr, C.indptr, C.indices,
                                                C.data, Cpts, splitting, max_row)

    # Build restriction operator
    nnz = R_rowptr[-1]
    R_colinds = np.zeros(nnz, dtype='int32')

    # Block matrix
    if isspmatrix_bsr(A):
        R_data = np.zeros(nnz*blocksize*blocksize, dtype=A.dtype)
        amg_core.block_approx_ideal_restriction_pass2(R_rowptr, R_colinds, R_data, A.indptr,
                                                      A.indices, A.data.ravel(), C.indptr,
                                                      C.indices, C.data, Cpts, splitting,
                                                      blocksize)
        R = bsr_matrix((R_data.reshape(nnz,blocksize,blocksize), R_colinds, R_rowptr),
                        blocksize=[blocksize,blocksize], shape=[nc*blocksize,A.shape[0]])
    # Not block matrix
    else:
        R_data = np.zeros(nnz, dtype=A.dtype)
        amg_core.approx_ideal_restriction_pass2(R_rowptr, R_colinds, R_data, A.indptr,
                                                A.indices, A.data, C.indptr, C.indices,
                                                C.data, Cpts, splitting)
        R = csr_matrix((R_data, R_colinds, R_rowptr), shape=[nc,A.shape[0]])

    R.eliminate_zeros()
    return R



