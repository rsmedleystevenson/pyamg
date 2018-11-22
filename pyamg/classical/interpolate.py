"""Classical AMG Interpolation methods"""
__docformat__ = "restructuredtext en"

from warnings import warn
import numpy as np
from scipy.sparse import csr_matrix, bsr_matrix, isspmatrix_csr, \
        isspmatrix_bsr, SparseEfficiencyWarning, eye, hstack, vstack, diags
from pyamg import amg_core
from pyamg.relaxation.relaxation import boundary_relaxation
from pyamg.strength import classical_strength_of_connection
from pyamg.util.utils import filter_matrix_rows, UnAmal
import numba

__all__ = ['direct_interpolation', 'standard_interpolation',
           'one_point_interpolation', 'injection_interpolation',
           'neumann_ideal_interpolation', 'neumann_AIR',
           'local_AIR', 'distance_two_interpolation']


@numba.jit(cache=True, nopython=True)
def pinv_nla_jit(A):
    return np.linalg.pinv(A)

def direct_interpolation(A, C, splitting, theta=None, norm='min', cost=[0]):
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
    theta : float in [0,1), default None
        theta value defining strong connections in a classical AMG sense. Provide if
        different SOC used for P than for CF-splitting; otherwise, theta = None. 
    norm : string, default 'abs'
        Norm used in redefining classical SOC. Options are 'min' and 'abs' for CSR matrices,
        and 'min', 'abs', and 'fro' for BSR matrices. See strength.py for more information.

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
    if not isspmatrix_csr(C):
        raise TypeError('Expected csr_matrix SOC matrix, C.')

    # Block BSR format. Transfer A to CSR and the splitting and SOC matrix to have
    # DOFs corresponding to CSR A
    if isspmatrix_bsr(A):
        temp_A = A.tocsr()
        temp_A.eliminate_zeros()
        splitting0 = splitting * np.ones((A.blocksize[0],1), dtype='intc')
        splitting0 = np.reshape(splitting0, (np.prod(splitting0.shape),), order='F')
        if theta is not None:
            C0 = classical_strength_of_connection(A, theta=theta, norm=norm, cost=cost)
            C0 = UnAmal(C0, A.blocksize[0], A.blocksize[1])
        else:
            C0 = UnAmal(C, A.blocksize[0], A.blocksize[1])
        C0 = C0.tocsr()
        C0.eliminate_zeros()

        # Interpolation weights are computed based on entries in A, but subject to the
        # sparsity pattern of C.  So, copy the entries of A into sparsity pattern of C.
        C0.data[:] = 1.0
        C0 = C0.multiply(temp_A)

        P_indptr = np.empty_like(temp_A.indptr)
        amg_core.rs_direct_interpolation_pass1(temp_A.shape[0], C0.indptr, C0.indices, 
                                               splitting0, P_indptr)

        nnz = P_indptr[-1]
        P_colinds = np.empty(nnz, dtype=P_indptr.dtype)
        P_data = np.empty(nnz, dtype=A.dtype)

        amg_core.rs_direct_interpolation_pass2(temp_A.shape[0], temp_A.indptr, temp_A.indices,
                                               temp_A.data, C0.indptr, C0.indices, C0.data,
                                               splitting0, P_indptr, P_colinds, P_data)

        nc = np.sum(splitting0)
        n = A.shape[0]
        P = csr_matrix((P_data, P_colinds, P_indptr), shape=[n,nc])
        return P.tobsr(blocksize=A.blocksize)

    # CSR format
    else:
        if theta is not None:
            C0 = classical_strength_of_connection(A, theta=theta, norm=norm, cost=cost)
        else:
            C0 = C.copy()
        C0.eliminate_zeros()

        # Interpolation weights are computed based on entries in A, but subject to the
        # sparsity pattern of C.  So, copy the entries of A into sparsity pattern of C.
        C0.data[:] = 1.0
        C0 = C0.multiply(A)

        P_indptr = np.empty_like(A.indptr)
        amg_core.rs_direct_interpolation_pass1(A.shape[0], C0.indptr, C0.indices, 
                                               splitting, P_indptr)
        nnz = P_indptr[-1]
        P_colinds = np.empty(nnz, dtype=P_indptr.dtype)
        P_data = np.empty(nnz, dtype=A.dtype)

        amg_core.rs_direct_interpolation_pass2(A.shape[0], A.indptr, A.indices,
                                               A.data, C0.indptr, C0.indices, C0.data,
                                               splitting, P_indptr, P_colinds, P_data)

        nc = np.sum(splitting)
        n = A.shape[0]
        return csr_matrix((P_data, P_colinds, P_indptr), shape=[n,nc])


def standard_interpolation(A, C, splitting, theta=None, norm='min', modified=True, cost=[0]):
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
    theta : float in [0,1), default None
        theta value defining strong connections in a classical AMG sense. Provide if
        different SOC used for P than for CF-splitting; otherwise, theta = None. 
    norm : string, default 'abs'
        Norm used in redefining classical SOC. Options are 'min' and 'abs' for CSR matrices,
        and 'min', 'abs', and 'fro' for BSR matrices. See strength.py for more information.
    modified : bool, default True
        Use modified classical interpolation. More robust if RS coarsening with second
        pass is not used for CF splitting. Ignores interpolating from strong F-connections
        without a common C-neighbor.

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
    if not isspmatrix_csr(C):
        raise TypeError('Expected csr_matrix SOC matrix, C.')

    nc = np.sum(splitting)
    n = A.shape[0]

    # Block BSR format. Transfer A to CSR and the splitting and SOC matrix to have
    # DOFs corresponding to CSR A
    if isspmatrix_bsr(A):
        temp_A = A.tocsr()
        splitting0 = splitting * np.ones((A.blocksize[0],1), dtype='intc')
        splitting0 = np.reshape(splitting0, (np.prod(splitting0.shape),), order='F')
        if theta is not None:
            C0 = classical_strength_of_connection(A, theta=theta, norm=norm, cost=cost)
            C0 = UnAmal(C0, A.blocksize[0], A.blocksize[1])
        else:
            C0 = UnAmal(C, A.blocksize[0], A.blocksize[1])
        C0 = C0.tocsr()

        # Use modified standard interpolation by ignoring strong F-connections that do
        # not have a common C-point.
        if modified:
            amg_core.remove_strong_FF_connections(temp_A.shape[0], C0.indptr, C0.indices,
                                                  C0.data, splitting)
        C0.eliminate_zeros()

        # Interpolation weights are computed based on entries in A, but subject to
        # the sparsity pattern of C.  So, copy the entries of A into the
        # sparsity pattern of C.
        C0.data[:] = 1.0
        C0 = C0.multiply(temp_A)

        P_indptr = np.empty_like(temp_A.indptr)
        amg_core.rs_standard_interpolation_pass1(temp_A.shape[0], C0.indptr,
                                                 C0.indices, splitting0, P_indptr)
        nnz = P_indptr[-1]
        P_colinds = np.empty(nnz, dtype=P_indptr.dtype)
        P_data = np.empty(nnz, dtype=temp_A.dtype)

        if modified:
            amg_core.mod_standard_interpolation_pass2(temp_A.shape[0], temp_A.indptr, temp_A.indices,
                                                      temp_A.data, C0.indptr, C0.indices,
                                                      C0.data, splitting0, P_indptr,
                                                      P_colinds, P_data)
        else:
            amg_core.rs_standard_interpolation_pass2(temp_A.shape[0], temp_A.indptr, temp_A.indices,
                                                     temp_A.data, C0.indptr, C0.indices,
                                                     C0.data, splitting0, P_indptr,
                                                     P_colinds, P_data)

        nc = np.sum(splitting0)
        n = A.shape[0] 
        P = csr_matrix((P_data, P_colinds, P_indptr), shape=[n,nc])
        return P.tobsr(blocksize=A.blocksize)

    # CSR format
    else:
        if theta is not None:
            C0 = classical_strength_of_connection(A, theta=theta, norm=norm, cost=cost)
        else:
            C0 = C.copy()

        # Use modified standard interpolation by ignoring strong F-connections that do
        # not have a common C-point.
        if modified:
            amg_core.remove_strong_FF_connections(A.shape[0], C0.indptr, C0.indices, C0.data, splitting)
        C0.eliminate_zeros()

        # Interpolation weights are computed based on entries in A, but subject to
        # the sparsity pattern of C.  So, copy the entries of A into the
        # sparsity pattern of C.
        C0.data[:] = 1.0
        C0 = C0.multiply(A)

        P_indptr = np.empty_like(A.indptr)
        amg_core.rs_standard_interpolation_pass1(A.shape[0], C0.indptr,
                                                 C0.indices, splitting, P_indptr)
        nnz = P_indptr[-1]
        P_colinds = np.empty(nnz, dtype=P_indptr.dtype)
        P_data = np.empty(nnz, dtype=A.dtype)

        if modified:
            amg_core.mod_standard_interpolation_pass2(A.shape[0], A.indptr, A.indices,
                                                      A.data, C0.indptr, C0.indices,
                                                      C0.data, splitting, P_indptr,
                                                      P_colinds, P_data)
        else:
            amg_core.rs_standard_interpolation_pass2(A.shape[0], A.indptr, A.indices,
                                                     A.data, C0.indptr, C0.indices,
                                                     C0.data, splitting, P_indptr,
                                                     P_colinds, P_data)
        nc = np.sum(splitting)
        n = A.shape[0]
        return csr_matrix((P_data, P_colinds, P_indptr), shape=[n,nc])


def distance_two_interpolation(A, C, splitting, theta=None, norm='min', plus_i=True, cost=[0]):
    """Create prolongator using distance-two AMG interpolation (extended+i interpolaton).

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format
    C : {csr_matrix}
        Strength-of-Connection matrix
        Must have zero diagonal
    splitting : array
        C/F splitting stored in an array of length N
    theta : float in [0,1), default None
        theta value defining strong connections in a classical AMG sense. Provide if
        different SOC used for P than for CF-splitting; otherwise, theta = None. 
    norm : string, default 'abs'
        Norm used in redefining classical SOC. Options are 'min' and 'abs' for CSR matrices,
        and 'min', 'abs', and 'fro' for BSR matrices. See strength.py for more information.
    plus_i : bool, default True
        Use "Extended+i" interpolation from [0] as opposed to "Extended" interpolation. Typically
        gives better interpolation with minimal added expense.

    Returns
    -------
    P : {csr_matrix}
        Prolongator using standard interpolation

    References
    ----------
    [0] "Distance-Two Interpolation for Parallel Algebraic Multigrid,"
       H. De Sterck, R. Falgout, J. Nolting, U. M. Yang, (2007).

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
    if not isspmatrix_csr(C):
        raise TypeError('Expected csr_matrix SOC matrix, C.')

    nc = np.sum(splitting)
    n = A.shape[0]

    # Block BSR format. Transfer A to CSR and the splitting and SOC matrix to have
    # DOFs corresponding to CSR A
    if isspmatrix_bsr(A):
        temp_A = A.tocsr()
        splitting0 = splitting * np.ones((A.blocksize[0],1), dtype='intc')
        splitting0 = np.reshape(splitting0, (np.prod(splitting0.shape),), order='F')
        if theta is not None:
            C0 = classical_strength_of_connection(A, theta=theta, norm=norm, cost=cost)
            C0 = UnAmal(C0, A.blocksize[0], A.blocksize[1])
        else:
            C0 = UnAmal(C, A.blocksize[0], A.blocksize[1])
        C0 = C0.tocsr()
        C0.eliminate_zeros()

        # Interpolation weights are computed based on entries in A, but subject to
        # the sparsity pattern of C.  So, copy the entries of A into the
        # sparsity pattern of C.
        C0.data[:] = 1.0
        C0 = C0.multiply(temp_A)

        P_indptr = np.empty_like(temp_A.indptr)
        amg_core.distance_two_amg_interpolation_pass1(temp_A.shape[0], C0.indptr,
                                                      C0.indices, splitting0, P_indptr)
        nnz = P_indptr[-1]
        P_colinds = np.empty(nnz, dtype=P_indptr.dtype)
        P_data = np.empty(nnz, dtype=temp_A.dtype)
        if plus_i:
            amg_core.extended_plusi_interpolation_pass2(temp_A.shape[0], temp_A.indptr, temp_A.indices,
                                                        temp_A.data, C0.indptr, C0.indices,
                                                        C0.data, splitting0, P_indptr,
                                                        P_colinds, P_data)
        else:
            amg_core.extended_interpolation_pass2(temp_A.shape[0], temp_A.indptr, temp_A.indices,
                                                  temp_A.data, C0.indptr, C0.indices,
                                                  C0.data, splitting0, P_indptr,
                                                  P_colinds, P_data)
        nc = np.sum(splitting0)
        n = A.shape[0] 
        P = csr_matrix((P_data, P_colinds, P_indptr), shape=[n,nc])
        return P.tobsr(blocksize=A.blocksize)

    # CSR format
    else:
        if theta is not None:
            C0 = classical_strength_of_connection(A, theta=theta, norm=norm, cost=cost)
        else:
            C0 = C.copy()
        C0.eliminate_zeros()

        # Interpolation weights are computed based on entries in A, but subject to
        # the sparsity pattern of C.  So, copy the entries of A into the
        # sparsity pattern of C.
        C0.data[:] = 1.0
        C0 = C0.multiply(A)

        P_indptr = np.empty_like(A.indptr)
        amg_core.distance_two_amg_interpolation_pass1(A.shape[0], C0.indptr,
                                                      C0.indices, splitting, P_indptr)
        nnz = P_indptr[-1]
        P_colinds = np.empty(nnz, dtype=P_indptr.dtype)
        P_data = np.empty(nnz, dtype=A.dtype)
        if plus_i:
            amg_core.extended_plusi_interpolation_pass2(A.shape[0], A.indptr, A.indices,
                                                        A.data, C0.indptr, C0.indices,
                                                        C0.data, splitting, P_indptr,
                                                        P_colinds, P_data)
        else:
            amg_core.extended_interpolation_pass2(A.shape[0], A.indptr, A.indices,
                                                  A.data, C0.indptr, C0.indices,
                                                  C0.data, splitting, P_indptr,
                                                  P_colinds, P_data)
        nc = np.sum(splitting)
        n = A.shape[0]
        return csr_matrix((P_data, P_colinds, P_indptr), shape=[n,nc])


def injection_interpolation(A, splitting, cost=[0]):
    """ Create interpolation operator by injection, that is C-points are
    interpolated by value and F-points are not interpolated.

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
            warn("Implicit conversion of A to csr", SparseEfficiencyWarning)
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


def one_point_interpolation(A, C, splitting, by_val=False, cost=[0]):
    """ Create one-point interpolation operator, that is C-points are
    interpolated by value and F-points are interpolated by value from
    their strongest-connected C-point neighbor.

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format
    C : {csr_matrix}
        Strength-of-Connection matrix (does not need zero diagonal)
    by_val : bool
        For CSR matrices only right now, use values of -Afc in interp as an
        approximation to P_ideal. If false, F-points are interpolated by value
        with weight 1.
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
            warn("Implicit conversion of A to csr", SparseEfficiencyWarning)
            n = A.shape[0]
            blocksize = 1
        except:
            raise TypeError("Invalid matrix type, must be CSR or BSR.")

    nc = np.sum(splitting)
    P_rowptr = np.empty((n+1,), dtype='int32') # P: n x nc, at most 'n' nnz
    P_colinds = np.empty((n,),dtype='int32')
    P_data = np.empty((n,),dtype=A.dtype)

    #amg_core.one_point_interpolation(P_rowptr, P_colinds, A.indptr,
    #                                 A.indices, A.data, splitting)
    if blocksize == 1:
        if by_val:
            amg_core.one_point_interpolation(P_rowptr, P_colinds, P_data, A.indptr,
                                     A.indices, A.data, splitting)
            return csr_matrix((P_data,P_colinds,P_rowptr), shape=[n,nc])
        else:
            amg_core.one_point_interpolation(P_rowptr, P_colinds, P_data, C.indptr,
                                     C.indices, C.data, splitting)
            P_data = np.ones((n,), dtype=A.dtype)
            return csr_matrix((P_data,P_colinds,P_rowptr), shape=[n,nc])
    else:
        amg_core.one_point_interpolation(P_rowptr, P_colinds, P_data, C.indptr,
                         C.indices, C.data, splitting)
        P_data = np.array(n*[np.identity(blocksize, dtype=A.dtype)], dtype=A.dtype)
        return bsr_matrix((P_data,P_colinds,P_rowptr), blocksize=[blocksize,blocksize],
                          shape=[blocksize*n,blocksize*nc])


def neumann_AIR(A, splitting, theta=0.025, degree=1, post_theta=0, cost=[0]):
    """ Approximate ideal restriction using a truncated Neumann expansion for A_ff^{-1},
    where 
        R = [-Acf*D, I],   where
        D = \sum_{i=0}^degree Lff^i

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format
    splitting : array
        C/F splitting stored in an array of length N
    theta : float : default 0.025
        Compute approximation to ideal restriction for C, where C has rows filtered
        with tolerance theta, that is for j s.t.
            |C_ij| <= theta * |C_ii|        --> C_ij = 0.
        Helps keep R sparse. 
    degree : int in [0,4] : default 1
        Degree of Neumann expansion. Only supported up to degree 4.

    Returns
    -------
    Approximate ideal restriction in CSR format.

    Notes
    -----
    Does not support block matrices.
    """

    Cpts = np.array(np.where(splitting == 1)[0], dtype='int32')
    Fpts = np.array(np.where(splitting == 0)[0], dtype='int32')
    nf0 = len(Fpts)

    # Convert block CF-splitting into scalar CF-splitting so that we can access
    # submatrices of BSR matrix A
    if isspmatrix_bsr(A):
        bsize = A.blocksize[0]
        Cpts *= bsize
        Fpts *= bsize
        Cpts0 = Cpts
        Fpts0 = Fpts
        for i in range(1,bsize):
            Cpts = np.hstack([Cpts,Cpts0+i])
            Fpts = np.hstack([Fpts,Fpts0+i])
        Cpts.sort()
        Fpts.sort()
    
    nc = Cpts.shape[0]
    nf = Fpts.shape[0]
    n = A.shape[0]
    C = csr_matrix(A, copy=True)
    if theta > 0.0:
        filter_matrix_rows(C, theta, diagonal=True, lump=False)

    # Expand sparsity pattern for R
    C.data[np.abs(C.data)<1e-16] = 0
    C.eliminate_zeros()

    Acf = C[Cpts,:][:,Fpts]
    Lff = -C[Fpts,:][:,Fpts]
    if isspmatrix_bsr(A):
        bsize = A.blocksize[0]
        Lff = Lff.tobsr(blocksize=[bsize,bsize])

        rows = np.zeros(Lff.indices.shape[0])
        for i in range(0,nf0):
            rows[Lff.indptr[i]:Lff.indptr[i+1]] = i
        rows = rows-Lff.indices[:]
        diag = np.nonzero(rows == 0)
        D_data = Lff.data[diag,:,:]
        # Set diagonal block to zero in Lff
        Lff.data[diag,:,:] = 0.0
        for i in range(0,nf0):
            D_data[i] = -pinv_nla_jit(D_data[i])
        
        #D_data = np.empty((nf0,bsize,bsize))
        #for i in range(0,nf0):
        #    offset = np.where(Lff.indices[Lff.indptr[i]:Lff.indptr[i+1]]==i)[0][0]
        #    # Save (pseudo)inverse of diagonal block
        #    #D_data[i] = -np.linalg.pinv(Lff.data[Lff.indptr[i]+offset])
        #    D_data[i] = -pinv_nla_jit(Lff.data[Lff.indptr[i]+offset])
        #    # Set diagonal block to zero in Lff
        #    Lff.data[Lff.indptr[i]+offset][:] = 0.0
        Dff_inv = bsr_matrix((D_data,np.arange(0,nf0),np.arange(0,nf0+1)),blocksize=[bsize,bsize])
        Lff = Dff_inv*Lff
    else:
        pts = np.arange(0,nf)
        D_data = -1.0/Lff.diagonal()
        Lff[pts,pts] = 0.0
        Lff.eliminate_zeros()
        Dff_inv = csr_matrix((D_data,np.arange(0,nf),np.arange(0,nf+1)))
        Lff = Dff_inv*Lff

    # Form Neuman approximation to Aff^{-1}
    Z = eye(nf,format='csr')
    if degree >= 1:
        Z += Lff
    if degree >= 2:
        Z += Lff*Lff
    if degree >= 3:
        Z += Lff*Lff*Lff
    if degree == 4:
        Z += Lff*Lff*Lff*Lff
    if degree > 4:
        raise ValueError("Only sparsity degree 0-4 supported.")
    Z = Z*Dff_inv

    # Multiply Acf by approximation to Aff^{-1}
    Z = -Acf*Z

    if post_theta > 0.0:
        if not isspmatrix_csr(Z):
            Z = Z.tocsr()
        filter_matrix_rows(Z, post_theta, diagonal=False, lump=False)

    # Get sizes and permutation matrix from [F, C] block
    # ordering to natural matrix ordering.
    permute = eye(n,format='csr')
    permute.indices = np.concatenate((Fpts,Cpts))

    # Form R = [Z, I], reorder and return
    R = hstack([Z, eye(nc, format='csr')])
    if isspmatrix_bsr(A):
        R = bsr_matrix(R * permute, blocksize=[bsize,bsize])
    else:
        R = csr_matrix(R * permute)
    return R


def scaled_Afc_interpolation(A, splitting, theta=0.0, cost=[0]):
    """ Approximate ideal interpolation using a scaled Afc: 
        P = [-D*Afc; I],   where
        D is such that P has row-sum 1

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format
    splitting : array
        C/F splitting stored in an array of length N
    theta : float : default 0.025
        Compute approximation to ideal restriction for C, where C has rows filtered
        with tolerance theta, that is for j s.t.
            |C_ij| <= theta * |C_ii|        --> C_ij = 0.
        Helps keep R sparse. 
    degree : int in [0,4] : default 1
        Degree of Neumann expansion. Only supported up to degree 4.

    Returns
    -------
    Approximate ideal interpolation in CSR format.

    Notes
    -----
    Does not support block matrices.
    """
    A = A.tocsr()
    warn("Implicit conversion of A to csr", SparseEfficiencyWarning)

    if theta > 0.0:
        C = csr_matrix(A, copy=True)
        filter_matrix_rows(C, theta, diagonal=True, lump=False)
    else:
        C = A

    Cpts = np.array(np.where(splitting == 1)[0], dtype='int32')
    Fpts = np.array(np.where(splitting == 0)[0], dtype='int32')
    nc = Cpts.shape[0]
    nf = Fpts.shape[0]
    n = C.shape[0]

    # Expand sparsity pattern for R
    C.data[np.abs(C.data)<1e-16] = 0
    C.eliminate_zeros()
    Afc = C[Fpts,:][:,Cpts]
    Afc = Afc.tocsr()
    rowsums = Afc.sum(1)
    rowsums[np.abs(rowsums) < 1e-15] = 1
    rowsums = 1.0/rowsums
    D = diags(rowsums.A1,0,format='csr')
    Afc = D*Afc

    # Get sizes and permutation matrix from [F, C] block
    # ordering to natural matrix ordering.
    permute = eye(n,format='csr')
    permute.indices = np.concatenate((Fpts,Cpts))
    permute = permute.T

    # Form R = [P, I], reorder and return
    P = vstack([Afc, eye(nc, format='csr')])
    P = csr_matrix(permute * P)
    return P


def neumann_ideal_interpolation(A, splitting, theta=0.0, degree=1, cost=[0]):
    """ Approximate ideal interpolation using a truncated Neumann expansion for A_ff^{-1},
    where 
        P = [-D*Afc; I],   where
        D = \sum_{i=0}^degree Lff^i

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format
    splitting : array
        C/F splitting stored in an array of length N
    theta : float : default 0.025
        Compute approximation to ideal restriction for C, where C has rows filtered
        with tolerance theta, that is for j s.t.
            |C_ij| <= theta * |C_ii|        --> C_ij = 0.
        Helps keep R sparse. 
    degree : int in [0,4] : default 1
        Degree of Neumann expansion. Only supported up to degree 4.

    Returns
    -------
    Approximate ideal interpolation in CSR format.

    Notes
    -----
    Does not support block matrices.
    """
    A = A.tocsr()
    warn("Implicit conversion of A to csr", SparseEfficiencyWarning)

    if theta > 0.0:
        C = csr_matrix(A, copy=True)
        filter_matrix_rows(C, theta, diagonal=True, lump=False)
    else:
        C = A

    Cpts = np.array(np.where(splitting == 1)[0], dtype='int32')
    Fpts = np.array(np.where(splitting == 0)[0], dtype='int32')
    nc = Cpts.shape[0]
    nf = Fpts.shape[0]
    n = C.shape[0]

    # Expand sparsity pattern for R
    C.data[np.abs(C.data)<1e-16] = 0
    C.eliminate_zeros()

    Lff = -C[Fpts,:][:,Fpts]
    pts = np.arange(0,nf)
    Lff[pts,pts] = 0.0
    Lff.eliminate_zeros()
    Afc = C[Fpts,:][:,Cpts]

    # Form Neuman approximation to Aff^{-1}
    W = eye(nf,format='csr')
    if degree >= 1:
        W += Lff
    if degree >= 2:
        W += Lff*Lff
    if degree >= 3:
        W += Lff*Lff*Lff
    if degree == 4:
        W += Lff*Lff*Lff*Lff
    if degree > 4:
        raise ValueError("Only sparsity degree 0-4 supported.")

    # Multiply Acf by approximation to Aff^{-1}
    W = -W*Afc

    # Get sizes and permutation matrix from [F, C] block
    # ordering to natural matrix ordering.
    permute = eye(n,format='csr')
    permute.indices = np.concatenate((Fpts,Cpts))
    permute = permute.T

    # Form R = [P, I], reorder and return
    P = vstack([W, eye(nc, format='csr')])
    P = csr_matrix(permute * P)
    return P


def local_AIR(A, splitting, theta=0.1, norm='abs', degree=1, use_gmres=False,
                                  maxiter=10, precondition=True, cost=[0]):
    """ Compute approximate ideal restriction by setting RA = 0, within the
    sparsity pattern of R. Sparsity pattern of R for the ith row (i.e. ith
    C-point) is the set of all strongly connected F-points, or the max_row
    *most* strongly connected F-points.

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR or BSR format
    splitting : array
        C/F splitting stored in an array of length N
    theta : float, default 0.1
        Solve local system for each row of R for all values
            |A_ij| >= 0.1 * max_{i!=k} |A_ik|
    degree : int, default 1
        Expand sparsity pattern for R by considering strongly connected
        neighbors within 'degree' of a given node. Only supports degree 1 and 2.
    use_gmres : bool
        Solve local linear system for each row of R using GMRES
    maxiter : int
        Maximum number of GMRES iterations
    precondition : bool
        Diagonally precondition GMRES

    Returns
    -------
    Approximate ideal restriction, R, in same sparse format as A.

    Notes
    -----
    - This was the original idea for approximating ideal restriction. In practice,
      however, a Neumann approximation is typically used.
    - Supports block bsr matrices as well.
    """

    # Get SOC matrix containing neighborhood to be included in local solve
    if isspmatrix_bsr(A):
        C = classical_strength_of_connection(A=A, theta=theta, block='amalgamate', norm=norm)
        blocksize = A.blocksize[0]
    elif isspmatrix_csr(A):
        blocksize = 1
        C = classical_strength_of_connection(A=A, theta=theta, block=None, norm=norm)
    else:
        try:
            A = A.tocsr()
            warn("Implicit conversion of A to csr", SparseEfficiencyWarning)
            C = classical_strength_of_connection(A=A, theta=theta, block=None, norm=norm)
            blocksize = 1
        except:
            raise TypeError("Invalid matrix type, must be CSR or BSR.")

    Cpts = np.array(np.where(splitting == 1)[0], dtype='int32')
    nc = Cpts.shape[0]
    n = C.shape[0]

    R_rowptr = np.empty(nc+1, dtype='int32')
    amg_core.approx_ideal_restriction_pass1(R_rowptr, C.indptr, C.indices,
                                            Cpts, splitting, degree)       

    # Build restriction operator
    nnz = R_rowptr[-1]
    R_colinds = np.zeros(nnz, dtype='int32')

    # Block matrix
    if isspmatrix_bsr(A):
        R_data = np.zeros(nnz*blocksize*blocksize, dtype=A.dtype)
        amg_core.block_approx_ideal_restriction_pass2(R_rowptr, R_colinds, R_data, A.indptr,
                                                      A.indices, A.data.ravel(), C.indptr,
                                                      C.indices, C.data, Cpts, splitting,
                                                      blocksize, degree, use_gmres, maxiter,
                                                      precondition)
        R = bsr_matrix((R_data.reshape(nnz,blocksize,blocksize), R_colinds, R_rowptr),
                        blocksize=[blocksize,blocksize], shape=[nc*blocksize,A.shape[0]])
    # Not block matrix
    else:
        R_data = np.zeros(nnz, dtype=A.dtype)
        amg_core.approx_ideal_restriction_pass2(R_rowptr, R_colinds, R_data, A.indptr,
                                                A.indices, A.data, C.indptr, C.indices,
                                                C.data, Cpts, splitting, degree, use_gmres, maxiter,
                                                precondition)            
        R = csr_matrix((R_data, R_colinds, R_rowptr), shape=[nc,A.shape[0]])

    R.eliminate_zeros()
    return R
